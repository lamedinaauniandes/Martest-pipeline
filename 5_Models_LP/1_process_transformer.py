import os 
import sys 
import argparse 
import configparser 
from pathlib import Path 
import pandas as pd
import json
import re 
import logging 
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
# from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    Trainer, 
    TrainingArguments, 
    default_data_collator,
    PreTrainedTokenizerFast,
)
from conditional_transformer import CondDataset, CondGPT2


# Agregar path del módulo tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools_testing_rl.path_logs import get_path_log_files, get_path_files


parser = argparse.ArgumentParser()
parser.add_argument('--autor',type=str)
parser.add_argument('--values_agt',type=bool,default=False)
parser.add_argument('--values_trnng',type=bool,default=False)
parser.add_argument('--combine')
parser.add_argument('--abstract_level', type=int, default=4)
parser.add_argument('--num_gpu',type=int)
parser.add_argument('--gymenv', type=str)   #LunarLander-v2, CartPole-v1
args = parser.parse_args()

assert args.num_gpu !=None, 'Please, indicate number of gpu.'
assert args.gymenv !=None, 'Please, indicate environment LunarLander-v2, CartPole-v1 .'

base_dir = Path(os.getcwd()).resolve()
config = configparser.ConfigParser()
# print(os.path.join(base_dir,'config.ini'))
config.read(os.path.join(base_dir,'config.ini'))
strf_now = datetime.now().strftime('%S%M%H%d%m%Y')

logging.basicConfig(
    filename=os.path.join(
        base_dir,
        config.get('general','models_lp_logs'),
        f'trainning_conditionaltransformer_{args.autor}_{strf_now}.log'
        ),
    filemode = 'a',
    format = '%(asctime)s-%(levelname)s-%(message)s',
    level=logging.DEBUG, 
)


class process_model: 

    def __init__(self,
                 autor, 
                 values_agt, 
                 values_trnng, 
                 combine, 
                 abstract_level,
                 ):
        
        number_gpu = args.num_gpu
        self.device = torch.device(f"cuda:{number_gpu}" if torch.cuda.is_available() else "cpu")
        self.autor = autor
        self.config = config
        self.especial_words = ['1','0','True','[UNK]','[BOS]']
        self.abstract_level = abstract_level
        
        if values_agt: type_file = ''
        elif values_trnng: type_file = 'trnng_'
        elif combine: type_file = 'combine_'

        ### abstract states dictionary 
        self.path_abstract_states_dir = os.path.join(os.getcwd(), config.get(autor, 'path_abstract_states_dir'))
        self.path_abstract_states_file, date_file = get_path_files(
            autor, 
            self.path_abstract_states_dir, 
            type_file, 
            'abstract_states', 
            'json', 
            self.abstract_level,
        )
        file = open(self.path_abstract_states_file,'r')
        self.dict_abstract_states = json.load(file)
        print(f'abstract states path: {self.path_abstract_states_file}')
        logging.debug(f'abstract states path: {self.path_abstract_states_file}')
        ### abstract episodes table
        self.path_abstract_dataset_dir = config.get(autor, 'path_abstract_datasets') 
        self.path_abstract_dataset_file, _ = get_path_files(
            autor, 
            self.path_abstract_dataset_dir, 
            type_file, 
            'abstract_episodes', 
            'csv', 
            self.abstract_level,
        )
        self.df_abstract_states = pd.read_csv(self.path_abstract_dataset_file,sep=';')
        print(f'abstract episodes path: {self.path_abstract_dataset_file}')
        logging.debug(f'abstract episodes path: {self.path_abstract_dataset_file}')
        
        ### define path to save model
        save_model_dir = config.get(self.autor,'path_to_model_dir')
        name = f'model_condtransformer_{self.abstract_level}_{type_file}{self.autor}_{date_file}'
        self.save_model_file = os.path.join(
            base_dir, 
            save_model_dir,
            name
        )
        logging.debug(f'path to save model: {self.save_model_file}')


        ### get dataset for conditional transformer
        self.dataset = self.buil_dataset()

        ### training conditional transformer
        self.training_conditional_transformer()

        
        return 
    
    def training_conditional_transformer(self): 
        logging.debug('training transformer ...')
        print('training transformer ...')

        config = GPT2Config(
            vocab_size=len(self.fast_tokenizer),
            n_embd=self.config.getint(self.autor,'n_embd'),
            n_layer=self.config.getint(self.autor,'num_layer'),
            n_head=self.config.getint(self.autor,'n_head'),
            bos_token_id=self.fast_tokenizer.pad_token_id,
            eos_token_id=self.fast_tokenizer.pad_token_id,
            pad_token_id=self.fast_tokenizer.pad_token_id
        )
        model = CondGPT2(config)
        model = model.to(self.device)


        training_args = TrainingArguments(
            output_dir = self.save_model_file,
            num_train_epochs = self.config.getint(self.autor,'num_train_epochs'),
            per_device_train_batch_size =self.config.getint(self.autor,'batch_size'), 
            save_steps=500, 
            save_total_limit=2, 
        )
        
        trainer = Trainer(
            model = model,
            args= training_args, 
            train_dataset=self.dataset,
            data_collator = default_data_collator
        )

        trainer.train()
        model.save_pretrained(training_args.output_dir)
        self.fast_tokenizer.save_pretrained(training_args.output_dir)
        logging.debug(f'model saved: {training_args.output_dir}')
        print(f'model saved: {training_args.output_dir}')
    
    def buil_dataset(self):
        logging.debug('building dataset ...')
        print('building dataset ...')
        def clean_text(row):
            return '[BOS] '+re.sub(r'\[\[.*?\]\]', '[UNK]', row['parse_abstract_states']) + ' True'
        
        # def scale(row,min_reward,max_reward):
        #     return (row['reward_mean'] - min_reward)/(max_reward - min_reward)

        def scale(row,min_reward,max_reward):
            reward = 200 if row['reward_mean']>200 else row['reward_mean']
            reward = 0 if reward<0 else reward
            return (reward - min_reward)/(max_reward - min_reward)

        #### get and prepare data
        
        texts = self.df_abstract_states.apply(clean_text,axis=1)

        min_reward_scale = config.getint(self.autor,'min_reward_scale')
        max_reward_scale = config.getint(self.autor,'max_reward_scale')
        rewards = self.df_abstract_states.apply(lambda row:scale(row,min_reward_scale,max_reward_scale),axis=1)
        prob_fault = self.df_abstract_states['prob_fault']

        ##### define vocabulary 'tengo que mejorar esta parte gym.action_space ... '
        if args.gymenv == 'LunarLander-v2':
            especial_words = ["1","0","2","3","True",'[UNK]','[BOS]']
        if args.gymenv == 'CartPole-v1':
             especial_words = ["1","0","True",'[UNK]','[BOS]']
        #####
        vocabulary_map = {f'w{i+1}':abstract_class for i,abstract_class in enumerate(self.dict_abstract_states.keys()) }
        vocabulary = list(vocabulary_map.keys()) + especial_words 
        vocab =  {word:idx for idx,word in enumerate(vocabulary)}
        tokenizer = Tokenizer(WordLevel(vocab,unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()


        self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.fast_tokenizer.add_special_tokens({
            'unk_token': '[UNK]',
            'pad_token': '[PAD]',
            'bos_token': '[BOS]'
        })

        encodings = self.fast_tokenizer(
            list(texts),
            padding="max_length",
            truncation=True, 
            max_length=1024, 
            return_tensors='pt'
        )

        ######## PRUEBA
        print('testing dataset')
        ids = [int(t) for t in list(encodings['input_ids'][0])]
        print(tokenizer.decode(ids))
        print(texts[0])

        dataset = CondDataset(encodings,rewards,prob_fault)

        return dataset









    



if __name__=='__main__': 
    
    process_model(
        autor=args.autor, 
        values_agt=args.values_agt, 
        values_trnng=args.values_trnng,
        combine=args.combine, 
        abstract_level=args.abstract_level, 

    )
    




