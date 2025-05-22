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
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import GPT2Config, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import PreTrainedTokenizerFast
from conditional_transformer import CondDataset, CondGPT2

# Agregar path del módulo tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools_testing_rl.path_logs import get_path_log_files, get_path_files,get_path_model

parser = argparse.ArgumentParser()
parser.add_argument('--autor',type=str)
parser.add_argument('--values_agt',type=bool,default=False)
parser.add_argument('--values_trnng',type=bool,default=False)
parser.add_argument('--combine')
parser.add_argument('--abstract_level', type=int, default=4)
parser.add_argument('--num_gpu',type=int)
# parser.add_argument('--gymenv', type=str)   #LunarLander-v2, CartPole-v1
args = parser.parse_args()

assert args.num_gpu !=None, 'Please, indicate number of gpu.'

base_dir = Path(os.getcwd()).resolve()
config = configparser.ConfigParser()
# print(os.path.join(base_dir,'config.ini'))
config.read(os.path.join(base_dir,'config.ini'))
strf_now = datetime.now().strftime('%S%M%H%d%m%Y')

logging.basicConfig(
    filename=os.path.join(
        base_dir,
        config.get('general','models_lp_logs'),
        f'generate_vulnerabilities_{args.autor}_{strf_now}.log'
        ),
    filemode = 'a',
    format = '%(asctime)s-%(levelname)s-%(message)s',
    level=logging.DEBUG, 
)


class generate_vulnerabilities: 

    def __init__(self,
                 autor, 
                 values_agt, 
                 values_trnng, 
                 combine, 
                 abstract_level,
                 ):

        number_gpu = args.num_gpu
        self.device = torch.device(f"cuda:{number_gpu}" if torch.cuda.is_available() else "cpu")
        self.abstract_level = abstract_level
        self.autor = autor
        self.vulnerabilitie_max_length = config.getint(self.autor,'vulnerabilities_max_length')

        if values_agt: type_file = ''
        elif values_trnng: type_file = 'trnng_'
        elif combine: type_file = 'combine_'

        ### get model
        self.path_model_dir = config.get(autor,'path_to_model_dir')
        self.path_to_model,date_file = get_path_model(
            autor = autor, 
            path_dir=self.path_model_dir,
            type_file=type_file,
            name_file='model_condtransformer',
            abstract_level=self.abstract_level,
        )
        self.path_to_model = os.path.join(
            base_dir, 
            self.path_to_model
        )

        ##### define pronostics vulnerabilities path
        name_file_pronostics = f'pronostic_vulnerabilities_{self.abstract_level}_{type_file}{self.autor}_{date_file}.log'
        self.path_file_pronostic = os.path.join(
            base_dir, 
            config.get(self.autor,'path_to_pronostic_dir'),
            name_file_pronostics
        )

        self.charge_model()
        
        self.generate()

        return 
    

    def generate(self): 
        number_gpu = args.num_gpu
        self.model_charged.eval()
        device = torch.device(f"cuda:{number_gpu}" if torch.cuda.is_available() else "cpu")
        self.model_charged.to(device)

        # 1) Prompt y tokenización inicial
        prompt = "[BOS]"
        inputs = self.fast_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model_charged.device)
        attention_mask = inputs["attention_mask"].to(self.model_charged.device)

        # 2) Tensor de condición (batch_size=1)
        reward_tensors     = [torch.tensor([i/200], device=self.model_charged.device) for i in range(100,201,10) ] # normalizado
        prob_fault_tensors = [torch.tensor([i/10], device=self.model_charged.device) for i in range(10,11)]

        file = open(self.path_file_pronostic , 'w', encoding='utf-8')
        num_test_case = 0
        for reward_tensor in reward_tensors: 
            for prob_fault_tensor in prob_fault_tensors: 
                num_test_case += 1
            
                outputs = self.model_charged.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    reward=reward_tensor,       
                    prob_fault=prob_fault_tensor,
                    max_length=int(self.vulnerabilitie_max_length),
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=20,
                )

                log_raw = f'reward: {reward_tensor}, prob_fault: {prob_fault_tensor}\n'
                for i, out in enumerate(outputs):
                    text = self.fast_tokenizer.decode(out, skip_special_tokens=True)
                    log_raw += f"test case {num_test_case}: {text}\n"
                    print(f"test case {num_test_case}:", text)

                print(f'saving pronostic in: {self.path_file_pronostic}')
                file.write(log_raw)

        


    def charge_model(self): 
        print('recharge model...')
        save_dir = self.path_to_model
        config = GPT2Config.from_pretrained(save_dir)
        self.model_charged = CondGPT2.from_pretrained(save_dir, config=config)
        self.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)

    
if __name__=='__main__': 

    generate_vulnerabilities(
        autor=args.autor, 
        values_agt=args.values_agt, 
        values_trnng=args.values_trnng,
        combine=args.combine, 
        abstract_level=args.abstract_level, 
    )
