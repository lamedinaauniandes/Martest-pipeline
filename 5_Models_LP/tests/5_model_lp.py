import argparse 
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--autor',type=str)
parser.add_argument('--values_agt',type=bool,default=False)
parser.add_argument('--values_trnng',type=bool,default=False)
parser.add_argument('--num_gpu',type=int,default=2)
parser.add_argument('--combine')
args = parser.parse_args()
# print(args.num_gpu,type(args.num_gpu))
number_gpu = args.num_gpu
tf.config.experimental.set_visible_devices(
    tf.config.experimental.list_physical_devices('GPU')[number_gpu],
    'GPU'
)
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding
from tensorflow.keras.models import Sequential
import configparser 
import os 
from pathlib import Path 
import pandas as pd
import json
import re 
import numpy as np
import logging 
from datetime import datetime

base_dir = Path(os.getcwd()).resolve()
config = configparser.ConfigParser()
print(os.path.join(base_dir,'config.ini'))
config.read(os.path.join(base_dir,'config.ini'))
strf_now = datetime.now().strftime('%M%H%d%m%Y')

logging.basicConfig(
    filename=os.path.join(
        base_dir,
        config.get('general','models_lp_logs'),
        f'models_lp_{args.autor}_{strf_now}.log'
        ),
    filemode = 'a',
    format = '%(asctime)s-%(levelname)s-%(message)s',
    level=logging.DEBUG, 
)


class model_lp:

    def __init__(self,
        autor: str, 
        values_agt: bool, 
        values_trnng: bool,
        combine: bool,
    ):
        if values_agt: 
            type_file = ''
        if values_trnng: 
            type_file = 'trnng_'
        if combine: 
            type_file = 'combine_'

        str_ftime = config.get(autor,'strftime_data_file')

        name_file_abstract_datasets = f"abstract_episodes_{type_file}{autor}_{str_ftime}.csv"
        path_abstract_datasets_dir = config.get(autor,'path_abstract_datasets')
        path_file_abstract_datasets = os.path.join(base_dir,path_abstract_datasets_dir,name_file_abstract_datasets)

        name_file_abstract_states = f"abstract_states_{type_file}{autor}_{str_ftime}.json"
        path_abstract_states_dir = config.get(autor,'path_abstract_states_dir')
        path_file_abstract_states = os.path.join(base_dir,path_abstract_states_dir,name_file_abstract_states)
         
        name_file_LSTM_model = f'LSTM_model_{type_file}{autor}_{str_ftime}'
        
        path_file_LSTM_model = os.path.join(
            base_dir,
            config.get(autor,'path_models_LSTM'),
            name_file_LSTM_model
        )

        logging.debug(f'path LSTM model: {path_file_LSTM_model}')

        f = open(path_file_abstract_states,'r')
        dict_abstract_states = json.load(f)
        
        df_abstract_states = pd.read_csv(path_file_abstract_datasets,sep=';')

        actions_set = ["1","0"]  # cambiar el conjunto de estados para hacerlo configurable!!!!
        vocabulary = list(dict_abstract_states.keys()) + actions_set

        tokenizer = self.tokenization(vocabulary,actions_set,dict_abstract_states)
        predictors,labels,max_sequence_length = self.get_predictors_labels(tokenizer,df_abstract_states)
        
        #### train & save the model
        self.train_LSTM_model(
            total_words= len(vocabulary)+1,
            max_sequence_length=max_sequence_length - 1, 
            predictors=predictors,
            labels=labels,
            epochs=10,    #### hacerlo modificable??? yo creo que si
            path_save_model_lstm=path_file_LSTM_model,
        )

    def tokenization(self,
        vocabulary: list,
        actions_set: list, 
        dict_abstract_states: dict,
        ):
        logging.debug(f'building tokenizer ...')
        tokenizer = Tokenizer()

        tokenizer.word_index = {
            f'w{i+1}' if class_abstract in dict_abstract_states.keys() else class_abstract : i + 1 for i,class_abstract in enumerate(vocabulary)
       }

        return tokenizer        

    def get_predictors_labels(self,tokenizer,df_abstrac_states):
            logging.debug(f'building predictors and labels ...')
            input_sequences = []

            logging.debug(f'shape df_abstract_states: {df_abstrac_states.shape}')
            for index,row in df_abstrac_states.iterrows():
                token_list = tokenizer.texts_to_sequences([row['parse_abstract_states']])[0]
                #### me preocupa este doble for!!! 
                for i in range(1,len(token_list)):
                    logging.debug(f'index: {index},index token list: {i}') 
                    partial_sequence = token_list[:i+1]
                    input_sequences.append(partial_sequence)
            max_sequence_length = max([len(sequence) for sequence in input_sequences])
            input_sequences = np.array(
                pad_sequences(input_sequences,maxlen=max_sequence_length,padding='pre')
            )
            predictors = input_sequences[:,:-1]
            labels = input_sequences[:,-1]
            labels = to_categorical(
                labels, 
                num_classes= len(tokenizer.word_index) + 1 
            )
            logging.debug(f'end predictors and labels')

            return predictors,labels,max_sequence_length
        
    def train_LSTM_model(self,
        total_words,
        max_sequence_length,
        predictors,
        labels,
        epochs,
        path_save_model_lstm,
        ):
        logging.debug(f'training model ...')
        model = Sequential()
        model.add(Embedding(total_words,50,input_length=max_sequence_length))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(total_words,activation='softmax'))
        logging.debug(f'sumary model: {model.summary()}')
        model.compile(loss='categorical_crossentropy',optimizer='adam')
        logging.debug('training ...')
        print('debug1')
        model.fit(predictors,labels,epochs=epochs,verbose=1) # verbose = 1 .log
        print('debug2')
        logging.debug(f'saving model: {path_save_model_lstm}')
        model.save(path_save_model_lstm)

if __name__ == '__main__':

    model_lp(
        args.autor,
        args.values_agt,
        args.values_trnng,
        args.combine,
    )
    
