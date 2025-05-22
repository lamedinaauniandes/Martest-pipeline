import os 
from pathlib import Path
import configparser
import logging
import argparse
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score, 
    confusion_matrix
)
import json
import re
import joblib

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools_testing_rl.path_logs import (
    get_path_files,
    get_path_log_files,
)


base_dir = Path(os.getcwd()).resolve()
config = configparser.ConfigParser()
config.read(os.path.join(base_dir,'config.ini'))
strf_now = datetime.now().strftime('%S%M%H%d%m%Y')

parser = argparse.ArgumentParser()
parser.add_argument('--autor',type=str)
parser.add_argument('--values_agt',type=bool,default=False)
parser.add_argument('--values_trnng',type=bool,default=False)
parser.add_argument('--combine',type=bool,default=False)
parser.add_argument('--abstract_level',type=int,default=4)


args = parser.parse_args()

logging.basicConfig(
    filename=os.path.join(
        base_dir,
        config.get('general','random_forest_logs'),
        f'random_forest_{args.autor}_{strf_now}.log'
        ),
    filemode = 'a',
    format = '%(asctime)s-%(levelname)s-%(message)s',
    level=logging.DEBUG, 
)

class random_forest_model: 

    def __init__(self,autor,values_agt,values_trnng,combine,abstract_level):
        self.autor = autor
        self.values_agt = values_agt
        self.values_trnng = values_trnng 
        self.combine = combine
        self.abstract_level = abstract_level
            
        if values_agt: 
            type_file = ''
        if values_trnng: 
            type_file = 'trnng_'
        if combine: 
            type_file = 'combine_'

        ##### get binary episodes file and get df
        path_binary_episodes, file_time = get_path_files(
            autor=self.autor,
            path_dir=config.get(autor,'path_binary_episodes_dir'),
            type_file=type_file,
            name_file='binary_episodes',
            extension='csv',
            abstract_level=self.abstract_level
        )

        #### define path file random forest model, '4../models'
        name_file_model = f'random_forest_model_{abstract_level}_{type_file}{autor}_{file_time}.plk'
        path_model_rndf = os.path.join(
            base_dir, 
            config.get(autor,'path_random_forest_models_dir'),
            name_file_model,
        )
        
        logging.debug(f'path to binary episodes file: {path_binary_episodes}')
        print(f'path to binary episodes file: {path_binary_episodes}')
        logging.debug(f'path to model: {path_model_rndf}')
        print(f'path to model: {path_model_rndf}')


        ### training model 
        self.training_model_random_forest(
            path_binary_episodes=path_binary_episodes,
            path_model_rndf=path_model_rndf
            )
        
        #### get to general table
        name_file_abstract_episodes = f'abstract_episodes_{abstract_level}_{type_file}{autor}_{file_time}.csv'
        path_file_abstract_episodes= os.path.join(
            base_dir, 
            config.get(autor,'path_abstract_episodes_dir'),
            name_file_abstract_episodes
        )
        ### get to abstract states(class) .json
        name_file_abstract_states = f'abstract_states_{abstract_level}_{type_file}{autor}_{file_time}.json'
        path_abstract_states_dir = config.get(autor,'path_abstract_states_dir')
        path_file_abstract_states = os.path.join(
            base_dir,
            path_abstract_states_dir,
            name_file_abstract_states
        )
        
        ### path to save new table wit prob fault and with abstract episodes with 'wi' tokens
        ### path to 4.../datasets
        path_to_abstractprobabilityepisodes = os.path.join(
            base_dir, 
            config.get(autor,'path_abstract_datasets'),
            name_file_abstract_episodes, 
        )
        
        self.construct_probability_table(
            path_model_saved = path_model_rndf,
            path_binary_episodes = path_binary_episodes,
            path_file_abstract_episodes = path_file_abstract_episodes,
            path_file_abstract_states = path_file_abstract_states, 
            path_to_abstractprobabilityepisodes = path_to_abstractprobabilityepisodes 
        )
    
    

    def construct_probability_table(
    self,
    path_model_saved,
    path_binary_episodes,
    path_file_abstract_episodes,
    path_file_abstract_states,
    path_to_abstractprobabilityepisodes,
    ):
        
        logging.debug(f'path_model_saved: {path_model_saved}')
        logging.debug(f'path_binary_episodes: {path_binary_episodes}')
        logging.debug(f'path_file_abstract_episodes: {path_file_abstract_episodes}')

        model,columns = joblib.load(path_model_saved)
        df_binary = pd.read_csv(path_binary_episodes,sep=';')
        df_abstract_episodes = pd.read_csv(path_file_abstract_episodes,sep=';')

        df_binary = df_binary[columns]
        arr_prob = model.predict_proba(df_binary)
        logging.debug(arr_prob)
        logging.debug(arr_prob.shape)
        
        #####
        if arr_prob.shape[1] == 2: 
            arr_prob = arr_prob[:,1]

        #######################
        ### agregate prob_fault column
        #######################
        df_abstract_episodes['prob_fault'] = arr_prob
        

        ########## 
        ### craeating parse_abstract_states column for tokenization 'Wo'
        ##########
   
        f = open(path_file_abstract_states,'r')
        abstract_states = json.load(f)
        vocabulary_list = list(abstract_states.keys()) + [1,0]
        vocabulario = {abstract_class: f'w{i+1}' if (abstract_class != 1 and abstract_class != 0 ) else str(abstract_class) for i,abstract_class in enumerate(vocabulary_list)}

        def parse_vocabulary(row):
            text = row['abstract_episode']
            text = re.sub(r' \d\.\d ','',text)
            text = re.sub(r'False|True','',text)
            for key,item in vocabulario.items():
                text = text.replace(str(key),str(item))
            text = re.sub(r' *-?\d+\.\d+(?:e-\d+)? ','',text)
            text = re.sub(' 100 ','',text)
            return text

        df_abstract_episodes['parse_abstract_states'] = df_abstract_episodes.apply(parse_vocabulary,axis=1)


        df_abstract_episodes.to_csv(path_to_abstractprobabilityepisodes,sep=';')

        return df_abstract_episodes

    def training_model_random_forest(
            self, 
            path_binary_episodes,
            path_model_rndf,
    ):
        print(f'training model for :{path_binary_episodes}')
        logging.debug(f'training model for: {path_binary_episodes}')

        df = pd.read_csv(path_binary_episodes,sep=';')

        ### split binary file dataset 
        x_train,x_test,y_train,y_test = train_test_split(df.drop('fail',axis=1),df['fail'],test_size=0.5)
        #### random forest model
        rnd_clf = RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
        )

        #### trainning modle
        print(f'training model ...')
        logging.debug(f'training model ...')
        rnd_clf.fit(x_train,y_train)
        y_pred = rnd_clf.predict(x_test)

        logging.debug(f"""
            {rnd_clf.__class__.__name__}
            len y_test: {                                                                                                                                                                                                                                                                                                                       y_test.shape}, shape y_pred: {y_pred.shape}
            accuracy: {accuracy_score(y_test,y_pred)}
            presicion score: {precision_score(y_test,y_pred)}
            recall score: {recall_score(y_test,y_pred)}
            confusion matrix:\n 
                {confusion_matrix(y_test,y_pred)}
        """)

        #### save model
        print(f'saving model in: {path_model_rndf}')
        logging.debug(f'saving model in: {path_model_rndf}')

        joblib.dump((rnd_clf,x_test.columns.to_list()),path_model_rndf)

        return rnd_clf,x_train,x_test,y_train,y_test


if __name__=='__main__': 
    
    random_forest_model(
        autor=args.autor,
        values_agt=args.values_agt, 
        values_trnng=args.values_trnng, 
        combine=args.combine,
        abstract_level = args.abstract_level,
    )
        







