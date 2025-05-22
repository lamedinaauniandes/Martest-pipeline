import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging.config
from pathlib import Path
import json
import logging
import configparser
from datetime import datetime
from tools_testing_rl.path_logs import (
    get_path_files,
    get_path_log_files,
)
import pandas as pd
import re
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--autor',type=str)
parser.add_argument('--values_agt',type=bool,default=False)
parser.add_argument('--values_trnng',type=bool,default=False)
parser.add_argument('--combine',type=bool,default=False)
parser.add_argument('--abstract_level',type=int,default=4)

args = parser.parse_args()

base_dir = Path(os.getcwd()).resolve()
config = configparser.ConfigParser()
path_config = os.path.join(base_dir,'config.ini')
config.read(path_config)

strf_now = datetime.now().strftime('%S%M%H%d%m%Y')
path_logs = os.path.join(
    config.get('general','abstract_binary_logs'),
    f'probability_fault_episodes_{args.autor}_{strf_now}.log',
)

logging.basicConfig(
    filename=path_logs,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)


class crt_binary_table:
    
    def __init__(self,autor:str,values_agt:bool,values_trnng:bool,combine:bool,abstract_level:int):
        self.autor = autor
        
        if values_agt:
            type_file = ''
        if values_trnng: 
            type_file = 'trnng_'
        if combine: 
            type_file = 'combine_'
        
        ### get abstract episodes table path 
        self.path_abstracepi_file,self.date_str_format = get_path_files(
            self.autor,
            os.path.join(base_dir,config.get(self.autor,'path_abstract_episodes_dir')),
            type_file,
            'abstract_episodes',
            'csv',
            abstract_level,
        )
        #### get abstract states path
        self.path_abstractstates_file,_ = get_path_files(
            self.autor, 
            os.path.join(base_dir,config.get(self.autor,'path_abstract_states_dir')),
            type_file,
            'abstract_states',
            'json',
            abstract_level,
        )
        ### define path file of binary episodes
        path_binary_episodes = os.path.join(
            base_dir,
            config.get(autor,'path_binary_episodes_dir'),
            f'binary_episodes_{abstract_level}_{type_file}{self.autor}_{self.date_str_format}.csv'
        )
        #### build binary table
        self.build_binaryabstractepisodes_table(
            self.path_abstracepi_file,
            self.path_abstractstates_file,
            path_binary_episodes,
            config.get(self.autor,'regex_pttrn_logs_state')
        )
    
        
        
    def build_binaryabstractepisodes_table(self,
    path_abstrcteps: str,
    path_abstractstat: str,
    path_binary_episodes: str, 
    pattern: str,
    ):
        logging.debug(f'build binary table, path abstract episode: {path_abstrcteps}')
        
        def fail_episode(row):
            return 1 if row['reward_mean'] < 200 else 0

        
        abstract_states_file = open(path_abstractstat,'r')
        abstract_states = json.load(abstract_states_file)

        df_abstract_episodes = pd.read_csv(path_abstrcteps,sep=';')

        abstract_episodes_keys = list(abstract_states.keys())
        df_table_binary_episodes = pd.DataFrame(columns=(abstract_episodes_keys) + ['fail']) 
        
        df_abstract_episodes['fail'] = df_abstract_episodes.apply(fail_episode,axis=1)
        

        for index,row in df_abstract_episodes.iterrows():
            logging.debug(f'building binary episode: {index}')
            abstract_episode_text = row['abstract_episode']
            # logging.debug(f'episdio abstracto: \n {abstract_episode_text}')
            abstract_episodes_row = re.finditer(pattern,abstract_episode_text)
            abstract_episdoes_list = list(set([state.group(0) for state in abstract_episodes_row]))
            # logging.debug(f'lista de episodios:\n {abstract_episdoes_list}')
            binary_row = {state: [1] if state in abstract_episdoes_list else [0] for state in abstract_episodes_keys}
            binary_row['fail'] = row['fail']
            df_new = pd.DataFrame(binary_row)
            df_table_binary_episodes = pd.concat([df_table_binary_episodes,df_new],ignore_index=True)

        df_table_binary_episodes.to_csv(path_binary_episodes,sep=';')
        
    

if __name__=='__main__':
    crt_binary_table(
        autor=args.autor,
        values_agt=args.values_agt, 
        values_trnng=args.values_trnng, 
        combine=args.combine,
        abstract_level=args.abstract_level,
    )



