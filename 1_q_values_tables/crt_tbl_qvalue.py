import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import re
import numpy as np
import pandas as pd
import configparser
from datetime import datetime
import logging
import argparse
from tools_testing_rl.path_logs import get_path_log_files
import gym

# --- Argumentos CLI ---
parser = argparse.ArgumentParser()
parser.add_argument('--autor', type=str, required=True)
parser.add_argument('--values_agt', type=bool, default=False)
parser.add_argument('--values_trnng', type=bool, default=False)
parser.add_argument('--combine', type=bool, default=False)
parser.add_argument('--gymenv', type=str)   #LunarLander-v2, CartPole-v1
args = parser.parse_args()

assert args.gymenv!=None ,'we need environment argument --gymenv , ex: LunarLander-v2, CartPole-v1 , etc'

# --- Configuración ---
config = configparser.ConfigParser()
config.read('config.ini')
strfmt_now = datetime.now().strftime("%S%M%H%d%m%Y")

logging.basicConfig(
    filename=os.path.join(os.getcwd(), config.get('general', 'q_values_logs'), f'q_value_tbl_{args.autor}_{strfmt_now}.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
)

# --- Clase principal ---
class CreateQValueTable:

    def __init__(self, autor, create_q_values_agent, create_q_values_trnng, combine):
        self.autor = autor

        if combine:
            type_file = 'combine_'
        elif create_q_values_trnng:
            type_file = 'trnng_'
        else:
            type_file = ''

        # Paths
        path_logs_dir = config.get(autor, 'path_logs_agent')
        self.path_file_log, self.date_str_format = get_path_log_files(
            path_dir=path_logs_dir,
            type_file=type_file,
            name_file='log',
            extension='log',
        )

        path_q_values_tables_dir = config.get(autor, 'path_q_values_tables_dir')
        self.output_csv_path = os.path.join(
            path_q_values_tables_dir,
            f'q_value_{type_file}{autor}_{self.date_str_format}.csv'
        )

        # Acciones del entorno
        env = gym.make(args.gymenv)
        self.set_actions = list(range(env.action_space.n))
        print(f'Set actions: {self.set_actions}')
        logging.debug(f'Set actions: {self.set_actions}')

        # Iniciar construcción
        self.build_qvalue()

    def build_qvalue(self):
        print('Building Q-values table...')
        logging.debug('Building Q-values table...')

        with open(self.path_file_log, 'r') as f:
            text = f.read()

        pattern_state = config.get(self.autor, 'regex_pttrn_logs_state')
        logging.debug(f'Regex pattern: {pattern_state}')

        # Encontrar todos los estados únicos
        all_states = re.findall(pattern_state, text)
        unique_states = list(set(all_states))  # Puedes ordenarlos si lo necesitas

        logging.debug(f'Found {len(unique_states)} unique states.')

        # Crear estructura del DataFrame
        columns_actions = [f'action {i}' for i in self.set_actions]
        columns_num_actions = [f'num action {i}' for i in self.set_actions]
        columns = columns_actions + columns_num_actions
        q_values_df = pd.DataFrame(0.0, index=unique_states, columns=columns)

        # Expresión para extraer valores y acción
        state_action_pattern = re.compile(rf"({pattern_state}) +(\d) +(-?\d+\.?\d+) *")

        ## lectura por linea
        logging.debug(f"total lines: {len(open(self.path_file_log, 'r').readlines())}")
        num_line = 0
        with open(self.path_file_log, 'r') as file: 
            line = file.readline()
            while line:
                num_line += 1
                logging.debug(f'num_line: {num_line}')
                matches = state_action_pattern.findall(line)
                for index,match in enumerate(matches):
                    state,action,_ = match
                    q_value = sum([float(x[2]) for x in matches[index:]])
                    q_values_df.at[state,f'num action {action}'] += 1
                    q_values_df.at[state, f'action {action}'] += (1/q_values_df.at[state,f'num action {action}'])*(q_value - q_values_df.at[state, f'action {action}'])
                line = file.readline()
        q_values_df.to_csv(self.output_csv_path, sep=';')
        logging.info(f'Q-values table saved to {self.output_csv_path}')


if __name__ == '__main__':
    logging.debug('Starting local execution...')
    CreateQValueTable(
        autor=args.autor,
        create_q_values_agent=args.values_agt,
        create_q_values_trnng=args.values_trnng,
        combine=args.combine,
    )
