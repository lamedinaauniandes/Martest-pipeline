import os
import sys
import json
import logging
import argparse
import configparser
from pathlib import Path
from datetime import datetime

import gym
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools_testing_rl.path_logs import get_path_files


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--autor', type=str, required=True)
parser.add_argument('--values_agt', type=bool, default=False)
parser.add_argument('--values_trnng', type=bool, default=False)
parser.add_argument('--combine', type=bool, default=False)
parser.add_argument('--gymenv', type=str)
parser.add_argument('--abstract_level', type=int, default=4)
args = parser.parse_args()

assert args.gymenv!=None ,'we need environment argument --gymenv , ex: LunarLander-v2, CartPole-v1 , etc'

# Load config
config = configparser.ConfigParser()
base_dir = Path.cwd()
config.read(base_dir / 'config.ini')

# Logging
timestamp = datetime.now().strftime('%S%M%H%d%m%Y')
log_file = os.path.join(
    config.get('general', 'abstract_states_logs'),
    f'absract_states_{args.autor}_{timestamp}.log'
)
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)


class AbstractStateCreator:
    def __init__(self, autor, create_agent, create_training, combine, abstract_level):
        self.autor = autor
        self.abstract_level = abstract_level

        if combine:
            type_file = 'combine_'
        elif create_training:
            type_file = 'trnng_'
        else:
            type_file = ''

        path_q_values, file_time = get_path_files(
            autor=self.autor,
            path_dir=config.get(autor, 'path_q_values_tables_dir'),
            type_file=type_file,
            name_file='q_value',
            extension='csv',
            abstract_level=self.abstract_level
        )
        logging.debug(f'Q-values file: {path_q_values} | Timestamp: {file_time}')

        path_output = os.path.join(
            config.get(autor, 'path_abstract_states_dir'),
            f'abstract_states_{self.abstract_level}_{type_file}{autor}_{file_time}.json'
        )

        # Create gym environment to determine actions
        self.set_actions = list(range(gym.make(args.gymenv).action_space.n))

        # Process
        self.create(path_q_values, path_output)

    def abstract_function(self, state, action, q_values):
        value = q_values.loc[state][f'action {action}']
        return np.floor(float(value) / self.abstract_level)

    def abstract_states(self, q_values):
        dic_abstract_states = {}
        logging.debug(f'cant states: {len(q_values.index)}')
        num_state = 0
        for state in q_values.index:
            num_state += 1
            logging.debug(f'Processing state: {num_state}')
            done = False
            for rep in dic_abstract_states.keys():
                if all(
                    self.abstract_function(state, a, q_values) == self.abstract_function(rep, a, q_values)
                    for a in self.set_actions
                ):
                    dic_abstract_states[rep].append(state)
                    done = True
                    break
            if not done:
                dic_abstract_states[state] = []

        return dic_abstract_states

    def create(self, input_path, output_path):
        logging.debug('Starting abstract state creation...')
        q_values = pd.read_csv(input_path, sep=';', index_col=0)
        abstract_dict = self.abstract_states(q_values)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(abstract_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    AbstractStateCreator(
        autor=args.autor,
        create_agent=args.values_agt,
        create_training=args.values_trnng,
        combine=args.combine,
        abstract_level=args.abstract_level
    )
