import os
import sys
import logging
import configparser
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar path del módulo tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools_testing_rl.path_logs import get_path_log_files, get_path_files

# Argumentos del script
parser = argparse.ArgumentParser()
parser.add_argument('--autor', type=str)
parser.add_argument('--values_agt', type=bool, default=False)
parser.add_argument('--values_trnng', type=bool, default=False)
parser.add_argument('--combine', type=bool, default=False)
parser.add_argument('--abstract_level', type=int, default=4)
args = parser.parse_args()

# Configuración
base_dir = Path(os.getcwd()).resolve()
config = configparser.ConfigParser()
config.read(os.path.join(base_dir, 'config.ini'))

# Configuración de logs
strf_now = datetime.now().strftime('%S%M%H%d%m%Y')
path_logs = os.path.join(
    config.get('general', 'abstract_episodes_logs'),
    f'abstract_episodes__{args.autor}_{strf_now}.log',
)
logging.basicConfig(
    filename=path_logs,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

class AbstractEpisodeCreator:

    def __init__(self, autor: str, agt: bool, trnng: bool, combine: bool, level: int):
        self.autor = autor
        self.abstract_level = level
        self.state_pattern = config.get(autor,'regex_pttrn_logs_state')
        logging.debug(f'pattern state: {self.state_pattern}')

        # Definir tipo de archivo según flags
        type_file = ''
        if trnng: type_file = 'trnng_'
        elif combine: type_file = 'combine_'

        self.path_logs_agent_dir = os.path.join(os.getcwd(), config.get(autor, 'path_logs_agent'))
        self.path_abstract_states_dir = os.path.join(os.getcwd(), config.get(autor, 'path_abstract_states_dir'))

        # Obtener paths
        self.path_logs_agent_file, self.date_str_format_agt = get_path_log_files(
            self.path_logs_agent_dir, type_file, 'log', 'log'
        )
        self.path_abstract_states_file, _ = get_path_files(
            autor, self.path_abstract_states_dir, type_file, 'abstract_states', 'json', level
        )

        # Definir path de salida
        self.path_abstract_output = os.path.join(
            config.get(autor, 'path_abstract_episodes_dir'),
            f'abstract_episodes_{level}_{type_file}{autor}_{self.date_str_format_agt}.csv'
        )

        logging.debug(f'Archivo de salida: {self.path_abstract_output}')
        logging.debug(f'Archivo de salida: {self.path_abstract_output}')

        self.create_file()

    def create_file(self):
        logging.debug("Iniciando creación de episodios abstractos")

        # Leer log original
        
        # Reemplazar estados concretos por abstractos
        with open(self.path_abstract_states_file, 'r') as f:
            abstract_states = json.load(f)

        logging.debug(f"num lines log file: {len( open(self.path_logs_agent_file, 'r').readlines() )}")

        with open(self.path_logs_agent_file, 'r') as f:
            line = f.readline()
            log_content = ''
            regex_pattern_state = re.compile(rf'({self.state_pattern})')
            num_line = 0
            while line:
                num_line += 1
                logging.debug(f'creating abstract lines: {num_line}') 
                for match in regex_pattern_state.finditer(line):
                    state = match.groups()[0]
                    ## buscar a que llave pertenece el estado para reemplazarlo 
                    for idx,(key,values) in enumerate(abstract_states.items(),1):
                        if state == key: 
                            break
                        elif state in values: 
                            line = line.replace(state,key)
                            break      
                log_content += f'{line}\n'
                line = f.readline()

        # Guardar log transformado
        transformed_log_path = self.path_abstract_output.replace('.csv', '.log')
        with open(transformed_log_path, 'w') as f:
            f.write(log_content)
            

        logging.debug('Extrayendo información para CSV...')
        abstract_data = {}

        with open(transformed_log_path, 'r') as f:
            for idx, line in enumerate(f, 1):
                logging.debug(f'id line abstract episodes: {idx}')
                match = re.search(r'\d: (.+True) .*reward: +(-?[\d\.]+)', line)
                if match:
                    abstract_episode, reward = match.group(1), float(match.group(2))
                    if abstract_episode not in abstract_data:
                        abstract_data[abstract_episode] = {'rewards': [], 'count': 0}
                    abstract_data[abstract_episode]['rewards'].append(reward)
                    abstract_data[abstract_episode]['count'] += 1
                logging.debug(f'Procesada línea {idx}')

        # Crear DataFrame final
        logging.debug('creating abstract episodes table ... ')
        df = pd.DataFrame([
            {
                'abstract_episode': k,
                'reward_mean': np.mean(v['rewards']),
                'prob_fault': np.nan,
                'num_episodes': v['count']
            } for k, v in abstract_data.items()
        ])

        df.to_csv(self.path_abstract_output, sep=';', index=False)
        logging.info(f"Archivo creado: {self.path_abstract_output}")

if __name__ == '__main__':
    logging.debug('Llamando a AbstractEpisodeCreator...')
    AbstractEpisodeCreator(
        autor=args.autor,
        agt=args.values_agt,
        trnng=args.values_trnng,
        combine=args.combine,
        level=args.abstract_level
    )
