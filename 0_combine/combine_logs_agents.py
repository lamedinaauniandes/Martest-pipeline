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
import math
import random
from operator import itemgetter

parser =  argparse.ArgumentParser()
parser.add_argument('--autor',type=str)

args = parser.parse_args()
base_dir = Path(os.getcwd()).resolve()

print(f'base dir: {base_dir}')

config = configparser.ConfigParser()
config.read('config.ini')
strfmt_now = datetime.now().strftime('%S%M%H%d%m%Y')

logging.basicConfig(
    filename=os.path.join(os.getcwd(),config.get('general','combine_logs'),f'combine_{args.autor}_{strfmt_now}.log'),
    filemode='a',
    format='%(asctime)s-%(levelname)s-%(message)s',
    level=logging.DEBUG,
)

class Combine_logs:

    def __init__(self,
                 autor:str,
                 ):

        self.autor = autor

        ## get path file log
        path_logs_dir = config.get(autor,'path_logs_agent')
        self.path_file_log,self.date_str_format = get_path_log_files(
                path_dir = path_logs_dir,
                type_file = '',
                name_file = 'log',
                extension = 'log',
        )
        logging.debug(f'path to file log: {self.path_file_log}')
        print(f'path to file log: {self.path_file_log}')

        ## get path file log trnng
        self.path_file_log_trnng,_ = get_path_log_files(
                path_dir = path_logs_dir,
                type_file = 'trnng_',
                name_file = 'log',
                extension = 'log',
        )
        logging.debug(f'path to trainning file logs: {self.path_file_log_trnng}')
        print(f'path to trainning file logs: {self.path_file_log_trnng}')

        ## define path file log combine
        name_file_logs_combine = f'log_combine_{self.date_str_format}.log'
        self.path_file_log_combine = os.path.join(
            base_dir,
            path_logs_dir,
            name_file_logs_combine
        )
        logging.debug(f'path to file logs combine: {self.path_file_log_combine}')
        print(f'path to file logs combine: {self.path_file_log_combine}')

        ### taking lines of training file log
        self.lines_taked = self.take_training_lines()
        self.combine()
        ###

    def take_training_lines(self):
        numerator_function = lambda x: x*(x+1)/2
        with open(self.path_file_log_trnng,'r') as file:
            lines = file.readlines()
            num_lines_trnng = len(lines)
            weigth_probability_lines = [i/numerator_function(num_lines_trnng) for i in range(1,num_lines_trnng+1)]
            ind_lines = [i for i in range(num_lines_trnng)]
            print(f'num lines trnng: {num_lines_trnng}')
            self.num_lines_trnng_taked = math.ceil(num_lines_trnng*0.3)   ### may be put the percentage as configuration parameter
            print(f'num trnng lines taked: {self.num_lines_trnng_taked}')
            logging.debug(f'num trnng lines taked: {self.num_lines_trnng_taked}')
            ind_lines_taked = random.choices(ind_lines,weigth_probability_lines,k=self.num_lines_trnng_taked)
            lines_taked = list(itemgetter(*ind_lines_taked)(lines))
            return lines_taked

    def combine(self):
        print('Combine file execution logs and training log ...')
        logging.debug('Combine file execution logs and training log ...')

        file_logs_combine = open(self.path_file_log_combine,'a')
        num_file_logs_lines = len(open(self.path_file_log,'r').readlines())
        with open(self.path_file_log,'r') as file:
            # num_file_logs_lines = len(file.readlines())
            # num_file_logs_lines = 10000
            print(f'num lines in log file: {num_file_logs_lines}')
            rate_lines = math.floor(num_file_logs_lines/self.num_lines_trnng_taked)
            print(f'rate of lines: {rate_lines}')
            line_log = file.readline()
            i = 1   ### num line of execution log
            j = 0   ### num line of training log
            while line_log:
                print(f'lines execution log: {i}')
                file_logs_combine.write(line_log.strip()+'\n')
                print(f'num trnng lines taked: {self.num_lines_trnng_taked}')
                if i%rate_lines == 0 and j<self.num_lines_trnng_taked:
                    print(f'adding training lines: {j}')
                    file_logs_combine.write(self.lines_taked[j]+'\n')
                    j += 1
                i += 1
                line_log = file.readline()

        print('combine logs finished. ')
        logging.debug('combine logs finished. ')
        return

if __name__=='__main__':
    logging.debug('local execution ...')
    Combine_logs(
     autor=args.autor
    )