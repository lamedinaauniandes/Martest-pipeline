import os
from datetime import datetime,date
import logging 
import re 

def get_near_date(lambda_function,path_dir): 
    print(f'getting near date from dir: {path_dir}')
    logging.debug(f'getting near date from dir: {path_dir}')
    list_files = os.listdir(path_dir)
    list_files_autor = list(map(lambda_function,list_files))
    date_files = list(map(lambda file: datetime.strptime(re.search('(\d{2,})',file).group(0),"%M%H%d%m%Y"),list_files_autor))
    near_date = min(date_files,key=lambda fecha: abs(fecha - datetime.now()))
    print(f'near date: {near_date}')
    return datetime.strftime(near_date,'%M%H%d%m%Y')

def get_path_files(autor,
                   path_dir,
                   type_file,
                   name_file,
                   extension,
                   abstract_level,
                   ):
    print(f'getting files from dir: {path_dir}')
    print(f'abstract level: {abstract_level}')
    logging.debug(f'getting files from dir: {path_dir}')
    logging.debug(f'abstract level: {abstract_level}')
    
    if type_file == '':
        flambda = (lambda file: file if (autor in file) and ('trnng' not in file) and ('combine' not in file) else date.max.strftime("%M%H%d%m%Y"))
    if type_file == 'trnng_':
        flambda = (lambda file: file if (autor in file) and ('trnng' in file) else date.max.strftime("%M%H%d%m%Y") )
    if type_file == 'combine_':
        flambda = (lambda file: file if (autor in file) and ('combine' in file) else date.max.strftime("%M%H%d%m%Y") )
    date_str_format = get_near_date(flambda,path_dir)
    
    if name_file != 'q_value':  ## q-values table doesn't need abstract level 
        path_file = os.path.join(path_dir,f'{name_file}_{abstract_level}_{type_file}{autor}_{date_str_format}.{extension}')
    else:
        print("getting q-values table doesn't need abstract level ...|")
        path_file = os.path.join(path_dir,f'{name_file}_{type_file}{autor}_{date_str_format}.{extension}')

    logging.debug(f'path {name_file} file: {path_file}, date: {date_str_format}')
    print(f'path {name_file} file: {path_file},  date: {date_str_format}')
    return path_file,date_str_format


def get_path_log_files(path_dir,type_file,name_file,extension):
    logging.debug(path_dir)
    if type_file == '':
        flambda = (lambda file: file if ('trnng' not in file) else date.max.strftime("%M%H%d%m%Y"))
    if type_file == 'trnng_':
        flambda = (lambda file: file if ('trnng' in file) else date.max.strftime("%M%H%d%m%Y") )
    if type_file == 'combine_':
        flambda = (lambda file: file if ('combine' in file) else date.max.strftime("%M%H%d%m%Y") )
        pass 
    date_str_format = get_near_date(flambda,path_dir)
    path_file = os.path.join(path_dir,f'{name_file}_{type_file}{date_str_format}.{extension}')
    logging.debug(f'path log file: {path_file}, date: {date_str_format}')
    print(f'path log file: {path_file}, date: {date_str_format}')
    
    return path_file,date_str_format


def get_path_model(autor,
                   path_dir,
                   type_file,
                   name_file,
                   abstract_level,
                   ):
    print(f'getting files from dir: {path_dir}')
    print(f'abstract level: {abstract_level}')
    logging.debug(f'getting files from dir: {path_dir}')
    logging.debug(f'abstract level: {abstract_level}')
    
    if type_file == '':
        flambda = (lambda file: file if (autor in file) and ('trnng' not in file) and ('combine' not in file) else date.max.strftime("%M%H%d%m%Y"))
    if type_file == 'trnng_':
        flambda = (lambda file: file if (autor in file) and ('trnng' in file) else date.max.strftime("%M%H%d%m%Y") )
    if type_file == 'combine_':
        flambda = (lambda file: file if (autor in file) and ('combine' in file) else date.max.strftime("%M%H%d%m%Y") )
    date_str_format = get_near_date(flambda,path_dir)
    
    path_file = os.path.join(path_dir,f'{name_file}_{abstract_level}_{type_file}{autor}_{date_str_format}')

    logging.debug(f'path {name_file} file: {path_file}, date: {date_str_format}')
    print(f'path {name_file} file: {path_file},  date: {date_str_format}')
    return path_file,date_str_format


######################
## outdated functions
#######################

def get_paths_abstrc_episodes_csv(autor,path_dir):
    """ 
    correspond to module 3 
    """
    logging.debug(path_dir)
    f_autor_exe = (lambda file: file if (autor in file) and ('trnng' not in file) else date.max.strftime("%M%H%d%m%Y"))
    f_autor_trnng = (lambda file: file if (autor in file) and ('trnng' in file) else date.max.strftime("%M%H%d%m%Y") )
    date_str_format_trnng = get_near_date(f_autor_trnng,path_dir)
    date_str_format_agt = get_near_date(f_autor_exe,path_dir)
    path_logs_trnng = os.path.join(path_dir,f'abstract_episodes_{autor}_{date_str_format_trnng}_trnng.csv')
    path_logs_agt = os.path.join(path_dir,f'abstract_episodes_{autor}_{date_str_format_agt}.csv')
    logging.debug(f'path abstract states file trnng: {path_logs_trnng}, path abstract states agt: {path_logs_agt}')
    return path_logs_agt,path_logs_trnng,date_str_format_agt,date_str_format_trnng


def get_paths_logs(autor,path_dir):
    """
    correspond to module 1
    """
    f_autor_exe = (lambda file: file if ('trnng' not in file) else date.max.strftime("%M%H%d%m%Y"))
    f_autor_trnng = (lambda file: file if ('trnng' in file) else date.max.strftime("%M%H%d%m%Y") )
    date_str_format_trnng = get_near_date(f_autor_trnng,path_dir)
    date_str_format_agt = get_near_date(f_autor_exe,path_dir)
    path_logs_trnng = os.path.join(path_dir,f'log_trnng_{date_str_format_trnng}.log')
    path_logs_agt = os.path.join(path_dir,f'log_{date_str_format_agt}.log')
    logging.debug(f'path logs file trnng: {path_logs_trnng}, path logs agt: {path_logs_agt}')
    return path_logs_agt,path_logs_trnng,date_str_format_agt,date_str_format_trnng


def get_paths_abstrc_states_json(autor,path_dir):
    """ 
    correspond to module 2 
    """
    f_autor_exe = (lambda file: file if (autor in file) and ('trnng' not in file) else date.max.strftime("%M%H%d%m%Y"))
    f_autor_trnng = (lambda file: file if (autor in file) and ('trnng' in file) else date.max.strftime("%M%H%d%m%Y") )
    date_str_format_trnng = get_near_date(f_autor_trnng,path_dir)
    date_str_format_agt = get_near_date(f_autor_exe,path_dir)
    path_logs_trnng = os.path.join(path_dir,f'abstract_states_{autor}_trnng_{date_str_format_trnng}.json')
    path_logs_agt = os.path.join(path_dir,f'abstract_states_{autor}_{date_str_format_agt}.json')
    logging.debug(f'path abstract states file trnng: {path_logs_trnng}, path abstract states agt: {path_logs_agt}')
    return path_logs_agt,path_logs_trnng,date_str_format_agt,date_str_format_trnng