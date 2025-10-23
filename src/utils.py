import numpy as np
import pandas  as pd
from pathlib import Path
import logging
import json
from typing import Optional, Any, List, Dict
import yaml

# logging and filepaths setup
base_dir = Path.cwd()

Path('logs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)

logs_dir = base_dir / 'logs'
plot_dir = base_dir / 'plots'

log_path = logs_dir / 'utils.log'

log = logging.getLogger('Utility')
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S')

def setup_logger(name: str, log_file: str, level = logging.INFO):
    '''Set up logging
    
    Args:
        name : name of the logging file
        log_file : file to which logs will be logged
        level : level of the log (e.g. info, error, warning, etc)
    '''
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s',datefmt='%H:%M:%S')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(level)
    return logger

# ----------load data from a csv file------------
def load_file(filename : str) -> pd.DataFrame:
    """Loads the csv file into the python environment as a pandas DataFrame
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        df = pd.read_csv(filename)
        log.info(f'Data successfully loaded from {filename} | Shape : {df.shape}')
        return df
    except FileNotFoundError:
        log.error(f'File not found! Check filepath and try again!')
        raise
    except Exception as e:
        log.error(f'Error loading file : {e}')

# ---------Save Data -JSON----------
def dump_json(filename : str, data : Optional[Any]):
    '''Saves data in a json file
    
    Args: 
        filename : The output file name
        data : data to be saved

    Raises: 
        FileNotFoundError: If the file doesn't exist
        JSONDecodeError: If there is an error decoding the JSON file
    '''
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
            log.info(f'Data saved to {filename}')
    except FileNotFoundError:
        log.error('File not found! Check file path and try again')
        raise
    except json.JSONDecodeError as e:
        log.error(f'Error decoding JSON file : {e}')
        raise

# ------------Load Data - JSON-----------------
def load_json(filename : str) -> Any:
    '''Loads data from a json file
    
    Args:
        filename : The filename from which data is loaed

    Raises: 
        FileNotFoundError: If the file doesn't exist
        JSONDecodeError: If there is an error decoding the JSON file
    '''
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            log.info(f'Data successfully loaded from {e}')
            return data
    except FileNotFoundError:
        log.error('File Not Found! Check file path and try again')
        raise
    except json.JSONDecodeError as e:
        log.error(f'Error decoding JSON file : {e}')
        raise

# ---------------Read data - YAML file ------------
def read_yaml(filename : str) -> Any:
    '''Reads data from a yaml file
    
    Args: 
        filename :The filename from which data is read

    Raises: 
        FileNotFoundError: If the file doesn't exist
        JSONDecodeError: If there is an error decoding the JSON file
    '''
    try:
        with open(filename,'r') as file:
            config = yaml.safe_load(file)
            log.info(f'Data successfully read from {filename}')
            return config
    except FileNotFoundError:
        log.error('File Not Found! Check file path and try again')
        raise
    except yaml.YAMLError as e:
        log.info(f'YAML parsing error : {e}')
        raise