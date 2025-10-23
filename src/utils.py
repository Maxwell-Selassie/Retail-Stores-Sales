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

log = logging.getLogger('EDA')
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S')

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
    '''
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
            log.info(f'Data saved to {filename}')
    except FileNotFoundError:
        log.error('File not found! Check file path and try again')
    except json.JSONDecodeError as e:
        log.error(f'Error : {e}')
        raise

# ------------Load Data - JSON-----------------
def load_json(filename : str) -> Any:
    '''Loads data from a json file
    
    Args:
        filename : The filename from which data is loaed
    '''
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            log.info(f'Data successfully loaded from {e}')
    except FileNotFoundError:
        log.error('File Not Found! Check file path and try again')
    except json.JSONDecodeError as e:
        log.error(f'Error : {e}')
        raise

# ---------------Read data - YAML file ------------
def read_yaml(filename : str) -> Any:
    '''Reads data from a yaml file
    
    Args: 
        filename :The filename from which data is read
    '''
    try:
        with open(filename,'r') as file:
            config = yaml.safe_load(file)
            log.info(f'Data successfully read from {filename}')
    except FileNotFoundError:
        log.error('File Not Found! Check file path and try again')
    except yaml.error as e:
        log.info(f'Error : {e}')
        raise