'''
    Utility functions for data science pipelines
    Provides logging, file I/O and common logger functions 
'''
import numpy as np
import pandas  as pd
from pathlib import Path
import logging
import json
from typing import Optional, Any, List, Dict
import yaml
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# filepaths setup
base_dir = Path.cwd()

# ensure directories exist
list_of_directories = ['logs','data','data/raw','data/processed','plots','config']
for directory in list_of_directories:
    Path(directory).mkdir(exist_ok=True)

logs_dir = base_dir / 'logs'
plot_dir = base_dir / 'plots'
config_dir = base_dir / 'config'
data_dir = base_dir / 'data'


def setup_logger(name: str, log_file: str | Path, level = logging.INFO) -> logging.log:
    '''Set up a dedicated logger with file handlers and console handlers
    
    Args:
        name : Logger name
        log_file : Path to log file
        level : llogging level (default = INFO)

    Returns:
        configured logger instance

    Example:
        >> logger = setup_logger(name='MyModule',log_file='logs/mymodule.log')
        >> logger.info('Processing started...)
    '''
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger 
    
    elif not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s',datefmt='%H:%M:%S')
        # file handler with rotation (creates new log files at midnight)
        file_handler = TimedRotatingFileHandler(
            filename= log_file,
            when='midnight',
            interval=1,
            backupCount=7 # keep log files for 7 days
        )
        file_handler.suffix = '%Y%m%d'
        file_handler.setFormatter(formatter)
        # console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(level)
        logger.propagate = False # don't propagate to root logger
    return logger

# logging setup
log_path = logs_dir / 'utils.log'
log = setup_logger(name='Utility', log_file=log_path, level=logging.INFO)

# ----------load data from a csv file------------
def load_csv_file(filename : str) -> pd.DataFrame:
    """Loads the csv file into the python environment as a pandas DataFrame
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError : If dataframe is empty
        pd.errors.ParseError : If csv file is malformed
    """
    try:
        df = pd.read_csv(filename)
        log.info(f'Data successfully loaded from {filename} | Shape : {df.shape}')
        return df
    except FileNotFoundError:
        log.error(f'File not found! Check filepath and try again!')
        raise
    except pd.errors.EmptyDataError:
        log.error(f'x File is empty : {filename}')
        raise
    except pd.errors.ParserError as e:
        log.error(f'x CSV parsing error : {e}')
        raise

def save_csv(data: pd.DataFrame, filename : str | Path, index : bool = False) -> None:
    '''Saves dataframe to a csv file
    
    Args:
        data : The dataframe to be saved
        filename : Output file path
        index : whether to save index column (default=false)
        
    Example:
        save_csv(data = df, filename= 'data/processed/cleaned_data.csv')
    '''
    try:
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, index=index)
        log.info(f'Data saved to {filepath} | Shape : {data.shape}')
    except Exception as e:
        log.error(f'Error saving CSV : {e}')
            

# ---------Save Data -JSON----------
def dump_json(filename : str, data : Optional[Any]):
    '''Saves data in a json file
    
    Args: 
        filename : The output file name
        data : data to be saved (must be JSON serializable)

    Raises: 
        FileNotFoundError: If the file doesn't exist
        TypeError: If data is not JSON serializable
    '''
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
            log.info(f'Data saved to {filename}')
    except FileNotFoundError:
        log.error('File not found! Check file path and try again')
        raise
    except TypeError as e:
        log.error(f'Data is not JSON serializable : {e}')
        raise
    except Exception as e:
        log.error(f'x Error saving JSON data : {e}')

# ------------Load Data - JSON-----------------
def load_json(filename : str | Path) -> Any:
    '''Loads data from a json file
    
    Args:
        filename : Path to JSON file

    Raises: 
        FileNotFoundError: If the file doesn't exist
        JSONDecodeError: If there is an error decoding the JSON file
    '''
    try:
        filepath = Path(filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
        log.info(f'Data successfully loaded from {e}')
        return data
    except FileNotFoundError:
        log.error('File Not Found! Check file path and try again')
        raise
    except json.JSONDecodeError as e:
        log.error(f'Error decoding JSON file : {e}')
        raise
    except Exception as e:
        log.error(f'Error loading JSON : {e}')
        raise

# ---------------Read data - YAML file ------------
def read_yaml(filename : str | Path) -> Dict[str, Any]:
    '''Reads configuration from yaml file
    
    Args: 
        filename : The filepath to YAML file

    Returns:
        Dictionary with configuration

    Raises: 
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If YAML file is malformed
    '''
    try:
        filepath = Path(filename)
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)
            log.info(f'Data successfully read from {filename}')
            return config
    except FileNotFoundError:
        log.error('File Not Found! Check file path and try again')
        raise
    except yaml.YAMLError as e:
        log.error(f'YAML parsing error : {e}')
        raise
    except Exception as e:
        log.error(f'x Error reading YAML : {e}')

def save_yaml(filename : str | Path, data: Dict[str,Any]) -> None:
    '''Saves data to YAML file

    Args:
        filename: Output file path
        data : Dictionary with configuration

    Examples:
        save_yaml(data=data, filename='config/config_file.yaml)
    '''
    try:
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        log.info(f'YAML config saved to {filepath}')
    except Exception as e:
        log.error(f'Error saving YAML file : {e}')

def save_metadata(metadata : Dict[str, Any], filename : str | Path) -> None:
    '''Saves processing metadata from reproducibility
    
    Args:
        metadata: dictionary with processing metadata
        filename : output file path    
    '''
    format = '%Y-%m-%d %H:%M:%S'
    metadata['timestamp'] = datetime.now().strftime(format)
    metadata['python_version'] = __import__('sys').version
    metadata['pandas_version'] = pd.__version__
    metadata['numpy_version'] = np.__version__
    dump_json(filename, metadata)
    log.info('Metadata saved with timestamp')

def validate_schema(df: pd.DataFrame, required_cols : Optional[List]) -> None:
    '''Validates dataframe structure
    
    Args:
        df: dataframe to validate
        required_cols : list of required column names
        
    Raises:
        ValueError: if validation fails
    '''
    if df.empty:
        raise ValueError('Dataframe is empty')
    
    if required_cols:
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns : {missing_cols}')
    log.info(f'Dataframe validated | Shape : {df.shape}')

def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    '''Generate quick data quality profile
    
    Args:
        df = Dataframe to profile
        
    Returns:
        Dictionary with data profile and data quality metrics
    '''
    profile = {
        'shape' : df.shape,
        'memory_usage' : df.memory_usage(deep=True).sum() / 1024 ** 2,
        'missing_values' : df.isnull().sum().to_dict(),
        'duplicates' : df.duplicated().sum(),
        'numeric_cols' : df.select_dtypes(include=[np.number]).columns,
        'categorical_cols' : df.select_dtypes(exclude=[np.number]).columns,
        'dtypes' : df.dtypes.astype('str').to_dict()
    }
    log.info(f'Data profile generated for dataframe with shape : {df.shape}')
    return profile

