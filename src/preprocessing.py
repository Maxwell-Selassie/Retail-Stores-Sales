'''
Data preprocessing pipeline for a retail store sales
Read configurations from eda_preprocessing.yaml and apply transformations

Author: Maxwell Selassie Hiamatsu
Date: 24/10/2025
'''
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import warnings
import logging
warnings.filterwarnings('ignore')
from datetime import datetime

# import utilities
from utils import (
        setup_logger,load_file,read_yaml,save_metadata,save_csv
)

# setup logging
base_dir = Path.cwd()
logs_dir = base_dir / 'logs'

Path('logs').mkdir(exist_ok=True)
logs_path = logs_dir / 'preprocessing.log'
log = setup_logger(name='Preprocessing',log_file=logs_path,level=logging.INFO)

class DataPreprocessor():
    '''
    Production-ready preprocessing pipeline driven by YAML configuration.

    This class handles:
    - Missing Data imputation
    - Outliers handling
    - Duplicates handling
    - Feature encoding
    - Feature scaling
    - Data quality fixes
    - Datetime feature extractions

    All decisions are documented in config/eda_preprocessing.yaml
    '''
    def __init__(self, config_path: str | Path = 'config/eda_preprocessing.yaml'):
        '''
        Initialize preprocessor with configuration

        Args:
            config_path: Path to configuration file
        '''
        log.info('='*70)
        log.info('INITIALIZING PREPROCESSING PIPELINE')
        log.info('='*70)
        
        self.config = read_yaml(config_path)
        self.config_path = config_path
        
		# Initialize transformers
        self.scalers = None
        self.target_encoders = {}
        
		# Track transformations
        self.transformation_log = []
        self.metadata = {
            'config_version' : self.config['project']['version'],
            'config_file' : config_path,
            'timestamp_start' : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		}
        
        log.info(f'Project : {self.config['project']['name']}')
        log.info(f'Version : {self.config['project']['version']}')
        log.info('Configurations loaded successfully')
        

