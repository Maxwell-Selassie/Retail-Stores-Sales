import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

base_dir = Path(__file__).resolve().parents[1]
logs_dir = base_dir / 'logs'

Path('logs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)

log_path = logs_dir / 'Exploratory_data_analysis.log'

log = logging.getLogger(__name__)
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S')

def load_file(filename : str = 'data/retail_store_sales.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        log.info(f'Data successfully loaded from {filename} file')
        return df
    except FileNotFoundError:
        log.error(f'File not found! Check filepath and try again!')
        return None
    
if __name__ == '__main__':
    load_file()