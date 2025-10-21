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
plot_dir = base_dir / 'plots'

Path('logs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)

log_path = logs_dir / 'Exploratory_data_analysis.log'

log = logging.getLogger(__name__)
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S')

def load_file(filename : str = 'data/retail_store_sales.csv') -> pd.DataFrame:
    '''Loads the csv file into the python environment as a pandas environment'''
    try:
        df = pd.read_csv(filename)
        log.info(f'Data successfully loaded from {filename} file')
        return df
    except FileNotFoundError:
        log.error(f'File not found! Check filepath and try again!')
        return None
    
def summary_overview(df : pd.DataFrame):
    '''A short, descriptive summary of the dataset'''
    log.info(f'Number of observations : {df.shape[0]}')
    log.info(f'Number of features : {df.shape[1]}')
    describe = df.describe(include='all').T
    log.info(f'{describe}')
    return {
        'Observations' : df.shape[0],
        'Features' : df.shape[1],
        'Description' : df.describe(include='all')
    }
    
def missing_values(df : pd.DataFrame):
    '''Analyze the missing values in the dataset'''
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_values' : missing,
        'missing_pct' : missing_pct.round(2)
    })
    log.info(f'{missing_df}')
    # plt.figure(figsize=(12,7))
    # sns.barplot(data=missing_df, y='missing_values',color='indigo',alpha=0.7)
    # plt.title('Missing Values', fontsize=12, fontweight='bold')
    # plt.ylabel('Frequency')
    # plt.savefig(plot_dir / 'missing_values.png', dpi=300)
    # plt.show()
    return missing_df

def plt_missing_values(df : pd.DataFrame):
    df['missing_values'].plot(kind='barh',figsize=(12,7),title='Distribution of missing values',
            xlabel='Frequency')
    plt.savefig(plot_dir / 'missing_values.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    df = load_file()
    summary_overview(df)
    missing_df = missing_values(df)
    plt_missing_values(missing_df)