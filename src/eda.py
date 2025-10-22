# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# logging and filepaths setup
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

# ------------load dataset from a csv file-------------
def load_file(filename : str = 'data/retail_store_sales.csv') -> pd.DataFrame:
    '''Loads the csv file into the python environment as a pandas environment'''
    try:
        df = pd.read_csv(filename)
        log.info(f'Data successfully loaded from {filename} file')
        return df
    except FileNotFoundError:
        log.error(f'File not found! Check filepath and try again!')
        return None

# ------a short descriptive statistical summary of the dataset----------  
def summary_overview(df : pd.DataFrame):
    '''A short, descriptive summary of the dataset'''
    log.info(f'Number of observations : {df.shape[0]}')
    log.info(f'Number of features : {df.shape[1]}')
    describe = df.describe(include='all').T[['min','max','mean','std']]
    log.info(f'\n{describe}')
    return {
        'Observations' : df.shape[0],
        'Features' : df.shape[1],
        'Description' : describe
    }

# -----------numerical columns - their minimum and maximum average values-----------
def numeric_columns(df : pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numeric_cols,1):
        log.info(f'{i}. {col:<15} | Min : {df[col].min():<3} | Max : {df[col].max()}') 
    return numeric_cols

# ---------categorical columns - number of unique values each category contains
def categorical_columns(df : pd.DataFrame):
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for i, col in enumerate(categorical_cols,1):
        uniques = df[col].unique()
        log.info(f'{i}. {col:<20} | Unique : {df[col].nunique():<3} | Examples : {uniques[:4]}')
    return categorical_cols
    
# -----------detect outliers using the IQR method------------
def outliers_detection(df: pd.DataFrame, col: str):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return lower_bound, upper_bound, outliers

# ----------outlier summary - return lower range, upper range and number of outliers------
def outlier_summary(df: pd.DataFrame, numeric_cols : list[str]):
    for i, col in enumerate(numeric_cols,1):
        lower, upper, outlier = outliers_detection(df, col)
        log.info(f'{i}. {col:<15} | Number of outliers : {len(outlier):<3} | Range : ({lower} - {upper})')

# ----------- missing values - missing percentages -------------
def missing_values(df : pd.DataFrame):
    '''Analyze the missing values in the dataset'''
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_values' : missing,
        'missing_pct' : missing_pct.round(2)
    })
    log.info(f'Dataset shape: {df.shape}, Missing columns: {missing_df.index.tolist()}')
    return missing_df

# ------------plot missing values -----------
def plt_missing_values(missing_summary : pd.DataFrame):
    if missing_summary['missing_values'].sum() == 0:
        log.info(f'No missing values detected. Skipping plots')
    else:
        missing_summary['missing_values'].plot(kind='barh',figsize=(12,7),title='Distribution of missing values',
                xlabel='Frequency',color='indigo')
        plt.savefig(plot_dir / 'missing_values.png', dpi=300)
        log.info('Image successfully saved!')
        plt.show()

# -----------check for duplicates in the dataset--------------
def duplicate_data(df : pd.DataFrame):
    duplicates = df[df.duplicated()]
    log.info(f'Number of duplicates : {len(duplicates)}')
    if len(duplicates) == 0:
        log.info('No duplicates found')
    else:
        return duplicates
    
# ---------correlation matrix ------------
def plt_heatmap(df : pd.DataFrame):
    plt.figure(figsize=(12,7))
    corr = df.corr(numeric_only=True, method='spearman')
    sns.heatmap(data=corr, fmt='.2f', annot=True, cmap='Blues',cbar=False)
    plt.title('Correlation HeatMap')
    plt.savefig(plot_dir / 'heatmap.png', dpi = 300)
    plt.tight_layout()
    plt.show()
    log.info(f'Correlation heatmap successfully plotted and saved!')

# ----------plot numerical historgrams---------------
def plt_histogram(df : pd.DataFrame, numeric_cols: list[str]):
    plt.figure(figsize=(12,7))
    for col in numeric_cols:
        sns.histplot(data=df, x=col, kde=True, color='indigo', alpha=0.7)
        plt.title(f'Distribution of {col}',fontsize=14,fontweight='bold')
        plt.ylabel('Frequency',fontsize=10,fontweight='bold')
        plt.tight_layout()
        plt.grid(True,alpha=0.3)
        plt.savefig(f'{plot_dir}/plt_{col}.png',dpi=300)
        log.info(f'{col} histogram successfully plotted and saved')
        plt.show()

# -----------plot boxplots---------
def plt_boxplots(df : pd.DataFrame, numeric_cols : list[str]):
    plt.figure(figsize=(12,7))
    for col in numeric_cols:
        sns.boxplot(data=df,y=col,linecolor='green',color='indigo')
        plt.title(f'-Boxplots - {col}')
        plt.tight_layout()
        plt.grid(True,alpha=0.3)
        plt.savefig(f'{plot_dir}/boxplot_{col}.png',dpi=300)
        log.info(f'{col} boxplot successfully plotted and saved!')
        plt.show()

# ------main-----
def run_eda(filename: str = 'data/retail_stores_sales.csv'):
    df = load_file()
    overview = summary_overview(df)
    missing_df = missing_values(df)
    plt_data = plt_missing_values(missing_df)
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    duplicates = duplicate_data(df)
    outliers = outlier_summary(df, num_cols)
    correlation = plt_heatmap(df)
    histogram = plt_histogram(df, num_cols)
    boxplot = plt_boxplots(df,num_cols)

    return {
        'data' : df,
        'overview' : overview,
        'missing_df' : missing_df,
        'plt_data' : plt_data,
        'num_cols' : num_cols,
        'cat_cols' : cat_cols,
        'duplicates' : duplicates,
        'outliers' : outliers,
        'correlation' : correlation,
        'histogram' : histogram,
        'boxplot' : boxplot
    }

if __name__ == '__main__':
    results = run_eda()
    df = results['data']