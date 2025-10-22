# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from typing import Dict,Any,List,Optional
import warnings
warnings.filterwarnings('ignore')

# logging and filepaths setup
base_dir = Path.cwd()

Path('logs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)

logs_dir = base_dir / 'logs'
plot_dir = base_dir / 'plots'

log_path = logs_dir / 'Exploratory_data_analysis.log'

log = logging.getLogger(__name__)
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S')

# --------Types/dataclass-----------
@dataclass
class EDAResults:
    data : pd.DataFrame
    overview : Dict[str,Any]
    missing_summary : pd.DataFrame
    numeric_columns : List[str]
    categorical_columns : List[str]
    duplicates : Optional[pd.DataFrame]
    outlier_summary : Dict[str,Dict[str,Any]]

# ------------Utility - load dataset from a csv file-------------
def load_file(filename : str = 'data/retail_store_sales.csv') -> pd.DataFrame:
    '''Loads the csv file into the python environment as a pandas environment'''
    try:
        df = pd.read_csv(filename)
        log.info(f'Data successfully loaded from {filename} file')
        return df
    except FileNotFoundError:
        log.error(f'File not found! Check filepath and try again!')
        raise

# --------------Validation---------------
def validation_schema(df: pd.DataFrame, required_columns : Optional[List[str]]) -> None:
    '''Check presence of required columns and basic type expectations
    Raises a valueError if required columns are missing'''
    if required_columns is None:
        return 
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        log.error(f'Missing required columns : {missing_cols}')
        raise ValueError(f'Missing required columns : {missing_cols}')
    log.info('Schema validation passed! All required columns present')



# ------a short descriptive statistical summary of the dataset----------  
def summary_overview(df : pd.DataFrame):
    '''A short, descriptive summary of the dataset'''
    describe = df.describe().T[['min','max','mean','std']].round(4)
    overview = {
        'Observations' : df.shape[0],
        'Features' : df.shape[1],
        'Description' : describe.to_dict(orient='index')
    }
    observations = overview['Observations']
    features = overview['Features']
    log.info(f'Overview : observations = {observations} | Features = {features}')
    return overview

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
        return None
    else:
        missing_summary['missing_values'].plot(kind='barh',figsize=(12,7),title='Distribution of missing values',
                xlabel='Frequency',color='indigo')
        output_path = f'{plot_dir}_missing_values.png'
        plt.savefig(output_path, dpi=300)
        log.info(f'Missing values plot successfully plotted and saved to {output_path}')
        plt.show()

# -----------check for duplicates in the dataset--------------
def duplicate_data(df : pd.DataFrame):
    duplicates = df[df.duplicated()]
    log.info(f'Number of duplicates : {len(duplicates)}')
    if len(duplicates) == 0:
        log.info('No duplicates found')
        return None
    else:
        return duplicates
    
# ---------correlation matrix ------------
def plt_heatmap(df : pd.DataFrame):
    corr = df.corr(numeric_only=True, method='spearman')
    if corr.isnull().all().all():
        log.info(f'Correlation Matrix is empty or contains only NaNs. Skipping heatmap.')
        return None
    
    plt.figure(figsize=(12,7))
    sns.heatmap(data=corr, fmt='.2f', annot=True, cmap='Blues',cbar=False)
    plt.title('Spearman Correlation HeatMap')
    plt.savefig(plot_dir / 'heatmap.png', dpi = 300)
    plt.tight_layout()
    plt.show()
    log.info(f'Correlation heatmap successfully plotted and saved!')
    plt.close()

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
        plt.close()

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
        plt.close()

# -----business sanity checks------------
def business_sanity_checks(df: pd.DataFrame):
    results = {}
    required = {'Total Spent','Price Per Unit','Quantity'}
    if required.issubset(df.columns):
        calc_total = (df['Price Per Unit'] * df['Quantity']).replace([np.inf, -np.inf], np.nan)
        match_rate = float(np.isclose(calc_total.fillna(-1), df["Total Spent"].fillna(-2)).mean()) * 100
        df['total_spent_match_pct'] = round(match_rate,2)
        log.info(f"Total Spent matches Price*Quantity for {match_rate:.2f}% of rows")
    else:
        log.info("Skipping Total Spent sanity check - required columns not present")
    return results

# ------main-----
def run_eda(filename: str = 'data/retail_store_sales.csv'):
    df = load_file(filename)
    overview = summary_overview(df)
    missing_df = missing_values(df)
    plt_data = plt_missing_values(missing_df)
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    duplicates = duplicate_data(df)
    outliers = outlier_summary(df, num_cols)
    histogram = plt_histogram(df, num_cols)
    boxplot = plt_boxplots(df,num_cols)
    sanity = business_sanity_checks(df)

    overview['sanity_checks'] = sanity


    results = EDAResults(
        data= df,
        overview = overview,
        missing_summary = missing_df,
        numeric_columns = num_cols,
        categorical_columns = cat_cols,
        duplicates = duplicates,
        outlier_summary = outliers
        )
    log.info('EDA run completed successfully!')
    return results

if __name__ == '__main__':
    run_eda()