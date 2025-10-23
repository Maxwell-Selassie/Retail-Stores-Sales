# import libraries
from __future__ import annotations


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
import argparse
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
    """Container for EDA results
    
    Attributes:
        data: The original DataFrame
        overview: Dictionary with dataset statistics
        missing_summary: DataFrame showing missing value analysis
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
        duplicates: DataFrame of duplicate rows (None if no duplicates)
        outlier_summary: Dictionary with outlier statistics per column
        cardinality_analysis: Dictionary of high-cardinality columns
        constant_features: List of quasi-constant feature names
        sanity_checks: Dictionary of business logic validation results
    """
    data : pd.DataFrame
    overview : Dict[str,Any]
    missing_summary : pd.DataFrame
    numeric_columns : List[str]
    categorical_columns : List[str]
    duplicates : Optional[pd.DataFrame]
    outlier_summary : Dict[str,Dict[str,Any]]
    cardinality_analysis : Dict[str,int]
    constant_features : List[str]
    sanity_checks : Dict[str,Any]

# ------------Utility - load dataset from a csv file-------------
def load_file(filename : str = 'data/retail_store_sales.csv') -> pd.DataFrame:
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

# --------------Validation---------------
def validation_schema(df: pd.DataFrame, required_columns : Optional[List[str]]) -> None:
    '''Check presence of required columns and basic type expectations
        Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Raises:
        ValueError: If required columns are missing
    '''
    if required_columns is None:
        log.info(f'No schema validation required')
        return 
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        log.error(f'Missing required columns : {missing_cols}')
        raise ValueError(f'Missing required columns : {missing_cols}')
    log.info('Schema validation passed! All required columns present')



# ------a short descriptive statistical summary of the dataset----------  
def summary_overview(df : pd.DataFrame) -> Dict[str,Any]:
    '''A short, descriptive summary of the dataset
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary containing dataset statistics'''
    describe = df.describe().T[['min','max','mean','std']].round(4)
    overview = {
        'Observations' : df.shape[0],
        'Features' : df.shape[1],
        'memory_usage_MB' : round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        'Description' : describe.to_dict(orient='index')
    }
    observations = overview['Observations']
    features = overview['Features']
    memory_usage = overview['memory_usage_MB']
    log.info(f'Overview : observations = {observations} | Features = {features} | Memory = {memory_usage}')
    return overview

# -----------numerical columns - their minimum and maximum average values-----------
def numeric_columns(df : pd.DataFrame) -> List[str]:
    """Identify and log statistics for numeric columns
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of numeric column names
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    log.info(f'Found {len(numeric_cols)} numeric columns')
    for i, col in enumerate(numeric_cols,1):
        log.info(f'{i}. {col:<15} | Min : {df[col].min():<3} | Max : {df[col].max():<3} | Mean : {df[col].mean():.2f}') 
    return numeric_cols

# ---------categorical columns - number of unique values each category contains
def categorical_columns(df : pd.DataFrame):
    """Identify and log statistics for categorical columns
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of categorical column names
    """
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    log.info(f'Found {len(categorical_cols)} categorical columns')
    for i, col in enumerate(categorical_cols,1):
        uniques = df[col].unique()
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
        log.info(f'{i}. {col:<20} | Unique : {df[col].nunique():<3} | Mode : {mode_value} | Examples : {uniques[:4]}')
    return categorical_cols
    
# -----------detect outliers using the IQR method------------
def outliers_detection(df: pd.DataFrame, col: str) -> tuple:
    """Detect outliers using the Interquartile Range (IQR) method
    
    Args:
        df: DataFrame containing the column
        col: Column name to check for outliers
        
    Returns:
        Tuple of (lower_bound, upper_bound, outliers_dataframe)
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return lower_bound, upper_bound, outliers

# ----------outlier summary - return lower range, upper range and number of outliers------
def outlier_summary(df: pd.DataFrame, numeric_cols : list[str]):
    """Generate comprehensive outlier analysis for numeric columns
    
    Args:
        df: DataFrame to analyze
        numeric_cols: List of numeric column names
        
    Returns:
        Dictionary with outlier statistics per column
    """
    summary = {}
    for i, col in enumerate(numeric_cols,1):
        lower, upper, outlier = outliers_detection(df, col)
        outlier_pct = round(len(outlier) / len(df) * 100, 2) if len(df) > 0 else 0
        summary[col] = {
            'outlier_count' : len(outlier),
            'outlier_pct' : outlier_pct,
            'lower_bound' : round(lower, 2),
            'upper_bound' : round(upper, 2)
        }
        log.info(f'{i}. {col:<15} | Number of outliers : {len(outlier):<3} | Range : ({lower} - {upper})')
    return summary

# ----------- missing values - missing percentages -------------
def missing_values(df : pd.DataFrame) -> pd.DataFrame:
    '''Analyze the missing values in the dataset
        Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing value statistics'''
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        log.info('No missing values detected in dataset')
        return pd.DataFrame(columns=['missing_values','missing_pct'])
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_values' : missing,
        'missing_pct' : missing_pct.round(2)
    })
    log.info(f'Dataset shape: {df.shape}, Missing columns: {missing_df.index.tolist()}')
    return missing_df

# ------------plot missing values -----------
def plt_missing_values(missing_summary : pd.DataFrame) -> None:
    """Visualize missing value distribution
    
    Args:
        missing_summary: DataFrame from missing_values() function
    """
    if missing_summary['missing_values'].sum() == 0:
        log.info(f'No missing values detected. Skipping plots')
        return None
    try:
        missing_summary['missing_values'].plot(kind='barh',figsize=(12,7),title='Distribution of missing values',
                xlabel='Frequency',color='indigo')
        output_path = f'{plot_dir}_missing_values.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f'Missing values plot successfully plotted and saved to {output_path}')
        plt.show()
        plt.close()
    except Exception as e:
        log.error(f'Error creating missing value plots : {e}')
        plt.close()

# -----------check for duplicates in the dataset--------------
def duplicate_data(df : pd.DataFrame) -> Optional[pd.DataFrame]:
    """Check for duplicate rows in the dataset
    
    Args:
        df: DataFrame to check
        
    Returns:
        DataFrame of duplicate rows or None if no duplicates
    """
    duplicates = df[df.duplicated()]
    log.info(f'Number of duplicates : {len(duplicates)}')
    if len(duplicates) == 0:
        log.info('No duplicates found')
        return None
    else:
        return duplicates
    
# ---------correlation matrix ------------
def plt_heatmap(df : pd.DataFrame) -> None:
    """Generate and save correlation heatmap
    
    Args:
        df: DataFrame to analyze
    """

    numeric_columns = df.select_dtypes(include=[np.number])
    if numeric_columns.shape[1] < 2:
        log.info('Less than 2 numeric columns. Skipping heatmap')

    corr = df.corr(numeric_only=True, method='spearman')
    if corr.isnull().all().all():
        log.info(f'Correlation Matrix is empty or contains only NaNs. Skipping heatmap.')
        return None
    try:
        plt.figure(figsize=(12,7))
        sns.heatmap(data=corr, fmt='.2f', annot=True, cmap='Blues',cbar=False)
        plt.title('Spearman Correlation HeatMap')
        plt.savefig(plot_dir / 'heatmap.png', dpi = 300)
        plt.tight_layout()
        plt.show()
        log.info(f'Correlation heatmap successfully plotted and saved!')
        plt.close()
    except Exception as e:
        log.error(f'Error creating heatmap: {e}')
        plt.close()

# ----------plot numerical historgrams---------------
def plt_histogram(df : pd.DataFrame, numeric_cols: list[str]) -> None:
    """Generate histograms for all numeric columns
    
    Args:
        df: DataFrame to plot
        numeric_cols: List of numeric column names
    """
    for col in numeric_cols:
        try:
            plt.figure(figsize=(12,7))
            sns.histplot(data=df, x=col, kde=True, color='indigo', alpha=0.7)
            plt.title(f'Distribution of {col}',fontsize=14,fontweight='bold')
            plt.ylabel('Frequency',fontsize=10,fontweight='bold')
            plt.tight_layout()
            plt.grid(True,alpha=0.3)
            plt.savefig(f'{plot_dir}/plt_{col}.png',dpi=300)
            log.info(f'{col} histogram successfully plotted and saved')
            plt.show()
            plt.close()
        except Exception as e:
            plt.error(f'Error creating histogram for {col} : {e}')
            plt.close()

# -----------plot boxplots---------
def plt_boxplots(df : pd.DataFrame, numeric_cols : list[str]) -> None:
    """Generate boxplots for all numeric columns
    
    Args:
        df: DataFrame to plot
        numeric_cols: List of numeric column names
    """
    for col in numeric_cols:
        try:
            plt.figure(figsize=(12,7))
            sns.boxplot(data=df,y=col,linecolor='green',color='indigo')
            plt.title(f'-Boxplots - {col}')
            plt.tight_layout()
            plt.grid(True,alpha=0.3)
            plt.savefig(f'{plot_dir}/boxplot_{col}.png',dpi=300)
            log.info(f'{col} boxplot successfully plotted and saved!')
            plt.show()
            plt.close()
        except Exception as e:
            log.error(f'Error plotting boxplot for {col} : {e}')
            plt.close()


# ---------cardinality analysis-----------
def cardinality_analysis(df: pd.DataFrame, categorical_cols : List[str],
                        threshold : int = 50) -> Dict[str,int]:
    """Flag high-cardinality categorical columns
    
    High cardinality features (many unique values) may need special encoding
    or could be identifiers rather than true features.
    
    Args:
        df: DataFrame to analyze
        categorical_cols: List of categorical column names
        threshold: Number of unique values to flag as high cardinality
        
    Returns:
        Dictionary of high-cardinality columns and their unique counts
    """
    high_card_cols = {}
    for col in categorical_cols:
        uniques = df[col].nunique()
        if uniques > threshold:
            high_card_cols[col] = uniques
            log.warning(f'{col} has {uniques} unique values - may need special encoding or could be an ID column')
    if not high_card_cols:
        log.info(f'No high cardinality columns detected. Threshold = {threshold}')
    return high_card_cols

#------------constant features detection---------
def constant_features(df: pd.DataFrame, threshold: int = 0.95) -> List[str]:
    """Detect features where one value dominates (quasi-constant)
    
    Columns where 95%+ of values are the same provide little information
    and can often be dropped.
    
    Args:
        df: DataFrame to analyze
        threshold: Proportion threshold for flagging (default 0.95 = 95%)
        
    Returns:
        List of quasi-constant column names
    """
    constants = []
    for col in df.columns:
        try:
            top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq > threshold:
                constants.append(col)
                log.warning(f'{col} is quasi contant - ({top_freq:.1} same value)')
        except Exception as e:
            continue
    if not constants:
        log.info(f'No quasi constants detected in dataset (Threshold = {threshold})')
    return constants

# -----business sanity checks------------
def business_sanity_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate business logic and data consistency
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
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
def run_eda(filename: str = 'data/retail_store_sales.csv', required_cols: Optional[List[str]] = None) -> EDAResults:
    '''Execute the complete EDA pipeline
    
    Args:
        filename : filename of csv file to analyze
        required_cols : optional list of columns that must be present in the dataset
        
    Returns:
        EDAResults dataclass containing all analysis results
    '''
    log.info('='*50)
    log.info('Starting EDA pipeline')
    log.info('='*50)
    # load and validate 
    df = load_file(filename)
    validate = validation_schema(df, required_cols)

    # basic overview
    overview = summary_overview(df)

    # missing values analysis
    missing_df = missing_values(df)
    plt_data = plt_missing_values(missing_df)

    # column type identification
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)

    # heatmap visualization
    plt_heatmap(df)

    # data quality checks
    duplicates = duplicate_data(df)
    outliers = outlier_summary(df, num_cols)
    high_card = cardinality_analysis(df, cat_cols)
    constants = constant_features(df)
    sanity = business_sanity_checks(df)

    # visualizations
    plt_histogram(df, num_cols)
    plt_boxplots(df,num_cols)


    overview['sanity_checks'] = sanity
    overview['data_quality'] = {
        'duplicate_rows' : duplicates.shape[0] if duplicates is not None else 0,
        'high_card_cols' : list(high_card.keys()),
        'quasi_constant_cols' : constants
    }


    results = EDAResults(
        data= df,
        overview = overview,
        missing_summary = missing_df,
        numeric_columns = num_cols,
        categorical_columns = cat_cols,
        duplicates = duplicates,
        outlier_summary = outliers,
        cardinality_analysis= high_card,
        constant_features=constants,
        sanity_checks=sanity
        )
    log.info('='*50)
    log.info('EDA run completed successfully!')
    log.info('='*50)
    return results

# -------------command line interface------------
def args_parse():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(description="Run production-grade EDA pipeline on retail stores sales",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog= """Examples:
                python eda.py --file data/sales.csv
                python eda.py -f data/sales.csv --required-cols "Total Spent" "Quantity"
            """)
    parser.add_argument("--file", "-f", help="CSV file path to analyze", required=True)
    parser.add_argument(
    "--required-cols",
    "-r",
    nargs="*",
    help="List of required columns to validate against",
    default=None,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parse()
    try: 
        eda_result = run_eda(args.file, args.required_cols)

        # export brief overview and missing summary for quick inspection
        overview_path = logs_dir / 'eda_overview.json'
        with open(overview_path,'w') as file:
            json.dump(eda_result.overview, file, indent=4)

        if not eda_result.missing_summary.empty:
            missing_path = logs_dir / 'missing_summary.csv'
            eda_result.missing_summary.to_csv(missing_path)
            log.info(f'Wrote overview to {overview_path} and missing summary to {missing_path}')

        # export outlier summary
        outlier_path = logs_dir / 'outlier_summary.json'
        with open(outlier_path, 'w') as file:
            json.dump(eda_result.outlier_summary,file, indent=4)

        log.info(f'Exported results to {logs_dir}')
        print(f"\n✅ EDA completed successfully!")
        print(f"📊 Results saved to: {logs_dir}")
        print(f"📈 Plots saved to: {plot_dir}")
        
    except Exception as e:
        log.exception(f'EDA failed: {e}')
        print(f"\n❌ EDA failed: {e}")
        print(f"Check log file for details: {log_path}")
        raise