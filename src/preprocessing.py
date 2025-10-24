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
        setup_logger,load_csv_file,read_yaml,save_metadata,save_csv
)

# setup logging
base_dir = Path.cwd()
logs_dir = base_dir / 'logs'
models_dir = base_dir / 'models'

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
        self.scaler = None
        self.target_encoders = {}
        
		# Track transformations
        self.transformation_log = []
        self.metadata = {
            'config_version' : self.config['project']['version'],
            'config_file' : config_path,
            'timestamp_start' : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		}

        project_name = self.config['project']['name']
        version_name = self.config['project']['version']
        
        log.info(f'Project : {project_name}')
        log.info(f'Version : {version_name}')
        log.info('Configurations loaded successfully')
        
    def _log_transformation(self, step: str, details: Dict[str, Any]):
        """Log transformation step for audit trail"""
        entry = {
            'step': step,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **details
        }
        self.transformation_log.append(entry)
        log.info(f"✓ {step}: {details}")
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial data validation"""
        log.info('-'*70)
        log.info('STEP 1: DATA VALIDATION')
        log.info('-'*70)
        
        initial_shape = df.shape
        log.info(f"Initial shape: {initial_shape}")
        
        # Check for completely empty dataframe
        if df.empty:
            raise ValueError("DataFrame is empty!")
        
        self._log_transformation('data_validation', {
            'initial_rows': initial_shape[0],
            'initial_columns': initial_shape[1],
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        })
        
        return df
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns based on config"""
        log.info('-'*70)
        log.info('STEP 2: DROPPING COLUMNS')
        log.info('-'*70)
        
        if 'columns_to_drop' not in self.config or not self.config['columns_to_drop']:
            log.info("No columns to drop")
            return df
        
        columns_to_drop = []
        for item in self.config['columns_to_drop']:
            col = item['column']
            reason = item['reason']
            
            if col in df.columns:
                columns_to_drop.append(col)
                log.info(f"Dropping '{col}': {reason}")
            else:
                log.warning(f"Column '{col}' not found in dataframe")
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            self._log_transformation('drop_columns', {
                'columns_dropped': columns_to_drop,
                'remaining_columns': df.shape[1]
            })
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on config"""
        log.info('-'*70)
        log.info('STEP 3: HANDLING MISSING VALUES')
        log.info('-'*70)
        
        missing_before = df.isnull().sum().sum()
        log.info(f"Total missing values before: {missing_before}")
        
        # Handle numeric columns
        if 'missing_values' in self.config and 'numeric' in self.config['missing_values']:
            for col, strategy in self.config['missing_values']['numeric'].items():
                if col not in df.columns:
                    log.warning(f"Column '{col}' not found, skipping")
                    continue
                
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    log.info(f"'{col}': No missing values")
                    continue
                
                method = strategy['method']
                log.info(f"'{col}': {missing_count} missing ({missing_count/len(df)*100:.2f}%) - Method: {method}")
                
                if method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif method == 'calculated':
                    # Handle Total Spent calculation
                    if col == 'Total Spent' or col == 'Total_Spent':
                        price_col = 'Price Per Unit' if 'Price Per Unit' in df.columns else 'Price_Per_Unit'
                        qty_col = 'Quantity'
                        if price_col in df.columns and qty_col in df.columns:
                            mask = df[col].isnull()
                            df.loc[mask, col] = df.loc[mask, price_col] * df.loc[mask, qty_col]
                            log.info(f"Calculated {mask.sum()} missing values from {price_col} * {qty_col}")
        
        # Handle categorical columns
        if 'missing_values' in self.config and 'categorical' in self.config['missing_values']:
            for col, strategy in self.config['missing_values']['categorical'].items():
                if col not in df.columns:
                    log.warning(f"Column '{col}' not found, skipping")
                    continue
                
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    log.info(f"'{col}': No missing values")
                    continue
                
                method = strategy['method']
                log.info(f"'{col}': {missing_count} missing ({missing_count/len(df)*100:.2f}%) - Method: {method}")
                
                if method == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif method == 'constant' or 'not specified' in method.lower():
                    fill_value = strategy.get('fill_value', 'Not Specified')
                    df[col].fillna(fill_value, inplace=True)
                    log.info(f"Filled with: '{fill_value}'")
                elif method == 'calculated':
                    # Handle Item calculation from Category
                    if col == 'Item' and 'Category' in df.columns:
                        # For now, fill with 'Unknown_' + Category
                        mask = df[col].isnull()
                        df.loc[mask, col] = 'Unknown_' + df.loc[mask, 'Category'].astype(str)
                        log.info(f"Calculated {mask.sum()} missing items from Category")
        
        missing_after = df.isnull().sum().sum()
        log.info(f"Total missing values after: {missing_after}")
        
        self._log_transformation('handle_missing_values', {
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'missing_removed': int(missing_before - missing_after)
        })
        
        return df
    
    def _fix_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data quality issues"""
        log.info('-'*70)
        log.info('STEP 4: DATA QUALITY FIXES')
        log.info('-'*70)
        
        # Recalculate Total Spent to ensure consistency
        if 'data_quality' in self.config:
            price_col = 'Price Per Unit' if 'Price Per Unit' in df.columns else 'Price_Per_Unit'
            qty_col = 'Quantity'
            total_col = 'Total Spent' if 'Total Spent' in df.columns else 'Total_Spent'
            
            if all(col in df.columns for col in [price_col, qty_col, total_col]):
                # Calculate expected total
                calculated_total = df[price_col] * df[qty_col]
                
                # Find mismatches
                mismatches = ~np.isclose(calculated_total, df[total_col], rtol=1e-5)
                mismatch_count = mismatches.sum()
                
                if mismatch_count > 0:
                    log.warning(f"Found {mismatch_count} mismatches in Total Spent")
                    log.info(f"Recalculating Total Spent = {price_col} * {qty_col}")
                    df[total_col] = calculated_total
                    
                    self._log_transformation('fix_total_spent', {
                        'mismatches_fixed': int(mismatch_count),
                        'percentage': round(mismatch_count/len(df)*100, 2)
                    })
                else:
                    log.info("✓ Total Spent calculation verified - all correct")
        
        return df
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        log.info('-'*70)
        log.info('STEP 5: DATETIME FEATURE EXTRACTION')
        log.info('-'*70)
        
        if 'datetime' not in self.config or not self.config['datetime'].get('date_column'):
            log.info("No datetime columns to process")
            return df
        
        date_columns = self.config['datetime']['date_column']
        if isinstance(date_columns, str):
            date_columns = [date_columns]
        
        date_format = self.config['datetime'].get('date_format', ['%Y-%m-%d'])[0]
        features_to_extract = self.config['datetime'].get('extract_features', [])
        
        for date_col in date_columns:
            if date_col not in df.columns:
                log.warning(f"Date column '{date_col}' not found")
                continue
            
            log.info(f"Processing '{date_col}'...")
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
            
            # Extract features
            extracted = []
            if 'year' in features_to_extract:
                df['year'] = df[date_col].dt.year
                extracted.append('year')
            
            if 'month' in features_to_extract:
                df['month'] = df[date_col].dt.month
                extracted.append('month')
            
            if 'day' in features_to_extract:
                df['day'] = df[date_col].dt.day
                extracted.append('day')
            
            if 'day_of_week' in features_to_extract:
                df['day_of_week'] = df[date_col].dt.dayofweek
                extracted.append('day_of_week')
            
            if 'quarter' in features_to_extract:
                df['quarter'] = df[date_col].dt.quarter
                extracted.append('quarter')
            
            if 'Is_weekend' in features_to_extract or 'is_weekend' in features_to_extract:
                df['Is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
                extracted.append('Is_weekend')
            
            if 'Is_month_start' in features_to_extract or 'is_month_start' in features_to_extract:
                df['Is_month_start'] = df[date_col].dt.is_month_start.astype(int)
                extracted.append('Is_month_start')
            
            if 'Is_month_end' in features_to_extract or 'is_month_end' in features_to_extract:
                df['Is_month_end'] = df[date_col].dt.is_month_end.astype(int)
                extracted.append('Is_month_end')
            
            log.info(f"Extracted features: {extracted}")
            
            # Drop original column if specified
            if self.config['datetime'].get('drop_column', False):
                df = df.drop(columns=[date_col])
                log.info(f"Dropped original column '{date_col}'")
        
        self._log_transformation('datetime_extraction', {
            'features_extracted': extracted,
            'new_columns': len(extracted)
        })
        
        return df
    
    def _encode_features(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Encode categorical features"""
        log.info('-'*70)
        log.info('STEP 6: FEATURE ENCODING')
        log.info('-'*70)
        
        if 'encoding' not in self.config:
            log.info("No encoding configuration found")
            return df
        
        # One-hot encoding
        if 'one_hot' in self.config['encoding'] or 'onehot' in self.config['encoding']:
            onehot_config = self.config['encoding'].get('one_hot', self.config['encoding'].get('onehot', []))
            
            for item in onehot_config:
                col = item['column']
                if col not in df.columns:
                    log.warning(f"Column '{col}' not found, skipping")
                    continue
                
                drop_first = item.get('drop_first', True)
                log.info(f"One-hot encoding '{col}' (drop_first={drop_first})")
                
                # Perform one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                
                log.info(f"Created {len(dummies.columns)} dummy columns")
        
        # Target encoding (only during fit, needs target variable)
        if 'target' in self.config['encoding'] and target is not None:
            for item in self.config['encoding']['target']:
                col = item['column']
                if col not in df.columns:
                    log.warning(f"Column '{col}' not found, skipping")
                    continue
                
                smoothing = item.get('smoothing', 1.0)
                log.info(f"Target encoding '{col}' (smoothing={smoothing})")
                
                # Initialize encoder if not exists
                if col not in self.target_encoders:
                    self.target_encoders[col] = TargetEncoder(smoothing=smoothing)
                    df[col] = self.target_encoders[col].fit_transform(df[col], target)
                    log.info(f"Fitted and transformed '{col}'")
                else:
                    df[col] = self.target_encoders[col].transform(df[col])
                    log.info(f"Transformed '{col}' using existing encoder")
        
        self._log_transformation('encoding', {
            'onehot_columns': len(self.config['encoding'].get('one_hot', [])),
            'target_encoded': len(self.target_encoders)
        })
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features"""
        log.info('-'*70)
        log.info('STEP 7: FEATURE SCALING')
        log.info('-'*70)
        
        if 'scaling' not in self.config:
            log.info("No scaling configuration found")
            return df
        
        method = self.config['scaling'].get('method', 'standard')
        columns_to_scale = self.config['scaling'].get('columns', [])
        exclude = self.config['scaling'].get('exclude', [])
        
        # Filter columns that exist and are not excluded
        columns_to_scale = [col for col in columns_to_scale if col in df.columns and col not in exclude]
        
        if not columns_to_scale:
            log.info("No columns to scale")
            return df
        
        log.info(f"Scaling method: {method}")
        log.info(f"Columns to scale: {columns_to_scale}")
        
        if self.scaler is None:
            # Fit scaler
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                log.warning(f"Scaling method '{method}' not implemented, using standard")
                self.scaler = StandardScaler()
            
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            log.info("✓ Scaler fitted and applied")
        else:
            # Transform only
            df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
            log.info("✓ Scaler applied (using existing fit)")
        
        self._log_transformation('scaling', {
            'method': method,
            'columns_scaled': columns_to_scale,
            'scaler_fitted': self.scaler is not None
        })
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit preprocessor on training data and transform
        
        Args:
            df: Training DataFrame
            target_col: Name of target column (for target encoding)
            
        Returns:
            Transformed DataFrame
        """
        log.info('\n' + '='*70)
        log.info('STARTING FIT_TRANSFORM (TRAINING DATA)')
        log.info('='*70)
        
        # Separate target if provided
        target = df[target_col] if target_col and target_col in df.columns else None
        
        # Apply transformations
        df = self._validate_data(df)
        df = self._drop_columns(df)
        df = self._handle_missing_values(df)
        df = self._fix_data_quality(df)
        df = self._extract_datetime_features(df)
        df = self._encode_features(df, target)
        df = self._scale_features(df)
        
        # Save transformers
        self._save_transformers()
        
        # Save metadata
        self.metadata['timestamp_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['final_shape'] = df.shape
        self.metadata['transformations'] = self.transformation_log
        save_metadata(self.metadata, self.config['paths']['preprocessing_metadata'])
        
        log.info('='*70)
        log.info('✓ FIT_TRANSFORM COMPLETED')
        log.info(f"Final shape: {df.shape}")
        log.info('='*70)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        log.info('\n' + '='*70)
        log.info('STARTING TRANSFORM (TEST/NEW DATA)')
        log.info('='*70)
        
        # Apply transformations (without fitting)
        df = self._validate_data(df)
        df = self._drop_columns(df)
        df = self._handle_missing_values(df)
        df = self._fix_data_quality(df)
        df = self._extract_datetime_features(df)
        df = self._encode_features(df, target=None)  # No target for test data
        df = self._scale_features(df)
        
        log.info('='*70)
        log.info('✓ TRANSFORM COMPLETED')
        log.info(f"Final shape: {df.shape}")
        log.info('='*70)
        
        return df
    
    def _save_transformers(self):
        """Save fitted transformers to disk"""
        Path('models').mkdir(exist_ok=True)
        
        
        if self.scaler:
            scaler_path = self.config['paths']['scaler_path']
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            log.info(f"✓ Scaler saved to {scaler_path}")
        
        if self.target_encoders:
            encoder_path = self.config['paths']['encoder_path']
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.target_encoders, f)
            log.info(f"✓ Encoders saved to {encoder_path}")
    
    def load_transformers(self):
        """Load fitted transformers from disk"""
        scaler_path = Path(self.config['paths']['scaler_path'])
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            log.info(f"✓ Scaler loaded from {scaler_path}")
        
        encoder_path = Path(self.config['paths']['encoder_path'])
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.target_encoders = pickle.load(f)
            log.info(f"✓ Encoders loaded from {encoder_path}")


def main():
    """Main preprocessing execution"""
    # Load configuration
    config = read_yaml('config/preprocessing_config.yaml')
    
    # Load raw data
    log.info("Loading raw data...")
    df = load_csv_file(config['paths']['raw_data'])
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor('config/preprocessing_config.yaml')
    
    # Fit and transform (assuming no target column for now)
    df_processed = preprocessor.fit_transform(df)
    
    # Save processed data
    save_csv(df_processed, config['paths']['processed_data'])
    
    log.info(f"\n✅ Preprocessing completed successfully!")
    log.info(f"📁 Processed data saved to: {config['paths']['processed_data']}")
    log.info(f"📊 Final shape: {df_processed.shape}")
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print(f"📁 Output: {config['paths']['processed_data']}")
    print(f"📊 Shape: {df_processed.shape}")
    print("="*70)


if __name__ == '__main__':
    main()