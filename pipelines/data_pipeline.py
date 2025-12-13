"""
Data Processing Pipeline for Telco Customer Churn Analysis

This pipeline handles the complete data processing workflow:
- Data ingestion from CSV
- Data cleaning and transformation  
- Feature engineering
- Train/test splitting
- Artifact saving

Module: CC6058ES Big Data and Visualisation
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data processing pipeline for churn prediction.
    Handles ingestion, cleaning, transformation, and splitting.
    """
    
    def __init__(self, config: Dict = None, use_mlflow: bool = False):
        """Initialize the data pipeline with configuration."""
        self.config = config or self._default_config()
        self.raw_data = None
        self.cleaned_data = None
        self.processed_data = None
        self.pipeline_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # MLflow tracking
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = None
        
        if self.use_mlflow:
            self._setup_mlflow()
        
        logger.info(f"Pipeline initialized - Timestamp: {self.pipeline_timestamp}")
        if self.use_mlflow:
            logger.info("MLflow tracking: ENABLED")
    
    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            from utils.mlflow_utils import MLflowTracker
            self.mlflow_tracker = MLflowTracker()
            logger.info("MLflow tracker initialized")
        except ImportError as e:
            logger.warning(f"MLflow not available: {e}")
            self.use_mlflow = False
    
    def _default_config(self) -> Dict:
        """Return default pipeline configuration."""
        return {
            'target_column': 'Churn',
            'id_column': 'customerID',
            'test_size': 0.2,
            'random_state': 42,
            'numeric_columns': ['tenure', 'MonthlyCharges', 'TotalCharges'],
            'categorical_columns': [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
        }
    
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Raw DataFrame
        """
        logger.info(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.raw_data = pd.read_csv(file_path)
        
        logger.info(f"Data loaded successfully")
        logger.info(f"  Shape: {self.raw_data.shape}")
        logger.info(f"  Columns: {len(self.raw_data.columns)}")
        
        return self.raw_data
    
    def explore_data(self, df: pd.DataFrame = None) -> Dict:
        """
        Perform exploratory data analysis.
        
        Args:
            df: DataFrame to explore (uses raw_data if not provided)
            
        Returns:
            Dictionary with exploration results
        """
        df = df if df is not None else self.raw_data
        
        if df is None:
            raise ValueError("No data available. Run ingest_data first.")
        
        exploration = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        # Target distribution
        if self.config['target_column'] in df.columns:
            target_dist = df[self.config['target_column']].value_counts()
            exploration['target_distribution'] = target_dist.to_dict()
            exploration['churn_rate'] = (target_dist.get('Yes', 0) / len(df)) * 100
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            exploration['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        logger.info("Data exploration completed")
        logger.info(f"  Total records: {exploration['shape'][0]:,}")
        logger.info(f"  Total features: {exploration['shape'][1]}")
        logger.info(f"  Missing values: {sum(exploration['missing_values'].values())}")
        
        return exploration
    
    def clean_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and data type issues.
        
        Args:
            df: DataFrame to clean (uses raw_data if not provided)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy() if df is not None else self.raw_data.copy()
        
        if df is None:
            raise ValueError("No data available. Run ingest_data first.")
        
        initial_rows = len(df)
        logger.info("Starting data cleaning...")
        
        # Remove ID column (not a feature)
        if self.config['id_column'] in df.columns:
            df = df.drop(columns=[self.config['id_column']])
            logger.info(f"  Removed ID column: {self.config['id_column']}")
        
        # Handle TotalCharges - convert to numeric
        if 'TotalCharges' in df.columns:
            # Replace blank strings with NaN
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            logger.info("  Converted TotalCharges to numeric")
        
        # Drop rows with missing TotalCharges (new customers with 0 tenure)
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        
        rows_removed = initial_rows - len(df)
        logger.info(f"  Rows removed: {rows_removed}")
        logger.info(f"  Missing values handled: {missing_before - missing_after}")
        
        self.cleaned_data = df
        logger.info(f"Data cleaning completed - Final shape: {df.shape}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            df: DataFrame to transform (uses cleaned_data if not provided)
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy() if df is not None else self.cleaned_data.copy()
        
        if df is None:
            raise ValueError("No data available. Run clean_data first.")
        
        logger.info("Starting feature engineering...")
        
        # Tenure buckets
        df['tenure_bucket'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12', '13-24', '25-48', '49-72'],
            include_lowest=True
        )
        logger.info("  Created tenure_bucket feature")
        
        # Monthly charges buckets
        df['charges_bucket'] = pd.cut(
            df['MonthlyCharges'],
            bins=[0, 30, 60, 90, 120],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        logger.info("  Created charges_bucket feature")
        
        # Service count - count of services subscribed
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        def count_services(row):
            count = 0
            for col in service_cols:
                if col in row.index:
                    if row[col] in ['Yes', 'DSL', 'Fiber optic']:
                        count += 1
            return count
        
        df['service_count'] = df.apply(count_services, axis=1)
        logger.info("  Created service_count feature")
        
        # Average monthly revenue per tenure month
        df['avg_charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
        logger.info("  Created avg_charges_per_month feature")
        
        # High value customer flag
        df['high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
        logger.info("  Created high_value feature")
        
        # Contract risk score (higher for month-to-month)
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        df['contract_risk'] = df['Contract'].map(contract_risk)
        logger.info("  Created contract_risk feature")
        
        self.processed_data = df
        logger.info(f"Feature engineering completed - New shape: {df.shape}")
        
        return df
    
    def encode_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            df: DataFrame to encode (uses processed_data if not provided)
            
        Returns:
            Encoded DataFrame
        """
        df = df.copy() if df is not None else self.processed_data.copy()
        
        if df is None:
            raise ValueError("No data available. Run engineer_features first.")
        
        logger.info("Starting feature encoding...")
        
        # Binary encoding for target
        if self.config['target_column'] in df.columns:
            df[self.config['target_column']] = df[self.config['target_column']].map({'Yes': 1, 'No': 0})
            logger.info(f"  Encoded target: {self.config['target_column']}")
        
        # Binary encoding for simple yes/no columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
        logger.info(f"  Encoded binary columns: {len(binary_cols)}")
        
        # Gender encoding
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
            logger.info("  Encoded gender column")
        
        # One-hot encoding for multi-category columns
        multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                         'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod',
                         'tenure_bucket', 'charges_bucket']
        
        existing_cols = [col for col in multi_cat_cols if col in df.columns]
        if existing_cols:
            df = pd.get_dummies(df, columns=existing_cols, drop_first=True)
            logger.info(f"  One-hot encoded columns: {len(existing_cols)}")
        
        logger.info(f"Feature encoding completed - Final shape: {df.shape}")
        
        return df
    
    def split_data(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df: DataFrame to split (uses encoded data if not provided)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        if df is None:
            raise ValueError("No data available. Run encode_features first.")
        
        logger.info("Splitting data into train/test sets...")
        
        # Separate features and target
        target_col = self.config['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Testing set: {X_test.shape}")
        logger.info(f"  Train churn rate: {y_train.mean()*100:.2f}%")
        logger.info(f"  Test churn rate: {y_test.mean()*100:.2f}%")
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, X_train, X_test, y_train, y_test, output_dir: str) -> Dict[str, str]:
        """
        Save processed data artifacts.
        
        Args:
            X_train, X_test, y_train, y_test: Split datasets
            output_dir: Directory to save artifacts
            
        Returns:
            Dictionary with saved file paths
        """
        logger.info(f"Saving artifacts to: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped subdirectory
        run_dir = os.path.join(output_dir, f"run_{self.pipeline_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save datasets
        artifacts = {}
        
        # Training data
        X_train.to_csv(os.path.join(run_dir, 'X_train.csv'), index=False)
        artifacts['X_train'] = os.path.join(run_dir, 'X_train.csv')
        
        X_test.to_csv(os.path.join(run_dir, 'X_test.csv'), index=False)
        artifacts['X_test'] = os.path.join(run_dir, 'X_test.csv')
        
        y_train.to_csv(os.path.join(run_dir, 'y_train.csv'), index=False)
        artifacts['y_train'] = os.path.join(run_dir, 'y_train.csv')
        
        y_test.to_csv(os.path.join(run_dir, 'y_test.csv'), index=False)
        artifacts['y_test'] = os.path.join(run_dir, 'y_test.csv')
        
        # Save pipeline metadata
        metadata = {
            'timestamp': self.pipeline_timestamp,
            'config': self.config,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'features': list(X_train.columns)
        }
        
        with open(os.path.join(run_dir, 'pipeline_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        artifacts['metadata'] = os.path.join(run_dir, 'pipeline_metadata.json')
        
        logger.info(f"Artifacts saved successfully - {len(artifacts)} files")
        
        return artifacts
    
    def run(self, data_path: str, output_dir: str = None) -> Dict:
        """
        Execute the complete data pipeline.
        
        Args:
            data_path: Path to input CSV file
            output_dir: Directory to save artifacts (optional)
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()        
        # Start MLflow run if enabled
        if self.use_mlflow and self.mlflow_tracker:
            self.mlflow_tracker.start_run(
                run_name="data_pipeline",
                tags={'pipeline': 'data_processing'}
            )
        
        try:
            # Step 1: Ingest data
            self.ingest_data(data_path)
            
            # Step 2: Explore data
            exploration = self.explore_data()
            
            # Step 3: Clean data
            cleaned = self.clean_data()
            
            # Step 4: Engineer features
            engineered = self.engineer_features()
            
            # Step 5: Encode features
            encoded = self.encode_features(engineered)
            
            # Step 6: Split data
            X_train, X_test, y_train, y_test = self.split_data(encoded)
            
            # Step 7: Save artifacts (if output directory provided)
            artifacts = {}
            if output_dir:
                artifacts = self.save_artifacts(X_train, X_test, y_train, y_test, output_dir)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'status': 'success',
                'timestamp': self.pipeline_timestamp,
                'duration_seconds': duration,
                'exploration': exploration,
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'artifacts': artifacts,
                'data': {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
            }
            
            logger.info("=" * 60)
            logger.info(f"PIPELINE COMPLETED - Duration: {duration:.2f}s")
            logger.info("=" * 60)
            
            return results
            
        finally:
            # End MLflow run
            if self.use_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.end_run()


def main(use_mlflow: bool = False):
    """Main entry point for the data pipeline."""
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'raw', 'Telco_Customer_Churn_Dataset.csv')
    output_dir = os.path.join(project_root, 'artifacts', 'data_pipeline')
    
    # Initialize and run pipeline
    pipeline = DataPipeline(use_mlflow=use_mlflow)
    results = pipeline.run(data_path, output_dir)
    
    print("\nPipeline Results:")
    print(f"  Status: {results['status']}")
    print(f"  Duration: {results['duration_seconds']:.2f} seconds")
    print(f"  Train Shape: {results['train_shape']}")
    print(f"  Test Shape: {results['test_shape']}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    args = parser.parse_args()
    
    main(use_mlflow=args.mlflow)
