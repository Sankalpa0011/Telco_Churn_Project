"""
Data Ingestion Module for Telco Customer Churn Analysis

Handles loading and initial validation of data from various sources.
"""

import os
import logging
import pandas as pd
from typing import Optional, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestor:
    """Base class for data ingestion."""
    
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load(self, source: str) -> pd.DataFrame:
        """Load data from source."""
        raise NotImplementedError
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate loaded data."""
        raise NotImplementedError


class CSVDataIngestor(DataIngestor):
    """CSV file data ingestor."""
    
    def __init__(self, required_columns: List[str] = None):
        super().__init__()
        self.required_columns = required_columns or []
    
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading CSV file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = pd.read_csv(file_path, **kwargs)
        
        self.metadata = {
            'source': file_path,
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'file_size_mb': os.path.getsize(file_path) / (1024**2)
        }
        
        logger.info(f"Loaded {self.metadata['rows']:,} rows, {self.metadata['columns']} columns")
        
        return self.data
    
    def validate(self, df: pd.DataFrame = None) -> Dict:
        """
        Validate the loaded data.
        
        Args:
            df: DataFrame to validate (uses self.data if not provided)
            
        Returns:
            Validation results dictionary
        """
        df = df if df is not None else self.data
        
        if df is None:
            raise ValueError("No data to validate")
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for required columns
        if self.required_columns:
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                validation['is_valid'] = False
                validation['errors'].append(f"Missing required columns: {missing}")
        
        # Check for empty DataFrame
        if len(df) == 0:
            validation['is_valid'] = False
            validation['errors'].append("DataFrame is empty")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 10:
            validation['warnings'].append(f"High missing value percentage: {missing_pct:.1f}%")
        
        return validation


class TelcoDataIngestor(CSVDataIngestor):
    """Specialized ingestor for Telco Churn dataset."""
    
    REQUIRED_COLUMNS = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    def __init__(self):
        super().__init__(required_columns=self.REQUIRED_COLUMNS)
    
    def load_and_validate(self, file_path: str) -> pd.DataFrame:
        """Load and validate Telco churn data in one step."""
        df = self.load(file_path)
        validation = self.validate(df)
        
        if not validation['is_valid']:
            raise ValueError(f"Data validation failed: {validation['errors']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(warning)
        
        return df
    
    def get_data_summary(self) -> Dict:
        """Get a summary of the loaded Telco data."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        df = self.data
        
        summary = {
            'total_customers': len(df),
            'churn_rate': (df['Churn'] == 'Yes').mean() * 100 if 'Churn' in df.columns else None,
            'avg_tenure': df['tenure'].mean() if 'tenure' in df.columns else None,
            'avg_monthly_charges': df['MonthlyCharges'].mean() if 'MonthlyCharges' in df.columns else None,
            'contract_distribution': df['Contract'].value_counts().to_dict() if 'Contract' in df.columns else None
        }
        
        return summary


def load_telco_data(file_path: str) -> pd.DataFrame:
    """
    Convenience function to load Telco churn data.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Loaded and validated DataFrame
    """
    ingestor = TelcoDataIngestor()
    return ingestor.load_and_validate(file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "../data/raw/Telco_Customer_Churn_Dataset.csv"
    
    try:
        df = load_telco_data(file_path)
        print(f"\nData loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nColumn dtypes:\n{df.dtypes}")
    except Exception as e:
        print(f"Error: {e}")
