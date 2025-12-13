"""
Feature Engineering Module for Telco Customer Churn Analysis

Handles feature creation, transformation, and encoding.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main class for feature engineering operations."""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure-based features."""
        df = df.copy()
        
        # Tenure buckets
        df['tenure_bucket'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12_months', '13-24_months', '25-48_months', '49-72_months'],
            include_lowest=True
        )
        
        # Tenure in years
        df['tenure_years'] = df['tenure'] / 12
        
        # Is new customer (less than 6 months)
        df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
        
        # Is loyal customer (more than 48 months)
        df['is_loyal_customer'] = (df['tenure'] > 48).astype(int)
        
        logger.info("Created tenure features")
        return df
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial/charges-based features."""
        df = df.copy()
        
        # Charges bucket
        df['charges_bucket'] = pd.cut(
            df['MonthlyCharges'],
            bins=[0, 35, 70, 90, 120],
            labels=['Low', 'Medium', 'High', 'Premium'],
            include_lowest=True
        )
        
        # Average charges per month of tenure
        df['avg_charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Charges difference from total expected
        expected_total = df['MonthlyCharges'] * df['tenure']
        df['charges_difference'] = df['TotalCharges'] - expected_total
        
        # High value customer
        median_charges = df['MonthlyCharges'].median()
        df['is_high_value'] = (df['MonthlyCharges'] > median_charges).astype(int)
        
        logger.info("Created financial features")
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create service-related features."""
        df = df.copy()
        
        # List of service columns
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count total services
        def count_services(row):
            count = 0
            for col in service_cols:
                if col in row.index:
                    val = row[col]
                    if val in ['Yes', 'DSL', 'Fiber optic']:
                        count += 1
            return count
        
        df['total_services'] = df.apply(count_services, axis=1)
        
        # Has internet service
        df['has_internet'] = df['InternetService'].apply(
            lambda x: 0 if x == 'No' else 1
        )
        
        # Has phone service
        df['has_phone'] = df['PhoneService'].apply(
            lambda x: 1 if x == 'Yes' else 0
        )
        
        # Has premium services (security, backup, protection, support)
        premium_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        
        def has_premium(row):
            for col in premium_cols:
                if col in row.index and row[col] == 'Yes':
                    return 1
            return 0
        
        df['has_premium_services'] = df.apply(has_premium, axis=1)
        
        # Has streaming services
        streaming_cols = ['StreamingTV', 'StreamingMovies']
        
        def has_streaming(row):
            for col in streaming_cols:
                if col in row.index and row[col] == 'Yes':
                    return 1
            return 0
        
        df['has_streaming'] = df.apply(has_streaming, axis=1)
        
        logger.info("Created service features")
        return df
    
    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contract-related features."""
        df = df.copy()
        
        # Contract risk score (month-to-month is highest risk)
        contract_risk = {
            'Month-to-month': 3,
            'One year': 2,
            'Two year': 1
        }
        df['contract_risk_score'] = df['Contract'].map(contract_risk)
        
        # Is month-to-month
        df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
        
        # Payment risk (electronic check has highest churn)
        payment_risk = {
            'Electronic check': 3,
            'Mailed check': 2,
            'Bank transfer (automatic)': 1,
            'Credit card (automatic)': 1
        }
        df['payment_risk_score'] = df['PaymentMethod'].map(payment_risk)
        
        # Is automatic payment
        df['is_auto_payment'] = df['PaymentMethod'].apply(
            lambda x: 1 if 'automatic' in x.lower() else 0
        )
        
        logger.info("Created contract features")
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic-based features."""
        df = df.copy()
        
        # Family status
        df['has_family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
        
        # Single senior
        df['is_single_senior'] = (
            (df['SeniorCitizen'] == 1) & 
            (df['Partner'] == 'No') & 
            (df['Dependents'] == 'No')
        ).astype(int)
        
        logger.info("Created demographic features")
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = self.create_tenure_features(df)
        df = self.create_financial_features(df)
        df = self.create_service_features(df)
        df = self.create_contract_features(df)
        df = self.create_demographic_features(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, 
                          columns: List[str] = None,
                          method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (auto-detect if None)
            method: 'onehot' or 'label'
            
        Returns:
            Encoded DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
            logger.info(f"One-hot encoded {len(columns)} columns")
        
        elif method == 'label':
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            logger.info(f"Label encoded {len(columns)} columns")
        
        return df
    
    def scale_numeric(self, df: pd.DataFrame,
                     columns: List[str] = None,
                     method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: 'standard' or 'minmax'
            
        Returns:
            Scaled DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers['numeric'] = scaler
        
        logger.info(f"Scaled {len(columns)} numeric columns using {method}")
        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for feature engineering."""
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("Import and use FeatureEngineer class or engineer_features function")
