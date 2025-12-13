"""
Telco Customer Churn Prediction - Source Package

This package contains modules for:
- Data ingestion and validation
- Feature engineering
- Model building
- Model evaluation
"""

from .data_ingestion import CSVDataIngestor, TelcoDataIngestor, load_telco_data
from .feature_engineering import FeatureEngineer, engineer_features
from .model_building import ModelFactory, create_model, get_all_models
from .model_evaluation import ModelEvaluator, compare_models

__all__ = [
    'CSVDataIngestor',
    'TelcoDataIngestor',
    'load_telco_data',
    'FeatureEngineer',
    'engineer_features',
    'ModelFactory',
    'create_model',
    'get_all_models',
    'ModelEvaluator',
    'compare_models'
]

__version__ = '1.0.0'
