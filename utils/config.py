import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_data_paths() -> Dict[str, str]:
    """Return standard data paths."""
    return {
        'raw_data': str(PROJECT_ROOT / 'data' / 'raw'),
        'processed_data': str(PROJECT_ROOT / 'data' / 'processed'),
        'artifacts': str(PROJECT_ROOT / 'artifacts'),
        'models': str(PROJECT_ROOT / 'models'),
        'notebooks': str(PROJECT_ROOT / 'notebooks')
    }


def get_default_config() -> Dict[str, Any]:
    """Return default configuration settings."""
    return {
        'data': {
            'raw_file': 'Telco_Customer_Churn_Dataset.csv',
            'target_column': 'Churn',
            'id_column': 'customerID',
            'test_size': 0.2,
            'random_state': 42
        },
        'features': {
            'numeric_columns': ['tenure', 'MonthlyCharges', 'TotalCharges'],
            'categorical_columns': [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ],
            'binary_columns': ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        },
        'models': {
            'logistic_regression': {
                'enabled': True,
                'params': {'max_iter': 1000, 'random_state': 42}
            },
            'decision_tree': {
                'enabled': True,
                'params': {'max_depth': 10, 'random_state': 42}
            },
            'random_forest': {
                'enabled': True,
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            },
            'gradient_boosting': {
                'enabled': True,
                'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
            }
        },
        'evaluation': {
            'primary_metric': 'f1_score',
            'threshold': 0.5
        },
        'kafka': {
            'bootstrap_servers': 'localhost:9092',
            'topic': 'customer-events',
            'consumer_group': 'churn-predictor'
        }
    }


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return defaults.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = PROJECT_ROOT / 'config.yaml'
    
    if os.path.exists(config_path):
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        logger.info("Using default configuration")
        return get_default_config()


def save_config(config: Dict[str, Any], config_path: str = None):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    if config_path is None:
        config_path = PROJECT_ROOT / 'config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to: {config_path}")


class Config:
    """Configuration class with dot-notation access."""
    
    def __init__(self, config_dict: Dict = None):
        self._config = config_dict or load_config()
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict:
        return self._config


# Global config instance
config = Config()


if __name__ == "__main__":
    print("Project Configuration")
    print("=" * 40)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nData Paths:")
    for name, path in get_data_paths().items():
        print(f"  {name}: {path}")
    print(f"\nDefault Config Keys: {list(get_default_config().keys())}")
