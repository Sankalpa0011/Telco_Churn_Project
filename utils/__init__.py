"""
Telco Customer Churn Prediction - Utilities Package

Contains utility modules for configuration and common operations.
"""

from .config import Config, load_config, get_data_paths, get_project_root

__all__ = [
    'Config',
    'load_config',
    'get_data_paths',
    'get_project_root'
]
