"""
Model Building Module for Telco Customer Churn Prediction

Contains model builders for various ML algorithms.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBuilder(ABC):
    """Abstract base class for model builders."""
    
    @abstractmethod
    def build(self, **kwargs) -> Any:
        """Build and return a model instance."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict:
        """Return default parameters for the model."""
        pass


class LogisticRegressionBuilder(ModelBuilder):
    """Builder for Logistic Regression model."""
    
    def get_default_params(self) -> Dict:
        return {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs',
            'class_weight': 'balanced'
        }
    
    def build(self, **kwargs) -> LogisticRegression:
        params = self.get_default_params()
        params.update(kwargs)
        logger.info(f"Building LogisticRegression with params: {params}")
        return LogisticRegression(**params)


class DecisionTreeBuilder(ModelBuilder):
    """Builder for Decision Tree model."""
    
    def get_default_params(self) -> Dict:
        return {
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    def build(self, **kwargs) -> DecisionTreeClassifier:
        params = self.get_default_params()
        params.update(kwargs)
        logger.info(f"Building DecisionTree with params: {params}")
        return DecisionTreeClassifier(**params)


class RandomForestBuilder(ModelBuilder):
    """Builder for Random Forest model."""
    
    def get_default_params(self) -> Dict:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
    
    def build(self, **kwargs) -> RandomForestClassifier:
        params = self.get_default_params()
        params.update(kwargs)
        logger.info(f"Building RandomForest with params: {params}")
        return RandomForestClassifier(**params)


class GradientBoostingBuilder(ModelBuilder):
    """Builder for Gradient Boosting model."""
    
    def get_default_params(self) -> Dict:
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        }
    
    def build(self, **kwargs) -> GradientBoostingClassifier:
        params = self.get_default_params()
        params.update(kwargs)
        logger.info(f"Building GradientBoosting with params: {params}")
        return GradientBoostingClassifier(**params)


class SVMBuilder(ModelBuilder):
    """Builder for Support Vector Machine model."""
    
    def get_default_params(self) -> Dict:
        return {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    def build(self, **kwargs) -> SVC:
        params = self.get_default_params()
        params.update(kwargs)
        logger.info(f"Building SVM with params: {params}")
        return SVC(**params)


class KNNBuilder(ModelBuilder):
    """Builder for K-Nearest Neighbors model."""
    
    def get_default_params(self) -> Dict:
        return {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'minkowski',
            'n_jobs': -1
        }
    
    def build(self, **kwargs) -> KNeighborsClassifier:
        params = self.get_default_params()
        params.update(kwargs)
        logger.info(f"Building KNN with params: {params}")
        return KNeighborsClassifier(**params)


class ModelFactory:
    """Factory class for creating models."""
    
    _builders = {
        'logistic_regression': LogisticRegressionBuilder,
        'decision_tree': DecisionTreeBuilder,
        'random_forest': RandomForestBuilder,
        'gradient_boosting': GradientBoostingBuilder,
        'svm': SVMBuilder,
        'knn': KNNBuilder
    }
    
    @classmethod
    def get_available_models(cls) -> list:
        """Return list of available model names."""
        return list(cls._builders.keys())
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model parameters
            
        Returns:
            Model instance
        """
        if model_name not in cls._builders:
            raise ValueError(f"Unknown model: {model_name}. Available: {cls.get_available_models()}")
        
        builder = cls._builders[model_name]()
        return builder.build(**kwargs)
    
    @classmethod
    def get_default_params(cls, model_name: str) -> Dict:
        """Get default parameters for a model."""
        if model_name not in cls._builders:
            raise ValueError(f"Unknown model: {model_name}")
        
        builder = cls._builders[model_name]()
        return builder.get_default_params()


def create_model(model_name: str, **kwargs) -> Any:
    """Convenience function to create a model."""
    return ModelFactory.create_model(model_name, **kwargs)


def get_all_models(**kwargs) -> Dict[str, Any]:
    """Create instances of all available models."""
    models = {}
    for name in ModelFactory.get_available_models():
        models[name] = ModelFactory.create_model(name, **kwargs)
    return models


if __name__ == "__main__":
    print("Available Models:")
    for model_name in ModelFactory.get_available_models():
        print(f"  - {model_name}")
        params = ModelFactory.get_default_params(model_name)
        print(f"    Default params: {params}")
