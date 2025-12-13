"""
Model Training Pipeline for Telco Customer Churn Prediction

This pipeline handles the complete model training workflow:
- Loading processed data
- Training multiple ML models
- Model evaluation and comparison
- Saving trained models and metrics
- MLflow experiment tracking

"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end model training pipeline for churn prediction.
    Trains multiple models and selects the best performer.
    """
    
    def __init__(self, config: Dict = None, use_mlflow: bool = False):
        """Initialize the training pipeline with configuration."""
        self.config = config or self._default_config()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.pipeline_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # MLflow tracking
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = None
        
        if self.use_mlflow:
            self._setup_mlflow()
        
        logger.info(f"Training Pipeline initialized - Timestamp: {self.pipeline_timestamp}")
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
        """Return default training configuration."""
        return {
            'random_state': 42,
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
            'evaluation_metric': 'f1_score'
        }
    
    def load_data(self, data_dir: str = None, 
                  X_train: pd.DataFrame = None, X_test: pd.DataFrame = None,
                  y_train: pd.Series = None, y_test: pd.Series = None) -> Tuple:
        """
        Load training and testing data.
        
        Args:
            data_dir: Directory containing processed data files
            X_train, X_test, y_train, y_test: Direct data (if already loaded)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if X_train is not None:
            logger.info("Using provided data directly")
            return X_train, X_test, y_train, y_test
        
        if data_dir is None:
            raise ValueError("Either data_dir or direct data must be provided")
        
        logger.info(f"Loading data from: {data_dir}")
        
        # Find latest run directory
        run_dirs = [d for d in os.listdir(data_dir) if d.startswith('run_')]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in {data_dir}")
        
        latest_run = sorted(run_dirs)[-1]
        run_path = os.path.join(data_dir, latest_run)
        
        logger.info(f"Using run: {latest_run}")
        
        # Load data files
        X_train = pd.read_csv(os.path.join(run_path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(run_path, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(run_path, 'y_train.csv')).squeeze()
        y_test = pd.read_csv(os.path.join(run_path, 'y_test.csv')).squeeze()
        
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _get_model_instance(self, model_name: str, params: Dict) -> Any:
        """Get a model instance based on name and parameters."""
        model_classes = {
            'logistic_regression': LogisticRegression,
            'decision_tree': DecisionTreeClassifier,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_classes[model_name](**params)
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, 
                   y_train: pd.Series, params: Dict = None) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            params: Model parameters (optional)
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        # Get parameters from config if not provided
        if params is None:
            params = self.config['models'].get(model_name, {}).get('params', {})
        
        # Create and train model
        start_time = datetime.now()
        model = self._get_model_instance(model_name, params)
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.models[model_name] = model
        logger.info(f"  {model_name} trained in {training_time:.2f}s")
        
        return model
    
    def evaluate_model(self, model: Any, model_name: str, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        self.results[model_name] = metrics
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train and evaluate all configured models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            Dictionary with all model results
        """
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)
        
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', True):
                logger.info(f"Skipping {model_name} (disabled)")
                continue
            
            try:
                # Train model
                model = self.train_model(model_name, X_train, y_train, model_config.get('params'))
                
                # Evaluate model
                self.evaluate_model(model, model_name, X_test, y_test)
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                self.results[model_name] = {'error': str(e)}
        
        return self.results
    
    def select_best_model(self, metric: str = None) -> Tuple[str, Any]:
        """
        Select the best performing model based on a metric.
        
        Args:
            metric: Evaluation metric to use for selection
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        metric = metric or self.config['evaluation_metric']
        
        logger.info(f"Selecting best model based on: {metric}")
        
        best_score = -1
        best_name = None
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
            
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = model_name
        
        if best_name is None:
            raise ValueError("No valid models found")
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"Best model: {best_name} ({metric}: {best_score:.4f})")
        
        return best_name, self.best_model
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance from a model.
        
        Args:
            model_name: Name of the model (uses best model if not provided)
            
        Returns:
            DataFrame with feature importances
        """
        model_name = model_name or self.best_model_name
        model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} doesn't support feature importance")
            return None
        
        # Need feature names - get from last training data
        # This is a simplified version; in production, store feature names during training
        feature_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_df
    
    def save_artifacts(self, output_dir: str, X_train: pd.DataFrame = None) -> Dict[str, str]:
        """
        Save trained models and results.
        
        Args:
            output_dir: Directory to save artifacts
            X_train: Training data (for feature names)
            
        Returns:
            Dictionary with saved file paths
        """
        logger.info(f"Saving artifacts to: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped subdirectory
        run_dir = os.path.join(output_dir, f"run_{self.pipeline_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        artifacts = {}
        
        # Save models
        models_dir = os.path.join(run_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            artifacts[f'model_{model_name}'] = model_path
        
        # Save best model separately
        if self.best_model is not None:
            best_path = os.path.join(run_dir, 'best_model.joblib')
            joblib.dump(self.best_model, best_path)
            artifacts['best_model'] = best_path
        
        # Save results
        results_path = os.path.join(run_dir, 'evaluation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in results.items()
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        artifacts['results'] = results_path
        
        # Save metadata
        metadata = {
            'timestamp': self.pipeline_timestamp,
            'best_model': self.best_model_name,
            'config': self.config,
            'model_names': list(self.models.keys())
        }
        
        if X_train is not None:
            metadata['features'] = list(X_train.columns)
        
        metadata_path = os.path.join(run_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        artifacts['metadata'] = metadata_path
        
        # Save comparison summary
        summary = self.get_comparison_summary()
        summary_path = os.path.join(run_dir, 'model_comparison.csv')
        summary.to_csv(summary_path, index=False)
        artifacts['comparison'] = summary_path
        
        logger.info(f"Artifacts saved - {len(artifacts)} files")
        
        return artifacts
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get a summary comparison of all models.
        
        Returns:
            DataFrame with model comparison
        """
        summary_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
            
            summary_data.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1_Score': results.get('f1_score', 0),
                'ROC_AUC': results.get('roc_auc', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('F1_Score', ascending=False)
        
        return summary_df
    
    def run(self, data_dir: str = None, output_dir: str = None,
           X_train: pd.DataFrame = None, X_test: pd.DataFrame = None,
           y_train: pd.Series = None, y_test: pd.Series = None) -> Dict:
        """
        Execute the complete training pipeline.
        
        Args:
            data_dir: Directory containing processed data
            output_dir: Directory to save artifacts
            X_train, X_test, y_train, y_test: Direct data (optional)
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Start MLflow run if enabled
        if self.use_mlflow and self.mlflow_tracker:
            self.mlflow_tracker.start_run(
                run_name="training_pipeline",
                tags={'pipeline': 'training'}
            )
        
        try:
            # Step 1: Load data
            X_train, X_test, y_train, y_test = self.load_data(
                data_dir, X_train, X_test, y_train, y_test
            )
            
            # Log data info to MLflow
            if self.use_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.log_params({
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'num_features': X_train.shape[1]
                })
            
            # Step 2: Train all models
            self.train_all_models(X_train, y_train, X_test, y_test)
            
            # Step 3: Select best model
            best_name, best_model = self.select_best_model()
            
            # Log best model to MLflow
            if self.use_mlflow and self.mlflow_tracker:
                best_metrics = self.results[best_name]
                self.mlflow_tracker.log_metrics({
                    'best_accuracy': best_metrics.get('accuracy', 0),
                    'best_precision': best_metrics.get('precision', 0),
                    'best_recall': best_metrics.get('recall', 0),
                    'best_f1_score': best_metrics.get('f1_score', 0),
                    'best_roc_auc': best_metrics.get('roc_auc', 0)
                })
                self.mlflow_tracker.log_params({'best_model': best_name})
                self.mlflow_tracker.log_model(best_model)
            
            # Step 4: Save artifacts
            artifacts = {}
            if output_dir:
                artifacts = self.save_artifacts(output_dir, X_train)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Print comparison summary
            print("\n" + "=" * 60)
            print("MODEL COMPARISON SUMMARY")
            print("=" * 60)
            print(self.get_comparison_summary().to_string(index=False))
            print("=" * 60)
            
            results = {
                'status': 'success',
                'timestamp': self.pipeline_timestamp,
                'duration_seconds': duration,
                'best_model': best_name,
                'best_model_metrics': self.results[best_name],
                'all_results': self.results,
                'artifacts': artifacts,
                'models': self.models
            }
            
            logger.info("=" * 60)
            logger.info(f"TRAINING COMPLETED - Duration: {duration:.2f}s")
            logger.info(f"Best Model: {best_name}")
            logger.info("=" * 60)
            
            return results
            
        finally:
            # End MLflow run
            if self.use_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.end_run()


def main(use_mlflow: bool = False):
    """Main entry point for the training pipeline."""
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'artifacts', 'data_pipeline')
    output_dir = os.path.join(project_root, 'artifacts', 'training_pipeline')
    
    # Initialize and run pipeline
    pipeline = TrainingPipeline(use_mlflow=use_mlflow)
    results = pipeline.run(data_dir=data_dir, output_dir=output_dir)
    
    print("\nTraining Results:")
    print(f"  Status: {results['status']}")
    print(f"  Duration: {results['duration_seconds']:.2f} seconds")
    print(f"  Best Model: {results['best_model']}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    args = parser.parse_args()
    
    main(use_mlflow=args.mlflow)
