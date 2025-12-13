import os
import logging
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mlflow_config() -> Dict[str, Any]:
    """Get MLflow configuration settings."""
    return {
        'tracking_uri': os.environ.get('MLFLOW_TRACKING_URI', 'file:./mlruns'),
        'experiment_name': 'telco_churn_prediction',
        'run_name_prefix': 'churn_run',
        'artifact_path': 'model',
        'model_registry_name': 'telco_churn_model',
        'tags': {
            'project': 'telco_churn',
            'module': 'telco_churn',
            'version': '1.0.0'
        }
    }


class MLflowTracker:
    """MLflow tracking utilities for experiment management and model versioning."""
    
    def __init__(self, config: Dict = None):
        """Initialize MLflow tracker."""
        self.config = config or get_mlflow_config()
        self.run = None
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Initialize MLflow tracking with configuration."""
        tracking_uri = self.config.get('tracking_uri', 'file:./mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        experiment_name = self.config.get('experiment_name', 'telco_churn_prediction')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            logger.warning("Continuing without MLflow tracking")
    
    def start_run(self, run_name: Optional[str] = None, 
                 tags: Optional[Dict[str, str]] = None) -> Optional[mlflow.ActiveRun]:
        """Start a new MLflow run."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if run_name is None:
                run_name_prefix = self.config.get('run_name_prefix', 'run')
                run_name = f"{run_name_prefix} | {timestamp}"
            else:
                run_name = f"{run_name} | {timestamp}"
            
            # Merge default tags
            default_tags = self.config.get('tags', {})
            if tags:
                default_tags.update(tags)
            
            self.run = mlflow.start_run(run_name=run_name, tags=default_tags)
            logger.info(f"Started MLflow run: {run_name} (ID: {self.run.info.run_id})")
            
            return self.run
            
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            return None
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("MLflow run ended")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            # Convert non-string values
            clean_params = {}
            for k, v in params.items():
                if isinstance(v, (list, dict)):
                    clean_params[k] = str(v)
                else:
                    clean_params[k] = v
            
            mlflow.log_params(clean_params)
            logger.debug(f"Logged {len(clean_params)} parameters")
            
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        try:
            # Filter numeric values only
            numeric_metrics = {
                k: float(v) for k, v in metrics.items() 
                if isinstance(v, (int, float, np.number))
            }
            
            mlflow.log_metrics(numeric_metrics, step=step)
            logger.debug(f"Logged {len(numeric_metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_model(self, model, artifact_path: str = None, 
                 registered_model_name: str = None):
        """Log a sklearn model to MLflow."""
        try:
            artifact_path = artifact_path or self.config.get('artifact_path', 'model')
            registered_name = registered_model_name or self.config.get('model_registry_name')
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_name
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")
            
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact file to MLflow."""
        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path)
                logger.debug(f"Logged artifact: {local_path}")
            else:
                logger.warning(f"Artifact not found: {local_path}")
                
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
    
    def log_data_pipeline_metrics(self, dataset_info: Dict[str, Any]):
        """Log data pipeline specific metrics."""
        try:
            metrics = {
                'dataset_rows': dataset_info.get('total_rows', 0),
                'training_rows': dataset_info.get('train_rows', 0),
                'test_rows': dataset_info.get('test_rows', 0),
                'num_features': dataset_info.get('num_features', 0),
                'missing_values': dataset_info.get('missing_values', 0),
                'churn_rate': dataset_info.get('churn_rate', 0)
            }
            self.log_metrics(metrics)
            
            params = {
                'test_size': dataset_info.get('test_size', 0.2),
                'random_state': dataset_info.get('random_state', 42)
            }
            self.log_params(params)
            
            logger.info("Logged data pipeline metrics")
            
        except Exception as e:
            logger.error(f"Error logging data pipeline metrics: {e}")
    
    def log_training_metrics(self, model, metrics: Dict[str, Any], 
                            params: Dict[str, Any]):
        """Log training metrics and model."""
        try:
            self.log_params(params)
            self.log_metrics(metrics)
            self.log_model(model)
            
            logger.info("Logged training metrics and model")
            
        except Exception as e:
            logger.error(f"Error logging training metrics: {e}")
    
    def log_evaluation_metrics(self, metrics: Dict[str, float], 
                              confusion_matrix_path: str = None):
        """Log evaluation metrics and confusion matrix."""
        try:
            self.log_metrics(metrics)
            
            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                self.log_artifact(confusion_matrix_path, "evaluation")
            
            logger.info("Logged evaluation metrics")
            
        except Exception as e:
            logger.error(f"Error logging evaluation metrics: {e}")


def setup_mlflow_autolog():
    """Enable MLflow autologging for sklearn."""
    try:
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True
        )
        logger.info("MLflow autologging enabled")
    except Exception as e:
        logger.error(f"Error enabling autolog: {e}")


def create_mlflow_run_tags(pipeline_name: str, model_type: str = None) -> Dict[str, str]:
    """Create standard tags for MLflow runs."""
    tags = {
        'pipeline': pipeline_name,
        'timestamp': datetime.now().isoformat(),
        'project': 'telco_churn',
        'module': 'telco_churn'
    }
    
    if model_type:
        tags['model_type'] = model_type
    
    return tags


def get_latest_model(model_name: str = None):
    """Get the latest version of a registered model."""
    try:
        model_name = model_name or 'telco_churn_model'
        client = mlflow.tracking.MlflowClient()
        
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = max(versions, key=lambda x: int(x.version))
            logger.info(f"Found latest model: {model_name} v{latest.version}")
            return latest
        else:
            logger.warning(f"No versions found for model: {model_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting latest model: {e}")
        return None


def load_model_from_registry(model_name: str = None, version: str = None):
    """Load a model from MLflow registry."""
    try:
        model_name = model_name or 'telco_churn_model'
        
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from: {model_uri}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


if __name__ == "__main__":
    # Test MLflow setup
    tracker = MLflowTracker()
    
    with tracker.start_run("test_run"):
        tracker.log_params({'test_param': 'value'})
        tracker.log_metrics({'test_metric': 0.95})
    
    print("MLflow test completed!")
