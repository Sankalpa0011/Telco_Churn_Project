"""
Inference Pipeline for Telco Customer Churn Prediction

This pipeline handles batch and real-time inference:
- Loading trained models
- Making predictions on new data
- Batch processing capabilities
- Prediction logging and monitoring

Module: CC6058ES Big Data and Visualisation
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Inference pipeline for making churn predictions.
    Supports both batch and single predictions.
    """
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to trained model file
            config: Pipeline configuration
        """
        self.config = config or self._default_config()
        self.model = None
        self.model_path = model_path
        self.feature_names = None
        self.pipeline_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.prediction_history = []
        
        logger.info(f"Inference Pipeline initialized - Timestamp: {self.pipeline_timestamp}")
    
    def _default_config(self) -> Dict:
        """Return default inference configuration."""
        return {
            'probability_threshold': 0.5,
            'high_risk_threshold': 0.7,
            'log_predictions': True,
            'batch_size': 1000
        }
    
    def load_model(self, model_path: str = None) -> Any:
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to model file (joblib format)
            
        Returns:
            Loaded model
        """
        model_path = model_path or self.model_path
        
        if model_path is None:
            raise ValueError("Model path must be provided")
        
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path
        
        logger.info(f"Model loaded successfully: {type(self.model).__name__}")
        
        return self.model
    
    def load_metadata(self, metadata_path: str) -> Dict:
        """
        Load training metadata including feature names.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            Metadata dictionary
        """
        logger.info(f"Loading metadata from: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'features' in metadata:
            self.feature_names = metadata['features']
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        return metadata
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        df = data.copy()
        
        # Ensure all expected features are present
        if self.feature_names is not None:
            missing_cols = set(self.feature_names) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing features: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    df[col] = 0
            
            # Reorder columns to match training
            df = df[self.feature_names]
        
        return df
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]], 
               return_proba: bool = True) -> Dict:
        """
        Make predictions on input data.
        
        Args:
            data: Input data for prediction
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # Preprocess input
        df = self.preprocess_input(data)
        
        logger.info(f"Making predictions on {len(df)} records")
        
        # Make predictions
        predictions = self.model.predict(df)
        
        result = {
            'predictions': predictions.tolist(),
            'prediction_labels': ['Churn' if p == 1 else 'No Churn' for p in predictions]
        }
        
        # Get probabilities if available
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df)
            result['probabilities'] = probabilities.tolist()
            result['churn_probability'] = probabilities[:, 1].tolist()
            
            # Risk categorization
            result['risk_category'] = [
                self._categorize_risk(p) for p in probabilities[:, 1]
            ]
        
        # Log predictions if configured
        if self.config['log_predictions']:
            self._log_predictions(result)
        
        return result
    
    def predict_single(self, customer_data: Dict) -> Dict:
        """
        Make prediction for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Prediction result
        """
        result = self.predict(customer_data)
        
        return {
            'prediction': result['predictions'][0],
            'prediction_label': result['prediction_labels'][0],
            'churn_probability': result.get('churn_probability', [None])[0],
            'risk_category': result.get('risk_category', [None])[0]
        }
    
    def predict_batch(self, data: pd.DataFrame, batch_size: int = None) -> Dict:
        """
        Make predictions on a large batch of data.
        
        Args:
            data: DataFrame with customer data
            batch_size: Number of records per batch
            
        Returns:
            Dictionary with all predictions
        """
        batch_size = batch_size or self.config['batch_size']
        
        logger.info(f"Batch prediction: {len(data)} records in batches of {batch_size}")
        
        all_predictions = []
        all_probabilities = []
        all_prediction_labels = []
        all_risk_categories = []
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            result = self.predict(batch)
            
            all_predictions.extend(result['predictions'])
            if 'prediction_labels' in result:
                all_prediction_labels.extend(result['prediction_labels'])
            if 'churn_probability' in result:
                all_probabilities.extend(result['churn_probability'])
            if 'risk_category' in result:
                all_risk_categories.extend(result['risk_category'])
            
            logger.info(f"Processed batch {i // batch_size + 1}")
        
        return_result = {
            'predictions': all_predictions,
            'total_records': len(data),
            'churn_count': sum(all_predictions),
            'churn_rate': sum(all_predictions) / len(data) * 100
        }
        
        if all_prediction_labels:
            return_result['prediction_labels'] = all_prediction_labels
        if all_probabilities:
            return_result['probabilities'] = all_probabilities
            return_result['churn_probability'] = all_probabilities
        if all_risk_categories:
            return_result['risk_category'] = all_risk_categories
        
        return return_result
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize churn probability into risk levels."""
        if probability >= self.config['high_risk_threshold']:
            return 'High Risk'
        elif probability >= self.config['probability_threshold']:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    def _log_predictions(self, result: Dict):
        """Log predictions to history."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'count': len(result['predictions']),
            'churn_count': sum(result['predictions']),
            'avg_probability': np.mean(result.get('churn_probability', [0]))
        }
        self.prediction_history.append(log_entry)
    
    def get_prediction_summary(self, result: Dict) -> str:
        """
        Get a human-readable summary of predictions.
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            Summary string
        """
        total = len(result['predictions'])
        churn_count = sum(result['predictions'])
        
        summary = f"""
Prediction Summary
==================
Total Records: {total}
Predicted Churn: {churn_count} ({churn_count/total*100:.1f}%)
Predicted Retained: {total - churn_count} ({(total-churn_count)/total*100:.1f}%)
"""
        
        if 'risk_category' in result:
            risk_counts = pd.Series(result['risk_category']).value_counts()
            summary += f"""
Risk Distribution:
  High Risk: {risk_counts.get('High Risk', 0)}
  Medium Risk: {risk_counts.get('Medium Risk', 0)}
  Low Risk: {risk_counts.get('Low Risk', 0)}
"""
        
        return summary
    
    def export_predictions(self, data: pd.DataFrame, result: Dict, 
                          output_path: str) -> str:
        """
        Export predictions to CSV file.
        
        Args:
            data: Original input data
            result: Prediction results
            output_path: Path to save CSV
            
        Returns:
            Path to saved file
        """
        output_df = data.copy()
        output_df['predicted_churn'] = result['predictions']
        output_df['prediction_label'] = result['prediction_labels']
        
        if 'churn_probability' in result:
            output_df['churn_probability'] = result['churn_probability']
        
        if 'risk_category' in result:
            output_df['risk_category'] = result['risk_category']
        
        output_df.to_csv(output_path, index=False)
        logger.info(f"Predictions exported to: {output_path}")
        
        return output_path
    
    def run(self, model_path: str, data_path: str, output_dir: str = None) -> Dict:
        """
        Execute the complete inference pipeline.
        
        Args:
            model_path: Path to trained model
            data_path: Path to data for prediction
            output_dir: Directory to save results
            
        Returns:
            Dictionary with inference results
        """
        logger.info("=" * 60)
        logger.info("STARTING INFERENCE PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Start MLflow run if enabled
        if self.use_mlflow and self.mlflow_tracker:
            self.mlflow_tracker.start_run(
                run_name="inference_pipeline",
                tags={'pipeline': 'inference'}
            )
        
        try:
            # Step 1: Load model
            self.load_model(model_path)
            
            # Step 2: Load data
            logger.info(f"Loading data from: {data_path}")
            data = pd.read_csv(data_path)
            logger.info(f"Data loaded: {data.shape}")
            
            # Step 3: Make predictions
            result = self.predict_batch(data)
            
            # Step 4: Generate summary
            summary = self.get_prediction_summary(result)
            print(summary)
            
            # Step 5: Export results
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'predictions_{self.pipeline_timestamp}.csv')
                self.export_predictions(data, result, output_path)
                result['output_file'] = output_path
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result['status'] = 'success'
            result['duration_seconds'] = duration
            result['timestamp'] = self.pipeline_timestamp
            
            logger.info("=" * 60)
            logger.info(f"INFERENCE COMPLETED - Duration: {duration:.2f}s")
            logger.info("=" * 60)
            
            return result
            
        finally:
            # End MLflow run
            if self.use_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.end_run()

class RealTimePredictor:
    """
    Real-time prediction handler for streaming scenarios.
    Optimized for low-latency single predictions.
    """
    
    def __init__(self, model_path: str):
        """Initialize with pre-loaded model."""
        self.model = joblib.load(model_path)
        self.prediction_count = 0
        logger.info("Real-time predictor initialized")
    
    def predict(self, features: Dict) -> Dict:
        """
        Make a single prediction with minimal latency.
        
        Args:
            features: Dictionary of customer features
            
        Returns:
            Prediction result
        """
        # Convert to array for faster prediction
        feature_values = np.array(list(features.values())).reshape(1, -1)
        
        prediction = self.model.predict(feature_values)[0]
        probability = self.model.predict_proba(feature_values)[0][1] if hasattr(self.model, 'predict_proba') else None
        
        self.prediction_count += 1
        
        return {
            'prediction': int(prediction),
            'churn': prediction == 1,
            'probability': float(probability) if probability else None,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main entry point for the inference pipeline."""
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'artifacts', 'training_pipeline')
    
    # Find latest model
    run_dirs = [d for d in os.listdir(model_path) if d.startswith('run_')]
    if run_dirs:
        latest_run = sorted(run_dirs)[-1]
        model_file = os.path.join(model_path, latest_run, 'best_model.joblib')
        
        if os.path.exists(model_file):
            # Load test data for demonstration
            data_path = os.path.join(project_root, 'artifacts', 'data_pipeline')
            data_runs = [d for d in os.listdir(data_path) if d.startswith('run_')]
            if data_runs:
                latest_data = sorted(data_runs)[-1]
                test_file = os.path.join(data_path, latest_data, 'X_test.csv')
                
                if os.path.exists(test_file):
                    # Example inference with batch prediction
                    pipeline = InferencePipeline(model_path=model_file)
                    pipeline.load_model()
                    
                    # Load test data
                    import pandas as pd
                    X_test = pd.read_csv(test_file)
                    
                    # Make predictions on first 10 records
                    sample_data = X_test.head(10)
                    results = pipeline.predict(sample_data)
                    
                    print(f"\nSample Predictions (First 10 records):")
                    print(f"Total Records: {len(results['predictions'])}")
                    print(f"Predicted Churns: {sum(results['predictions'])}")
                    print(f"Predicted Non-Churns: {len(results['predictions']) - sum(results['predictions'])}")
                    print("\nFirst 5 predictions:")
                    for i in range(min(5, len(results['predictions']))):
                        print(f"  Record {i+1}: {results['prediction_labels'][i]} (Prob: {results.get('churn_probability', [0])[i]:.2%})")
                else:
                    print("Test data not found.")
            else:
                print("No data runs found.")
        else:
            print("No trained model found. Run training_pipeline.py first.")
    else:
        print("No training runs found. Run training_pipeline.py first.")


if __name__ == "__main__":
    main()
