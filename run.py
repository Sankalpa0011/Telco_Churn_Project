import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline(use_mlflow: bool = False):
    """Run the data processing pipeline."""
    from pipelines.data_pipeline import DataPipeline
    
    logger.info("=" * 70)
    logger.info("STEP 1: DATA PROCESSING PIPELINE")
    logger.info("=" * 70)
    
    # Define paths
    data_path = os.path.join('data', 'raw', 'Telco_Customer_Churn_Dataset.csv')
    output_dir = os.path.join('artifacts', 'data_pipeline')
    
    # Run pipeline
    pipeline = DataPipeline(use_mlflow=use_mlflow)
    results = pipeline.run(data_path, output_dir)
    
    logger.info(f"Data pipeline completed successfully")
    logger.info(f"  Train shape: {results['train_shape']}")
    logger.info(f"  Test shape: {results['test_shape']}")
    
    return results


def run_training_pipeline(use_mlflow: bool = False):
    """Run the model training pipeline."""
    from pipelines.training_pipeline import TrainingPipeline
    
    logger.info("=" * 70)
    logger.info("STEP 2: MODEL TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Define paths
    data_dir = os.path.join('artifacts', 'data_pipeline')
    output_dir = os.path.join('artifacts', 'training_pipeline')
    
    # Run pipeline
    pipeline = TrainingPipeline(use_mlflow=use_mlflow)
    results = pipeline.run(data_dir=data_dir, output_dir=output_dir)
    
    logger.info(f"Training pipeline completed successfully")
    logger.info(f"  Best model: {results['best_model']}")
    
    return results


def run_inference_pipeline(use_mlflow: bool = False):
    """Run the inference pipeline on test data."""
    from pipelines.inference_pipeline import InferencePipeline
    
    logger.info("=" * 70)
    logger.info("STEP 3: INFERENCE PIPELINE")
    logger.info("=" * 70)
    
    # Find latest model
    training_dir = os.path.join('artifacts', 'training_pipeline')
    run_dirs = [d for d in os.listdir(training_dir) if d.startswith('run_')]
    
    if not run_dirs:
        logger.error("No trained models found. Run training pipeline first.")
        return None
    
    latest_run = sorted(run_dirs)[-1]
    model_path = os.path.join(training_dir, latest_run, 'best_model.joblib')
    
    # Find latest test data
    data_dir = os.path.join('artifacts', 'data_pipeline')
    data_runs = [d for d in os.listdir(data_dir) if d.startswith('run_')]
    latest_data_run = sorted(data_runs)[-1]
    test_data_path = os.path.join(data_dir, latest_data_run, 'X_test.csv')
    
    # Output directory
    output_dir = os.path.join('artifacts', 'inference_pipeline')
    
    # Run pipeline
    pipeline = InferencePipeline(model_path=model_path, use_mlflow=use_mlflow)
    results = pipeline.run(model_path, test_data_path, output_dir)
    
    logger.info(f"Inference pipeline completed successfully")
    logger.info(f"  Total predictions: {results['total_records']}")
    logger.info(f"  Predicted churn rate: {results['churn_rate']:.2f}%")
    
    return results


def run_full_pipeline(use_mlflow: bool = False):
    """Run the complete ML pipeline."""
    logger.info("#" * 70)
    logger.info("#" + " " * 20 + "TELCO CHURN PREDICTION SYSTEM" + " " * 19 + "#")
    logger.info("#" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if use_mlflow:
        logger.info("MLflow tracking: ENABLED")
    
    start_time = datetime.now()
    
    # Step 1: Data Processing
    data_results = run_data_pipeline(use_mlflow=use_mlflow)
    
    # Step 2: Model Training
    training_results = run_training_pipeline(use_mlflow=use_mlflow)
    
    # Step 3: Inference (optional demonstration)
    inference_results = run_inference_pipeline(use_mlflow=use_mlflow)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Final Summary
    logger.info("#" * 70)
    logger.info("#" + " " * 25 + "PIPELINE SUMMARY" + " " * 28 + "#")
    logger.info("#" * 70)
    logger.info(f"Total Duration: {duration:.2f} seconds")
    logger.info(f"Data Pipeline: {data_results['status']}")
    logger.info(f"Training Pipeline: {training_results['status']}")
    if inference_results:
        logger.info(f"Inference Pipeline: {inference_results['status']}")
    logger.info(f"Best Model: {training_results['best_model']}")
    logger.info(f"Best F1-Score: {training_results['best_model_metrics']['f1_score']:.4f}")
    logger.info("#" * 70)
    
    return {
        'data': data_results,
        'training': training_results,
        'inference': inference_results,
        'duration': duration
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Telco Customer Churn Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    # Run full pipeline
    python run.py --data-only        # Run data pipeline only
    python run.py --train-only       # Run training pipeline only
    python run.py --inference        # Run inference on test data
    python run.py --mlflow           # Run with MLflow tracking
        """
    )
    
    parser.add_argument('--data-only', action='store_true',
                       help='Run data processing pipeline only')
    parser.add_argument('--train-only', action='store_true',
                       help='Run training pipeline only')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference pipeline')
    parser.add_argument('--mlflow', action='store_true',
                       help='Enable MLflow experiment tracking')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.data_only:
            run_data_pipeline(use_mlflow=args.mlflow)
        elif args.train_only:
            run_training_pipeline(use_mlflow=args.mlflow)
        elif args.inference:
            run_inference_pipeline(use_mlflow=args.mlflow)
        else:
            run_full_pipeline(use_mlflow=args.mlflow)
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
