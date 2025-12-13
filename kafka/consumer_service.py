"""
Kafka Consumer for Real-Time Churn Prediction
Subscribes to customer events and performs real-time inference
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    ML-based churn prediction engine
    Loads trained model and performs real-time inference
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor with trained model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = None
        self.feature_columns = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        
        # Resolve a sensible default model path when not provided
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            tp_dir = os.path.join(project_root, 'artifacts', 'training_pipeline')
            latest_run = None
            if os.path.isdir(tp_dir):
                runs = [d for d in os.listdir(tp_dir) if d.startswith('run_')]
                if runs:
                    latest_run = sorted(runs)[-1]
            candidates = []
            if latest_run:
                candidates.extend([
                    os.path.join(tp_dir, latest_run, 'best_model.joblib'),
                    os.path.join(tp_dir, latest_run, 'models', 'logistic_regression.joblib'),
                    os.path.join(tp_dir, latest_run, 'models', 'random_forest.joblib'),
                ])
            # Legacy/default locations
            candidates.extend([
                os.path.join(project_root, 'models', 'best_model.joblib'),
                os.path.join(project_root, 'models', 'logistic_regression.joblib'),
                os.path.join(project_root, 'models', 'random_forest_model_for_balanced.pkl'),
            ])
            model_path = next((p for p in candidates if os.path.isfile(p)), None)
        
        if model_path:
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model at {model_path}: {e}\nUsing rule-based prediction. Run 'make train-pipeline' to regenerate models with the current library versions.")
                self.model = None
        else:
            logger.warning("No model file found. Using rule-based prediction. Run 'make train-pipeline' to create one.")
    
    def preprocess_event(self, event: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Preprocess customer event for prediction
        
        Args:
            event: Raw customer event dictionary
            
        Returns:
            Preprocessed feature array
        """
        try:
            # Extract features with basic encoding
            features = {
                'senior_citizen': event.get('senior_citizen', 0),
                'tenure': event.get('tenure', 0),
                'monthly_charges': event.get('monthly_charges', 0),
                'total_charges': event.get('total_charges', 0),
                'contract_monthly': 1 if event.get('contract') == 'Month-to-month' else 0,
                'internet_fiber': 1 if event.get('internet_service') == 'Fiber optic' else 0,
                'online_security_no': 1 if event.get('online_security') == 'No' else 0,
                'tech_support_no': 1 if event.get('tech_support') == 'No' else 0,
                'paperless_billing': 1 if event.get('paperless_billing') == 'Yes' else 0,
                'payment_electronic': 1 if 'Electronic' in event.get('payment_method', '') else 0
            }
            
            return np.array(list(features.values())).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None
    
    def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make churn prediction for a customer event
        
        Args:
            event: Customer event dictionary
            
        Returns:
            Prediction result with probability and risk score
        """
        # Rule-based prediction when model is not available
        risk_factors = 0
        
        # High-risk indicators based on EDA findings
        if event.get('contract') == 'Month-to-month':
            risk_factors += 2
        if event.get('tenure', 0) < 12:
            risk_factors += 2
        if event.get('internet_service') == 'Fiber optic':
            risk_factors += 1
        if event.get('online_security') == 'No':
            risk_factors += 1
        if event.get('tech_support') == 'No':
            risk_factors += 1
        if event.get('paperless_billing') == 'Yes':
            risk_factors += 1
        if 'Electronic' in event.get('payment_method', ''):
            risk_factors += 1
        if event.get('monthly_charges', 0) > 70:
            risk_factors += 1
        
        # Calculate risk score (0-1)
        max_risk = 10
        risk_score = min(risk_factors / max_risk, 1.0)
        
        # Determine prediction
        churn_prediction = 1 if risk_score >= 0.5 else 0
        confidence = risk_score if churn_prediction == 1 else (1 - risk_score)
        
        return {
            'customer_id': event.get('customer_id'),
            'prediction': churn_prediction,
            'churn_probability': round(risk_score, 4),
            'confidence': round(confidence, 4),
            'risk_category': 'High' if risk_score >= 0.7 else ('Medium' if risk_score >= 0.4 else 'Low'),
            'predicted_at': datetime.now(timezone.utc).isoformat(),
            'model_version': 'rule_based_v1.0'
        }


class ChurnKafkaConsumer:
    """
    Kafka Consumer for real-time churn prediction
    Subscribes to customer events and publishes predictions
    """
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9094',
                 input_topic: str = 'customer-events',
                 output_topic: str = 'churn-predictions',
                 group_id: str = 'churn-prediction-group'):
        """
        Initialize Kafka consumer
        
        Args:
            bootstrap_servers: Kafka broker address
            input_topic: Topic to consume from
            output_topic: Topic to publish predictions to
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.predictor = ChurnPredictor()
        self.predictions_made = 0
        self.high_risk_count = 0
        
        try:
            from confluent_kafka import Consumer, Producer
            
            self.consumer = Consumer({
                'bootstrap.servers': bootstrap_servers,
                'group.id': group_id,
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': True
            })
            
            self.producer = Producer({
                'bootstrap.servers': bootstrap_servers
            })
            
            self.consumer.subscribe([input_topic])
            logger.info(f"Consumer initialized - Subscribed to: {input_topic}")
            
        except ImportError:
            logger.warning("confluent-kafka not installed. Running in simulation mode.")
            self.consumer = None
            self.producer = None
    
    def process_message(self, message_value: str) -> Dict[str, Any]:
        """
        Process a single message and return prediction
        
        Args:
            message_value: JSON string of customer event
            
        Returns:
            Prediction result dictionary
        """
        try:
            event = json.loads(message_value)
            prediction = self.predictor.predict(event)
            
            self.predictions_made += 1
            if prediction['risk_category'] == 'High':
                self.high_risk_count += 1
            
            return prediction
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            return None
    
    def publish_prediction(self, prediction: Dict[str, Any]):
        """Publish prediction to output topic"""
        if self.producer and prediction:
            self.producer.produce(
                self.output_topic,
                key=prediction['customer_id'],
                value=json.dumps(prediction)
            )
            self.producer.poll(0)
    
    def consume_and_predict(self, max_messages: int = 1000, timeout: float = 1.0, idle_timeout: float = 5.0):
        """
        Consume messages and make predictions
        
        Args:
            max_messages: Maximum number of messages to process
            timeout: Poll timeout in seconds
        """
        logger.info(f"Starting consumption from {self.input_topic}")
        
        messages_processed = 0
        # Track consecutive empty polls to detect idle periods
        idle_polls = 0
        max_idle_polls = max(1, int(idle_timeout / timeout)) if timeout > 0 else float('inf')
        
        if self.consumer:
            while messages_processed < max_messages:
                msg = self.consumer.poll(timeout)
                
                if msg is None:
                    idle_polls += 1
                    if idle_polls >= max_idle_polls:
                        logger.info(f"No messages received for {idle_timeout} seconds, exiting consumption")
                        break
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                prediction = self.process_message(msg.value().decode('utf-8'))
                
                if prediction:
                    self.publish_prediction(prediction)
                    messages_processed += 1
                    # Reset idle counter when we see new data
                    idle_polls = 0
                    
                    if messages_processed % 100 == 0:
                        logger.info(f"Processed {messages_processed} messages, High-risk: {self.high_risk_count}")
        else:
            # Simulation mode
            logger.info("Running in simulation mode...")
            from .producer_service import CustomerEventGenerator
            
            generator = CustomerEventGenerator()
            
            for i in range(max_messages):
                event = generator.generate_customer_event()
                prediction = self.predictor.predict(event)
                messages_processed += 1
                
                if messages_processed % 100 == 0:
                    logger.info(f"[SIMULATED] Processed {messages_processed} messages")
        
        logger.info(f"Consumption complete: {messages_processed} messages processed")
        logger.info(f"Total predictions: {self.predictions_made}, High-risk customers: {self.high_risk_count}")

        if self.producer:
            # Ensure all produced predictions are delivered before exit
            self.producer.flush()
        
        return messages_processed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        return {
            'total_predictions': self.predictions_made,
            'high_risk_count': self.high_risk_count,
            'high_risk_percentage': round(self.high_risk_count / max(self.predictions_made, 1) * 100, 2)
        }


def main():
    """
    Main function for Kafka consumer functionality
    Consumes real messages from Kafka and publishes predictions
    """
    print("\n" + "="*60)
    print("TELCO CUSTOMER CHURN - KAFKA CONSUMER")
    print("="*60 + "\n")

    # Initialize consumer to match producer host listener
    consumer = ChurnKafkaConsumer(
        bootstrap_servers='localhost:9094',
        input_topic='customer-events',
        output_topic='churn-predictions',
        group_id='churn-prediction-group'
    )

    # Consume and predict from Kafka
    processed = consumer.consume_and_predict(max_messages=200, timeout=1.0)

    # Statistics
    stats = consumer.get_statistics()
    print("\nConsumer Statistics:")
    print(f"  Total Processed: {processed}")
    print(f"  Total Predictions: {stats['total_predictions']}")
    print(f"  High-Risk Customers: {stats['high_risk_count']}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
