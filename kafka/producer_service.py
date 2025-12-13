"""
Kafka Producer for Real-Time Customer Churn Streaming
Publishes customer events to Kafka topic for real-time processing
"""

import os
import json
import time
import random
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomerEventGenerator:
    """
    Generates customer events from the Telco churn dataset
    Simulates real-time customer activity for streaming pipeline
    """
    
    def __init__(self, data_path: str = None, seed: int = 42):
        """
        Initialize the event generator with customer dataset
        
        Args:
            data_path: Path to the customer churn CSV file
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Determine the correct data path
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            data_path = os.path.join(project_root, 'data', 'raw', 'Telco_Customer_Churn_Dataset.csv')
        
        # Load customer dataset
        self.dataset = pd.read_csv(data_path)
        self.dataset = self.dataset.dropna(subset=['TotalCharges'])
        self.dataset['TotalCharges'] = pd.to_numeric(self.dataset['TotalCharges'], errors='coerce')
        
        logger.info(f"Loaded {len(self.dataset)} customer records from dataset")
    
    def generate_customer_event(self) -> Dict[str, Any]:
        """
        Generate a single customer event by sampling from the dataset
        Adds event metadata for streaming context
        
        Returns:
            Dictionary containing customer data and event metadata
        """
        # Sample random customer
        idx = random.randint(0, len(self.dataset) - 1)
        customer = self.dataset.iloc[idx]
        
        # Convert to dictionary with proper types
        event = {
            'customer_id': str(customer['customerID']),
            'gender': str(customer['gender']),
            'senior_citizen': int(customer['SeniorCitizen']),
            'partner': str(customer['Partner']),
            'dependents': str(customer['Dependents']),
            'tenure': int(customer['tenure']),
            'phone_service': str(customer['PhoneService']),
            'multiple_lines': str(customer['MultipleLines']),
            'internet_service': str(customer['InternetService']),
            'online_security': str(customer['OnlineSecurity']),
            'online_backup': str(customer['OnlineBackup']),
            'device_protection': str(customer['DeviceProtection']),
            'tech_support': str(customer['TechSupport']),
            'streaming_tv': str(customer['StreamingTV']),
            'streaming_movies': str(customer['StreamingMovies']),
            'contract': str(customer['Contract']),
            'paperless_billing': str(customer['PaperlessBilling']),
            'payment_method': str(customer['PaymentMethod']),
            'monthly_charges': float(customer['MonthlyCharges']),
            'total_charges': float(customer['TotalCharges']),
            # Event metadata
            'event_id': f"evt_{customer['customerID']}_{int(time.time() * 1000)}",
            'event_timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': 'customer_activity'
        }
        
        return event
    
    def generate_batch(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Generate a batch of customer events
        
        Args:
            batch_size: Number of events to generate
            
        Returns:
            List of customer event dictionaries
        """
        return [self.generate_customer_event() for _ in range(batch_size)]


class ChurnKafkaProducer:
    """
    Kafka Producer for streaming customer events
    Uses confluent-kafka library for high-performance publishing
    """
    
    def __init__(self, bootstrap_servers: str = 'localhost:9094', topic: str = 'customer-events'):
        """
        Initialize Kafka producer
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic to publish to
        """
        try:
            from confluent_kafka import Producer
            
            self.producer = Producer({
                'bootstrap.servers': bootstrap_servers,
                'client.id': 'telco-churn-producer',
                'acks': 'all',
                'retries': 3,
                'retry.backoff.ms': 500
            })
            self.topic = topic
            self.event_generator = CustomerEventGenerator()
            logger.info(f"Kafka producer initialized - Server: {bootstrap_servers}, Topic: {topic}")
            
        except ImportError:
            logger.warning("confluent-kafka not installed. Running in simulation mode.")
            self.producer = None
            self.topic = topic
            self.event_generator = CustomerEventGenerator()
    
    def delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def publish_event(self, event: Dict[str, Any]) -> bool:
        """
        Publish a single event to Kafka topic
        
        Args:
            event: Customer event dictionary
            
        Returns:
            True if published successfully
        """
        try:
            if self.producer:
                message = json.dumps(event)
                self.producer.produce(
                    self.topic,
                    key=event['customer_id'],
                    value=message,
                    callback=self.delivery_callback
                )
                self.producer.poll(0)
                return True
            else:
                # Simulation mode
                logger.info(f"[SIMULATED] Published event: {event['event_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def stream_events(self, events_per_second: int = 10, duration_seconds: int = 60):
        """
        Stream customer events at specified rate
        
        Args:
            events_per_second: Number of events to publish per second
            duration_seconds: Total streaming duration
        """
        logger.info(f"Starting event stream: {events_per_second} events/sec for {duration_seconds} seconds")
        
        start_time = time.time()
        events_published = 0
        interval = 1.0 / events_per_second
        
        while time.time() - start_time < duration_seconds:
            event = self.event_generator.generate_customer_event()
            
            if self.publish_event(event):
                events_published += 1
                
                if events_published % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = events_published / elapsed
                    logger.info(f"Published {events_published} events ({rate:.2f} events/sec)")
            
            time.sleep(interval)
        
        # Flush remaining messages
        if self.producer:
            self.producer.flush()
        
        total_time = time.time() - start_time
        logger.info(f"Streaming complete: {events_published} events in {total_time:.2f} seconds")
        
        return events_published
    
    def publish_batch(self, batch_size: int = 1000) -> int:
        """
        Publish a batch of events
        
        Args:
            batch_size: Number of events to publish
            
        Returns:
            Number of successfully published events
        """
        logger.info(f"Publishing batch of {batch_size} events")
        
        events = self.event_generator.generate_batch(batch_size)
        successful = 0
        
        for event in events:
            if self.publish_event(event):
                successful += 1
        
        if self.producer:
            self.producer.flush()
        
        logger.info(f"Batch complete: {successful}/{batch_size} events published")
        return successful


def main():
    """
    Main function for Kafka producer functionality
    Runs the producer to publish customer events to Kafka
    """
    print("\n" + "="*60)
    print("TELCO CUSTOMER CHURN - KAFKA PRODUCER")
    print("="*60 + "\n")
    
    # Initialize producer
    producer = ChurnKafkaProducer(
        bootstrap_servers='localhost:9094',
        topic='customer-events'
    )
    
    # Generate sample events
    print("Generating sample customer events...\n")
    
    for i in range(5):
        event = producer.event_generator.generate_customer_event()
        print(f"Event {i+1}:")
        print(f"  Customer ID: {event['customer_id']}")
        print(f"  Contract: {event['contract']}")
        print(f"  Monthly Charges: ${event['monthly_charges']:.2f}")
        print(f"  Tenure: {event['tenure']} months")
        print(f"  Internet Service: {event['internet_service']}")
        print(f"  Event Timestamp: {event['event_timestamp']}")
        print()
    
    # Publish batch
    print("\nPublishing batch of 100 events...")
    published = producer.publish_batch(100)
    print(f"Successfully published: {published} events")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
