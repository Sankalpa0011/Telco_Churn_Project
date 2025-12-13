# Kafka Streaming Components

This module provides Kafka-based real-time streaming for customer churn prediction.

## Components

1. **producer_service.py** - Publishes customer events to Kafka
2. **consumer_service.py** - Consumes events and performs real-time prediction

## Architecture

```
Customer Data (CSV) 
       |
       v
[Producer Service] ---> [Kafka Topic: customer-events]
                                    |
                                    v
                        [Consumer Service]
                                    |
                                    v
                        [ML Prediction Engine]
                                    |
                                    v
                        [Kafka Topic: churn-predictions]
```

## Usage

### Prerequisites
- Docker / Docker Desktop (Windows/macOS) or Docker Engine (Linux) must be running before starting Kafka.
- Ensure ports 9092 (Kafka) and 8080 (Kafka UI) are available.

### Start Kafka via Docker Compose
You can start Kafka using Docker directly or via the project's Makefile:

```bash
make kafka-up
# or
# docker-compose up -d
```

### Start Producer
```python
from producer_service import ChurnKafkaProducer

producer = ChurnKafkaProducer(bootstrap_servers='localhost:9092')
producer.stream_events(events_per_second=10, duration_seconds=60)
```

### Start Consumer
```python
from consumer_service import ChurnKafkaConsumer

consumer = ChurnKafkaConsumer(bootstrap_servers='localhost:9092')
consumer.consume_and_predict(max_messages=1000)
```

**Tip:** If you prefer Docker Desktop, open the UI after starting the cluster to inspect the `telco-churn` cluster and topics (UI available at `http://localhost:8080`).
