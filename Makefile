.PHONY: all clean install setup data-pipeline train-pipeline inference-pipeline run-all help test notebooks kafka-up kafka-down export-aggregates serve-dashboard serve-d3-viz-dashboard

# Default Python interpreter
PYTHON = python
VENV = venv/Scripts/activate
VENV_UNIX = venv/bin/activate
RSCRIPT = Rscript

# Default target
all: help

# Help target
help:
	@echo "=============================================="
	@echo "  Telco Customer Churn Prediction System"
	@echo "  CC6058ES Big Data and Visualisation"
	@echo "=============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install          - Install dependencies and setup environment"
	@echo "  make setup            - Create project directories"
	@echo "  make clean            - Clean up artifacts and cache"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make data-pipeline    - Run data processing pipeline"
	@echo "  make train-pipeline   - Run model training pipeline"
	@echo "  make inference        - Run inference pipeline"
	@echo "  make run-all          - Run complete ML pipeline"
	@echo ""
	@echo "Notebook Commands:"
	@echo "  make notebooks        - Launch Jupyter notebook server"
	@echo ""
	@echo "Kafka Streaming Commands:"
	@echo "  make kafka-up         - Start Kafka infrastructure"
	@echo "  make kafka-down       - Stop Kafka infrastructure"
	@echo "  make kafka-reset      - Stop and wipe Kafka data for a fresh start"
	@echo "  make kafka-producer   - Run Kafka producer"
	@echo "  make kafka-consumer   - Run Kafka consumer"
	@echo ""
	@echo "MLflow Commands:"
	@echo "  make mlflow-ui        - Start MLflow tracking UI"
	@echo "  make mlflow-run       - Run pipeline with MLflow tracking"
	@echo "  make mlflow-clean     - Clean MLflow artifacts"
	@echo ""
	@echo "D3 Dashboard Commands:"
	@echo "  make export-aggregates - Export aggregate CSVs for D3 dashboard"
	@echo "  make serve-d3-viz-dashboard   - Start local HTTP server for dashboard (port 8000)"

# Install project dependencies (Windows)
install:
	@echo "Installing project dependencies..."
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv venv
	@echo "Installing dependencies..."
	venv\Scripts\python.exe -m pip install --upgrade pip
	venv\Scripts\pip install -r requirements.txt
	@echo ""
	@echo "Installation completed successfully!"
	@echo "To activate: venv\Scripts\activate"

# Install for Unix/Mac
install-unix:
	@echo "Installing project dependencies..."
	python3 -m venv venv
	. $(VENV_UNIX) && pip install --upgrade pip
	. $(VENV_UNIX) && pip install -r requirements.txt
	@echo "Installation completed!"
	@echo "To activate: source venv/bin/activate"

# Setup directories
setup:
	@echo "Creating project directories..."
	@if not exist "data\raw" mkdir "data\raw"
	@if not exist "data\processed" mkdir "data\processed"
	@if not exist "artifacts\data_pipeline" mkdir "artifacts\data_pipeline"
	@if not exist "artifacts\training_pipeline" mkdir "artifacts\training_pipeline"
	@if not exist "artifacts\inference_pipeline" mkdir "artifacts\inference_pipeline"
	@if not exist "artifacts\r_visualizations" mkdir "artifacts\r_visualizations"
	@if not exist "models" mkdir "models"
	@echo "Directories created successfully!"

# Clean up artifacts
clean:
	@echo "Cleaning up artifacts..."
	@if exist "artifacts\data_pipeline\*" del /Q "artifacts\data_pipeline\*"
	@if exist "artifacts\training_pipeline\*" del /Q "artifacts\training_pipeline\*"
	@if exist "artifacts\inference_pipeline\*" del /Q "artifacts\inference_pipeline\*"
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@if exist ".pytest_cache" rd /s /q ".pytest_cache"
	@echo "Cleanup completed!"

# Clean all (including models)
clean-all: clean
	@echo "Cleaning all including models..."
	@if exist "models\*" del /Q "models\*"
	@if exist "data\processed\*" del /Q "data\processed\*"
	@echo "Full cleanup completed!"

# Run data pipeline
data-pipeline:
	@echo "========================================"
	@echo "Running Data Processing Pipeline"
	@echo "========================================"
	$(PYTHON) -c "from pipelines.data_pipeline import main; main()"
	@echo "Data pipeline completed!"

# Run training pipeline
train-pipeline:
	@echo "========================================"
	@echo "Running Model Training Pipeline"
	@echo "========================================"
	$(PYTHON) -c "from pipelines.training_pipeline import main; main()"
	@echo "Training pipeline completed!"

# Run inference pipeline
inference:
	@echo "========================================"
	@echo "Running Inference Pipeline"
	@echo "========================================"
	$(PYTHON) -c "from pipelines.inference_pipeline import main; main()"
	@echo "Inference pipeline completed!"

# Run all pipelines
run-all:
	@echo "========================================"
	@echo "  Running Complete ML Pipeline"
	@echo "========================================"
	@echo ""
	@echo "Step 1: Data Processing"
	@echo "----------------------------------------"
	$(PYTHON) run.py --data-only
	@echo ""
	@echo "Step 2: Model Training"
	@echo "----------------------------------------"
	$(PYTHON) run.py --train-only
	@echo ""
	@echo "Step 3: Inference"
	@echo "----------------------------------------"
	$(PYTHON) run.py --inference
	@echo ""
	@echo "========================================"
	@echo "  All Pipelines Completed!"
	@echo "========================================"

# Run full pipeline using run.py
run:
	@echo "Running complete pipeline..."
	$(PYTHON) run.py

# Launch Jupyter notebooks
notebooks:
	@echo "Launching Jupyter Notebook..."
	jupyter notebook notebooks/

# Start Kafka infrastructure
kafka-up:
	@echo "Starting Kafka infrastructure..."
	docker-compose up -d
	@echo "Kafka started! UI available at http://localhost:8080"

# Stop Kafka infrastructure
kafka-down:
	@echo "Stopping Kafka infrastructure..."
	docker-compose down
	@echo "Kafka stopped!"

# Reset Kafka (down + remove volumes and local data)
kafka-reset:
	@echo "Resetting Kafka: stopping and wiping data..."
	docker-compose down -v
	@echo "Removing local Kafka data folders if present..."
	@if exist "docker\data" rd /s /q "docker\data"
	@if exist "docker\kafka-data" rd /s /q "docker\kafka-data"
	@if exist "docker\zookeeper-data" rd /s /q "docker\zookeeper-data"
	@echo "Kafka reset complete. Run 'make kafka-up' to start fresh."

# Kafka producer
kafka-producer:
	@echo "Running Kafka producer..."
	venv\Scripts\python.exe kafka/producer_service.py

# Kafka consumer
kafka-consumer:
	@echo "Running Kafka consumer..."
	venv\Scripts\python.exe kafka/consumer_service.py

# MLflow Commands
mlflow-ui:
	@echo "Starting MLflow UI..."
	mlflow ui --port 5000
	@echo "MLflow UI running at http://localhost:5000"

mlflow-run:
	@echo "Running pipeline with MLflow tracking..."
	$(PYTHON) run.py --mlflow

mlflow-clean:
	@echo "Cleaning MLflow artifacts..."
	@if exist "mlruns" rd /s /q "mlruns"
	@echo "MLflow artifacts cleaned!"

# Check project status
status:
	@echo "Project Status:"
	@echo "---------------"
	@echo "Data files:"
	@if exist "data\raw\*" (dir /B "data\raw") else (echo   No raw data)
	@echo ""
	@echo "Artifacts:"
	@if exist "artifacts\*" (dir /B "artifacts") else (echo   No artifacts)
	@echo ""
	@echo "Models:"
	@if exist "models\*" (dir /B "models") else (echo   No models)

# Export aggregate CSVs for dashboard
export-aggregates:
	@echo "Exporting aggregate CSVs..."
	venv\Scripts\python.exe scripts/export_aggregates.py
	@echo "Aggregates exported to d3_visualizations/data/"

# Serve dashboard locally
serve-d3-viz-dashboard:
	@echo "Starting local HTTP server on port 8000..."
	@echo "D3 UI running at http://localhost:8000/d3_visualizations/churn_dashboard.html"
	venv\Scripts\python.exe -m http.server 8000
