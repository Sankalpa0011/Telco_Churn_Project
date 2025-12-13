# Telco Customer Churn Analysis Project

## Project Overview

This project applies data processing and visualization techniques to predict customer churn in a telecommunications company. The solution demonstrates the use of big data processing, real-time streaming, and multiple visualization tools to generate actionable insights for business decision-makers.

### Business Problem Statement

**Customer churn** is a critical challenge for telecommunications companies. When customers leave for competitors, companies lose revenue and incur costs to acquire new customers. This project aims to:

1. **Identify factors** that contribute to customer churn
2. **Build predictive models** to identify at-risk customers
3. **Create interactive visualizations** for business stakeholders
4. **Design real-time streaming pipelines** for proactive intervention

### Dataset

- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 attributes including demographics, services, and billing
- **Target Variable:** Churn (Yes/No)
- **Churn Rate:** ~26.5%

---

## Project Structure

```
Telco_Churn_Project/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── Telco_Customer_Churn_Dataset.csv
│   └── processed/                    # Transformed data
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # EDA and data understanding
│   ├── 02_data_preprocessing.ipynb   # Feature engineering & cleaning
│   ├── 03_model_training.ipynb       # ML model training & evaluation
│   └── 04_data_processing_sql.ipynb  # SQL-style data querying
│
├── pipelines/                        # Data processing pipelines
│   ├── __init__.py                   # Package initialization
│   ├── data_pipeline.py              # Complete data processing pipeline
│   ├── training_pipeline.py          # Model training pipeline
│   ├── inference_pipeline.py         # Batch & real-time inference
│   └── spark_data_pipeline.py        # PySpark data pipeline
│
├── src/                              # Source modules
│   ├── __init__.py                   # Package initialization
│   ├── data_ingestion.py             # Data loading and validation
│   ├── feature_engineering.py        # Feature creation & transformation
│   ├── model_building.py             # Model factory & builders
│   └── model_evaluation.py           # Evaluation metrics & visualization
│
├── utils/                            # Utility modules
│   ├── __init__.py                   # Package initialization
│   └── config.py                     # Configuration management
│
├── kafka/                            # Real-time streaming
│   ├── __init__.py
│   ├── producer_service.py           # Kafka event producer
│   ├── consumer_service.py           # Kafka consumer with inference
│   └── README.md
│
├── sql/                              # SQL queries
│   └── data_processing_queries.sql   # Comprehensive SQL queries
│
├── r_scripts/                        # R visualizations
│   └── churn_visualization.R         # Decision Trees, Naive Bayes, Clustering
│
├── d3_visualizations/                # D3.js web visualizations
│   └── churn_dashboard.html          # Interactive dashboard
│
├── artifacts/                        # Model artifacts and outputs
│   ├── balance/                      # Balanced dataset splits
│   ├── imbalance/                    # Original imbalanced splits
│   ├── data_pipeline/                # Data pipeline outputs
│   ├── training_pipeline/            # Training artifacts
│   └── r_visualizations/             # R script outputs
│
├── models/                           # Trained ML models
│
├── config.yaml                       # Project configuration
├── run.py                            # Main entry point
├── docker-compose.yml                # Kafka infrastructure
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Repository & Version Control

**GitHub Repository:** https://github.com/Sankalpa0011/Telco_Churn_Project.git

This project is version-controlled using Git and hosted on GitHub. To clone the repository:

```bash
git clone https://github.com/Sankalpa0011/Telco_Churn_Project.git
cd Telco_Churn_Project
```

---

## Technologies Used

| Category | Technologies |
|----------|-------------|
| **Data Processing** | Python, Pandas, PySpark, SQL |
| **Machine Learning** | Scikit-learn, XGBoost, CatBoost |
| **Real-time Streaming** | Apache Kafka |
| **R Visualization** | ggplot2, rpart, e1071, caret |
| **Web Visualization** | D3.js |
| **Notebook Environment** | Jupyter Notebook |

---

## Real-Time Streaming Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Customer      │     │     Kafka       │     │    Consumer     │
│   Data Source   │────>│   Topic:        │────>│    Service      │
│   (CSV/API)     │     │ customer-events │     │    (ML Model)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │<────│     Kafka       │<────│   Predictions   │
│   (D3.js)       │     │   Topic:        │     │   Generated     │
│                 │     │ churn-predict   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Streaming Benefits for Business

1. **Real-time alerts** for high-risk customers
2. **Immediate intervention** opportunities
3. **Dynamic dashboard updates** for monitoring
4. **Scalable architecture** for growing customer base

---

## Key Findings

### Churn Risk Factors

| Factor | Churn Rate | Insight |
|--------|-----------|---------|
| Month-to-month contract | 42.7% | Highest risk || One-year contract | 11.3% | Moderate risk |
| Two-year contract | 2.8% | Lowest risk |
| Fiber optic internet | 41.9% | Service quality issue |
| DSL internet | 19.0% | Lower than fiber |
| Senior citizens | 41.7% | Higher risk group || Fiber optic internet | 41.9% | Higher than DSL |
| Electronic check payment | 45.3% | Less committed |
| Tenure < 12 months | 47.4% | New customers at risk |
| No tech support | 41.7% | Support matters |

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.80 | 0.66 | 0.54 | 0.59 | 0.84 |
| Decision Tree | 0.77 | 0.58 | 0.52 | 0.55 | 0.73 |
| Random Forest | 0.82 | 0.70 | 0.52 | 0.60 | 0.86 |
| XGBoost | 0.81 | 0.68 | 0.53 | 0.59 | 0.85 |

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Sankalpa0011/Telco_Churn_Project.git
cd Telco_Churn_Project
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Run the Complete Pipeline
```bash
# Run full pipeline (data processing + training + inference)
python run.py

# Or run individual pipelines
python run.py --data-only      # Data processing only
python run.py --train-only     # Training only
python run.py --inference      # Inference only
```

### 4. Pipeline Outputs
After running the pipeline, you will find:
- **Data Pipeline:** `artifacts/data_pipeline/run_<timestamp>/`
  - X_train.csv, X_test.csv, y_train.csv, y_test.csv
  - pipeline_metadata.json
- **Training Pipeline:** `artifacts/training_pipeline/run_<timestamp>/`
  - best_model.joblib
  - evaluation_results.json
  - model_comparison.csv
- **Inference Pipeline:** `artifacts/inference_pipeline/`
  - predictions_<timestamp>.csv

### 5. Notebooks
- **01_data_exploration.ipynb**: Perform exploratory data analysis.
- **02_data_preprocessing.ipynb**: Data cleaning, encoding, and feature selection.
- **03_model_training.ipynb**: Train and evaluate ML models.
- **04_data_processing_sql.ipynb**: SQL-style data querying demonstration.
- **03_model_training.ipynb**: Train and evaluate multiple models, save results.

Open notebooks in Jupyter or VS Code and run cells sequentially.

### 5. Kafka Streaming Setup

Before starting Kafka, ensure Docker is running on your machine:

- **Windows / macOS:** Start Docker Desktop and make sure the Docker Engine is running. On Windows, enable the WSL2 backend if using WSL.
- **Linux:** Ensure the Docker Engine is started (e.g., `systemctl start docker`).

Start Kafka services for real-time streaming:

```bash
# Start Kafka using Docker Compose (or use Makefile alias)
make kafka-up
# or
# docker-compose up -d

# Start producer to send customer events
python kafka/producer_service.py

# Start consumer to process events and predict churn
python kafka/consumer_service.py

# Access Kafka UI (Kafka UI service runs on port 8080 in docker-compose)
Open browser: http://localhost:8080
```

The `make kafka-up` command uses Docker Compose; you can monitor containers in Docker Desktop (Windows/macOS) or via `docker ps` on Linux. If you need to reset the Kafka environment, use `make kafka-reset` to wipe local data and start fresh.

### 6. D3.js Visualization Dashboard

Serve the interactive churn dashboard:

```bash
# Export aggregated data for D3 dashboard
python scripts/export_aggregates.py

# Serve D3 dashboard (Python 3)
python -m http.server 8000 --directory d3_visualizations

# Access dashboard
Open browser: http://localhost:8000/churn_dashboard.html
```

### 7. R Visualizations

Generate statistical visualizations using R:

```bash
# Install R packages (first time only)
Rscript -e "install.packages(c('ggplot2', 'rpart', 'e1071', 'caret', 'corrplot', 'factoextra'))"

# Run R visualization script
Rscript r_scripts/churn_visualization.R

# Outputs saved to: artifacts/r_visualizations/
```

---

## Project Features

- **Distributed Data Processing:** Apache Spark (PySpark) for scalable ETL pipelines
- **Real-Time Streaming:** Apache Kafka for customer event processing
- **Multiple ML Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost
- **Advanced Visualizations:** R (ggplot2) for statistical plots, D3.js for interactive dashboards
- **Experiment Tracking:** MLflow for model versioning and metrics
- **SQL Analytics:** Comprehensive SQL queries for data exploration
- **Balanced/Imbalanced Datasets:** Handles class imbalance with SMOTE
- **Model Evaluation:** Accuracy, precision, recall, F1-score, ROC curves, confusion matrices
- **Reproducible Workflow:** Containerized with Docker, automated with Makefiles

---

## System Requirements

- **Python:** 3.8 or higher
- **Docker:** For Kafka services (optional but recommended)
- **R:** 4.0+ for statistical visualizations
- **Memory:** Minimum 8GB RAM for Spark processing
- **OS:** Windows, macOS, or Linux

### Python Dependencies

See `requirements.txt` for complete list. Key packages:
- `pyspark` - Distributed data processing
- `kafka-python` - Kafka client
- `pandas`, `numpy` - Data manipulation
- `scikit-learn`, `xgboost`, `catboost` - Machine learning
- `mlflow` - Experiment tracking
- `jupyter` - Interactive notebooks

---

## Project Architecture

The project follows a modular architecture:

1. **Data Layer:** Raw CSV → Spark processing → Processed datasets
2. **Streaming Layer:** Kafka producers → Topics → Consumers
3. **Model Layer:** Training pipelines → MLflow tracking → Saved models
4. **Visualization Layer:** R scripts → Static plots, D3.js → Interactive dashboards
5. **Orchestration:** Docker Compose for Kafka, Make for pipeline automation

---

## References & Resources

- **Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **GitHub Repository:** [Sankalpa0011/Telco_Churn_Project](https://github.com/Sankalpa0011/Telco_Churn_Project)
- **Apache Spark:** [spark.apache.org](https://spark.apache.org/)
- **Apache Kafka:** [kafka.apache.org](https://kafka.apache.org/)
- **MLflow:** [mlflow.org](https://mlflow.org/)
- **D3.js:** [d3js.org](https://d3js.org/)

---

## License

This project is available for educational and research purposes.

---

## Contact

**Author:** Kavindu Sankalpa  
**Repository:** https://github.com/Sankalpa0011/Telco_Churn_Project  
**Date:** December 2025

