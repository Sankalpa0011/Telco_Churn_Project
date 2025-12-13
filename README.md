# Telco Customer Churn Analysis Project
## Big Data and Visualisation Coursework (CC6058ES)

**Student ID:** E285181  
**Module:** CC6058ES Big Data and Visualisation  
**Submission Date:** December 2025

---

## Project Overview

This project applies data processing and visualization techniques to predict customer churn in a telecommunications company. The solution demonstrates the use of big data processing, real-time streaming, and multiple visualization tools to generate actionable insights for business decision-makers.

### Business Problem Statement

**Customer churn** is a critical challenge for telecommunications companies. When customers leave for competitors, companies lose revenue and incur costs to acquire new customers. This project aims to:

1. **Identify factors** that contribute to customer churn
2. **Build predictive models** to identify at-risk customers
3. **Create interactive visualizations** for business stakeholders
4. **Design real-time streaming pipelines** for proactive intervention

### Dataset

- **Source:** Telco Customer Churn Dataset
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

## Assessment Criteria Mapping

### 1. Business Problem Context (15%)

The project addresses **customer churn prediction** for a telecommunications company:

- **Problem Statement:** Predict which customers are likely to churn to enable proactive retention strategies
- **Business Impact:** Reducing churn by 5% can increase profits by 25-95%
- **Key Questions:**
  - What factors drive customer churn?
  - Which customer segments are at highest risk?
  - How can real-time monitoring improve retention?

### 2. Data Processing and Querying (20%)

**Evidence Location:** `notebooks/`, `pipelines/`, `sql/`

- **Data Ingestion:** CSV loading with Pandas and PySpark
- **Data Cleaning:** Missing value handling, type conversions
- **Feature Engineering:** Tenure binning, service adoption scoring
- **SQL Queries:** Comprehensive analysis in `sql/data_processing_queries.sql`
- **Spark Processing:** Distributed processing in `pipelines/spark_data_pipeline.py`

### 3. Real-Time Data Streaming (15%)

**Evidence Location:** `kafka/`

- **Publish/Subscribe Pipeline:** Apache Kafka implementation
- **Producer Service:** Streams customer events from dataset
- **Consumer Service:** Real-time churn prediction
- **Architecture Diagram:** See Section below

### 4. Visualization and Reporting (30%)

**Evidence Location:** `r_scripts/`, `d3_visualizations/`, `notebooks/`

| Tool | Implementation |
|------|---------------|
| **R** | Decision Trees, Naive Bayes, Clustering (`r_scripts/churn_visualization.R`) |
| **D3.js** | Interactive dashboard (`d3_visualizations/churn_dashboard.html`) |
| **Python** | EDA visualizations in Jupyter notebooks |

### 5. Critical Evaluation and Reflection (20%)

See written report for detailed analysis of:
- Data quality challenges
- Model performance comparison
- Ethical considerations
- Scalability recommendations

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
| Month-to-month contract | 42.7% | Highest risk |
| Fiber optic internet | 41.9% | Higher than DSL |
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

## Visualization Comparison

| Aspect | R | Tableau | D3.js |
|--------|---|---------|-------|
| **Strengths** | Statistical modeling, reproducibility | Ease of use, interactivity | Full customization, web integration |
| **Limitations** | Steeper learning curve | License cost | Requires JavaScript knowledge |
| **Best For** | Statistical analysis, ML visualization | Business dashboards | Custom web visualizations |
| **Used In Project** | Decision Trees, Naive Bayes, Clustering | N/A | Interactive churn dashboard |

## Getting Started

### 1. Clone the Repository
```bash
git clone <repo-url>
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

### 5. Model Files
Trained models are saved in the `models/` directory for both balanced and imbalanced datasets, with filenames indicating the algorithm and data type.

## Key Features
- Handles both imbalanced and balanced datasets
- Multiple ML algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Model persistence using joblib
- Well-organized artifacts and reproducible workflow

## Requirements
- Python 3.7+
- See `requirements.txt` for all dependencies

## References
- [Kaggle Dataset: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

