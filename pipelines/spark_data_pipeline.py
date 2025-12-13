"""
PySpark Data Processing Pipeline for Telco Customer Churn Analysis
Demonstrates big data processing capabilities using Apache Spark
"""

import os
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import PySpark - if not available, use pandas fallback
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, 
        DoubleType, FloatType
    )
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
    )
    from pyspark.ml.classification import (
        LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
    )
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    PYSPARK_AVAILABLE = True
    logger.info("PySpark loaded successfully")
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning("PySpark not available - using pandas fallback")
    import pandas as pd
    import numpy as np


class SparkChurnPipeline:
    """
    PySpark-based data processing pipeline for customer churn analysis
    Handles large-scale data processing with distributed computing
    """
    
    def __init__(self, app_name: str = "TelcoChurnAnalysis"):
        """
        Initialize Spark session and pipeline configuration
        
        Args:
            app_name: Name for the Spark application
        """
        self.app_name = app_name
        self.spark = None
        self.df = None
        self.processed_df = None
        
        if PYSPARK_AVAILABLE:
            self._create_spark_session()
    
    def _create_spark_session(self):
        """Create and configure Spark session"""
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark session created: {self.app_name}")
    
    def load_data(self, file_path: str) -> None:
        """
        Load CSV data into Spark DataFrame
        
        Args:
            file_path: Path to the CSV file
        """
        logger.info(f"Loading data from: {file_path}")
        
        if PYSPARK_AVAILABLE and self.spark:
            self.df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(file_path)
            
            logger.info(f"Loaded {self.df.count()} rows with {len(self.df.columns)} columns")
        else:
            # Pandas fallback
            self.df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns (Pandas)")
    
    def show_schema(self) -> None:
        """Display DataFrame schema"""
        if PYSPARK_AVAILABLE and self.spark:
            print("\n=== Dataset Schema ===")
            self.df.printSchema()
        else:
            print("\n=== Dataset Info ===")
            print(self.df.info())
    
    def explore_data(self) -> Dict:
        """
        Perform exploratory data analysis
        
        Returns:
            Dictionary containing EDA metrics
        """
        logger.info("Performing exploratory data analysis")
        
        metrics = {}
        
        if PYSPARK_AVAILABLE and self.spark:
            # Row count
            metrics['total_rows'] = self.df.count()
            metrics['total_columns'] = len(self.df.columns)
            
            # Churn distribution
            churn_dist = self.df.groupBy("Churn").count().collect()
            metrics['churn_distribution'] = {row['Churn']: row['count'] for row in churn_dist}
            
            # Calculate churn rate
            total = metrics['total_rows']
            churned = metrics['churn_distribution'].get('Yes', 0)
            metrics['churn_rate'] = round((churned / total) * 100, 2)
            
            # Missing values
            missing = {}
            for col in self.df.columns:
                null_count = self.df.filter(F.col(col).isNull()).count()
                if null_count > 0:
                    missing[col] = null_count
            metrics['missing_values'] = missing
            
            # Numerical statistics
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            metrics['numerical_stats'] = {}
            for col in numerical_cols:
                if col in self.df.columns:
                    stats = self.df.select(
                        F.mean(col).alias('mean'),
                        F.min(col).alias('min'),
                        F.max(col).alias('max'),
                        F.stddev(col).alias('std')
                    ).collect()[0]
                    metrics['numerical_stats'][col] = {
                        'mean': round(float(stats['mean'] or 0), 2),
                        'min': round(float(stats['min'] or 0), 2),
                        'max': round(float(stats['max'] or 0), 2),
                        'std': round(float(stats['std'] or 0), 2)
                    }
        else:
            # Pandas fallback
            metrics['total_rows'] = len(self.df)
            metrics['total_columns'] = len(self.df.columns)
            metrics['churn_distribution'] = self.df['Churn'].value_counts().to_dict()
            metrics['churn_rate'] = round((self.df['Churn'] == 'Yes').mean() * 100, 2)
            metrics['missing_values'] = self.df.isnull().sum().to_dict()
        
        return metrics
    
    def clean_data(self) -> None:
        """Clean and prepare data for analysis"""
        logger.info("Cleaning data")
        
        if PYSPARK_AVAILABLE and self.spark:
            # Drop customerID
            if 'customerID' in self.df.columns:
                self.df = self.df.drop('customerID')
            
            # Convert TotalCharges to numeric and handle blanks
            self.df = self.df.withColumn(
                'TotalCharges',
                F.when(F.col('TotalCharges') == ' ', None)
                .otherwise(F.col('TotalCharges').cast('double'))
            )
            
            # Drop rows with null TotalCharges
            self.df = self.df.dropna(subset=['TotalCharges'])
            
            # Convert SeniorCitizen to categorical
            self.df = self.df.withColumn(
                'SeniorCitizen',
                F.when(F.col('SeniorCitizen') == 1, 'Yes').otherwise('No')
            )
            
            logger.info(f"Cleaned data: {self.df.count()} rows remaining")
        else:
            # Pandas fallback
            if 'customerID' in self.df.columns:
                self.df = self.df.drop(columns=['customerID'])
            
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            self.df = self.df.dropna(subset=['TotalCharges'])
            self.df['SeniorCitizen'] = self.df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
            
            logger.info(f"Cleaned data: {len(self.df)} rows remaining")
    
    def feature_engineering(self) -> None:
        """Create derived features for better predictions"""
        logger.info("Performing feature engineering")
        
        if PYSPARK_AVAILABLE and self.spark:
            # Tenure buckets
            self.df = self.df.withColumn(
                'tenure_bucket',
                F.when(F.col('tenure') <= 12, 'New')
                .when(F.col('tenure') <= 24, 'Growing')
                .when(F.col('tenure') <= 48, 'Established')
                .otherwise('Loyal')
            )
            
            # Average charges per month
            self.df = self.df.withColumn(
                'avg_charges_per_tenure',
                F.when(F.col('tenure') > 0, 
                       F.col('TotalCharges') / F.col('tenure'))
                .otherwise(F.col('MonthlyCharges'))
            )
            
            # High value customer flag
            self.df = self.df.withColumn(
                'high_value_customer',
                F.when(F.col('MonthlyCharges') > 70, 'Yes').otherwise('No')
            )
            
            # Service count
            service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            for col in service_cols:
                if col in self.df.columns:
                    self.df = self.df.withColumn(
                        f'{col}_flag',
                        F.when(F.col(col).isin(['Yes', 'Fiber optic', 'DSL']), 1).otherwise(0)
                    )
            
            service_flag_cols = [f'{col}_flag' for col in service_cols if f'{col}_flag' in self.df.columns]
            if service_flag_cols:
                self.df = self.df.withColumn(
                    'service_count',
                    sum([F.col(c) for c in service_flag_cols])
                )
            
            logger.info("Feature engineering completed")
        else:
            # Pandas fallback
            self.df['tenure_bucket'] = pd.cut(
                self.df['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['New', 'Growing', 'Established', 'Loyal']
            )
            
            self.df['avg_charges_per_tenure'] = self.df.apply(
                lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'],
                axis=1
            )
            
            self.df['high_value_customer'] = (self.df['MonthlyCharges'] > 70).map({True: 'Yes', False: 'No'})
    
    def run_sql_queries(self) -> Dict:
        """
        Execute SQL queries for analysis
        
        Returns:
            Dictionary containing query results
        """
        logger.info("Running SQL queries")
        results = {}
        
        if PYSPARK_AVAILABLE and self.spark:
            # Register as temp view
            self.df.createOrReplaceTempView("telco_churn")
            
            # Query 1: Churn by Contract
            query1 = """
                SELECT 
                    Contract,
                    COUNT(*) as total_customers,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
                    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
                FROM telco_churn
                GROUP BY Contract
                ORDER BY churn_rate DESC
            """
            results['churn_by_contract'] = self.spark.sql(query1).collect()
            
            # Query 2: Churn by Internet Service
            query2 = """
                SELECT 
                    InternetService,
                    COUNT(*) as total_customers,
                    ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges,
                    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
                FROM telco_churn
                GROUP BY InternetService
                ORDER BY churn_rate DESC
            """
            results['churn_by_internet'] = self.spark.sql(query2).collect()
            
            # Query 3: High-risk customers
            query3 = """
                SELECT 
                    tenure_bucket,
                    high_value_customer,
                    COUNT(*) as customer_count,
                    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
                FROM telco_churn
                GROUP BY tenure_bucket, high_value_customer
                ORDER BY churn_rate DESC
            """
            results['risk_segments'] = self.spark.sql(query3).collect()
            
            # Query 4: Revenue analysis
            query4 = """
                SELECT 
                    Churn,
                    COUNT(*) as customers,
                    ROUND(SUM(MonthlyCharges), 2) as total_monthly_revenue,
                    ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges,
                    ROUND(SUM(TotalCharges), 2) as lifetime_revenue
                FROM telco_churn
                GROUP BY Churn
            """
            results['revenue_analysis'] = self.spark.sql(query4).collect()
        
        return results
    
    def prepare_ml_data(self) -> Tuple:
        """
        Prepare data for machine learning
        
        Returns:
            Tuple of (train_df, test_df) for Spark or (X_train, X_test, y_train, y_test) for pandas
        """
        logger.info("Preparing data for ML")
        
        if PYSPARK_AVAILABLE and self.spark:
            # Define categorical columns
            categorical_cols = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_bucket'
            ]
            
            # Filter to columns that exist
            categorical_cols = [c for c in categorical_cols if c in self.df.columns]
            
            # StringIndexer for each categorical column
            indexers = [
                StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
                for col in categorical_cols
            ]
            
            # OneHotEncoder
            indexed_cols = [f"{col}_index" for col in categorical_cols]
            encoded_cols = [f"{col}_encoded" for col in categorical_cols]
            
            encoder = OneHotEncoder(
                inputCols=indexed_cols,
                outputCols=encoded_cols,
                handleInvalid="keep"
            )
            
            # Label indexer
            label_indexer = StringIndexer(
                inputCol="Churn",
                outputCol="label",
                handleInvalid="keep"
            )
            
            # Numerical columns
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            
            # Vector assembler
            assembler = VectorAssembler(
                inputCols=encoded_cols + numerical_cols,
                outputCol="features",
                handleInvalid="skip"
            )
            
            # Scaler
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features"
            )
            
            # Build pipeline
            pipeline = Pipeline(stages=indexers + [encoder, label_indexer, assembler, scaler])
            
            # Fit and transform
            pipeline_model = pipeline.fit(self.df)
            transformed_df = pipeline_model.transform(self.df)
            
            # Split data
            train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)
            
            logger.info(f"Training set: {train_df.count()} rows")
            logger.info(f"Test set: {test_df.count()} rows")
            
            return train_df, test_df
        else:
            # Pandas fallback
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            
            df_encoded = self.df.copy()
            
            # Encode categorical variables
            label_encoders = {}
            categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
            
            if 'Churn' in categorical_cols:
                categorical_cols.remove('Churn')
            
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
            
            # Prepare features and target
            X = df_encoded.drop(columns=['Churn'])
            y = (df_encoded['Churn'] == 'Yes').astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Test set: {len(X_test)} samples")
            
            return X_train, X_test, y_train, y_test
    
    def stop(self):
        """Stop the Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def run_pipeline_demo():
    """
    Demonstration of the Spark data pipeline
    """
    print("\n" + "="*70)
    print("TELCO CUSTOMER CHURN - PYSPARK DATA PROCESSING PIPELINE")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = SparkChurnPipeline(app_name="TelcoChurnDemo")
    
    # Load data
    data_path = "../data/raw/Telco_Customer_Churn_Dataset.csv"
    pipeline.load_data(data_path)
    
    # Show schema
    pipeline.show_schema()
    
    # Explore data
    print("\n=== Exploratory Data Analysis ===")
    metrics = pipeline.explore_data()
    print(f"Total Records: {metrics['total_rows']}")
    print(f"Total Columns: {metrics['total_columns']}")
    print(f"Churn Rate: {metrics['churn_rate']}%")
    print(f"Churn Distribution: {metrics['churn_distribution']}")
    
    # Clean data
    pipeline.clean_data()
    
    # Feature engineering
    pipeline.feature_engineering()
    
    # Run SQL queries
    print("\n=== SQL Query Results ===")
    sql_results = pipeline.run_sql_queries()
    
    if sql_results.get('churn_by_contract'):
        print("\nChurn by Contract Type:")
        for row in sql_results['churn_by_contract']:
            print(f"  {row['Contract']}: {row['churn_rate']}% churn rate ({row['total_customers']} customers)")
    
    if sql_results.get('revenue_analysis'):
        print("\nRevenue Analysis:")
        for row in sql_results['revenue_analysis']:
            print(f"  {row['Churn']}: {row['customers']} customers, ${row['total_monthly_revenue']:,.2f} monthly revenue")
    
    # Prepare ML data
    print("\n=== Machine Learning Data Preparation ===")
    ml_data = pipeline.prepare_ml_data()
    
    # Stop Spark
    pipeline.stop()
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_pipeline_demo()
