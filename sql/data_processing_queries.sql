-- SQL Data Processing Queries for Telco Customer Churn Analysis
-- Author: Student ID: E285181
-- Purpose: Demonstrate SQL querying capabilities for Big Data coursework
-- Dataset: Telco Customer Churn Dataset (7,043 records)

-- Data exploration queries

-- Query 1.1: Basic dataset overview
SELECT 
    COUNT(*) AS total_records,
    COUNT(DISTINCT customerID) AS unique_customers,
    COUNT(CASE WHEN Churn = 'Yes' THEN 1 END) AS churned_customers,
    COUNT(CASE WHEN Churn = 'No' THEN 1 END) AS retained_customers,
    ROUND(COUNT(CASE WHEN Churn = 'Yes' THEN 1 END) * 100.0 / COUNT(*), 2) AS churn_rate_percent
FROM telco_churn;

-- Query 1.2: Missing values analysis
SELECT 
    'TotalCharges' AS column_name,
    COUNT(*) AS total_records,
    SUM(CASE WHEN TotalCharges IS NULL OR TotalCharges = '' THEN 1 ELSE 0 END) AS missing_count,
    ROUND(SUM(CASE WHEN TotalCharges IS NULL OR TotalCharges = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS missing_percent
FROM telco_churn;

-- Query 1.3: Data type summary
SELECT 
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'telco_churn'
ORDER BY ordinal_position;

-- ============================================================
-- SECTION 2: CUSTOMER DEMOGRAPHIC ANALYSIS
-- ============================================================

-- Query 2.1: Churn by Gender
SELECT 
    gender,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS retained,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
GROUP BY gender
ORDER BY churn_rate DESC;

-- Query 2.2: Churn by Senior Citizen status
SELECT 
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS customer_type,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
GROUP BY SeniorCitizen
ORDER BY churn_rate DESC;

-- Query 2.3: Churn by Partner and Dependents
SELECT 
    Partner,
    Dependents,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
GROUP BY Partner, Dependents
ORDER BY churn_rate DESC;

-- ============================================================
-- SECTION 3: SERVICE ANALYSIS
-- ============================================================

-- Query 3.1: Churn by Internet Service type
SELECT 
    InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
GROUP BY InternetService
ORDER BY churn_rate DESC;

-- Query 3.2: Churn by Contract type
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(tenure), 1) AS avg_tenure_months,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
GROUP BY Contract
ORDER BY churn_rate DESC;

-- Query 3.3: Service adoption analysis
SELECT 
    'OnlineSecurity' AS service_name,
    OnlineSecurity AS service_status,
    COUNT(*) AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
WHERE InternetService != 'No'
GROUP BY OnlineSecurity

UNION ALL

SELECT 
    'TechSupport' AS service_name,
    TechSupport AS service_status,
    COUNT(*) AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
WHERE InternetService != 'No'
GROUP BY TechSupport

ORDER BY service_name, churn_rate DESC;

-- ============================================================
-- SECTION 4: FINANCIAL ANALYSIS
-- ============================================================

-- Query 4.1: Churn by Payment Method
SELECT 
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly,
    ROUND(AVG(CAST(TotalCharges AS DECIMAL)), 2) AS avg_total,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
WHERE TotalCharges != ''
GROUP BY PaymentMethod
ORDER BY churn_rate DESC;

-- Query 4.2: Monthly charges distribution by churn status
SELECT 
    Churn,
    COUNT(*) AS customers,
    ROUND(MIN(MonthlyCharges), 2) AS min_charges,
    ROUND(AVG(MonthlyCharges), 2) AS avg_charges,
    ROUND(MAX(MonthlyCharges), 2) AS max_charges,
    ROUND(STDDEV(MonthlyCharges), 2) AS std_charges
FROM telco_churn
GROUP BY Churn;

-- Query 4.3: Revenue at risk analysis
SELECT 
    Churn,
    COUNT(*) AS customer_count,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue,
    ROUND(SUM(CAST(TotalCharges AS DECIMAL)), 2) AS lifetime_revenue,
    ROUND(AVG(MonthlyCharges * 12), 2) AS projected_annual_revenue
FROM telco_churn
WHERE TotalCharges != ''
GROUP BY Churn;

-- ============================================================
-- SECTION 5: TENURE ANALYSIS
-- ============================================================

-- Query 5.1: Churn by tenure buckets
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-12 months (New)'
        WHEN tenure <= 24 THEN '13-24 months (Growing)'
        WHEN tenure <= 48 THEN '25-48 months (Established)'
        WHEN tenure <= 60 THEN '49-60 months (Loyal)'
        ELSE '60+ months (Very Loyal)'
    END AS tenure_bucket,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
GROUP BY 
    CASE 
        WHEN tenure <= 12 THEN '0-12 months (New)'
        WHEN tenure <= 24 THEN '13-24 months (Growing)'
        WHEN tenure <= 48 THEN '25-48 months (Established)'
        WHEN tenure <= 60 THEN '49-60 months (Loyal)'
        ELSE '60+ months (Very Loyal)'
    END
ORDER BY churn_rate DESC;

-- ============================================================
-- SECTION 6: HIGH-RISK CUSTOMER IDENTIFICATION
-- ============================================================

-- Query 6.1: High-risk customer profile
SELECT 
    customerID,
    gender,
    SeniorCitizen,
    Contract,
    tenure,
    MonthlyCharges,
    PaymentMethod,
    InternetService,
    OnlineSecurity,
    TechSupport,
    Churn
FROM telco_churn
WHERE 
    Contract = 'Month-to-month'
    AND tenure < 12
    AND InternetService = 'Fiber optic'
    AND OnlineSecurity = 'No'
    AND TechSupport = 'No'
ORDER BY MonthlyCharges DESC
FETCH FIRST 100 ROWS ONLY;

-- Query 6.2: Customer segmentation summary
SELECT 
    Contract,
    InternetService,
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS age_group,
    COUNT(*) AS segment_size,
    ROUND(AVG(MonthlyCharges), 2) AS avg_charges,
    ROUND(AVG(tenure), 1) AS avg_tenure,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate
FROM telco_churn
GROUP BY Contract, InternetService, SeniorCitizen
HAVING COUNT(*) > 50
ORDER BY churn_rate DESC;

-- ============================================================
-- SECTION 7: DATA TRANSFORMATION QUERIES
-- ============================================================

-- Query 7.1: Create transformed dataset with feature engineering
CREATE TABLE telco_churn_transformed AS
SELECT 
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    -- Tenure bucket
    CASE 
        WHEN tenure <= 12 THEN 'New'
        WHEN tenure <= 48 THEN 'Established'
        ELSE 'Loyal'
    END AS tenure_category,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    CAST(NULLIF(TotalCharges, '') AS DECIMAL(10,2)) AS TotalCharges,
    -- Service adoption score
    (CASE WHEN PhoneService = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN MultipleLines = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN InternetService != 'No' THEN 1 ELSE 0 END +
     CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END) AS service_adoption_score,
    -- Payment reliability
    CASE 
        WHEN PaymentMethod LIKE '%automatic%' THEN 'High'
        ELSE 'Low'
    END AS payment_reliability,
    Churn,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS churn_flag
FROM telco_churn
WHERE TotalCharges != '';

-- ============================================================
-- SECTION 8: AGGREGATION FOR DASHBOARDS
-- ============================================================

-- Query 8.1: Monthly summary statistics
SELECT 
    DATE_FORMAT(NOW(), '%Y-%m') AS report_month,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS revenue_at_risk
FROM telco_churn;

-- Query 8.2: Contract distribution pivot
SELECT 
    'Month-to-month' AS contract_type,
    COUNT(CASE WHEN Contract = 'Month-to-month' THEN 1 END) AS total,
    COUNT(CASE WHEN Contract = 'Month-to-month' AND Churn = 'Yes' THEN 1 END) AS churned
FROM telco_churn
UNION ALL
SELECT 
    'One year' AS contract_type,
    COUNT(CASE WHEN Contract = 'One year' THEN 1 END) AS total,
    COUNT(CASE WHEN Contract = 'One year' AND Churn = 'Yes' THEN 1 END) AS churned
FROM telco_churn
UNION ALL
SELECT 
    'Two year' AS contract_type,
    COUNT(CASE WHEN Contract = 'Two year' THEN 1 END) AS total,
    COUNT(CASE WHEN Contract = 'Two year' AND Churn = 'Yes' THEN 1 END) AS churned
FROM telco_churn;

-- End of SQL Processing Queries
