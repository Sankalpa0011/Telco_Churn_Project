# ============================================================
# R Visualization Script for Telco Customer Churn Analysis
# Author: Student ID: E285181
# Purpose: Decision Trees, Naive Bayes Classification, and Clustering
# Dataset: Telco Customer Churn Dataset
# ============================================================

# 1. SETUP ENVIRONMENT
# ============================================================

# Clear environment to start fresh
rm(list = ls())

# Set working directory
# NOTE: Update this path if you move the folder. 
work_dir <- "C:/Users/DELL/OneDrive/Desktop/BDV Coursework/New folder/Telco_Churn_Project/r_scripts"

if (dir.exists(work_dir)) {
  setwd(work_dir)
  cat("Working directory set to:", getwd(), "\n")
} else {
  warning("Directory not found. Please ensure the path is correct or set it manually via Session > Set Working Directory.")
}

# 2. ROBUST PACKAGE INSTALLATION & LOADING
# ============================================================

# List of required packages (ORDER MATTERS - dependencies first)
required_packages <- c(
  "parallelly",     # CRITICAL dependency for future/caret (install first!)
  "globals",        # Dependency for future
  "listenv",        # Dependency for future
  "future",         # Parallel processing
  "future.apply",   # Dependency for caret
  "proxy",          # Dependency for e1071
  "tidyverse",      # Data manipulation and visualization
  "caret",          # Machine learning framework
  "rpart",          # Decision trees
  "rpart.plot",     # Decision tree visualization
  "e1071",          # Naive Bayes classifier
  "ggplot2",        # Advanced plotting
  "corrplot",       # Correlation plots
  "gridExtra",      # Grid arrangements
  "scales",         # Scale functions
  "pROC",           # ROC curves
  "randomForest",   # Random Forest
  "cluster",        # Clustering algorithms
  "factoextra"      # Clustering visualization
)

# Identify missing packages
installed_pkgs <- installed.packages()[, "Package"]
missing_pkgs <- required_packages[!(required_packages %in% installed_pkgs)]

# Install missing packages (Force Binary to avoid Rtools error)
if (length(missing_pkgs) > 0) {
  message("Installing missing packages (Binary mode): ", paste(missing_pkgs, collapse = ", "))
  # type = "binary" is crucial for Windows users without Rtools
  for (pkg in missing_pkgs) {
    tryCatch({
      install.packages(pkg, dependencies = TRUE, type = "binary")
      cat("Installed:", pkg, "\n")
    }, error = function(e) {
      cat("Failed to install", pkg, ":", e$message, "\n")
    })
  }
}

# Load all packages with error handling
cat("\n=== Loading Packages ===\n")
load_errors <- c()
for (pkg in required_packages) {
  tryCatch({
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
    cat("Loaded:", pkg, "\n")
  }, error = function(e) {
    load_errors <<- c(load_errors, pkg)
    cat("FAILED to load:", pkg, "-", e$message, "\n")
  })
}

# Check if critical packages loaded successfully
critical_pkgs <- c("tidyverse", "caret", "rpart", "rpart.plot", "e1071", "ggplot2", "corrplot", "pROC", "cluster", "factoextra")
missing_critical <- critical_pkgs[critical_pkgs %in% load_errors]

if (length(missing_critical) > 0) {
  cat("\n*** WARNING: Some critical packages failed to load:", paste(missing_critical, collapse = ", "), "***\n")
  cat("Run this in the R console to fix:\n")
  cat(paste0('install.packages(c("', paste(missing_critical, collapse = '", "'), '"), type = "binary")\n'))
} else {
  cat("\nAll critical packages loaded successfully!\n")
}

# 3. DATA LOADING AND PREPROCESSING
# ============================================================

cat("\n=== Loading Telco Customer Churn Dataset ===\n")

# Check if file exists before reading
file_path <- "../data/raw/Telco_Customer_Churn_Dataset.csv"

if (file.exists(file_path)) {
  telco_data <- read.csv(file_path, stringsAsFactors = TRUE)
} else {
  stop("Dataset not found! Please check that 'Telco_Customer_Churn_Dataset.csv' is in the '../data/raw/' folder.")
}

# Display basic info
cat("\nDataset Dimensions:", dim(telco_data), "\n")
cat("Number of Rows:", nrow(telco_data), "\n")
cat("Number of Columns:", ncol(telco_data), "\n")

# Structure of data
str(telco_data)

# Summary statistics
summary(telco_data)

# 4. DATA CLEANING
# ============================================================

cat("\n=== Data Cleaning ===\n")

# Remove customerID (not a feature)
if ("customerID" %in% names(telco_data)) {
  telco_data <- telco_data[, -which(names(telco_data) == "customerID")]
}

# Convert TotalCharges to numeric (handles potential blank strings forcing it to factor)
telco_data$TotalCharges <- as.numeric(as.character(telco_data$TotalCharges))

# Check for missing values
missing_values <- colSums(is.na(telco_data))
cat("\nMissing Values:\n")
print(missing_values[missing_values > 0])

# Remove rows with missing values
telco_data <- na.omit(telco_data)
cat("\nRecords after cleaning:", nrow(telco_data), "\n")

# Convert target variable to factor
telco_data$Churn <- as.factor(telco_data$Churn)

# Create output directory for artifacts if it doesn't exist
if (!dir.exists("../artifacts/r_visualizations")) {
  dir.create("../artifacts/r_visualizations", recursive = TRUE)
}

# ============================================================
# OPTION: Set to TRUE to display plots in RStudio, FALSE to only save to files
# ============================================================
SHOW_PLOTS_IN_RSTUDIO <- TRUE

# Helper function to display plot in RStudio and save to file
save_and_show <- function(plot_obj, filename, width = 800, height = 600) {
  # Display in RStudio FIRST if enabled (so user sees it)
  if (SHOW_PLOTS_IN_RSTUDIO) {
    print(plot_obj)
    readline(prompt = "Press [Enter] to continue to next plot...")
  }
  
  # Save to PNG file using png() for consistent sizing
  png(filename, width = width, height = height, res = 100)
  print(plot_obj)
  dev.off()
  cat("Saved:", basename(filename), "\n")
}

# Helper function for base R plots (corrplot, rpart.plot, ROC)
save_and_show_base <- function(plot_func, filename, width = 800, height = 600) {
  # Display in RStudio FIRST if enabled
  if (SHOW_PLOTS_IN_RSTUDIO) {
    plot_func()
    readline(prompt = "Press [Enter] to continue to next plot...")
  }
  
  # Save to PNG file
  png(filename, width = width, height = height, res = 100)
  plot_func()
  dev.off()
  cat("Saved:", basename(filename), "\n")
}

# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

cat("\n=== Exploratory Data Analysis ===\n")

# Churn distribution
churn_dist <- table(telco_data$Churn)
cat("\nChurn Distribution:\n")
print(churn_dist)
cat("Churn Rate:", round(churn_dist["Yes"] / sum(churn_dist) * 100, 2), "%\n")

# Plot 1: Churn Distribution
p1 <- ggplot(telco_data, aes(x = Churn, fill = Churn)) +
  geom_bar(width = 0.6) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("No" = "#2E86AB", "Yes" = "#E94F37")) +
  labs(title = "Customer Churn Distribution",
       subtitle = "Telco Customer Churn Dataset",
       x = "Churn Status",
       y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
save_and_show(p1, "../artifacts/r_visualizations/01_churn_distribution.png")

# Plot 2: Churn by Contract Type
p2 <- ggplot(telco_data, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("No" = "#2E86AB", "Yes" = "#E94F37")) +
  labs(title = "Churn by Contract Type",
       x = "Contract Type",
       y = "Number of Customers") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))
save_and_show(p2, "../artifacts/r_visualizations/02_churn_by_contract.png")

# Plot 3: Monthly Charges by Churn Status
p3 <- ggplot(telco_data, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot() +
  scale_fill_manual(values = c("No" = "#2E86AB", "Yes" = "#E94F37")) +
  labs(title = "Monthly Charges Distribution by Churn Status",
       x = "Churn Status",
       y = "Monthly Charges ($)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))
save_and_show(p3, "../artifacts/r_visualizations/03_monthly_charges_boxplot.png")

# 6. DATA PREPARATION FOR MODELING
# ============================================================

cat("\n=== Preparing Data for Modeling ===\n")

# Create a numeric copy for correlation matrix
telco_numeric <- telco_data
cat_vars <- sapply(telco_numeric, is.factor)
for (col in names(telco_numeric)[cat_vars]) {
  telco_numeric[[col]] <- as.numeric(telco_numeric[[col]])
}

# Plot 4: Correlation matrix
cor_matrix <- cor(telco_numeric, use = "complete.obs")
corr_plot_func <- function() {
  corrplot(cor_matrix, method = "color", type = "upper",
           tl.col = "black", tl.srt = 45, tl.cex = 0.7,
           title = "Correlation Matrix - Telco Churn Features",
           mar = c(0, 0, 2, 0))
}
save_and_show_base(corr_plot_func, "../artifacts/r_visualizations/04_correlation_matrix.png", width = 1000, height = 800)

# Split data into training and testing sets
set.seed(42)
train_index <- createDataPartition(telco_data$Churn, p = 0.8, list = FALSE)
train_data <- telco_data[train_index, ]
test_data <- telco_data[-train_index, ]

cat("\nTraining Set Size:", nrow(train_data), "\n")
cat("Testing Set Size:", nrow(test_data), "\n")

# 7. DECISION TREE CLASSIFICATION
# ============================================================

cat("\n=== Decision Tree Classification ===\n")

# Train decision tree
dt_model <- rpart(Churn ~ ., data = train_data, method = "class",
                  control = rpart.control(cp = 0.01, maxdepth = 5))

# Plot 5: Decision tree
dt_plot_func <- function() {
  rpart.plot(dt_model, 
             type = 4, 
             extra = 104,
             fallen.leaves = TRUE,
             main = "Decision Tree for Customer Churn Prediction",
             box.palette = c("#2E86AB", "#E94F37"),
             shadow.col = "gray",
             nn = TRUE)
}
save_and_show_base(dt_plot_func, "../artifacts/r_visualizations/05_decision_tree.png", width = 1200, height = 800)

# Predictions
dt_pred <- predict(dt_model, test_data, type = "class")
dt_prob <- predict(dt_model, test_data, type = "prob")

# Confusion Matrix
dt_cm <- confusionMatrix(dt_pred, test_data$Churn, positive = "Yes")
cat("\nDecision Tree Results:\n")
print(dt_cm)

# Variable Importance
var_imp <- dt_model$variable.importance
var_imp_df <- data.frame(
  Variable = names(var_imp),
  Importance = var_imp
)
var_imp_df <- var_imp_df[order(-var_imp_df$Importance), ]

# Plot 6: Variable Importance
p6 <- ggplot(head(var_imp_df, 10), aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#2E86AB") +
  coord_flip() +
  labs(title = "Decision Tree - Top 10 Important Variables",
       x = "Variable",
       y = "Importance Score") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
save_and_show(p6, "../artifacts/r_visualizations/06_dt_variable_importance.png")

# 8. NAIVE BAYES CLASSIFICATION
# ============================================================

cat("\n=== Naive Bayes Classification ===\n")

# Train Naive Bayes model
nb_model <- naiveBayes(Churn ~ ., data = train_data)

# Predictions
nb_pred <- predict(nb_model, test_data)
nb_prob <- predict(nb_model, test_data, type = "raw")

# Confusion Matrix
nb_cm <- confusionMatrix(nb_pred, test_data$Churn, positive = "Yes")
cat("\nNaive Bayes Results:\n")
print(nb_cm)

# Plot 7: Confusion Matrix Comparison
dt_cm_table <- as.data.frame(dt_cm$table)
nb_cm_table <- as.data.frame(nb_cm$table)

p_cm1 <- ggplot(dt_cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "#2E86AB", high = "#E94F37") +
  labs(title = "Decision Tree", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

p_cm2 <- ggplot(nb_cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "#2E86AB", high = "#E94F37") +
  labs(title = "Naive Bayes", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Display in RStudio first
if (SHOW_PLOTS_IN_RSTUDIO) {
  grid.arrange(p_cm1, p_cm2, ncol = 2, 
               top = "Confusion Matrix Comparison: Decision Tree vs Naive Bayes")
  readline(prompt = "Press [Enter] to continue to next plot...")
}

# Save confusion matrix comparison
png("../artifacts/r_visualizations/07_confusion_matrix_comparison.png", width = 1000, height = 500, res = 100)
grid.arrange(p_cm1, p_cm2, ncol = 2, 
             top = "Confusion Matrix Comparison: Decision Tree vs Naive Bayes")
dev.off()
cat("Saved: 07_confusion_matrix_comparison.png\n")

# 9. ROC CURVES COMPARISON
# ============================================================

cat("\n=== ROC Curve Analysis ===\n")

# Plot 8: ROC curves
dt_roc <- roc(test_data$Churn, dt_prob[, "Yes"])
nb_roc <- roc(test_data$Churn, nb_prob[, "Yes"])

roc_plot_func <- function() {
  plot(dt_roc, col = "#2E86AB", lwd = 2, main = "ROC Curve Comparison")
  plot(nb_roc, col = "#E94F37", lwd = 2, add = TRUE)
  legend("bottomright", 
         legend = c(paste("Decision Tree (AUC:", round(auc(dt_roc), 3), ")"),
                    paste("Naive Bayes (AUC:", round(auc(nb_roc), 3), ")")),
         col = c("#2E86AB", "#E94F37"), lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")
}
save_and_show_base(roc_plot_func, "../artifacts/r_visualizations/08_roc_curves.png")

cat("\nAUC Scores:\n")
cat("Decision Tree AUC:", round(auc(dt_roc), 4), "\n")
cat("Naive Bayes AUC:", round(auc(nb_roc), 4), "\n")

# 10. CLUSTERING ANALYSIS
# ============================================================

cat("\n=== Customer Clustering Analysis ===\n")

# Prepare numeric data for clustering
cluster_vars <- c("tenure", "MonthlyCharges", "TotalCharges")
cluster_data <- scale(telco_numeric[, cluster_vars])

# Plot 9: Determine optimal number of clusters (Elbow Method)
p9 <- fviz_nbclust(cluster_data, kmeans, method = "wss") +
  labs(title = "Optimal Number of Clusters (Elbow Method)",
       subtitle = "Based on tenure, MonthlyCharges, TotalCharges") +
  theme_minimal()
save_and_show(p9, "../artifacts/r_visualizations/09_elbow_method.png")

# K-Means Clustering (k=4)
set.seed(42)
kmeans_result <- kmeans(cluster_data, centers = 4, nstart = 25)

# Plot 10: Visualize clusters
p10 <- fviz_cluster(kmeans_result, data = cluster_data,
                    palette = c("#2E86AB", "#E94F37", "#28A745", "#FFC107"),
                    geom = "point",
                    ellipse.type = "convex",
                    ggtheme = theme_minimal()) +
  labs(title = "Customer Segmentation (K-Means Clustering)",
       subtitle = "4 Clusters based on tenure, MonthlyCharges, TotalCharges")
save_and_show(p10, "../artifacts/r_visualizations/10_customer_clusters.png")

# Cluster profiling
telco_data$Cluster <- as.factor(kmeans_result$cluster)

cluster_profile <- telco_data %>%
  group_by(Cluster) %>%
  summarise(
    Count = n(),
    Avg_Tenure = mean(tenure),
    Avg_Monthly = mean(MonthlyCharges),
    Avg_Total = mean(TotalCharges),
    Churn_Rate = mean(Churn == "Yes") * 100
  )

cat("\nCluster Profiles:\n")
print(cluster_profile)

# 11. MODEL PERFORMANCE SUMMARY & SAVE
# ============================================================

cat("\n=== Model Performance Summary ===\n")

# Create summary dataframe
model_summary <- data.frame(
  Model = c("Decision Tree", "Naive Bayes"),
  Accuracy = c(dt_cm$overall["Accuracy"], nb_cm$overall["Accuracy"]),
  Sensitivity = c(dt_cm$byClass["Sensitivity"], nb_cm$byClass["Sensitivity"]),
  Specificity = c(dt_cm$byClass["Specificity"], nb_cm$byClass["Specificity"]),
  Precision = c(dt_cm$byClass["Precision"], nb_cm$byClass["Precision"]),
  F1_Score = c(dt_cm$byClass["F1"], nb_cm$byClass["F1"]),
  AUC = c(auc(dt_roc), auc(nb_roc))
)

cat("\nModel Comparison:\n")
print(model_summary)

# Plot 11: Model comparison
model_metrics <- model_summary %>%
  pivot_longer(cols = c(Accuracy, Sensitivity, Specificity, Precision, F1_Score, AUC),
               names_to = "Metric", values_to = "Value")

p11 <- ggplot(model_metrics, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("Decision Tree" = "#2E86AB", "Naive Bayes" = "#E94F37")) +
  labs(title = "Model Performance Comparison",
       subtitle = "Decision Tree vs Naive Bayes",
       x = "Metric",
       y = "Score") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = round(Value, 3)), position = position_dodge(0.9), 
            vjust = -0.5, size = 3)
save_and_show(p11, "../artifacts/r_visualizations/11_model_comparison.png", width = 1000, height = 600)

# Save tables
write.csv(model_summary, "../artifacts/r_visualizations/model_summary.csv", row.names = FALSE)
write.csv(cluster_profile, "../artifacts/r_visualizations/cluster_profiles.csv", row.names = FALSE)

# Completion Message
cat("\n" , paste(rep("=", 60), collapse = ""), "\n")
cat("R VISUALIZATION SCRIPT COMPLETED SUCCESSFULLY\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("All visualizations saved to: ../artifacts/r_visualizations/\n")