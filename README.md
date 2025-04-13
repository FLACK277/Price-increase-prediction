CPI Price Increase Prediction
This repository contains a machine learning project that predicts whether the Consumer Price Index (CPI) will increase based on historical Denver CPI data.
Project Overview
The project analyzes Denver CPI data and builds predictive models to determine whether prices will increase in the future. It implements both Decision Tree and Logistic Regression models with hyperparameter tuning to achieve optimal performance.
Features

Data preprocessing including handling missing values and feature encoding
Exploratory data analysis with visualizations
Implementation of Decision Tree and Logistic Regression models
Hyperparameter tuning using GridSearchCV
Model performance evaluation with accuracy metrics, classification reports, and confusion matrices
Visualizations of model performance and feature importance

Data
The project uses the denver_cpi.csv dataset which includes:

CPI values over different time periods
Area type information
Period types and regions
Percentage change metrics

Requirements

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn

Usage

Clone this repository
Place your denver_cpi.csv file in the appropriate directory
Run the main script:

python cpi_prediction.py
Outputs
The script generates several visualization files:

target_distribution.png: Distribution of price increases
cpi_trend.png: CPI trends over time by area type
correlation_matrix.png: Correlation between numerical features
dt_confusion_matrix.png: Decision Tree model confusion matrix
lr_confusion_matrix.png: Logistic Regression model confusion matrix
model_comparison.png: Comparison of model performances
dt_feature_importance.png: Feature importance for Decision Tree model

Results
Both models are evaluated for accuracy, precision, recall, and F1-score. The model comparison visualization helps identify which algorithm performs better for predicting CPI increases.
Future Work

Expand to other geographical regions
Incorporate additional economic indicators
Implement more advanced models (Random Forest, XGBoost)
Deploy as a web application for real-time predictions
