# Bank Churner Prediction

## Project Overview
A business manager of a consumer credit card portfolio is facing a steady loss of customers and wants to understand why. This notebook (`Bank_Churner_Prediction.ipynb`) explores customer demographics, account details, and engagement metrics to:

1. Analyze patterns of attrition (16.07% churn rate in our data)  
2. Identify key factors driving churn  
3. Train and evaluate a machine-learning model to predict which customers are at highest risk of leaving  
4. Provide actionable recommendations for targeted retention strategies  

## Business Problem
Customer attrition (or churn) erodes fee income, reduces cross-sell opportunities, and increases acquisition costs. By predicting which customers are most likely to churn, the bank can proactively engage high-risk segments with personalized offers and service enhancements to lower churn and boost lifetime value.

## Data Source
- **Kaggle “Credit Card Customers”** by Sakshi Goyal (10,000 records, 18 features)  
  https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data

## Notebook Contents
1. **Data Ingestion & Cleaning**  
   - Load data, inspect schema, check for missing values and duplicates  
2. **Exploratory Data Analysis (EDA)**  
   - Univariate plots (age, income, etc.)  
   - Churn rates by gender, card category, and other segments  
   - Correlation matrix and initial feature insights  
3. **Feature Engineering & Preprocessing**  
   - Encode categoricals, scale numeric features, handle imbalance  
4. **Model Training & Evaluation**  
   - Stratified train/test split  
   - Baseline Random Forest on original data  
   - Upsampling minority class and retraining  
   - Metrics: accuracy, precision, recall, F1, ROC/PR curves  
5. **Conclusions & Business Recommendations**  
   - Key drivers of churn (e.g. inactivity, product count, contact frequency)  
   - Model performance trade-offs  
   - Next steps: threshold tuning, hold-out validation, deployment plan  

## Requirements
- Python 3.7+  
- `pandas`, `numpy`, `scikit-learn`  
- `plotly`, `scikit-plot`, `matplotlib`

## Note: You can interact with the interactive plotly plots in Colab notebook, but they are not rendered on github. The link is included in the ipynb file.
