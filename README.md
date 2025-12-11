Customer Churn Prediction — CS210 Final Project - Adam Abdalla
CS210 – Data Management for Data Science
Project Overview

--This project analyzes telecom customer churn using a full data science pipeline--
Data Cleaning & Preprocessing (Python / Pandas)
Database Normalization + SQL Queries (SQLite)
Machine Learning Models (Logistic Regression & Random Forest)
Performance Evaluation & Feature Analysis
The goal is to understand why customers leave and build a model that predicts churn.

--Repository Structure--
Customer-Churn/
│
├── load_and_clean.py        # Cleans raw CSV, normalizes tables, builds churn.db
├── model_churn.py           # Trains ML models and prints metrics
│
├── Churn.csv                # Original dataset
├── cleaned_churn.csv        # Cleaned output file
├── churn.db                 # SQLite database with normalized tables
│
└── README.md                # Project documentation

--How to Run the Project--
1. Install required Python packages
pip install pandas numpy scikit-learn sqlite3 matplotlib seaborn

2. Run data cleaning + database creation
python load_and_clean.py

3. Run machine learning models
python model_churn.py

--Data Cleaning Summary--
Fixed inconsistent column names
Converted numeric fields
Mapped Churn: Yes/No → 1/0
Normalized dataset into 4 tables:
customers
services
contracts
churn_labels
Saved in churn.db.

--SQL Insights--
Churn Rate by Tariff Plan
Shows which plan results in higher churn.
Average Charges & Subscription Length by Churn
Indicates behavioral differences between churned and non-churned customers.

--Machine Learning Results--
Logistic Regression
Accuracy: ~87%
Identifies linear patterns

Random Forest
Accuracy: ~94%
Strong nonlinear classifier

--Key Findings--
-Top factors contributing to churn
Complaints
Subscription length
Seconds of use
Account status
Monthly activity levels

--Technologies Used--
Python
Pandas, NumPy
Scikit-Learn
SQLite
SQL
GitHub

--References--
CS210 Lecture Notes
Scikit-Learn Documentation
UCI Machine Learning Repository (for base churn dataset inspiration)

