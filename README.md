customer_churn_prediction/
│
├── data/
│   ├── raw/                 # Raw datasets (original, unmodified)
│   ├── processed/           # Cleaned and preprocessed datasets
│   └── external/            # Any external datasets (optional)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Load and save data
│   │   ├── data_cleaning.py         # Handle missing values, duplicates, outliers
│   │   └── feature_engineering.py   # Feature creation, encoding, scaling
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py                   # Visualization & correlation analysis
│   │   └── statistical_tests.py     # Chi-square, ANOVA, etc.
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_builder.py         # Train ML models
│   │   ├── model_tuning.py          # GridSearchCV or RandomizedSearchCV
│   │   ├── model_evaluation.py      # Metrics, confusion matrix, ROC curves
│   │   └── model_persistence.py     # Save/load models with joblib or pickle
│   │
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── api.py                   # FastAPI service for predictions
│   │   ├── monitor.py               # Monitor model drift or performance
│   │   └── retrain.py               # Retraining logic for MLOps
│   │
│   └── utils/
│       ├── __init__.py
│       └── eda_visualization.py               # General utility functions

│
├── reports/
│   ├── eda_report.pdf
│   ├── model_evaluation_report.pdf
│   └── final_project_report.pdf
│
├── models/
│   ├── final_model.pkl
│   └── model_metrics.json
│
├── app.py                           # Entry point for deployment (Flask/FastAPI)
├── config.yaml                      # Config for paths, model params, etc.
├── README.md
└── main.py                          # Main execution script for training pipeline


# How to Use the Customer Churn Prediction API (FastAPI)
open your terminal and run the following command to start the FastAPI server:

```
http://127.0.0.1:8000/docs
```

