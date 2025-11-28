# api/predict.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import os
import sys
from mangum import Mangum

# =========================
# 0) Path setup & utils import
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Import your utility modules
from src.utils.preprocessing import *
from src.utils.features_extension import *

# =========================
# 1) Load the trained model
# =========================

MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pkl")  # put model.pkl in project root
MODEL_PATH = os.path.abspath(MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model.pkl: {e}")

# =========================
# 2) Input schema
# =========================

class CustomerFeatures(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# =========================
# 3) Expected columns
# =========================

EXPECTED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "NEW_noProt", "NEW_Engaged", "NEW_Young_Not_Engaged",
    "NEW_FLAG_ANY_STREAMING", "NEW_FLAG_AutoPayment",
    "NEW_TotalServices", "NEW_AVG_Service_Fee"
]

# =========================
# 4) Helpers
# =========================

def ensure_expected_columns(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    df_final = df.copy()
    for col in expected_cols:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[expected_cols]
    return df_final

def preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = convert_to_numeric(df, "TotalCharges", fill_method="median")
    df = drop_column(df, "customerID")
    df_encoded, _ = encode_categorical_features_api(df)
    df_feature = feature_engineering(df_encoded)
    df_final = ensure_expected_columns(df_feature, EXPECTED_COLUMNS)
    return df_final

# =========================
# 5) FastAPI app
# =========================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained Telco churn model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Churn prediction API is live"}

@app.post("/api/predict")
def predict_churn(features: CustomerFeatures):
    raw_df = pd.DataFrame([features.dict()])

    try:
        processed_df = preprocess_input(raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    try:
        y_pred = model.predict(processed_df)[0]
        prob = float(model.predict_proba(processed_df)[0][1]) if hasattr(model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    return {
        "churn_class": int(y_pred),
        "churn_label": "Churn" if int(y_pred) == 1 else "No Churn",
        "churn_probability": prob,
    }

# =========================
# 6) Mangum handler for Vercel serverless
# =========================

handler = Mangum(app)
