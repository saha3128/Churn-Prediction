import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

pipe      = joblib.load("elbehiry_v3_final.pkl")
threshold = float(np.load("elbehiry_v3_threshold.npy"))

class Customer(BaseModel):
    Age: int
    Gender: str
    TenureMonths: int
    RecencyDays: int
    Frequency: int
    MonetaryValue: float
    SupportCalls: int

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    proba = pipe.predict_proba(df)[:, 1][0]
    prediction = int(proba >= threshold)
    return {
        "churn_probability": round(float(proba), 4),
        "will_churn": bool(prediction),
        "threshold_used": round(threshold, 4)
    }