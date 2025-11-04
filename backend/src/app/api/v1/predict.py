from fastapi import APIRouter
import joblib
import numpy as np
from src.app.schemas.request_response import TransactionRequest, PredictionResponse

router = APIRouter()

# Load saved artifacts
model = joblib.load("src/app/models/model.joblib")
preprocessor = joblib.load("src/app/models/preprocessing.joblib")

@router.post("/predict", response_model=PredictionResponse)
def predict_transaction(data: TransactionRequest):
    # Convert to dict
    data_dict = data.dict()
    
    # Convert into a DataFrame-like structure
    import pandas as pd
    df = pd.DataFrame([data_dict])

    # Preprocess
    X_processed = preprocessor.transform(df)

    # Predict
    y_pred = model.predict(X_processed)[0]
    y_proba = model.predict_proba(X_processed)[0][1]

    # Return JSON response
    return PredictionResponse(
        prediction="Fraud" if y_pred == 1 else "Legit",
        probability=round(float(y_proba), 4)
    )
