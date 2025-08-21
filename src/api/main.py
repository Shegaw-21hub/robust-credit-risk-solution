from fastapi import FastAPI
import joblib
import os
import numpy as np
from src.api.pydantic_models import CreditRiskRequest

MODEL_PATH = os.path.join("models", "rf_model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Credit Risk Prediction API")

@app.post("/predict")
def predict(request: CreditRiskRequest):
    # Convert request into numpy array in correct order
    data = np.array([[ 
        request.num__Recency,
        request.num__Frequency,
        request.num__MonetarySum,
        request.num__MonetaryMean,
        request.num__MonetaryStd,
        request.cat__ProductCategory_airtime,
        request.cat__ProductCategory_data_bundles,
        request.cat__ProductCategory_financial_services,
        request.cat__ProductCategory_movies,
        request.cat__ProductCategory_other,
        request.cat__ProductCategory_ticket,
        request.cat__ProductCategory_transport,
        request.cat__ProductCategory_tv,
        request.cat__ProductCategory_utility_bill,
        request.cat__ChannelId_ChannelId_1,
        request.cat__ChannelId_ChannelId_2,
        request.cat__ChannelId_ChannelId_3,
        request.cat__ChannelId_ChannelId_5,
        request.cat__PricingStrategy_0,
        request.cat__PricingStrategy_1,
        request.cat__PricingStrategy_2,
        request.cat__PricingStrategy_4,
        request.temp__TransactionHour,
        request.temp__TransactionDay,
        request.temp__TransactionMonth
    ]])

    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running!"}
