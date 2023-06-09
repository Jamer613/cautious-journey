# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("credit_api")

# Create input/output pydantic models
input_model = create_model("credit_api_input", **{'checking_status': 1, 'duration': 6.0, 'credit_history': 'critical/other existing credit', 'purpose': 'radio/tv', 'credit_amount': 338.0, 'savings_status': 3, 'employment': 4, 'installment_commitment': 4.0, 'other_parties': 'none', 'residence_since': 4.0, 'property_magnitude': 'car', 'age': 52.0, 'other_payment_plans': 'none', 'housing': 'own', 'existing_credits': 2.0, 'job': 'skilled', 'num_dependents': 1.0, 'own_telephone': 'none', 'foreign_worker': 'yes', 'sex': 'male', 'marital_status': 'single'})
output_model = create_model("credit_api_output", prediction='good')


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
