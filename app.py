from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Define your feature schema
class PassengerFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# 2. Load the trained pipeline once at startup
pipeline = joblib.load("full_pipeline.joblib")

# 3. Set up FastAPI app
app = FastAPI()

# 4. Write the predict endpoint
@app.post("/predict")
def predict(features: PassengerFeatures):
    # Convert input to DataFrame
    X = pd.DataFrame([features.dict()])
    # Predict (assumed classification; adapt for regression if needed)
    prediction = pipeline.predict(X)[0]
    # (Optional) Get probabilities, if you want
    # probabilities = pipeline.predict_proba(X)[0]
    return {"prediction": int(prediction)}  # cast to int if binary classifier

# Optional: health check endpoint
@app.get("/")
def root():
    return {"message": "Titanic Pipeline Inference API running!"}