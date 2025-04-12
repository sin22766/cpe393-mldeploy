from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return "ML Model is Running"

class PredictInput(BaseModel):
    features: list

class PredictOutput(BaseModel):
    prediction: int
    confidence: float

@app.post("/predict")
def predict(predict_input: PredictInput) -> PredictOutput:
    input_features = np.array(predict_input.features).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]
    return PredictOutput(
        prediction=int(prediction),
        confidence=float(prediction_proba[prediction])
    )