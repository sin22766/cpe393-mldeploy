from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")

app = FastAPI()

# Load the trained model
with open(model_path, "rb") as f:
    model: RandomForestClassifier = pickle.load(f)


@app.get("/")
def home():
    return "ML Model is Running"


@app.get("/health")
def health():
    return {"status": "ok"}


class PredictInput(BaseModel):
    features: list[tuple[float, float, float, float]] = Field(..., min_length=1)


class PredictOutput(BaseModel):
    prediction: int
    confidence: float


@app.post("/predict")
def predict(input: PredictInput) -> list[PredictOutput]:
    input_features = np.array(input.features)
    prediction_probas = model.predict_proba(input_features)

    result = [
        PredictOutput(prediction=int(np.argmax(prob)), confidence=float(np.max(prob)))
        for prob in prediction_probas
    ]

    return result
