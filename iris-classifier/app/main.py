from typing import Dict
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
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
    features: list[list[float]]


class PredictOutput(BaseModel):
    prediction: int
    confidence: float


def validate_features(features: list[list[float]]) -> PredictInput:
    if not features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Features is not exist."
        )

    if not isinstance(features, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Features should be a list."
        )

    if not all(len(i) == 4 for i in features):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Features should contain exactly 4 values.",
        )

    if not all(isinstance(i, (int, float)) for sublist in features for i in sublist):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Features should contain only numbers.",
        )

    return PredictInput(features=features)


@app.post("/predict")
def predict(input: Dict) -> list[PredictOutput]:
    predict_input = validate_features(input.get("features"))

    input_features = np.array(predict_input.features)
    prediction_probas = model.predict_proba(input_features)

    result = [
        PredictOutput(prediction=int(np.argmax(prob)), confidence=float(np.max(prob)))
        for prob in prediction_probas
    ]

    return result
