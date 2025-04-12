from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Load the trained model
with open("model.pkl", "rb") as f:
    model: RandomForestClassifier = pickle.load(f)

@app.get("/")
def home():
    return "ML Model is Running"

class PredictInput(BaseModel):
    features: list[list[float]]

class PredictOutput(BaseModel):
    prediction: int
    confidence: float


@app.post("/predict")
def predict(predict_input: PredictInput) -> list[PredictOutput]:
    input_features = np.array(predict_input.features)
    prediction_probas = model.predict_proba(input_features)
    
    result = [PredictOutput(
        prediction=int(np.argmax(prob)),
        confidence=float(np.max(prob))
    ) for prob in prediction_probas]

    return result
    

    