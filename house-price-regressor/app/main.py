import pickle
import os
from fastapi import FastAPI, HTTPException, status
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")

# Load the trained model
with open(model_path, "rb") as f:
    model: GradientBoostingRegressor = pickle.load(f)

app = FastAPI()


@app.get("/")
def home():
    return "ML Model is Running"


@app.get("/health")
def health():
    return {"status": "ok"}


class Feature(BaseModel):
    area: int = Field(..., gt=0, description="Area of the house in square units")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, description="Number of bathrooms")
    stories: int = Field(..., ge=1, description="Number of stories/floors")
    mainroad: bool = Field(
        ..., description="Whether the house is connected to a main road"
    )
    guestroom: bool = Field(..., description="Whether the house has a guest room")
    basement: bool = Field(..., description="Whether the house has a basement")
    hotwaterheating: bool = Field(
        ..., description="Whether the house has hot water heating"
    )
    airconditioning: bool = Field(
        ..., description="Whether the house has air conditioning"
    )
    parking: int = Field(..., ge=0, description="Number of parking spaces")
    prefarea: bool = Field(..., description="Whether the house is in a preferred area")
    furnishingstatus: float = Field(
        ..., ge=0, le=1, description="Status of furnishing in the house"
    )


class PredictInput(BaseModel):
    features: list[Feature] = Field(..., min_length=1)


class PredictOutput(BaseModel):
    prediction: float = Field(..., description="Predicted price of the house")


@app.post("/predict")
def predict(input: PredictInput) -> list[PredictOutput]:
    try:
        # Extract features
        features = input.model_dump()["features"]

        # Convert to DataFrame
        df = pd.DataFrame(features)

        # Make predictions
        predictions = model.predict(df).round(2)

        # Return the prediction
        return [
            PredictOutput(prediction=pred) for pred in predictions
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )
