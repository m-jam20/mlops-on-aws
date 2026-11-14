"""HTTP API that serves churn predictions."""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from churn import __version__, config

YesNo = Literal["Yes", "No"]
YesNoNoService = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer, with the same columns as the training data."""

    tenure: int = Field(ge=0, description="Months as a customer", examples=[5])
    MonthlyCharges: float = Field(ge=0, examples=[70.35])
    TotalCharges: float = Field(ge=0, examples=[351.75])
    SeniorCitizen: Literal[0, 1] = 0
    gender: Literal["Female", "Male"] = "Female"
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: YesNoNoService = "No"
    OnlineBackup: YesNoNoService = "No"
    DeviceProtection: YesNoNoService = "No"
    TechSupport: YesNoNoService = "No"
    StreamingTV: YesNoNoService = "No"
    StreamingMovies: YesNoNoService = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request.
    app.state.model = joblib.load(config.MODEL_PATH)
    yield


app = FastAPI(title="Churn API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])[config.FEATURES]
    probability = float(app.state.model.predict_proba(row)[0, 1])
    return Prediction(
        churn_probability=round(probability, 4),
        churn=probability >= 0.5,
        model_version=__version__,
    )
