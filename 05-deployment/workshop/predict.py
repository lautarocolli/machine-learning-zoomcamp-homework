import pickle
import uvicorn

from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# request
class Customer(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # Categorical fields with strict value validation
    gender: Literal['male', 'female']
    seniorcitizen: Literal[0, 1]
    partner: Literal['yes', 'no']
    dependents: Literal['yes', 'no']
    phoneservice: Literal['yes', 'no']
    multiplelines: Literal['no', 'yes', 'no_phone_service']
    internetservice: Literal['fiber_optic', 'dsl', 'no']
    onlinesecurity: Literal['no', 'yes', 'no_internet_service']
    onlinebackup: Literal['no', 'yes', 'no_internet_service']
    deviceprotection: Literal['no', 'yes', 'no_internet_service']
    techsupport: Literal['no', 'yes', 'no_internet_service']
    streamingtv: Literal['no', 'yes', 'no_internet_service']
    streamingmovies: Literal['no', 'yes', 'no_internet_service']
    contract: Literal['month-to-month', 'one_year', 'two_year']
    paperlessbilling: Literal['yes', 'no']
    paymentmethod: Literal[
        'electronic_check', 
        'mailed_check', 
        'bank_transfer_(automatic)', 
        'credit_card_(automatic)'
    ]

    # Numerical fields with ranges based on your stats
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

# response
class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title = 'churn-prediction')

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    churn_proba = pipeline.predict_proba([customer])[0,1]
    return churn_proba

@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    churn_proba = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=churn_proba,
        churn=bool(churn_proba >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8080, reload=True)