import pickle
import uvicorn

from typing import Dict, Any
from fastapi import FastAPI

app = FastAPI(title = 'churn-prediction')

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    churn_proba = pipeline.predict_proba([customer])[0,1]
    return churn_proba

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    churn_proba = predict_single(customer)
    return {
        "churn_probability": churn_proba,
        "churn": bool(churn_proba >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8080, reload=True)