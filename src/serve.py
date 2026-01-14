from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
from .model import load_model
from .explainer import explain

app = FastAPI()


class PredictRequest(BaseModel):
    features: List[float]
    feature_names: List[str]


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.features], columns=req.feature_names)
    model = load_model()
    preds = model.predict(df).tolist()
    ex = explain(model, df)
    return {"predictions": preds, "explanation": ex}
