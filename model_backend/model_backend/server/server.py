import datetime
from typing import List

import numpy as np
import psycopg2
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from model_backend.engine.predictor import Predictor


MODEL_PATH = "model_backend/artifacts/model.pth"


predictor = Predictor(MODEL_PATH)

def write_to_database(pred, label):
    conn = psycopg2.connect(
        dbname="mnist",
        user="mnist",
        password="mnist",
        host="mnist-db",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (timestamp, pred, label) VALUES (NOW(), %s, %s)",
        (pred, label)
    )
    conn.commit()

    cur.execute("""
            SELECT * 
            FROM history
            ORDER BY timestamp DESC
            LIMIT 10;
            """
                )

    res = cur.fetchall()

    cur.close()
    conn.close()

    timestamps = [timestamp for timestamp, _, _ in res]
    preds = [pred for _, pred, _ in res]
    labels = [label for _, _, label in res]

    return timestamps, preds, labels


app = FastAPI()


class PredData(BaseModel):
    image: List[List[int]]
    label: str

@app.post("/predict")
async def predict(data: PredData):
    image = np.asarray(data.image)
    label = data.label

    pred, confidence = predictor.predict(image)

    timestamps, preds, labels = write_to_database(pred, label)

    return {
        "label": pred,
        "confidence": confidence,
        "history": {"timestamps": timestamps, "pred": preds, "label": labels}}
