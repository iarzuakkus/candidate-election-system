from fastapi import FastAPI
from pydantic import BaseModel
from create_dataset import generate_data
from model_test import model_report
from SVM import build_model, model_predict  
from model_test import plot_decision_boundary
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from fastapi import Query
from typing import List
from enum import Enum

app = FastAPI(title="Candidate Election System API 👑")

# Global olarak veri sakla
df = generate_data()
df_train = df.copy()

# 🔹 Tahmin için input formatı
class CandidateInput(BaseModel):
    name: str
    experience: int
    technical_test_score: float

class MetricName(str, Enum):
    accuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    f1_score = "f1_score"
    confusion_matrix = "confusion_matrix"
    classification_report = "classification_report"
    all = "all"

@app.get("/data")
def get_dataset():
    global df_train
    return df_train.to_dict(orient="records")

@app.get("/data/summary")
def get_dataset_info():
    global df_train
    summary = {
        "rows": df_train.shape[0],
        "columns": df_train.shape[1],
        "column_names": list(df_train.columns),
        "null_values": df_train.isnull().sum().to_dict(),
        "data_types": df_train.dtypes.astype(str).to_dict()
    }
    return summary

@app.post("/train")
def train_model():
    global df_train
    model, scaler = build_model(df_train)
    return {"message": "✅ Model başarıyla df_train ile eğitildi ve kaydedildi."}


@app.get("/report")
def get_svm_metrics(metric: MetricName = Query(...)):
    return model_report(df_train, metric.value)

@app.post("/predict")
def predict(
    name: str = Query(...),
    experience: float = Query(...),
    technical_test_score: float = Query(...)
):
    global df_train

    prediction = model_predict(experience, technical_test_score)
    predicted_result = "İşe Alındı" if prediction[0] == 1 else "Reddedildi"

    new_row = {
        "name": name,
        "application_date": pd.Timestamp.now().date(),
        "experience": experience,
        "technical_test_score": technical_test_score,
        "hired": int(prediction[0])
    }

    df_train = pd.concat([df_train, pd.DataFrame([new_row])], ignore_index=True)

    return {
        "prediction": int(prediction[0]),
        "result": predicted_result,
        "saved_to_dataset": new_row
    }

@app.get("/plot")
def plot_model_boundary():
    model, scaler = build_model(df)
    X = df[['experience', 'technical_test_score']] 
    y = df['hired']
    return plot_decision_boundary(model, scaler, X, y)


#python -m uvicorn main:app --reload
#http://127.0.0.1:8000/docs