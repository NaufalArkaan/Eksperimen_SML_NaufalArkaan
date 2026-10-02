"""
FastAPI Model Serving API for Bank Marketing Dataset
Project: Dicoding Final Project - Sistem Machine Learning MLOps (Kriteria 4 Model Serving)
Author: NaufalArkaan
File: Monitoring dan Logging/app.py
"""

import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Bank Marketing Model Serving API",
    description="API untuk inferensi real-time model klasifikasi deposito berjangka Bank Marketing",
    version="1.0.0"
)

# Global variables for model and feature names
MODEL = None
FEATURE_NAMES = None


def load_latest_model():
    """
    Memuat model terbaik dari MLflow Tracking Store secara robust dengan path absolut.
    """
    global MODEL, FEATURE_NAMES

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_path = os.path.join(base_dir, "mlruns").replace("\\", "/")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")

    # Set feature names dari dataset preprocessing
    dataset_path = os.path.join(base_dir, "bank-full_preprocessing.csv")
    if os.path.exists(dataset_path):
        df_sample = pd.read_csv(dataset_path, nrows=2)
        if 'y' in df_sample.columns:
            FEATURE_NAMES = df_sample.drop(columns=['y']).columns.tolist()

    # Search runs di segenap eksperimen
    for exp_name in ["Bank_Marketing_Hyperparameter_Tuning", "Bank_Marketing_Baseline", "Bank_Marketing_MLProject"]:
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is not None:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                if not runs.empty:
                    last_run_id = runs.iloc[0]['run_id']
                    model_uri = f"runs:/{last_run_id}/model"
                    print(f"[Serving] Memuat model MLflow dari URI: {model_uri}")
                    MODEL = mlflow.sklearn.load_model(model_uri)
                    break
        except Exception as e:
            print(f"[Serving Warning] Error saat memuat dari {exp_name}: {e}")

    if MODEL is None:
        print("[Serving Warning] Model MLflow tidak ditemukan via tracking.")


@app.on_event("startup")
def startup_event():
    load_latest_model()


class PredictRequest(BaseModel):
    features: Union[List[float], List[int], Dict[str, Any]]


@app.get("/")
def read_root():
    if MODEL is None:
        load_latest_model()
    return {
        "service": "Bank Marketing Model Serving API",
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "num_features": len(FEATURE_NAMES) if FEATURE_NAMES else 40
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    global MODEL, FEATURE_NAMES

    if MODEL is None:
        load_latest_model()
        if MODEL is None:
            raise HTTPException(status_code=500, detail="Model belum siap atau gagal dimuat.")

    try:
        data = payload.features
        expected_len = len(FEATURE_NAMES) if FEATURE_NAMES else 40

        # Penanganan input list (40 fitur ter-scale)
        if isinstance(data, list):
            if len(data) != expected_len:
                raise HTTPException(
                    status_code=400,
                    detail=f"Jumlah fitur tidak sesuai. Diharapkan {expected_len} fitur, diterima {len(data)} fitur."
                )
            input_df = pd.DataFrame([data], columns=FEATURE_NAMES if FEATURE_NAMES else None)

        # Penanganan input dictionary
        elif isinstance(data, dict):
            input_df = pd.DataFrame([data])
            if FEATURE_NAMES:
                for col in FEATURE_NAMES:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[FEATURE_NAMES]

        else:
            raise HTTPException(status_code=400, detail="Format data tidak valid.")

        # Inference
        pred_code = int(MODEL.predict(input_df)[0])
        pred_label = "yes" if pred_code == 1 else "no"

        prob_no, prob_yes = 0.5, 0.5
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(input_df)[0]
            prob_no = round(float(probs[0]), 4)
            prob_yes = round(float(probs[1]), 4)

        return {
            "status": "success",
            "prediction_code": pred_code,
            "prediction_label": pred_label,
            "probability": {
                "no": prob_no,
                "yes": prob_yes
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saat melakukan prediksi: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
