"""
Baseline Model Training with MLflow Autolog
Project: Dicoding Final Project - Sistem Machine Learning MLOps (Kriteria 2 Baseline)
Author: NaufalArkaan
File: modelling.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import mlflow
import mlflow.sklearn


def get_preprocessing_dataset_path() -> str:
    """
    Cari lokasi file dataset hasil preprocessing bank-full_preprocessing.csv
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "Membangun_model", "bank-full_preprocessing.csv"),
        os.path.join(base_dir, "preprocessing", "bank-full_preprocessing.csv"),
        os.path.join(base_dir, "bank-full_preprocessing.csv"),
        "bank-full_preprocessing.csv",
        "preprocessing/bank-full_preprocessing.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Dataset hasil preprocessing 'bank-full_preprocessing.csv' tidak ditemukan! "
        "Jalankan script preprocessing terlebih dahulu."
    )


def main():
    print("=" * 60)
    print(" BASELINE MODEL TRAINING (MLflow Autolog Enabled)")
    print("=" * 60)

    # 1. Load Processed Dataset
    dataset_path = get_preprocessing_dataset_path()
    print(f"[1/6] Memuat processed dataset dari: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"      Dimensi dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

    if 'y' not in df.columns:
        raise KeyError("Kolom target 'y' tidak ditemukan pada dataset preprocessing.")

    # 2. Pemisahan Fitur (X) dan Target (y)
    X = df.drop(columns=['y'])
    y = df['y']

    # 3. Train-Test Split (80% Train, 20% Test, Stratified)
    print("[2/6] Membagi data menjadi Training Set (80%) dan Test Set (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"      - Data Train: {X_train.shape[0]} baris")
    print(f"      - Data Test : {X_test.shape[0]} baris")

    # 4. Setup MLflow Tracking & Autolog
    print("[3/6] Mengaktifkan MLflow Autolog...")
    mlflow.set_experiment("Bank_Marketing_Baseline")
    mlflow.autolog(log_models=True)

    # 5. Training Baseline Model
    print("[4/6] Melatih baseline model (RandomForestClassifier)...")
    baseline_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    with mlflow.start_run(run_name="Baseline_RandomForest") as run:
        run_id = run.info.run_id
        print(f"      - MLflow Run ID: {run_id}")

        # Fit model (autolog akan merekam parameter, metrik sklearn, & artefak model)
        baseline_model.fit(X_train, y_train)

        # Evaluasi
        print("[5/6] Evaluasi performa model pada Test Set...")
        y_pred = baseline_model.predict(X_test)
        y_proba = baseline_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        print("\n" + "-" * 40)
        print("METRIK PERFORMA BASELINE MODEL:")
        print("-" * 40)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print("-" * 40)
        print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

        # Log metrik tambahan jika diperlukan
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", auc)

        # 6. Validasi Re-loadability Model
        print("\n[6/6] Memvalidasi re-loadability model dari MLflow artifacts...")
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        sample_preds = loaded_model.predict(X_test.iloc[:5])
        print(f"      - Prediksi sampel dari reloaded model: {sample_preds}")
        print(f"      - Model berhasil di-load kembali dari: {model_uri}")

    print("\n" + "=" * 60)
    print(" BASELINE TRAINING SELESAI & BERHASIL DILOG KE MLFLOW")
    print("=" * 60)


if __name__ == "__main__":
    main()
