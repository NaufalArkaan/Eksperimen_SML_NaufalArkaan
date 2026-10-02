"""
Hyperparameter Tuning with Manual MLflow & DagsHub Integration
Project: Dicoding Final Project - Sistem Machine Learning MLOps (Kriteria 2 Advanced)
Author: NaufalArkaan
File: Membangun_model/modelling_tuning.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.sklearn


def print_log(msg: str):
    print(msg, flush=True)


def setup_mlflow_tracking():
    """
    Konfigurasi remote MLflow tracking via DagsHub jika environment variables tersedia.
    Jika tidak tersedia, gunakan local tracking store.
    """
    dagshub_username = os.environ.get("DAGSHUB_USERNAME")
    dagshub_repo = os.environ.get("DAGSHUB_REPO")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN") or os.environ.get("MLFLOW_TRACKING_PASSWORD")

    if dagshub_username and dagshub_repo and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        tracking_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print_log(f"[MLflow] Remote DagsHub MLflow Tracking Aktif: {tracking_uri}")
        return True
    else:
        print_log("[MLflow] Environment Variable DagsHub belum lengkap. Menggunakan Local MLflow Tracking.")
        return False


def get_preprocessing_dataset_path() -> str:
    """
    Cari lokasi file dataset hasil preprocessing bank-full_preprocessing.csv
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def generate_artifacts(model, X_test, y_test, feature_names, artifact_dir="temp_artifacts"):
    """
    Menghasilkan 5 artefak visualisasi & metadata evaluasi untuk dilog ke MLflow/DagsHub:
    1. confusion_matrix.png
    2. feature_importance.png
    3. classification_report.json
    4. model_metadata.json
    5. prediction_sample.csv
    """
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_paths = []

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No (0)', 'Yes (1)'], yticklabels=['No (0)', 'Yes (1)'])
    plt.title('Confusion Matrix - Tuned Model', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    artifact_paths.append(cm_path)

    # 2. Feature Importance Plot (Top 15 Fitur)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
        plt.barh(range(len(indices)), importances[indices][::-1], color='#2ecc71', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        fi_path = os.path.join(artifact_dir, "feature_importance.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        artifact_paths.append(fi_path)

    # 3. Classification Report (JSON)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(artifact_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    artifact_paths.append(report_path)

    # 4. Model Metadata (JSON)
    metadata = {
        "model_architecture": "RandomForestClassifier",
        "num_features": len(feature_names),
        "test_samples": len(y_test),
        "python_version": sys.version,
        "scikit_learn_version": "1.9.1",
        "mlflow_version": mlflow.__version__
    }
    meta_path = os.path.join(artifact_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    artifact_paths.append(meta_path)

    # 5. Sample Predictions CSV
    sample_df = X_test.iloc[:50].copy()
    sample_df['y_true'] = y_test.iloc[:50].values
    sample_df['y_pred'] = y_pred[:50]
    sample_df['y_prob_yes'] = y_proba[:50]
    sample_path = os.path.join(artifact_dir, "prediction_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    artifact_paths.append(sample_path)

    return artifact_dir, artifact_paths


def main():
    print_log("=" * 60)
    print_log(" HYPERPARAMETER TUNING & MANUAL MLFLOW/DAGSHUB LOGGING")
    print_log("=" * 60)

    # 0. Setup Tracking URI
    is_remote = setup_mlflow_tracking()

    # 1. Load Processed Dataset
    dataset_path = get_preprocessing_dataset_path()
    print_log(f"[1/7] Memuat processed dataset dari: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print_log(f"      Dimensi dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

    if 'y' not in df.columns:
        raise KeyError("Kolom target 'y' tidak ditemukan pada dataset preprocessing.")

    # 2. Pemisahan Fitur (X) dan Target (y)
    X = df.drop(columns=['y'])
    y = df['y']
    feature_names = X.columns.tolist()

    # 3. Train-Test Split (80% Train, 20% Test, Stratified)
    print_log("[2/7] Membagi data menjadi Training Set (80%) dan Test Set (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print_log(f"      - Data Train: {X_train.shape[0]} baris")
    print_log(f"      - Data Test : {X_test.shape[0]} baris")

    # 4. Grid Search Hyperparameter Tuning
    print_log("[3/7] Menjalankan GridSearchCV Hyperparameter Tuning...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [15, 25],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    base_rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_

    print_log("\n" + "-" * 40)
    print_log("HYPERPARAMETER TERBAIK (GRID SEARCH):")
    print_log("-" * 40)
    for k, v in best_params.items():
        print_log(f"  {k}: {v}")
    print_log(f"  Best CV F1-Score: {best_cv_f1:.4f}")
    print_log("-" * 40)

    # 5. Evaluasi pada Test Set
    print_log("\n[4/7] Evaluasi model terbaik pada Test Set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print_log("\n" + "-" * 40)
    print_log("METRIK PERFORMA TUNED MODEL (TEST SET):")
    print_log("-" * 40)
    print_log(f"  Accuracy : {acc:.4f}")
    print_log(f"  Precision: {prec:.4f}")
    print_log(f"  Recall   : {rec:.4f}")
    print_log(f"  F1-Score : {f1:.4f}")
    print_log(f"  ROC-AUC  : {auc:.4f}")
    print_log("-" * 40)

    # 6. Manual MLflow Logging
    print_log("\n[5/7] Melakukan Manual MLflow Logging (Parameters, Metrics, & Model)...")
    mlflow.set_experiment("Bank_Marketing_Hyperparameter_Tuning")

    with mlflow.start_run(run_name="Tuned_RandomForest_Manual_Logging") as run:
        run_id = run.info.run_id
        print_log(f"      - MLflow Run ID: {run_id}")

        # a. Manual logging parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_params(best_params)

        # b. Manual logging metrics
        mlflow.log_metric("best_cv_f1_score", best_cv_f1)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", auc)

        # c. Generate and log 5 manual artifacts
        print_log("[6/7] Membuat dan melog visualisasi & artefak evaluasi...")
        artifact_dir, artifact_files = generate_artifacts(best_model, X_test, y_test, feature_names)
        for art_path in artifact_files:
            mlflow.log_artifact(art_path)
            print_log(f"      - Artifact logged: {os.path.basename(art_path)}")

        # d. Manual model logging
        print_log("      - Melog model ke MLflow Model Registry Store...")
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example
        )

        # 7. Validasi Re-loadability Model
        print_log("\n[7/7] Memvalidasi re-loadability model dari MLflow artifacts...")
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        sample_preds = loaded_model.predict(X_test.iloc[:5])
        print_log(f"      - Prediksi sampel dari reloaded tuned model: {sample_preds}")
        print_log(f"      - Model berhasil di-load kembali dari: {model_uri}")

    # Membersihkan folder temporary artifact lokal
    if os.path.exists(artifact_dir):
        import shutil
        shutil.rmtree(artifact_dir, ignore_errors=True)

    print_log("\n" + "=" * 60)
    print_log(" HYPERPARAMETER TUNING & MANUAL LOGGING SELESAI SUNGGUH BERHASIL")
    print_log("=" * 60)


if __name__ == "__main__":
    main()
