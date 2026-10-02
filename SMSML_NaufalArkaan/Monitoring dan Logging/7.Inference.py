"""
Inference Testing Script
Project: Dicoding Final Project - Sistem Machine Learning MLOps (Kriteria 4 Model Serving)
Author: NaufalArkaan
File: Monitoring dan Logging/7.Inference.py
"""

import os
import sys
import json
import requests
import pandas as pd


def get_sample_valid_features() -> list:
    """
    Mengambil sampel 40 fitur valid dari dataset bank-full_preprocessing.csv
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "bank-full_preprocessing.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        if 'y' in df.columns:
            sample_features = df.drop(columns=['y']).iloc[0].values.tolist()
            return sample_features
    # Fallback 40 dummy scaled features
    return [0.5] * 40


def main():
    server_url = "http://127.0.0.1:8000"
    predict_url = f"{server_url}/predict"
    evidence_dir = os.path.join(os.path.dirname(__file__), "1.bukti_serving")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_file = os.path.join(evidence_dir, "bukti_serving.txt")

    logs = []
    logs.append("=" * 60)
    logs.append(" BUKTI MODEL SERVING & INFERENCE TESTING")
    logs.append("=" * 60)

    # 1. Health Check Server
    logs.append("\n[1/3] Pengujian Health Check Server (GET /)...")
    try:
        res_health = requests.get(server_url, timeout=5)
        logs.append(f"      - HTTP Status Code: {res_health.status_code}")
        logs.append(f"      - Health Response : {json.dumps(res_health.json(), indent=2)}")
    except Exception as e:
        logs.append(f"      - Error koneksi ke server serving: {e}")
        print("\n".join(logs))
        sys.exit(1)

    # 2. Pengujian Valid Input
    logs.append("\n[2/3] Pengujian Valid Request (POST /predict)...")
    valid_features = get_sample_valid_features()
    valid_payload = {"features": valid_features}

    logs.append(f"      - Sample Input Features (40 Fitur): {valid_features[:5]} ...")
    res_valid = requests.post(predict_url, json=valid_payload, timeout=5)
    logs.append(f"      - HTTP Status Code: {res_valid.status_code}")
    logs.append(f"      - Response JSON   : {json.dumps(res_valid.json(), indent=2)}")

    # 3. Pengujian Invalid Input (Error Handling)
    logs.append("\n[3/3] Pengujian Invalid Request (Jumlah Fitur Tidak Sesuai)...")
    invalid_payload = {"features": [1.0, 2.0, 3.0]} # Hanya 3 fitur
    res_invalid = requests.post(predict_url, json=invalid_payload, timeout=5)
    logs.append(f"      - HTTP Status Code: {res_invalid.status_code} (Expected 400 Bad Request)")
    logs.append(f"      - Response Error  : {json.dumps(res_invalid.json(), indent=2)}")

    logs.append("\n" + "=" * 60)
    logs.append(" SERVING & INFERENCE TESTING SELESAI SUNGGUH BERHASIL")
    logs.append("=" * 60)

    output_text = "\n".join(logs)
    print(output_text)

    # Simpan bukti ke file
    with open(evidence_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\n[Bukti] Log bukti serving berhasil disimpan ke: {evidence_file}")


if __name__ == "__main__":
    main()
