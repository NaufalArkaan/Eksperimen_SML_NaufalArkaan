"""
Traffic Generator for Bank Marketing Monitoring System
Project: Dicoding Final Project - Sistem Machine Learning MLOps
Author: NaufalArkaan
File: Monitoring dan Logging/generate_traffic.py
"""

import os
import time
import random
import requests
import pandas as pd

EXPORTER_URL = "http://127.0.0.1:8001"

def load_dataset_samples():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "bank-full_preprocessing.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        if 'y' in df.columns:
            df = df.drop(columns=['y'])
        return df.values.tolist()
    return [[random.uniform(-1.0, 1.0) for _ in range(40)] for _ in range(50)]

def main():
    print("=" * 60)
    print(" STARTING REAL-TIME MONITORING TRAFFIC GENERATION")
    print("=" * 60)
    
    samples = load_dataset_samples()
    num_samples = len(samples)
    print(f"[Traffic Generator] Loaded {num_samples} sample rows from dataset.")
    
    total_valid = 0
    total_yes = 0
    total_no = 0
    total_errors = 0
    total_exceptions = 0
    
    # Run 3 traffic waves to produce smooth time-series graphs over ~30 seconds
    for wave in range(1, 4):
        print(f"\n---> Wave {wave}/3: Sending batch of requests...")
        
        # 1. Valid predictions batch
        for i in range(40):
            row = samples[random.randint(0, num_samples - 1)]
            payload = {"features": row}
            try:
                res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=5)
                if res.status_code == 200:
                    total_valid += 1
                    lbl = res.json().get("prediction_label")
                    if lbl == "yes":
                        total_yes += 1
                    else:
                        total_no += 1
                else:
                    total_errors += 1
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(0.1) # 100ms interval for natural latency variation
            
        # 2. Invalid requests batch (triggers ml_request_error_total & ml_http_errors_total)
        for _ in range(3):
            payload = {"features": [0.1, 0.2, 0.3]} # Wrong length
            try:
                res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=5)
                if res.status_code == 400:
                    total_errors += 1
            except Exception as e:
                pass
            time.sleep(0.1)
            
        # 3. Simulated Exception batch (triggers ml_exception_total)
        try:
            res = requests.post(f"{EXPORTER_URL}/test_exception", timeout=5)
            if res.status_code == 500:
                total_exceptions += 1
        except Exception as e:
            pass
        
        print(f"  [Wave {wave} Done] Valid: {total_valid} (Yes: {total_yes}, No: {total_no}), Errors: {total_errors}, Exceptions: {total_exceptions}")
        time.sleep(2.0)
        
    print("\n" + "=" * 60)
    print(" TRAFFIC GENERATION COMPLETE!")
    print(f" Total Requests Sent: {total_valid + total_errors + total_exceptions}")
    print(f" Successful Predictions: {total_valid} (Yes: {total_yes}, No: {total_no})")
    print(f" Request Errors (400): {total_errors}")
    print(f" Simulated Exceptions (500): {total_exceptions}")
    print("=" * 60)

if __name__ == "__main__":
    main()
