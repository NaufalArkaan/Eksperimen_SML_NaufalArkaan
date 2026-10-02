"""
Prometheus Exporter for Bank Marketing Model Serving
Project: Dicoding Final Project - Sistem Machine Learning MLOps (Milestone 11 & 13)
Author: NaufalArkaan
File: Monitoring dan Logging/3.prometheus_exporter.py
"""

import time
import os
import requests
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import uvicorn
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY
)

app = FastAPI(
    title="Bank Marketing Prometheus Exporter",
    description="Exporter metrics Prometheus & Webhook Alerting untuk monitoring model serving Bank Marketing API",
    version="1.0.0"
)

# Target Model Serving URL
MODEL_SERVING_URL = os.getenv("MODEL_SERVING_URL", "http://127.0.0.1:8000")

# Store received webhook notifications in memory for verification
RECEIVED_NOTIFICATIONS = []

# ---------------------------------------------------------
# PROMETHEUS METRICS DEFINITIONS (11 Relevan Metrics)
# ---------------------------------------------------------

# 1. ml_request_total: Total incoming requests to ML system
ML_REQUEST_TOTAL = Counter(
    "ml_request_total",
    "Total incoming requests to the ML serving system",
    ["endpoint"]
)

# 2. ml_prediction_total: Total prediction calls processed
ML_PREDICTION_TOTAL = Counter(
    "ml_prediction_total",
    "Total prediction requests processed by the model",
    ["status"]
)

# 3. ml_prediction_yes_total: Total predictions for class 'yes' (subscribed)
ML_PREDICTION_YES_TOTAL = Counter(
    "ml_prediction_yes_total",
    "Total positive predictions ('yes' - deposit subscribed)"
)

# 4. ml_prediction_no_total: Total predictions for class 'no' (not subscribed)
ML_PREDICTION_NO_TOTAL = Counter(
    "ml_prediction_no_total",
    "Total negative predictions ('no' - deposit not subscribed)"
)

# 5. ml_request_error_total: Total request errors (invalid data, bad requests)
ML_REQUEST_ERROR_TOTAL = Counter(
    "ml_request_error_total",
    "Total request errors due to invalid payload or client side error",
    ["error_type"]
)

# 6. ml_request_latency_seconds: Total request latency duration histogram
ML_REQUEST_LATENCY_SECONDS = Histogram(
    "ml_request_latency_seconds",
    "Overall request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# 7. ml_prediction_latency_seconds: Model inference latency duration histogram
ML_PREDICTION_LATENCY_SECONDS = Histogram(
    "ml_prediction_latency_seconds",
    "Model inference latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# 8. ml_http_requests_total: Total HTTP requests processed by server
ML_HTTP_REQUESTS_TOTAL = Counter(
    "ml_http_requests_total",
    "Total HTTP requests handled by exporter HTTP server",
    ["method", "status_code"]
)

# 9. ml_http_errors_total: Total HTTP error responses (4xx, 5xx)
ML_HTTP_ERRORS_TOTAL = Counter(
    "ml_http_errors_total",
    "Total HTTP error responses (4xx and 5xx status codes)",
    ["status_code"]
)

# 10. ml_inference_total: Total successful model inference operations
ML_INFERENCE_TOTAL = Counter(
    "ml_inference_total",
    "Total successful model inference executions"
)

# 11. ml_exception_total: Total internal exceptions encountered
ML_EXCEPTION_TOTAL = Counter(
    "ml_exception_total",
    "Total internal exceptions encountered during request processing",
    ["exception_type"]
)


class PredictRequest(BaseModel):
    features: Union[List[float], List[int], Dict[str, Any]]


@app.get("/metrics")
def get_metrics():
    """Prometheus metrics scrape endpoint"""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
@app.get("/health")
def health_check():
    ML_HTTP_REQUESTS_TOTAL.labels(method="GET", status_code="200").inc()
    return {"exporter_status": "UP", "target_model_url": MODEL_SERVING_URL}


@app.post("/test_exception")
def test_exception():
    """Trigger simulated exception for metric verification"""
    ML_REQUEST_TOTAL.labels(endpoint="/test_exception").inc()
    ML_HTTP_REQUESTS_TOTAL.labels(method="POST", status_code="500").inc()
    ML_HTTP_ERRORS_TOTAL.labels(status_code="500").inc()
    ML_EXCEPTION_TOTAL.labels(exception_type="SimulatedException").inc()
    raise HTTPException(status_code=500, detail="Simulated exception for testing ml_exception_total")


@app.post("/predict")
def predict_proxy(payload: PredictRequest):
    """
    Proxy request ke Model Serving API dan catat semua metrics Prometheus.
    """
    start_req_time = time.time()
    ML_REQUEST_TOTAL.labels(endpoint="/predict").inc()

    try:
        start_infer_time = time.time()
        # Forward request ke model serving API (port 8000)
        res = requests.post(f"{MODEL_SERVING_URL}/predict", json={"features": payload.features}, timeout=5.0)
        infer_duration = time.time() - start_infer_time

        status_code_str = str(res.status_code)
        ML_HTTP_REQUESTS_TOTAL.labels(method="POST", status_code=status_code_str).inc()

        if res.status_code != 200:
            ML_HTTP_ERRORS_TOTAL.labels(status_code=status_code_str).inc()
            ML_REQUEST_ERROR_TOTAL.labels(error_type="model_api_error").inc()
            ML_PREDICTION_TOTAL.labels(status="error").inc()
            raise HTTPException(status_code=res.status_code, detail=res.text)

        data = res.json()
        pred_label = str(data.get("prediction_label", "")).lower()

        # Update prediction metrics
        ML_PREDICTION_LATENCY_SECONDS.observe(infer_duration)
        ML_PREDICTION_TOTAL.labels(status="success").inc()
        ML_INFERENCE_TOTAL.inc()

        if pred_label == "yes":
            ML_PREDICTION_YES_TOTAL.inc()
        elif pred_label == "no":
            ML_PREDICTION_NO_TOTAL.inc()

        req_duration = time.time() - start_req_time
        ML_REQUEST_LATENCY_SECONDS.labels(endpoint="/predict").observe(req_duration)

        return data

    except HTTPException as he:
        req_duration = time.time() - start_req_time
        ML_REQUEST_LATENCY_SECONDS.labels(endpoint="/predict").observe(req_duration)
        raise he

    except requests.exceptions.RequestException as re:
        req_duration = time.time() - start_req_time
        ML_REQUEST_LATENCY_SECONDS.labels(endpoint="/predict").observe(req_duration)
        ML_HTTP_ERRORS_TOTAL.labels(status_code="503").inc()
        ML_REQUEST_ERROR_TOTAL.labels(error_type="connection_error").inc()
        ML_EXCEPTION_TOTAL.labels(exception_type=type(re).__name__).inc()
        raise HTTPException(status_code=503, detail=f"Model serving unreachable: {str(re)}")

    except Exception as e:
        req_duration = time.time() - start_req_time
        ML_REQUEST_LATENCY_SECONDS.labels(endpoint="/predict").observe(req_duration)
        ML_HTTP_ERRORS_TOTAL.labels(status_code="500").inc()
        ML_REQUEST_ERROR_TOTAL.labels(error_type="internal_error").inc()
        ML_EXCEPTION_TOTAL.labels(exception_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Internal exporter error: {str(e)}")


# ---------------------------------------------------------
# GRAFANA ALERTING WEBHOOK NOTIFICATION RECEIVER
# ---------------------------------------------------------

@app.post("/webhook")
async def grafana_webhook(request: Request):
    """
    Receiver untuk Notifikasi Alert dari Grafana (Phase 13)
    """
    try:
        body = await request.json()
        status = body.get("status", "unknown")
        alerts = body.get("alerts", [])

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "alerts_count": len(alerts),
            "alerts": alerts
        }
        RECEIVED_NOTIFICATIONS.append(record)

        print(f"\n[Grafana Webhook Alert Received!] Status: {status.upper()} | Count: {len(alerts)}")
        for a in alerts:
            labels = a.get("labels", {})
            annotations = a.get("annotations", {})
            print(f" -> Alert Name: {labels.get('alertname', 'Unknown')}")
            print(f"    State     : {a.get('status', status)}")
            print(f"    Summary   : {annotations.get('summary', 'No summary')}")
            print(f"    Description: {annotations.get('description', 'No description')}")

        return {"status": "success", "message": "Notification received successfully", "record": record}
    except Exception as e:
        print(f"[Webhook Error]: {e}")
        return {"status": "error", "detail": str(e)}


@app.get("/notifications")
def get_notifications():
    """Endpoint untuk mengecek riwayat notifikasi alert yang diterima"""
    return {"notifications_count": len(RECEIVED_NOTIFICATIONS), "notifications": RECEIVED_NOTIFICATIONS}


if __name__ == "__main__":
    print("[Exporter] Starting Prometheus Exporter & Alert Webhook Receiver on http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
