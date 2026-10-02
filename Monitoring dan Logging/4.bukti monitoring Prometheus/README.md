# Laporan Monitoring Model Serving dengan Prometheus (Milestone 11)

**Project**: Dicoding Final Project - Sistem Machine Learning MLOps  
**Author**: NaufalArkaan  
**Phase**: Phase 11 — Prometheus  

---

## 1. Arsitektur Monitoring

```
[ Model API (FastAPI - Port 8000) ]
                 ↑
                 │ (Forward Request & Latency Check)
                 ↓
[ Prometheus Exporter (Port 8001) ]
                 ↑
                 │ (Scrape /metrics setiap 2s)
                 ↓
[ Prometheus Server (Port 9090) ]
```

---

## 2. File Konfigurasi dan Exporter

- **Prometheus Config**: `Monitoring dan Logging/2.prometheus.yml`
- **Prometheus Exporter**: `Monitoring dan Logging/3.prometheus_exporter.py`
- **Target Scraping**: `127.0.0.1:8001` (`job_name: bank_marketing_model_exporter`)

---

## 3. Daftar 11 Metrics Monitoring Model Serving

| No | Nama Metric | Tipe | Deskripsi & Fungsi Monitoring | Nilai Teruji |
|---|---|---|---|---|
| 1 | `ml_request_total` | Counter | Menghitung total HTTP request ke sistem ML serving | `/predict`: 26, `/test_exception`: 1 |
| 2 | `ml_prediction_total` | Counter | Menghitung total prediksi yang diproses model berdasarkan status | `success`: 22, `error`: 4 |
| 3 | `ml_prediction_yes_total` | Counter | Menghitung total prediksi kelas positif ('yes' / deposit subscribed) | 10 |
| 4 | `ml_prediction_no_total` | Counter | Menghitung total prediksi kelas negatif ('no' / deposit not subscribed) | 12 |
| 5 | `ml_request_error_total` | Counter | Menghitung error request akibat payload invalid/bad request | `model_api_error`: 4 |
| 6 | `ml_request_latency_seconds` | Histogram | Mengukur distribusi total latensi request (detik) | 26 samples (sum: 0.779s) |
| 7 | `ml_prediction_latency_seconds` | Histogram | Mengukur latensi inferensi murni model (detik) | 22 samples (sum: 0.734s) |
| 8 | `ml_http_requests_total` | Counter | Menghitung total request HTTP berdasarkan HTTP status code | `200`: 22, `400`: 4, `500`: 1 |
| 9 | `ml_http_errors_total` | Counter | Menghitung total HTTP error status responses (4xx/5xx) | `400`: 4, `500`: 1 |
| 10 | `ml_inference_total` | Counter | Menghitung total eksekusi inferensi model yang sukses | 22 |
| 11 | `ml_exception_total` | Counter | Menghitung total internal exception yang ditangkap di exporter | `SimulatedException`: 1 |

---

## 4. Bukti Tangkapan Layar (Screenshots)

Seluruh bukti screenshot query Prometheus tersimpan pada folder `Monitoring dan Logging/4.bukti monitoring Prometheus/`:

1. `01_prometheus_targets.png` — Bukti target Exporter dalam status **UP**
2. `02_ml_request_total.png` — Query metric `ml_request_total`
3. `03_ml_prediction_total.png` — Query metric `ml_prediction_total`
4. `04_ml_prediction_yes_total.png` — Query metric `ml_prediction_yes_total`
5. `05_ml_prediction_no_total.png` — Query metric `ml_prediction_no_total`
6. `06_ml_request_error_total.png` — Query metric `ml_request_error_total`
7. `07_ml_request_latency_seconds.png` — Query metric `ml_request_latency_seconds_count`
8. `08_ml_prediction_latency_seconds.png` — Query metric `ml_prediction_latency_seconds_count`
9. `09_ml_http_requests_total.png` — Query metric `ml_http_requests_total`
10. `10_ml_http_errors_total.png` — Query metric `ml_http_errors_total`
11. `11_ml_inference_total.png` — Query metric `ml_inference_total`
12. `12_ml_exception_total.png` — Query metric `ml_exception_total`
