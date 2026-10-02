# Laporan Monitoring Model Serving dengan Grafana (Milestone 12)

**Project**: Dicoding Final Project - Sistem Machine Learning MLOps  
**Author**: NaufalArkaan  
**Phase**: Phase 12 — Grafana  

---

## 1. Arsitektur Monitoring Grafana

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
                 ↑
                 │ (Datasource Proxy HTTP Query)
                 ↓
[ Grafana Dashboard "NaufalArkaan" (Port 3000) ]
```

---

## 2. Datasource & Configuration

- **Datasource Name**: `Prometheus` (Default)
- **URL**: `http://127.0.0.1:9090`
- **Dashboard Name**: **`NaufalArkaan`** *(WAJIB sesuai instruksi)*
- **Dashboard UID**: `naufalarkaan_ml_monitoring`
- **JSON Definition**: `Monitoring dan Logging/1.grafana_dashboard_NaufalArkaan.json`

---

## 3. Daftar 11 Panels Dashboard "NaufalArkaan"

| Panel ID | Nama Panel | Visualisasi | Target Metric / PromQL | Deskripsi |
|---|---|---|---|---|
| 1 | `1. Total Requests` | Stat | `sum(ml_request_total)` | Total request yang masuk ke API ML serving |
| 2 | `2. Total Predictions` | Stat | `sum(ml_prediction_total)` | Total eksekusi prediksi model |
| 3 | `3. Prediction Yes` | Gauge | `ml_prediction_yes_total` | Total prediksi positif ('yes' / deposit subscribed) |
| 4 | `4. Prediction No` | Gauge | `ml_prediction_no_total` | Total prediksi negatif ('no' / not subscribed) |
| 5 | `5. Error Count` | Stat | `sum(ml_request_error_total)` | Total error request akibat malformed payload |
| 6 | `6. Inference Count` | Stat | `ml_inference_total` | Total eksekusi inferensi sukses |
| 7 | `7. Request Latency (Average)` | Time series | `ml_request_latency_seconds_sum / ml_request_latency_seconds_count` | Rata-rata latensi per request (detik) |
| 8 | `8. Prediction Latency (Average)` | Time series | `ml_prediction_latency_seconds_sum / ml_prediction_latency_seconds_count` | Rata-rata latensi inferensi murni model (detik) |
| 9 | `9. HTTP Requests` | Bar chart | `ml_http_requests_total` | Distribusi HTTP request per method & status code |
| 10 | `10. HTTP Errors` | Stat | `sum(ml_http_errors_total)` | Total HTTP error status responses (4xx/5xx) |
| 11 | `11. Exception Count` | Stat | `sum(ml_exception_total)` | Total internal exception yang ditangkap di exporter |

---

## 4. Bukti Tangkapan Layar (Screenshots)

Seluruh bukti screenshot dashboard Grafana **`NaufalArkaan`** tersimpan pada folder `Monitoring dan Logging/5.bukti monitoring Grafana/`:

1. `01_grafana_dashboard_NaufalArkaan_full.png` — Bukti tampilan penuh Dashboard **NaufalArkaan** dengan seluruh 11 panel terisi data
2. `02_grafana_dashboard_NaufalArkaan_view.png` — Bukti visualisasi panel Time Series & Bar Chart pada Dashboard **NaufalArkaan**
3. `03_grafana_dashboard_NaufalArkaan_metrics.png` — Bukti panel Stat & Gauge pada Dashboard **NaufalArkaan**
4. `04_grafana_datasource_prometheus.png` — Bukti konfigurasi Datasource Prometheus yang terhubung di Grafana
