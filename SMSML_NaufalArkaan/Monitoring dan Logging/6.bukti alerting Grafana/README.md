# Laporan Monitoring Alerting Grafana (Milestone 13)

**Project**: Dicoding Final Project - Sistem Machine Learning MLOps  
**Author**: NaufalArkaan  
**Phase**: Phase 13 — Grafana Alerting  

---

## 1. Arsitektur Grafana Alerting & Notification Flow

```
[ Model Serving / Client Traffic ]
                 ↓
[ Prometheus Exporter (Port 8001) ]
                 ↓ (Scrape /metrics)
[ Prometheus Server (Port 9090) ]
                 ↓ (Evaluate Rules)
[ Grafana Alerting Engine (Port 3000) ]
                 ↓ (HTTP Webhook Notification)
[ Webhook Receiver Endpoint: http://127.0.0.1:8001/webhook ]
```

---

## 2. 3 Alert Rules Terkonfigurasi & Teruji

### 🚨 ALERT 1 — HIGH REQUEST ERROR RATE
- **UID**: `alert_high_request_error_rate`
- **PromQL Query**: `sum(ml_request_error_total)`
- **Condition**: `$B > 0` (Error count melebihi 0 request error)
- **Evaluation Period**: Evaluate `every 10s` for `0s` (Instant Alerting)
- **Summary**: `High Request Error Rate Detected`
- **Description**: `Jumlah error request ML serving melebihi threshold (> 0 errors)`

### 🚨 ALERT 2 — HIGH REQUEST LATENCY
- **UID**: `alert_high_request_latency`
- **PromQL Query**: `ml_request_latency_seconds_sum / ml_request_latency_seconds_count`
- **Condition**: `$B > 0.001` (Rata-rata latensi request melebihi 0.001 detik / 1 ms)
- **Evaluation Period**: Evaluate `every 10s` for `0s` (Instant Alerting)
- **Summary**: `High Request Latency Detected`
- **Description**: `Latensi rata-rata request ML serving melebihi threshold (> 0.001s / 1ms)`

### 🚨 ALERT 3 — HIGH INFERENCE ERROR / EXCEPTION
- **UID**: `alert_high_inference_exception`
- **PromQL Query**: `sum(ml_exception_total)`
- **Condition**: `$B > 0` (Internal exception count melebihi 0 exception)
- **Evaluation Period**: Evaluate `every 10s` for `0s` (Instant Alerting)
- **Summary**: `High Inference Exception Detected`
- **Description**: `Jumlah internal exception pada model inference melebihi threshold (> 0 exceptions)`

---

## 3. Contact Point & Notification Policy

- **Contact Point**: `MLOps Webhook Receiver` (Type: `webhook`, URL: `http://127.0.0.1:8001/webhook`)
- **Notification Policy**: Default Policy (`group_by: [alertname]`, `group_wait: 1s`, `group_interval: 2s`, `repeat_interval: 1m`)

---

## 4. Hasil Pengujian Siklus Alert (Testing Lifecycle)

1. **Trigger Alert**: Dikirimkan request bermasalah (payload 3 fitur daripada 40 fitur) dan request `/test_exception` untuk memicu kondisi error dan exception.
2. **Alert Firing**: Rule Grafana mendeteksi kondisi melebihi threshold dan berubah status menjadi **`Alerting (Firing)`**.
3. **Notifikasi Diterima**: Endpoint webhook Exporter `http://127.0.0.1:8001/webhook` menerima paylod notifikasi HTTP POST dari Grafana dengan status **`FIRING`**.
4. **Sistem Kembali Normal (Recovery)**: Trafik error dihentikan dan sistem kembali melayani request normal.
5. **Alert Resolved**: Grafana mendeteksi kondisi kembali normal, mengubah status alert menjadi **`OK`**, dan mengirimkan notifikasi webhook berstatus **`RESOLVED`**.

---

## 5. Bukti Tangkapan Layar (Screenshots)

Seluruh bukti screenshot Grafana Alerting tersimpan pada folder `Monitoring dan Logging/6.bukti alerting Grafana/`:

1. `01_alert_rules_list.png` — Bukti daftar 3 Alert Rules pada Grafana UI (`http://127.0.0.1:3000/alerting/list`)
2. `02_alert_1_high_request_error_rate.png` — Detail rincian rule **ALERT 1 — HIGH REQUEST ERROR RATE**
3. `03_alert_2_high_request_latency.png` — Detail rincian rule **ALERT 2 — HIGH REQUEST LATENCY**
4. `04_alert_3_high_inference_exception.png` — Detail rincian rule **ALERT 3 — HIGH INFERENCE ERROR / EXCEPTION**
5. `05_notification_policies.png` — Bukti konfigurasi Notification Policy di Grafana
6. `06_contact_points_webhook.png` — Bukti konfigurasi Contact Point Webhook Receiver (`http://127.0.0.1:8001/webhook`)
