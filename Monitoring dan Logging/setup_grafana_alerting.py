"""
Setup Grafana Alerting Rules & Contact Points for Milestone 13
Project: Dicoding Final Project - MLOps
Author: NaufalArkaan
File: Monitoring dan Logging/setup_grafana_alerting.py
"""

import requests
import json
import time

GRAFANA_URL = "http://127.0.0.1:3000"
AUTH = ("admin", "admin")
DS_UID = "PBFA97CFB590B2093"

print("[Grafana Alerting Setup] Connecting to Grafana...")

# 1. Get or Create Folder 'Alerting'
res_f = requests.post(f"{GRAFANA_URL}/api/folders", json={"title": "Alerting"}, auth=AUTH)
if res_f.status_code in [200, 409]:
    res_folders = requests.get(f"{GRAFANA_URL}/api/folders", auth=AUTH).json()
    folder_uid = res_folders[0]["uid"] if res_folders else "general"
else:
    folder_uid = "general"

print(f"Using Folder UID: {folder_uid}")

# 2. Create Webhook Contact Point
contact_point_payload = {
    "name": "MLOps Webhook Receiver",
    "type": "webhook",
    "settings": {
        "url": "http://127.0.0.1:8001/webhook",
        "httpMethod": "POST"
    }
}

res_cp = requests.post(f"{GRAFANA_URL}/api/v1/provisioning/contact-points", json=contact_point_payload, auth=AUTH)
print("Contact Point Status:", res_cp.status_code)

# 3. Set Default Notification Policy
policy_payload = {
    "receiver": "MLOps Webhook Receiver",
    "group_by": ["alertname"],
    "group_wait": "1s",
    "group_interval": "2s",
    "repeat_interval": "1m"
}
res_pol = requests.put(f"{GRAFANA_URL}/api/v1/provisioning/policies", json=policy_payload, auth=AUTH)
print("Notification Policy Status:", res_pol.status_code)

# 4. Create Alert Rules Function (Instant Firing for: 0s)
def create_alert_rule(uid, title, promql_expr, condition_expr, summary, description):
    rule_payload = {
        "uid": uid,
        "title": title,
        "folderUID": folder_uid,
        "ruleGroup": "Model Serving Alerts",
        "condition": "C",
        "data": [
            {
                "refId": "A",
                "queryType": "",
                "relativeTimeRange": {"from": 60, "to": 0},
                "datasourceUid": DS_UID,
                "model": {
                    "editorMode": "code",
                    "expr": promql_expr,
                    "instant": True,
                    "intervalMs": 1000,
                    "maxDataPoints": 43200,
                    "refId": "A"
                }
            },
            {
                "refId": "B",
                "queryType": "",
                "relativeTimeRange": {"from": 60, "to": 0},
                "datasourceUid": "-100",
                "model": {
                    "conditions": [
                        {
                            "evaluator": {"params": [0], "type": "gt"},
                            "operator": {"type": "and"},
                            "query": {"params": ["A"]},
                            "reducer": {"params": [], "type": "last"},
                            "type": "query"
                        }
                    ],
                    "datasource": {"type": "__expr__", "uid": "-100"},
                    "expression": "A",
                    "reducer": "last",
                    "refId": "B",
                    "type": "reduce"
                }
            },
            {
                "refId": "C",
                "queryType": "",
                "relativeTimeRange": {"from": 60, "to": 0},
                "datasourceUid": "-100",
                "model": {
                    "conditions": [
                        {
                            "evaluator": {"params": [0], "type": "gt"},
                            "operator": {"type": "and"},
                            "query": {"params": ["B"]},
                            "reducer": {"params": [], "type": "last"},
                            "type": "query"
                        }
                    ],
                    "datasource": {"type": "__expr__", "uid": "-100"},
                    "expression": condition_expr,
                    "refId": "C",
                    "type": "math"
                }
            }
        ],
        "noDataState": "OK",
        "execErrState": "OK",
        "for": "0s",
        "annotations": {
            "summary": summary,
            "description": description
        },
        "labels": {
            "severity": "critical",
            "service": "bank-marketing-ml"
        }
    }
    
    res = requests.put(f"{GRAFANA_URL}/api/v1/provisioning/alert-rules/{uid}", json=rule_payload, auth=AUTH)
    if res.status_code not in [200, 201]:
        res = requests.post(f"{GRAFANA_URL}/api/v1/provisioning/alert-rules", json=rule_payload, auth=AUTH)
    print(f"Alert Rule '{title}' Update Status:", res.status_code)


# ---------------------------------------------------------
# CREATE THE 3 REQUIRED ALERTS
# ---------------------------------------------------------

# ALERT 1 — HIGH REQUEST ERROR RATE
create_alert_rule(
    uid="alert_high_request_error_rate",
    title="ALERT 1 - HIGH REQUEST ERROR RATE",
    promql_expr="sum(ml_request_error_total)",
    condition_expr="$B > 0",
    summary="High Request Error Rate Detected",
    description="Jumlah error request ML serving melebihi threshold (> 0 errors)"
)

# ALERT 2 — HIGH REQUEST LATENCY
create_alert_rule(
    uid="alert_high_request_latency",
    title="ALERT 2 - HIGH REQUEST LATENCY",
    promql_expr="ml_request_latency_seconds_sum / ml_request_latency_seconds_count",
    condition_expr="$B > 0.001",
    summary="High Request Latency Detected",
    description="Latensi rata-rata request ML serving melebihi threshold (> 0.001s / 1ms)"
)

# ALERT 3 — HIGH INFERENCE ERROR / EXCEPTION
create_alert_rule(
    uid="alert_high_inference_exception",
    title="ALERT 3 - HIGH INFERENCE ERROR / EXCEPTION",
    promql_expr="sum(ml_exception_total)",
    condition_expr="$B > 0",
    summary="High Inference Exception Detected",
    description="Jumlah internal exception pada model inference melebihi threshold (> 0 exceptions)"
)

print("[Grafana Alerting Setup] All 3 alert rules updated with instant for:0s!")
