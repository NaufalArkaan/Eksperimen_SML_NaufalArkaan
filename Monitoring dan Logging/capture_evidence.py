"""
Automated Evidence Capture Script for Prometheus & Grafana Monitoring
Project: Dicoding Final Project - Sistem Machine Learning MLOps
Author: NaufalArkaan
File: Monitoring dan Logging/capture_evidence.py
"""

import os
import time
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMETHEUS_DIR = os.path.join(BASE_DIR, "Monitoring dan Logging", "4.bukti monitoring Prometheus")
GRAFANA_DIR = os.path.join(BASE_DIR, "Monitoring dan Logging", "5.bukti monitoring Grafana")
ALERTING_DIR = os.path.join(BASE_DIR, "Monitoring dan Logging", "6.bukti alerting Grafana")

os.makedirs(PROMETHEUS_DIR, exist_ok=True)
os.makedirs(GRAFANA_DIR, exist_ok=True)
os.makedirs(ALERTING_DIR, exist_ok=True)

def run():
    print("=" * 60)
    print(" AUTOMATED EVIDENCE CAPTURE FOR PROMETHEUS & GRAFANA")
    print("=" * 60)
    
    with sync_playwright() as p:
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        if os.path.exists(edge_path):
            browser = p.chromium.launch(executable_path=edge_path, headless=True)
        else:
            browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # ---------------------------------------------------------
        # 1. GRAFANA LOGIN & DASHBOARD SCREENSHOTS
        # ---------------------------------------------------------
        print("\n[Grafana] Logging into Grafana (http://127.0.0.1:3000)...")
        page.goto("http://127.0.0.1:3000/login")
        page.wait_for_selector("input[name='user']")
        page.fill("input[name='user']", "admin")
        page.fill("input[name='password']", "admin")
        page.click("button[type='submit']")
        
        # Handle "Skip" password change if prompted
        time.sleep(2)
        if page.query_selector("button:has-text('Skip')"):
            page.click("button:has-text('Skip')")
        
        time.sleep(3)
        print("[Grafana] Logged in successfully!")

        # 1.1 Datasource Screenshot
        print("[Grafana] Capturing Datasource status...")
        page.goto("http://127.0.0.1:3000/connections/datasources/edit/PBFA97CFB590B2093")
        time.sleep(3)
        page.screenshot(path=os.path.join(GRAFANA_DIR, "04_grafana_datasource_prometheus.png"))
        print(f" -> Saved 04_grafana_datasource_prometheus.png")

        # 1.2 Dashboard NaufalArkaan Full & Panel Views
        print("[Grafana] Navigating to Dashboard 'NaufalArkaan'...")
        dash_url = "http://127.0.0.1:3000/d/naufalarkaan_ml_monitoring/naufalarkaan?orgId=1&refresh=5s&from=now-15m&to=now"
        page.goto(dash_url)
        time.sleep(5) # Allow graphs and stats to load completely

        # Screenshot 1: Full Dashboard
        page.set_viewport_size({"width": 1920, "height": 1400})
        time.sleep(2)
        page.screenshot(path=os.path.join(GRAFANA_DIR, "01_grafana_dashboard_NaufalArkaan_full.png"), full_page=True)
        print(f" -> Saved 01_grafana_dashboard_NaufalArkaan_full.png")

        # Screenshot 2: Time Series & Graphs View
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.evaluate("window.scrollTo(0, 400)")
        time.sleep(2)
        page.screenshot(path=os.path.join(GRAFANA_DIR, "02_grafana_dashboard_NaufalArkaan_view.png"))
        print(f" -> Saved 02_grafana_dashboard_NaufalArkaan_view.png")

        # Screenshot 3: Stat & Gauge Metrics View
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        page.screenshot(path=os.path.join(GRAFANA_DIR, "03_grafana_dashboard_NaufalArkaan_metrics.png"))
        print(f" -> Saved 03_grafana_dashboard_NaufalArkaan_metrics.png")


        # ---------------------------------------------------------
        # 2. GRAFANA ALERTING SCREENSHOTS
        # ---------------------------------------------------------
        print("\n[Alerting] Capturing Grafana Alerting pages...")
        
        # Alert Rules List
        page.goto("http://127.0.0.1:3000/alerting/list")
        time.sleep(3)
        page.screenshot(path=os.path.join(ALERTING_DIR, "01_alert_rules_list.png"))
        print(f" -> Saved 01_alert_rules_list.png")

        # Alert 1 Detail
        page.goto("http://127.0.0.1:3000/alerting/alert_high_request_error_rate/edit")
        time.sleep(3)
        page.screenshot(path=os.path.join(ALERTING_DIR, "02_alert_1_high_request_error_rate.png"))
        print(f" -> Saved 02_alert_1_high_request_error_rate.png")

        # Alert 2 Detail
        page.goto("http://127.0.0.1:3000/alerting/alert_high_request_latency/edit")
        time.sleep(3)
        page.screenshot(path=os.path.join(ALERTING_DIR, "03_alert_2_high_request_latency.png"))
        print(f" -> Saved 03_alert_2_high_request_latency.png")

        # Alert 3 Detail
        page.goto("http://127.0.0.1:3000/alerting/alert_high_inference_exception/edit")
        time.sleep(3)
        page.screenshot(path=os.path.join(ALERTING_DIR, "04_alert_3_high_inference_exception.png"))
        print(f" -> Saved 04_alert_3_high_inference_exception.png")

        # Notification Policies
        page.goto("http://127.0.0.1:3000/alerting/routes")
        time.sleep(3)
        page.screenshot(path=os.path.join(ALERTING_DIR, "05_notification_policies.png"))
        print(f" -> Saved 05_notification_policies.png")

        # Contact Points
        page.goto("http://127.0.0.1:3000/alerting/notifications")
        time.sleep(3)
        page.screenshot(path=os.path.join(ALERTING_DIR, "06_contact_points_webhook.png"))
        print(f" -> Saved 06_contact_points_webhook.png")


        # ---------------------------------------------------------
        # 3. PROMETHEUS TARGETS & METRICS SCREENSHOTS
        # ---------------------------------------------------------
        print("\n[Prometheus] Capturing Prometheus Web UI (http://127.0.0.1:9090)...")
        
        # 3.1 Targets UP
        page.goto("http://127.0.0.1:9090/targets")
        time.sleep(3)
        page.screenshot(path=os.path.join(PROMETHEUS_DIR, "01_prometheus_targets.png"))
        print(f" -> Saved 01_prometheus_targets.png")

        # 3.2 Metrics Queries (11 Metrics)
        prom_queries = [
            ("ml_request_total", "02_ml_request_total.png"),
            ("ml_prediction_total", "03_ml_prediction_total.png"),
            ("ml_prediction_yes_total", "04_ml_prediction_yes_total.png"),
            ("ml_prediction_no_total", "05_ml_prediction_no_total.png"),
            ("ml_request_error_total", "06_ml_request_error_total.png"),
            ("ml_request_latency_seconds_count", "07_ml_request_latency_seconds.png"),
            ("ml_prediction_latency_seconds_count", "08_ml_prediction_latency_seconds.png"),
            ("ml_http_requests_total", "09_ml_http_requests_total.png"),
            ("ml_http_errors_total", "10_ml_http_errors_total.png"),
            ("ml_inference_total", "11_ml_inference_total.png"),
            ("ml_exception_total", "12_ml_exception_total.png")
        ]

        for expr, filename in prom_queries:
            query_url = f"http://127.0.0.1:9090/graph?g0.expr={expr}&g0.tab=1"
            page.goto(query_url)
            time.sleep(2)
            # Execute query if Execute button present
            if page.query_selector("button:has-text('Execute')"):
                page.click("button:has-text('Execute')")
                time.sleep(1)
            page.screenshot(path=os.path.join(PROMETHEUS_DIR, filename))
            print(f" -> Saved {filename} for expression '{expr}'")

        browser.close()
        print("\n" + "=" * 60)
        print(" ALL MONITORING & ALERTING SCREENSHOTS SUCCESSFULLY CAPTURED!")
        print("=" * 60)

if __name__ == "__main__":
    run()
