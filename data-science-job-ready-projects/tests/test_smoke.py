from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_generate_churn_data():
    proj = ROOT / "projects" / "01_customer_churn_prediction_saas"
    subprocess.check_call([sys.executable, "data/make_dataset.py"], cwd=proj)
    assert (proj / "data" / "churn_customers.csv").exists()

def test_generate_demand_and_train():
    proj = ROOT / "projects" / "02_retail_demand_forecasting"
    subprocess.check_call([sys.executable, "data/make_dataset.py"], cwd=proj)
    subprocess.check_call([sys.executable, "src/train_and_export.py"], cwd=proj)
    assert (proj / "models" / "demand_forecaster.joblib").exists()
    assert (proj / "reports" / "metrics.json").exists()

def test_generate_fraud_data():
    proj = ROOT / "projects" / "03_payment_fraud_detection"
    subprocess.check_call([sys.executable, "data/make_dataset.py"], cwd=proj)
    assert (proj / "data" / "transactions.csv").exists()

def test_generate_tickets_data():
    proj = ROOT / "projects" / "04_nlp_support_ticket_routing"
    subprocess.check_call([sys.executable, "data/make_dataset.py"], cwd=proj)
    assert (proj / "data" / "tickets.csv").exists()
