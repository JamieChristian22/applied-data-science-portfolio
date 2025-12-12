"""Run all projects to generate datasets and key artifacts."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("01_customer_churn_prediction_saas", [sys.executable, "data/make_dataset.py"]),
    ("02_retail_demand_forecasting", [sys.executable, "data/make_dataset.py"]),
    ("02_retail_demand_forecasting", [sys.executable, "src/train_and_export.py"]),
    ("03_payment_fraud_detection", [sys.executable, "data/make_dataset.py"]),
    ("04_nlp_support_ticket_routing", [sys.executable, "data/make_dataset.py"]),
]

def main():
    for proj, cmd in STEPS:
        cwd = ROOT / "projects" / proj
        print(f"==> {proj}: {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=cwd)
    print("Done.")

if __name__ == "__main__":
    main()
