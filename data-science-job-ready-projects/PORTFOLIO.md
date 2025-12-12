# Data Science Portfolio (Job-Ready)

Four complete, end-to-end projects designed to look like real work:
- clear business framing + metrics
- reproducible dataset generation
- modeling + evaluation + decision thresholds
- explainability / drivers
- production-style inference scripts
- automated smoke tests + CI

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Projects
1) **Customer Churn Prediction (Subscription SaaS)**  
2) **Retail Demand Forecasting (Store × SKU)**  
3) **Payment Fraud Detection (E-commerce)**  
4) **NLP Support Ticket Auto-Routing**

## One-command reproducibility
```bash
python run_all.py
```

## Tests / CI
- Local: `pytest -q`
- GitHub Actions: `.github/workflows/ci.yml`
