# 📊 Applied Data Science Portfolio

**Job-ready, end-to-end data science projects** demonstrating real-world problem solving across classification, forecasting, fraud detection, and NLP.

This portfolio emphasizes:
- Business-driven problem framing
- Reproducible data pipelines
- Model evaluation & decision thresholds
- Production-style artifacts and scripts
- CI-ready project structure

---

## 🚀 Portfolio Overview

This repository contains **four applied data science projects**, each built end-to-end with:
- Synthetic but realistic datasets
- Clear business objectives
- Feature engineering & modeling
- Evaluation tied to real decisions
- Saved model artifacts for reuse
- Clean, production-style structure

➡️ **Start here:** `PORTFOLIO.md` for a recruiter-friendly walkthrough of each project.

---

## 📁 Projects

| Project | Business Problem | Techniques |
|------|------------------|------------|
| **Customer Churn Prediction (SaaS)** | Identify at-risk customers and guide retention actions | Classification, probability calibration, threshold tuning, explainability |
| **Retail Demand Forecasting** | Forecast weekly demand to reduce stockouts and over-inventory | Time-based feature engineering, ML forecasting, model export |
| **Payment Fraud Detection** | Detect fraudulent transactions while minimizing business cost | Imbalanced learning, PR-AUC, cost-optimized thresholds |
| **NLP Support Ticket Auto-Routing** | Automatically route support tickets to the correct queue | TF-IDF, text classification, confusion matrix analysis |

All projects live under the `projects/` directory.

---

## ⚙️ Quick Start (Reproducible)

```bash
pip install -r requirements.txt
python run_all.py
```

This will:
- Generate datasets
- Train key models
- Export model artifacts
- Save metrics and reports

---

## 🧪 Testing & CI

### Local tests
```bash
pytest -q
```

### Continuous Integration
GitHub Actions automatically validates:
- Dataset generation
- Model training scripts
- Basic project integrity

CI configuration lives in `.github/workflows/`.

---

## 🛠 Tech Stack

- **Language:** Python  
- **Data:** pandas, NumPy  
- **Modeling:** scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)  
- **NLP:** TF-IDF, linear classifiers  
- **Workflow:** Pipelines, joblib artifacts, CI testing  

---

## 👤 Author

**Jamie Christian II**  
Applied Data Science | Analytics | Business-Driven Modeling  

- GitHub: https://github.com/JamieChristian22  
- LinkedIn: https://www.linkedin.com/in/jamiechristiananalytics/

---

## 📌 Final Notes

This portfolio is designed to reflect **real professional data science work**, not academic exercises.

Each project prioritizes:
- Decision-making impact
- Clarity of reasoning
- Practical deployment considerations
