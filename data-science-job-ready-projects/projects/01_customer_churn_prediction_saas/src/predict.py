import pandas as pd
import joblib

def load_model(path="models/churn_model.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], float(bundle["threshold"]), bundle["features"]

def score(df: pd.DataFrame, model, threshold: float):
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

if __name__ == "__main__":
    model, thr, features = load_model()
    sample = pd.DataFrame([{
        "tenure_months": 3,
        "monthly_charges": 105.0,
        "contract": "month-to-month",
        "payment_method": "paypal",
        "support_tickets_90d": 4,
        "onboarding_completed": 0,
        "product_adoption_score": 38.0,
        "usage_hours_week": 1.2,
        "late_payments_6m": 2
    }])[features]
    p, y = score(sample, model, thr)
    print({"churn_probability": float(p[0]), "churn_flag": int(y[0]), "threshold": float(thr)})
