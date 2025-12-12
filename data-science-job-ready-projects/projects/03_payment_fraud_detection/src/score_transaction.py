import pandas as pd
import joblib

def load(path="models/fraud_model.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], float(bundle["threshold"])

if __name__ == "__main__":
    model, thr = load()
    sample = pd.DataFrame([{
        "amount": 740.50,
        "hour": 2,
        "device": "mobile",
        "payment_method": "card",
        "country_risk": "high",
        "account_age_days": 12,
        "prior_chargebacks": 1,
        "ip_distance_km": 9500,
        "velocity_1h": 3,
        "failed_logins_24h": 4
    }])
    p = model.predict_proba(sample)[0,1]
    flag = int(p >= thr)
    print({"fraud_probability": float(p), "flag_for_review": flag, "threshold": thr})
