import numpy as np
import pandas as pd
from pathlib import Path

def sigmoid(x):
    return 1/(1+np.exp(-x))

def main(seed: int = 21, n: int = 12000, out_path: str = "data/transactions.csv"):
    rng = np.random.default_rng(seed)

    amount = np.clip(rng.lognormal(mean=3.2, sigma=0.7, size=n), 2, 3000)
    hour = rng.integers(0, 24, size=n)
    device = rng.choice(["mobile","desktop","tablet"], size=n, p=[0.62, 0.30, 0.08])
    payment = rng.choice(["card","paypal","bank_transfer","crypto"], size=n, p=[0.76, 0.16, 0.07, 0.01])
    country_risk = rng.choice(["low","medium","high"], size=n, p=[0.80, 0.15, 0.05])

    account_age_days = np.clip(rng.lognormal(mean=4.0, sigma=1.0, size=n), 1, 4000)
    prior_chargebacks = rng.poisson(lam=np.clip(0.06 + (payment=="crypto")*0.12 + (country_risk=="high")*0.12, 0.02, 0.5), size=n)
    ip_distance_km = np.clip(rng.lognormal(mean=5.0, sigma=1.1, size=n), 0, 20000)
    velocity_1h = rng.poisson(lam=np.clip(0.3 + (device=="mobile")*0.1, 0.05, 1.5), size=n)
    failed_logins_24h = rng.poisson(lam=np.clip(0.2 + (country_risk=="high")*0.8, 0.05, 2.0), size=n)

    risk = (
        0.9*(country_risk=="high").astype(int)
        + 0.35*(country_risk=="medium").astype(int)
        + 0.55*(payment=="crypto").astype(int)
        + 0.22*(payment=="bank_transfer").astype(int)
        + 0.20*(device=="mobile").astype(int)
        + 0.0009*(ip_distance_km-500)
        + 0.45*prior_chargebacks
        + 0.18*velocity_1h
        + 0.22*failed_logins_24h
        + 0.001*(amount-80)
        - 0.00035*(account_age_days-30)
        + 0.25*((hour>=0) & (hour<=5)).astype(int)
    )

    fraud_prob = sigmoid(-5.3 + risk)
    fraud = rng.binomial(1, p=np.clip(fraud_prob, 0.001, 0.35), size=n)

    df = pd.DataFrame({
        "tx_id": [f"T{900000+i}" for i in range(n)],
        "amount": amount.round(2),
        "hour": hour,
        "device": device,
        "payment_method": payment,
        "country_risk": country_risk,
        "account_age_days": account_age_days.round(1),
        "prior_chargebacks": prior_chargebacks,
        "ip_distance_km": ip_distance_km.round(1),
        "velocity_1h": velocity_1h,
        "failed_logins_24h": failed_logins_24h,
        "fraud": fraud
    })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path} (fraud rate ~{df['fraud'].mean():.2%})")

if __name__ == "__main__":
    main()
