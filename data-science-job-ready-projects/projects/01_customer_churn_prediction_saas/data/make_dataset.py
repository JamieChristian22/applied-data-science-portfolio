import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def main(seed: int = 42, n: int = 6000, out_path: str = "data/churn_customers.csv"):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 49, size=n)
    monthly_charges = np.clip(rng.normal(65, 20, size=n), 15, 180)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], size=n, p=[0.58, 0.27, 0.15])
    payment_method = rng.choice(["credit_card", "debit_card", "paypal", "bank_transfer"], size=n, p=[0.38, 0.22, 0.25, 0.15])

    support_tickets_90d = rng.poisson(lam=np.clip(1.0 + (monthly_charges-50)/60, 0.2, 3.0), size=n)
    onboarding_completed = rng.choice([0, 1], size=n, p=[0.22, 0.78])
    product_adoption_score = np.clip(rng.normal(62, 18, size=n) + 8*onboarding_completed - 0.3*support_tickets_90d, 0, 100)

    usage_hours_week = np.clip(rng.normal(5.5, 2.2, size=n) + 0.04*(product_adoption_score-60) - 0.05*(support_tickets_90d), 0, 30)
    late_payments_6m = rng.binomial(n=3, p=np.clip(0.08 + (contract=="month-to-month")*0.08 + (payment_method=="bank_transfer")*0.05, 0.02, 0.35), size=n)

    risk = (
        1.10*(contract=="month-to-month").astype(int)
        - 0.60*(contract=="two-year").astype(int)
        + 0.03*(monthly_charges-70)
        - 0.04*(tenure_months-12)
        + 0.20*support_tickets_90d
        - 0.035*(product_adoption_score-60)
        - 0.06*(usage_hours_week-5)
        + 0.55*late_payments_6m
        + 0.55*(1-onboarding_completed)
    )
    churn_prob = sigmoid(-2.2 + risk)
    churned = rng.binomial(n=1, p=np.clip(churn_prob, 0.01, 0.95), size=n)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "contract": contract,
        "payment_method": payment_method,
        "support_tickets_90d": support_tickets_90d,
        "onboarding_completed": onboarding_completed,
        "product_adoption_score": product_adoption_score.round(1),
        "usage_hours_week": usage_hours_week.round(2),
        "late_payments_6m": late_payments_6m,
        "churned": churned
    })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")

if __name__ == "__main__":
    from pathlib import Path
    main()
