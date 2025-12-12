# Customer Churn Prediction (Subscription SaaS)

## Business problem
Predict which customers will churn so the retention team can intervene with targeted actions.

## Deliverables
- Baseline vs stronger model
- Calibrated churn probabilities + tuned decision threshold
- Driver insights (permutation importance)
- Saved model artifact for integration

## Run
```bash
python data/make_dataset.py
```
Then run `notebooks/churn_modeling.ipynb`.
