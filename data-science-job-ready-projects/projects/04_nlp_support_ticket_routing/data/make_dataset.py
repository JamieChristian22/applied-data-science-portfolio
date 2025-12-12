import random
import pandas as pd
from pathlib import Path

CATEGORIES = ["billing", "technical", "account_access", "shipping", "refunds"]

TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription this month.",
        "My invoice shows an extra fee I don't recognize.",
        "Can you explain why my monthly price increased?",
        "Payment failed but my card works everywhere else.",
        "Do you offer annual billing and can you switch me?"
    ],
    "technical": [
        "The app crashes when I upload a file.",
        "I'm getting an error code when I try to log in from Chrome.",
        "Notifications stopped working after the last update.",
        "The dashboard is slow and pages time out.",
        "API requests return 500 errors intermittently."
    ],
    "account_access": [
        "I forgot my password and the reset link isn't arriving.",
        "My account is locked after too many attempts.",
        "Please change the email on my account.",
        "I can't access my workspace after switching phones.",
        "Two-factor authentication isn't working for me."
    ],
    "shipping": [
        "My order shows delivered but I never received it.",
        "Tracking hasn't updated in three days.",
        "Can you change the shipping address on my order?",
        "The package arrived damaged and the box was open.",
        "Do you ship internationally to Canada?"
    ],
    "refunds": [
        "I'd like a refund for my last purchase.",
        "Can you cancel my order and issue a refund?",
        "My return was delivered—when will I get my money back?",
        "I was promised a refund but haven't seen it yet.",
        "Refund to my card was declined—what can I do?"
    ]
}

VARIANTS = ["ASAP", "please", "today", "this week", "I'm frustrated", "thanks", "urgent", "help me"]

def augment(text: str) -> str:
    addons = random.sample(VARIANTS, k=random.randint(0,2))
    if addons:
        text = text + " " + " ".join(addons)
    if random.random() < 0.2:
        text = text.replace("can't", "cannot")
    if random.random() < 0.2:
        text = text.replace("I'd", "I would")
    return text

def main(seed: int = 11, n_per_class: int = 900, out_path: str = "data/tickets.csv"):
    random.seed(seed)
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_class):
            rows.append([augment(random.choice(TEMPLATES[cat])), cat])
    df = pd.DataFrame(rows, columns=["text", "category"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")

if __name__ == "__main__":
    main()
