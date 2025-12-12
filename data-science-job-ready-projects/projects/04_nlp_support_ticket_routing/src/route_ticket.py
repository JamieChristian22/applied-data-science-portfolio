import joblib

def load(path="models/ticket_router.joblib"):
    return joblib.load(path)

def route(text: str, model):
    proba = model.predict_proba([text])[0]
    labels = model.classes_
    top = proba.argsort()[::-1][:3]
    return [(labels[i], float(proba[i])) for i in top]

if __name__ == "__main__":
    model = load()
    example = "I was charged twice and my invoice looks wrong, please fix ASAP"
    print(route(example, model))
