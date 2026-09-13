import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import prepare_ticket_text


MODELS_DIR = PROJECT_ROOT / "models"

# Load Queue model
queue_vectorizer = joblib.load(
    MODELS_DIR / "queue_tfidf_vectorizer.pkl"
)

queue_model = joblib.load(
    MODELS_DIR / "queue_classifier.pkl"
)


# Load Priority model
priority_vectorizer = joblib.load(
    MODELS_DIR / "priority_tfidf_vectorizer.pkl"
)

priority_model = joblib.load(
    MODELS_DIR / "priority_classifier.pkl"
)


def predict_ticket(subject: str, body: str) -> dict:
    """Predict queue and priority for a customer support ticket."""

    text = prepare_ticket_text(subject, body)

    queue_features = queue_vectorizer.transform([text])
    queue_prediction = queue_model.predict(queue_features)[0]

    priority_features = priority_vectorizer.transform([text])
    priority_prediction = priority_model.predict(priority_features)[0]

    return {
        "queue": queue_prediction,
        "priority": priority_prediction
    }


if __name__ == "__main__":

    subject = "Internet connection problem"

    body = """
    My internet connection keeps disconnecting.
    I cannot access the service and the problem has
    continued for several hours.
    """

    result = predict_ticket(subject, body)

    print("Prediction")
    print("----------")
    print("Queue   :", result["queue"])
    print("Priority:", result["priority"])