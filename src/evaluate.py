from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def evaluate_model(
    model,
    vectorizer,
    X_test,
    y_test,
    model_name
):
    """Evaluate a classification model."""

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():

    print("Loading data...")

    df = prepare_data()

    # -------------------------
    # Queue Evaluation
    # -------------------------

    X_queue = df["text"]
    y_queue = df["queue"]

    _, X_queue_test, _, y_queue_test = train_test_split(
        X_queue,
        y_queue,
        test_size=0.20,
        random_state=42,
        stratify=y_queue
    )

    queue_vectorizer = joblib.load(
        MODELS_DIR / "queue_tfidf_vectorizer.pkl"
    )

    queue_model = joblib.load(
        MODELS_DIR / "queue_classifier.pkl"
    )

    evaluate_model(
        queue_model,
        queue_vectorizer,
        X_queue_test,
        y_queue_test,
        "Queue Classification - Linear SVM"
    )

    # -------------------------
    # Priority Evaluation
    # -------------------------

    X_priority = df["text"]
    y_priority = df["priority"]

    _, X_priority_test, _, y_priority_test = train_test_split(
        X_priority,
        y_priority,
        test_size=0.20,
        random_state=42,
        stratify=y_priority
    )

    priority_vectorizer = joblib.load(
        MODELS_DIR / "priority_tfidf_vectorizer.pkl"
    )

    priority_model = joblib.load(
        MODELS_DIR / "priority_classifier.pkl"
    )

    evaluate_model(
        priority_model,
        priority_vectorizer,
        X_priority_test,
        y_priority_test,
        "Priority Classification - Linear SVM"
    )


if __name__ == "__main__":
    main()