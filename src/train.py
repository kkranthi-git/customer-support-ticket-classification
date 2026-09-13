from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocessing import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def train_queue_model(df):
    """Train the Queue classification model."""

    X = df["text"]
    y = df["queue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LinearSVC(
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


def train_priority_model(df):
    """Train the Priority classification model."""

    X = df["text"]
    y = df["priority"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LinearSVC(
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


def main():

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")

    df = prepare_data()

    print(f"Training data shape: {df.shape}")

    print("\nTraining Queue model...")

    queue_vectorizer, queue_model = train_queue_model(df)

    joblib.dump(
        queue_vectorizer,
        MODELS_DIR / "queue_tfidf_vectorizer.pkl"
    )

    joblib.dump(
        queue_model,
        MODELS_DIR / "queue_classifier.pkl"
    )

    print("Queue model saved.")

    print("\nTraining Priority model...")

    priority_vectorizer, priority_model = train_priority_model(df)

    joblib.dump(
        priority_vectorizer,
        MODELS_DIR / "priority_tfidf_vectorizer.pkl"
    )

    joblib.dump(
        priority_model,
        MODELS_DIR / "priority_classifier.pkl"
    )

    print("Priority model saved.")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()