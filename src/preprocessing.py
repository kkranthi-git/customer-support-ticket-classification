from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_PATH = (
    RAW_DATA_DIR
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)


def load_raw_data() -> pd.DataFrame:
    """Load the raw customer support ticket dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare ticket text for NLP modeling."""

    df = df.copy()

    # Keep English tickets for the current models
    df = df[df["language"] == "en"].copy()

    # Handle missing text
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Combine subject and body
    df["text"] = (
        df["subject"] + " " + df["body"]
    ).str.strip()

    # Remove empty tickets
    df = df[df["text"] != ""].copy()

    return df


def prepare_data() -> pd.DataFrame:
    """Load and prepare the dataset."""

    df = load_raw_data()

    df = prepare_text_data(df)

    return df


if __name__ == "__main__":
    df = prepare_data()

    print("Preprocessing completed.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())