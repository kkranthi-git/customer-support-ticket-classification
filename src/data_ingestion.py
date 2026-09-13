from pathlib import Path
import zipfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def extract_zip(zip_path: Path) -> None:
    """Extract a ZIP file into the raw data directory."""

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)


def find_csv_file() -> Path:
    """Find the expected customer support ticket dataset."""

    expected_file = (
        RAW_DATA_DIR
        / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    )

    if not expected_file.exists():
        raise FileNotFoundError(
            f"Expected dataset not found: {expected_file}"
        )

    return expected_file


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the CSV dataset into a pandas DataFrame."""

    return pd.read_csv(file_path)


def ingest_data() -> pd.DataFrame:
    """Extract and load the raw dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = list(RAW_DATA_DIR.glob("*.zip"))

    if not zip_files:
        print("No ZIP file found. Searching for CSV files...")

    for zip_file in zip_files:
        print(f"Extracting: {zip_file.name}")
        extract_zip(zip_file)

    csv_file = find_csv_file()

    print(f"Loading: {csv_file.name}")

    df = load_data(csv_file)

    print("\nData ingestion completed.")
    print(f"File: {csv_file.name}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = ingest_data()

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())