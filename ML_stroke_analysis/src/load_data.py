"""Load dataset dari CSV."""
import pandas as pd
from src.config import RAW_DATA


def load_raw_data(path=RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_data] Shape: {df.shape}")
    print(f"[load_data] Stroke distribution:\n{df['stroke'].value_counts()}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())