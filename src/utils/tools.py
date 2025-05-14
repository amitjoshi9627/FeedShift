from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(data: pd.DataFrame, path: str | Path) -> None:
    data.to_csv(path, index=False)
