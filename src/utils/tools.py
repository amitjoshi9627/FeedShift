from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(data: pd.DataFrame, path: str | Path):
    data.to_csv(path, index=False)


def harmonic_mean(a: np.ndarray, b: np.ndarray):
    # Handle zero values
    if np.all(a) == 0 and np.all(b) == 0:
        return 0
    if np.all(a) == 0 or np.all(b) == 0:
        return 0
    return 2 * a * b / (a + b)
