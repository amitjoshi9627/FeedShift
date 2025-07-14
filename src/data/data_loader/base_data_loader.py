import pandas as pd

from src.config.paths import RAW_DATA_PATH
from src.data.preprocessors import BasePreprocessor
from src.utils.tools import load_csv


class BaseDataLoader:
    def __init__(self, path: str = RAW_DATA_PATH) -> None:
        self.raw_data = load_csv(path)

    @property
    def processed_data(self) -> pd.DataFrame:
        return BasePreprocessor().process_data(self.raw_data)
