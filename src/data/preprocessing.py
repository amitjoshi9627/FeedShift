import pandas as pd

from src.config.paths import PROCESSED_DATA
from src.data.constants import DataCols
from src.utils.tools import save_csv


class FeedShiftPreprocessor:
    def __init__(self) -> None:
        self.processed_data = pd.DataFrame()
        self.processed_text = list()
        self.processed_data_path = PROCESSED_DATA

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = raw_data.copy(deep=True)
        self.processed_data = self._remove_duplicates(data)
        save_csv(self.processed_data, self.processed_data_path)
        return self.processed_data

    @staticmethod
    def _remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        return data.drop_duplicates(subset=[DataCols.TEXT])
