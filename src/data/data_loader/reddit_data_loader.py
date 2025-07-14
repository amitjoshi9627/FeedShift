import pandas as pd

from src.data.data_loader import BaseDataLoader
from src.data.preprocessors import RedditPreprocessor


class RedditDataLoader(BaseDataLoader):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @property
    def processed_data(self) -> pd.DataFrame:
        return RedditPreprocessor().process_data(self.raw_data)
