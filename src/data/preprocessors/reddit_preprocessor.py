import pandas as pd

from src.data.constants import RedditDataCols
from src.data.preprocessors import BasePreprocessor


class RedditPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # dropping duplicates
        processed_data = raw_data.copy(deep=True).drop_duplicates()

        # removing extra characters
        processed_data[RedditDataCols.PROCESSED_TEXT] = processed_data[RedditDataCols.TEXT].apply(self.clean_text)
        processed_data[RedditDataCols.PROCESSED_TITLE] = processed_data[RedditDataCols.TITLE].apply(self.clean_text)

        return self.post_processing(processed_data, RedditDataCols.TIMESTAMP)

    def process_text(self, text: str) -> str:
        return self.clean_text(text)
