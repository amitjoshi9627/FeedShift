from pathlib import Path

import pandas as pd

from src.data.constants import RedditDataCols
from src.data.data_loader import RedditDataLoader
from src.engine import BaseEngine
from src.ranking.ranker import TextRanker


class RedditEngine(BaseEngine):
    def __init__(self, path: str | Path | None) -> None:
        super().__init__(path)

    @staticmethod
    def _init_dataloader(path: str):
        return RedditDataLoader(path)

    @staticmethod
    def _init_ranker(data: pd.DataFrame):
        return TextRanker(data, timestamp_col=RedditDataCols.TIMESTAMP, text_col=RedditDataCols.TITLE)
