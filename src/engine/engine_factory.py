from typing import Optional

import pandas as pd

from src.config.paths import RAW_DATA_PATH
from src.data.constants import RedditDataCols
from src.data.data_loader import FeedShiftDataLoader, RedditDataLoader, BaseDataLoader
from src.data.preprocessing import RedditPreprocessor
from src.ranking.constants import (
    DEFAULT_TOXICITY_STRICTNESS,
)
from src.ranking.ranker import TextRanker


class FeedShiftEngine:
    def __init__(self, path: str = RAW_DATA_PATH) -> None:
        self.data_loader = RedditDataLoader(path)

    def run(
        self,
        interests: list[str],
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
        diversity_strength: float = 0.9,
    ) -> pd.DataFrame:
        text_ranker = TextRanker(self.data_loader.processed_data, timestamp_col=RedditDataCols.TIMESTAMP, text_col=RedditDataCols.PROCESSED_TITLE)
        return text_ranker.rerank(interests, toxicity_strictness, diversity_strength=diversity_strength)




if __name__ == "__main__":
    engine = FeedShiftEngine()
    result = engine.run(interests=["Technology", "Science"]).head()
    print(result)
