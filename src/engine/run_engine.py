from typing import Optional

import pandas as pd

from src.data.data_loader import FeedShiftDataLoader
from src.ranking.ranker import FeedShiftTextRanker


class FeedShiftEngine:
    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        self.data_loader = FeedShiftDataLoader(data)

    def run(self) -> pd.DataFrame:
        text_ranker = FeedShiftTextRanker(self.data_loader.processed_data)
        return text_ranker.rerank()


if __name__ == "__main__":
    engine = FeedShiftEngine()
    print(engine.run().head())
