from typing import Optional

import pandas as pd

from src.data.data_loader import FeedShiftDataLoader
from src.ranking.constants import DEFAULT_TOXICITY_STRICTNESS
from src.ranking.ranker import FeedShiftTextRanker


class FeedShiftEngine:
    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        self.data_loader = FeedShiftDataLoader(data)

    def run(
        self,
        interests: list[str],
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
    ) -> pd.DataFrame:
        text_ranker = FeedShiftTextRanker(self.data_loader.processed_data)
        return text_ranker.rerank(interests, toxicity_strictness)


if __name__ == "__main__":
    engine = FeedShiftEngine()
    print(engine.run(interests=[]).head())
