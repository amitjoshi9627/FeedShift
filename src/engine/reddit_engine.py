from abc import ABC

import pandas as pd

from src.data.constants import RedditDataCols
from src.data.ingestors.reddit_ingestor import RedditIngestor
from src.engine import BaseEngine
from src.ranking.constants import DEFAULT_TOXICITY_STRICTNESS
from src.ranking.ranker import TextRanker


class RedditEngine(BaseEngine, ABC):
    def __init__(self, subreddit: str) -> None:
        self.ingestor = RedditIngestor()
        data = self.ingestor.ingest(subreddit)
        self.text_ranker = TextRanker(data, timestamp_col=RedditDataCols.TIMESTAMP, text_col=RedditDataCols.TITLE)

    def run(
        self,
        interests: list[str] | None = None,
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
        diversity_strength: float = 0.9,
    ) -> pd.DataFrame:
        interests = interests if interests is not None else []
        return self.text_ranker.rerank(interests, toxicity_strictness, diversity_strength=diversity_strength)
