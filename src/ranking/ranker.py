from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.data.constants import DataCols
from src.models.embedder import FeedShiftEmbeddor
from src.ranking.constants import RankingWeight


class FeedShiftTextRanker:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.dates = data[DataCols.TIMESTAMP]
        self.embeddor = FeedShiftEmbeddor()

    def rerank(self) -> pd.DataFrame:
        embeddings = self._get_embeddings(self.data[DataCols.TEXT].tolist())
        self.data[DataCols.SCORES] = self._get_score(self.dates, embeddings).round(1)
        self.data = self.data.sort_values(by=DataCols.SCORES, ascending=False)
        return self.data

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        return self.embeddor(texts)

    def _get_score(self, dates: Iterable[str], embeddings: np.ndarray) -> np.ndarray:
        uniqueness_score = self._get_uniqueness_score(embeddings)
        freshness_score = self._get_freshness_score(dates)
        return RankingWeight.UNIQUENESS * uniqueness_score + RankingWeight.FRESHNESS * freshness_score

    @staticmethod
    def _get_uniqueness_score(embeddings: np.ndarray) -> np.ndarray:
        # Every point represent how much a sentence is similar to all other sentences
        similarity_score = np.mean(cosine_similarity(embeddings), axis=1)
        return 1 - MinMaxScaler().fit_transform(similarity_score.reshape(-1, 1))

    @staticmethod
    def _get_freshness_score(dates: Iterable[str]) -> np.ndarray:
        dates = pd.to_datetime(dates)
        age_days = (dates.max() - dates).dt.seconds.values / (60 * 60 * 24)
        return 1 - MinMaxScaler().fit_transform(age_days.reshape(-1, 1))



