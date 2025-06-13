import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.data.constants import DataCols
from src.models.detoxifier import FeedShiftDetoxified
from src.models.embedder import FeedShiftEmbeddor
from src.ranking.constants import DEFAULT_TOXICITY_STRICTNESS, RankingWeight


class FeedShiftTextRanker:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.dates = data[DataCols.TIMESTAMP]
        self.texts = self.data[DataCols.TEXT].tolist()
        self.embeddor = FeedShiftEmbeddor()
        self.detoxifier = FeedShiftDetoxified()

    def rerank(
        self, toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS
    ) -> pd.DataFrame:
        self.data[DataCols.SCORES] = self._get_score(toxicity_strictness).round(1)
        self.data = self.data.sort_values(by=DataCols.SCORES, ascending=False)
        return self.data

    def _get_score(self, toxicity_strictness: float) -> np.ndarray:
        uniqueness_score = self._get_uniqueness_score()
        freshness_score = self._get_freshness_score()
        toxicity_score = self._get_toxicity_score()
        return (
            RankingWeight.UNIQUENESS * uniqueness_score
            + RankingWeight.FRESHNESS * freshness_score
            + RankingWeight.TOXICITY * toxicity_strictness * toxicity_score
        )

    def _get_uniqueness_score(self) -> np.ndarray:
        embeddings = self.embeddor(self.texts)
        # Every point represent how much a sentence is similar to all other sentences
        similarity_score = np.mean(cosine_similarity(embeddings), axis=1)
        return 1 - MinMaxScaler().fit_transform(similarity_score.reshape(-1, 1))

    def _get_freshness_score(self) -> np.ndarray:
        dates = pd.to_datetime(self.dates)
        age_days = (dates.max() - dates).dt.seconds.values / (60 * 60 * 24)
        return 1 - MinMaxScaler().fit_transform(age_days.reshape(-1, 1))

    def _get_toxicity_score(self) -> np.ndarray:
        toxicity_scores = np.array(self.detoxifier.toxicity_score(self.texts))
        return 1 - toxicity_scores
