from datetime import datetime

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
        self.text_embeddings = self.embeddor.encode(self.texts)
        self.detoxifier = FeedShiftDetoxified()

    def rerank(
        self,
        interests: list[str],
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
    ) -> pd.DataFrame:
        self.data[DataCols.SCORES] = self._get_score(
            toxicity_strictness, interests
        ).round(1)
        self.data[DataCols.TIMESTAMP] = self.dates.apply(self._format_date)
        self.data = self.data.sort_values(by=DataCols.SCORES, ascending=False)
        return self.data

    @staticmethod
    def _format_date(date_string: str):
        dt = datetime.fromisoformat(date_string.replace("Z", ""))
        return dt.strftime("%b %d, %Y at %H:%M")

    def _get_score(
        self, toxicity_strictness: float, interests: list[str] | None = None
    ) -> np.ndarray:
        uniqueness_score = self._get_uniqueness_score()
        freshness_score = self._get_freshness_score()
        toxicity_score = self._get_toxicity_score()
        interests_score = self._get_interests_score(interests)

        return (
            RankingWeight.UNIQUENESS * uniqueness_score
            + RankingWeight.FRESHNESS * freshness_score
            + RankingWeight.TOXICITY * toxicity_strictness * toxicity_score
            + RankingWeight.INTERESTS * interests_score
        )

    def _get_uniqueness_score(self) -> np.ndarray:
        # Every point represent how much a sentence is similar to all other sentences
        similarity_score = np.mean(cosine_similarity(self.text_embeddings), axis=1)
        return 1 - MinMaxScaler().fit_transform(similarity_score.reshape(-1, 1))

    def _get_freshness_score(self) -> np.ndarray:
        dates = pd.to_datetime(self.dates)
        age_days = (dates.max() - dates).dt.seconds.values / (60 * 60 * 24)
        return 1 - MinMaxScaler().fit_transform(age_days.reshape(-1, 1))

    def _get_toxicity_score(self) -> np.ndarray:
        toxicity_scores = np.array(self.detoxifier.toxicity_score(self.texts))
        return 1 - toxicity_scores

    def _get_interests_score(self, interests: list[str]) -> np.ndarray:
        if not interests:
            return np.zeros(len(self.texts)).reshape(-1, 1)
        interests_embedding = self.embeddor(interests)
        interests_score = np.array(
            [
                max(
                    cosine_similarity(
                        embedding.reshape(1, -1), interest_embedding.reshape(1, -1)
                    ).flatten()
                    for interest_embedding in interests_embedding
                )
                for embedding in self.text_embeddings
            ]
        )
        return MinMaxScaler().fit_transform(interests_score.reshape(-1, 1))
