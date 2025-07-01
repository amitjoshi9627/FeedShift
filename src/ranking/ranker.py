from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.data.constants import DataCols
from src.models.detoxifier import FeedShiftDetoxified
from src.models.embedder import FeedShiftEmbeddor
from src.ranking.constants import (
    DEFAULT_TOXICITY_STRICTNESS,
    RankingWeight,
    SIMILAR_POSTS_ALPHA,
    DEFAULT_DIVERSITY_STRENGTH,
)


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
        diversity_strength: float = DEFAULT_DIVERSITY_STRENGTH,
    ) -> pd.DataFrame:
        self.data[DataCols.SCORES] = self._get_score(interests, toxicity_strictness, diversity_strength).round(1)
        self.data[DataCols.TIMESTAMP] = self.dates.apply(self._format_date)
        self.data = self.data.sort_values(by=DataCols.SCORES, ascending=False)
        return self.data

    @staticmethod
    def _format_date(date_string: str):
        dt = datetime.fromisoformat(date_string.replace("Z", ""))
        return dt.strftime("%b %d, %Y at %H:%M")

    def _get_score(
        self,
        interests: list[str],
        toxicity_strictness: float,
        diversity_strength: float,
    ) -> np.ndarray:
        uniqueness_score = self._get_uniqueness_score()
        freshness_score = self._get_freshness_score()
        toxicity_score = self._get_toxicity_score()
        interests_score = self._get_interests_score(interests)
        diversity_score = self._get_diversity_score(interests_score, diversity_strength=diversity_strength)
        return (
            RankingWeight.UNIQUENESS * uniqueness_score
            + RankingWeight.FRESHNESS * freshness_score
            + RankingWeight.TOXICITY * toxicity_strictness * toxicity_score
            + RankingWeight.INTERESTS * interests_score
            + RankingWeight.DIVERSITY * diversity_score
        )

    def _get_uniqueness_score(self) -> np.ndarray:
        # Every point represent how much a sentence is similar to all other sentences
        similarity_matrix = cosine_similarity(self.text_embeddings)

        # Ignoring self similarity
        np.fill_diagonal(similarity_matrix, 0)

        global_similarity_score = np.mean(similarity_matrix, axis=1)
        local_similarity_score = np.max(similarity_matrix, axis=1)

        similarity_score = (
            2 * (global_similarity_score * local_similarity_score) / (global_similarity_score + local_similarity_score)
        )
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
                np.mean(
                    [
                        cosine_similarity(embedding.reshape(1, -1), interest_embedding.reshape(1, -1)).flatten()
                        for interest_embedding in interests_embedding
                    ]
                )
                for embedding in self.text_embeddings
            ]
        )
        return MinMaxScaler().fit_transform(interests_score.reshape(-1, 1))

    @staticmethod
    def _get_diversity_score(interests_score: np.ndarray, diversity_strength: float) -> np.ndarray:
        scores = interests_score.copy()

        # Similar Posts
        mask = (scores >= 0.8) & (scores <= 1.0)
        scores[mask] -= SIMILAR_POSTS_ALPHA * diversity_strength * scores[mask]

        # Near Posts
        diversity_strength_near = diversity_strength if diversity_strength <= 0.5 else (1 - diversity_strength / 2)
        mask = (scores >= 0.6) & (scores < 0.8)
        scores[mask] += diversity_strength_near * (1 - scores[mask])

        # Diverse Posts
        mask = (scores >= 0.4) & (scores < 0.6)
        scores[mask] += diversity_strength * (1 - scores[mask])
        return scores
