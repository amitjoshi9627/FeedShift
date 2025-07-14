import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
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
    FRESHNESS_HALF_LIFE,
)
from src.utils.tools import harmonic_mean


class TextRanker:
    def __init__(
        self,
        data: pd.DataFrame,
        timestamp_col: str = DataCols.TIMESTAMP,
        text_col: str = DataCols.PROCESSED_TEXT,
    ) -> None:
        self.data = data
        self.dates = data[timestamp_col].values
        self.texts = self.data[text_col].tolist()

        self.embeddor = FeedShiftEmbeddor()
        self.text_embeddings = self.embeddor.encode(self.texts)
        self.detoxifier = FeedShiftDetoxified()

    def rerank(
        self,
        interests: list[str],
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
        diversity_strength: float = DEFAULT_DIVERSITY_STRENGTH,
    ) -> pd.DataFrame:
        print("Starting Re Ranking...")
        self.data[DataCols.RECOMMENDATION_SCORE] = self._get_score(
            interests, toxicity_strictness, diversity_strength
        ).round(1)
        self.data = self.data.sort_values(by=DataCols.RECOMMENDATION_SCORE, ascending=False)
        return self.data

    def _get_score(
        self,
        interests: list[str],
        toxicity_strictness: float,
        diversity_strength: float,
    ) -> np.ndarray:
        st = time.time()
        self.data[DataCols.UNIQUENESS_SCORE] = self._get_uniqueness_score()
        print(f"Uniqueness Scoring took - {round(time.time() - st, 2)} Seconds")
        st = time.time()
        self.data[DataCols.FRESHNESS_SCORE] = self._get_freshness_score()
        print(f"Freshness Scoring took - {round(time.time() - st, 2)} Seconds")
        st = time.time()
        self.data[DataCols.TOXICITY_SCORE] = self._get_toxicity_score()
        print(f"Toxicity Scoring took - {round(time.time() - st, 2)} Seconds")
        st = time.time()
        self.data[DataCols.INTERESTS_SCORE] = self._get_interests_score(interests)
        print(f"Interests Scoring took - {round(time.time() - st, 2)} Seconds")
        st = time.time()
        self.data[DataCols.DIVERSITY_SCORE] = self._get_diversity_score(
            self.data[DataCols.INTERESTS_SCORE], diversity_strength=diversity_strength
        )
        print(f"Diversity Scoring took - {round(time.time() - st, 2)} Seconds")

        return (
            RankingWeight.UNIQUENESS * self.data[DataCols.UNIQUENESS_SCORE]
            + RankingWeight.FRESHNESS * self.data[DataCols.FRESHNESS_SCORE]
            - toxicity_strictness * self.data[DataCols.TOXICITY_SCORE]
            + RankingWeight.INTERESTS * self.data[DataCols.INTERESTS_SCORE]
            + RankingWeight.DIVERSITY * self.data[DataCols.DIVERSITY_SCORE]
        )

    @staticmethod
    def _auto_eps(distance_matrix):
        # Calculate 95th percentile of non-diagonal distances
        flat_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        return np.percentile(flat_dist, 95)

    @lru_cache
    def _get_uniqueness_score(self) -> np.ndarray:
        # Every point represent how much a sentence is similar to all other sentences
        distance_matrix = np.maximum(0, 1 - cosine_similarity(self.text_embeddings))

        # Ignoring self similarity
        np.fill_diagonal(distance_matrix, 0)
        eps = self._auto_eps(distance_matrix)
        clustering = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit(distance_matrix)
        labels = clustering.labels_
        clustering_uniqueness_score = np.ones(len(self.text_embeddings))

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 1:
                continue

            # finding the most central point
            cluster_distance = distance_matrix[cluster_indices][:, cluster_indices]
            central_idx = cluster_indices[np.argmin(cluster_distance.sum(axis=1))]
            for idx in cluster_indices:
                if idx != central_idx:
                    clustering_uniqueness_score[idx] = max(0.1, distance_matrix[idx][central_idx])

        global_distance_score = np.mean(distance_matrix, axis=1)

        uniqueness_score = harmonic_mean(global_distance_score, clustering_uniqueness_score)
        return MinMaxScaler().fit_transform(uniqueness_score.reshape(-1, 1))

    @lru_cache
    def _get_freshness_score(self, half_life_days: int = FRESHNESS_HALF_LIFE) -> np.ndarray:
        """
        Calculates a freshness score for dates using an exponential decay model.

        The score is 1.0 for the most recent date and decays exponentially
        based on the provided half-life.

        Args:
            half_life_days (float): The number of days it takes for the freshness score
                                    to halve. A larger value means slower decay.

        Returns:
            np.ndarray: An array of freshness scores, where 1.0 is freshest and
                        values decay towards 0.
        """
        dates = pd.to_datetime(self.dates)
        age_days = (dates.max() - dates).days
        freshness_score = np.power(0.5, age_days / half_life_days).values
        return MinMaxScaler().fit_transform(freshness_score.reshape(-1, 1))

    @lru_cache
    def _get_toxicity_score(self) -> np.ndarray:
        return np.array(self.detoxifier.toxicity_score(self.texts))

    def _get_interests_score(self, interests: list[str]) -> np.ndarray:
        if not interests:
            return np.zeros(len(self.texts)).reshape(-1, 1)
        similarity = cosine_similarity(self.text_embeddings, self.embeddor(interests) + 1e-6)
        interests_score = np.mean(similarity, axis=1)
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
