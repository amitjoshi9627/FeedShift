"""
Content ranking and scoring system for feed personalization.

This module implements a comprehensive ranking algorithm that scores content
based on multiple factors including uniqueness, freshness, toxicity, user interests,
and diversity to create personalized and high-quality feeds.
"""

import logging
import time
from functools import lru_cache
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.data.constants import DataCols
from src.models.detoxifier import FeedShiftDetoxified
from src.models.embedder import FeedShiftEmbeddor
from src.ranking.constants import (
    DEFAULT_DIVERSITY_STRENGTH,
    DEFAULT_TOXICITY_STRICTNESS,
    FRESHNESS_HALF_LIFE,
    SIMILAR_POSTS_ALPHA,
    RankingWeight,
)
from src.utils.tools import harmonic_mean

# Configure logger
logger = logging.getLogger(__name__)


class TextRanker:
    """
    Advanced text ranking system for content scoring and personalization.

    This class implements a multi-factor ranking algorithm that evaluates content
    based on uniqueness, freshness, toxicity, user interests, and diversity metrics.
    It uses machine learning models for embeddings and toxicity detection to provide
    sophisticated content scoring capabilities.

    The ranking system combines multiple scoring components:
    - Uniqueness: Measures content novelty and avoids duplicates
    - Freshness: Prioritizes recent content with exponential decay
    - Toxicity: Filters harmful content based on strictness settings
    - Interests: Aligns content with user preferences using embeddings
    - Diversity: Promotes varied content to avoid echo chambers

    Attributes:
        data (pd.DataFrame): Input data containing posts to rank
        dates (np.ndarray): Timestamp values for freshness calculation
        texts (List[str]): List of text content for processing
        embeddor (FeedShiftEmbeddor): Text embedding model for similarity
        text_embeddings (np.ndarray): Precomputed embeddings for all texts
        detoxifier (FeedShiftDetoxified): Toxicity detection model
    """

    def __init__(
        self,
        data: pd.DataFrame,
        timestamp_col: str = DataCols.TIMESTAMP,
        text_col: str = DataCols.PROCESSED_TEXT,
    ) -> None:
        """
        Initialize the text ranker with data and ML models.

        Sets up the ranking system with input data and initializes the required
        machine learning models for embeddings and toxicity detection. Precomputes
        text embeddings for efficiency.

        Args:
            data (pd.DataFrame): DataFrame containing posts to rank with required columns
            timestamp_col (str): Column name containing timestamps for freshness scoring.
                Defaults to DataCols.TIMESTAMP.
            text_col (str): Column name containing text content for processing.
                Defaults to DataCols.PROCESSED_TEXT.

        Raises:
            KeyError: If required columns are missing from the data
            Exception: If model initialization fails

        Example:
            >>> ranker = TextRanker(posts_df, timestamp_col='created_at', text_col='content')
        """
        logger.info(f"Initializing TextRanker with {len(data)} posts")

        try:
            # Validate required columns
            if timestamp_col not in data.columns:
                raise KeyError(f"Timestamp column '{timestamp_col}' not found in data")
            if text_col not in data.columns:
                raise KeyError(f"Text column '{text_col}' not found in data")

            self.data = data.copy()  # Work with a copy to avoid modifying original
            self.dates = data[timestamp_col].values
            self.texts = data[text_col].fillna("").tolist()  # Handle NaN values

            logger.debug(f"Extracted {len(self.texts)} texts for processing")

            # Initialize ML models
            logger.info("Initializing embedding model")
            self.embeddor = FeedShiftEmbeddor()

            logger.info("Generating text embeddings")
            start_time = time.time()
            self.text_embeddings = self.embeddor.encode(self.texts)
            embedding_time = time.time() - start_time
            logger.info(f"Generated {len(self.text_embeddings)} embeddings in {embedding_time:.2f} seconds")

            logger.info("Initializing toxicity detection model")
            self.detoxifier = FeedShiftDetoxified()

            logger.info("TextRanker initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TextRanker: {e}")
            raise

    def rerank(
        self,
        interests: List[str],
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
        diversity_strength: float = DEFAULT_DIVERSITY_STRENGTH,
    ) -> pd.DataFrame:
        """
        Rerank the content based on multiple scoring factors.

        Applies the complete ranking pipeline including uniqueness, freshness,
        toxicity, interests, and diversity scoring. Returns the data sorted by
        the final recommendation score.

        Args:
            interests (List[str]): List of user interest keywords/topics for
                content personalization. Empty list applies no interest filtering.
            toxicity_strictness (float): Strictness level for toxicity filtering
                (0.0 = no filtering, 1.0 = maximum filtering).
                Defaults to DEFAULT_TOXICITY_STRICTNESS.
            diversity_strength (float): Strength of diversity promotion algorithm
                (0.0 = no diversity, 1.0 = maximum diversity).
                Defaults to DEFAULT_DIVERSITY_STRENGTH.

        Returns:
            pd.DataFrame: Reranked DataFrame with recommendation scores,
                sorted by final score in descending order

        Raises:
            Exception: If any scoring component fails

        Example:
            >>> ranker = TextRanker(posts_df)
            >>> ranked_posts = ranker.rerank(
            ...     interests=['technology', 'science'],
            ...     toxicity_strictness=0.8,
            ...     diversity_strength=0.6
            ... )
        """
        logger.info("Starting content reranking process")
        logger.info(
            f"Parameters: interests={len(interests)} items, "
            f"toxicity_strictness={toxicity_strictness}, "
            f"diversity_strength={diversity_strength}"
        )

        try:
            start_time = time.time()

            # Calculate final recommendation scores
            recommendation_scores = self._get_score(interests, toxicity_strictness, diversity_strength)

            # Add scores to data and sort
            self.data[DataCols.RECOMMENDATION_SCORE] = recommendation_scores.round(3)
            self.data = self.data.sort_values(by=DataCols.RECOMMENDATION_SCORE, ascending=False).reset_index(drop=True)

            total_time = time.time() - start_time
            logger.info(f"Reranking completed in {total_time:.2f} seconds")
            logger.info(f"Top score: {self.data[DataCols.RECOMMENDATION_SCORE].iloc[0]:.3f}")

            return self.data

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise

    def _get_score(
        self,
        interests: List[str],
        toxicity_strictness: float,
        diversity_strength: float,
    ) -> np.ndarray:
        """
        Calculate composite recommendation scores using weighted components.

        Computes individual scoring components and combines them using predefined
        weights to create final recommendation scores. Includes performance timing
        for each component.

        Args:
            interests (List[str]): User interest keywords
            toxicity_strictness (float): Toxicity filtering strength
            diversity_strength (float): Diversity promotion strength

        Returns:
            np.ndarray: Array of final recommendation scores

        Raises:
            Exception: If any scoring component fails
        """
        logger.debug("Computing individual scoring components")

        try:
            # Uniqueness scoring
            start_time = time.time()
            self.data[DataCols.UNIQUENESS_SCORE] = self._get_uniqueness_score()
            uniqueness_time = time.time() - start_time
            logger.debug(f"Uniqueness scoring completed in {uniqueness_time:.2f} seconds")

            # Freshness scoring
            start_time = time.time()
            self.data[DataCols.FRESHNESS_SCORE] = self._get_freshness_score()
            freshness_time = time.time() - start_time
            logger.debug(f"Freshness scoring completed in {freshness_time:.2f} seconds")

            # Toxicity scoring
            start_time = time.time()
            self.data[DataCols.TOXICITY_SCORE] = self._get_toxicity_score()
            toxicity_time = time.time() - start_time
            logger.debug(f"Toxicity scoring completed in {toxicity_time:.2f} seconds")

            # Interests scoring
            start_time = time.time()
            self.data[DataCols.INTERESTS_SCORE] = self._get_interests_score(interests)
            interests_time = time.time() - start_time
            logger.debug(f"Interests scoring completed in {interests_time:.2f} seconds")

            # Diversity scoring
            start_time = time.time()
            self.data[DataCols.DIVERSITY_SCORE] = self._get_diversity_score(
                np.array(self.data[DataCols.INTERESTS_SCORE]), diversity_strength=diversity_strength
            )
            diversity_time = time.time() - start_time
            logger.debug(f"Diversity scoring completed in {diversity_time:.2f} seconds")

            # Combine scores with weights
            final_scores = (
                RankingWeight.UNIQUENESS * self.data[DataCols.UNIQUENESS_SCORE]
                + RankingWeight.FRESHNESS * self.data[DataCols.FRESHNESS_SCORE]
                - toxicity_strictness * self.data[DataCols.TOXICITY_SCORE]  # Subtract toxicity
                + RankingWeight.INTERESTS * self.data[DataCols.INTERESTS_SCORE]
                + RankingWeight.DIVERSITY * self.data[DataCols.DIVERSITY_SCORE]
            )

            logger.debug(f"Final score range: {final_scores.min():.3f} to {final_scores.max():.3f}")
            return final_scores.values

        except Exception as e:
            logger.error(f"Score calculation failed: {e}")
            raise

    @staticmethod
    def _auto_eps(distance_matrix: np.ndarray) -> float:
        """
        Automatically determine optimal epsilon parameter for DBSCAN clustering.

        Calculates the 95th percentile of pairwise distances to set a reasonable
        threshold for clustering similar content together.

        Args:
            distance_matrix (np.ndarray): Symmetric distance matrix

        Returns:
            float: Optimal epsilon value for DBSCAN
        """
        # Extract upper triangular distances (excluding diagonal)
        flat_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        eps = float(np.percentile(flat_distances, 95))
        logger.debug(f"Auto-calculated DBSCAN epsilon: {eps:.4f}")
        return eps

    @lru_cache(maxsize=1)
    def _get_uniqueness_score(self) -> np.ndarray:
        """
        Calculate uniqueness scores based on content similarity clustering.

        Uses DBSCAN clustering on cosine distances to identify similar content
        groups. Penalizes duplicate content while preserving the most central
        (representative) item in each cluster.

        Returns:
            np.ndarray: Normalized uniqueness scores (0-1, higher = more unique)

        Raises:
            Exception: If clustering or scoring calculations fail
        """
        logger.debug("Calculating uniqueness scores using DBSCAN clustering")

        try:
            # Calculate cosine distance matrix
            distance_matrix = np.maximum(0, 1 - cosine_similarity(self.text_embeddings))
            np.fill_diagonal(distance_matrix, 0)  # Remove self-similarity

            # Perform clustering
            eps = self._auto_eps(distance_matrix)
            clustering = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit(distance_matrix)
            labels = clustering.labels_

            # Initialize uniqueness scores
            uniqueness_scores = np.ones(len(self.text_embeddings))

            # Process clusters to penalize similar content
            unique_labels = set(labels)
            cluster_count = len([label for label in unique_labels if label != -1])
            logger.debug(f"Found {cluster_count} content clusters")

            for cluster_id in unique_labels:
                if cluster_id == -1:  # Noise points (unique content)
                    continue

                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) <= 1:
                    continue

                # Find most central point in cluster
                cluster_distances = distance_matrix[cluster_indices][:, cluster_indices]
                central_idx = cluster_indices[np.argmin(cluster_distances.sum(axis=1))]

                # Penalize non-central points based on distance to center
                for idx in cluster_indices:
                    if idx != central_idx:
                        uniqueness_scores[idx] = max(0.1, distance_matrix[idx][central_idx])

            # Combine with global distance scores
            global_distance_scores = np.mean(distance_matrix, axis=1)
            final_uniqueness = harmonic_mean(global_distance_scores, uniqueness_scores)

            # Normalize to 0-1 range
            normalized_scores = MinMaxScaler().fit_transform(final_uniqueness.reshape(-1, 1)).flatten()

            logger.debug(f"Uniqueness score range: {normalized_scores.min():.3f} to {normalized_scores.max():.3f}")
            return normalized_scores.reshape(-1, 1)

        except Exception as e:
            logger.error(f"Uniqueness scoring failed: {e}")
            raise

    @lru_cache(maxsize=1)
    def _get_freshness_score(self, half_life_minutes: int = FRESHNESS_HALF_LIFE) -> np.ndarray:
        """
        Calculate freshness scores using exponential decay model.

        Implements an exponential decay function where the most recent content
        gets a score of 1.0 and older content decays based on the half-life parameter.

        Args:
            half_life_minutes (int): Number of minutes for score to halve.
                Defaults to FRESHNESS_HALF_LIFE constant.

        Returns:
            np.ndarray: Normalized freshness scores (0-1, higher = more recent)

        Raises:
            Exception: If date parsing or calculation fails
        """
        logger.debug(f"Calculating freshness scores with {half_life_minutes}min half-life")

        try:
            # Convert timestamps to datetime
            dates = pd.to_datetime(self.dates, errors="coerce")

            # Handle invalid dates
            if dates.isnull().any():
                logger.warning(f"Found {dates.isnull().sum()} invalid timestamps")
                dates = dates.fillna(dates.max())  # Fill with most recent valid date

            # Calculate age in minutes
            max_date = dates.max()
            age_minutes = (max_date - dates).total_seconds() / 60

            # Apply exponential decay
            freshness_scores = np.power(0.5, age_minutes / half_life_minutes)

            # Normalize scores
            normalized_scores = MinMaxScaler().fit_transform(freshness_scores.values.reshape(-1, 1)).flatten()

            logger.debug(f"Freshness score range: {normalized_scores.min():.3f} to {normalized_scores.max():.3f}")
            logger.debug(f"Oldest content age: {age_minutes.max():.1f} minutes")

            return normalized_scores.reshape(-1, 1)

        except Exception as e:
            logger.error(f"Freshness scoring failed: {e}")
            raise

    @lru_cache(maxsize=1)
    def _get_toxicity_score(self) -> np.ndarray:
        """
        Calculate toxicity scores for content filtering.

        Uses the toxicity detection model to assess harmful content.
        Higher scores indicate more toxic content that should be filtered.

        Returns:
            np.ndarray: Toxicity scores (0-1, higher = more toxic)

        Raises:
            Exception: If toxicity detection fails
        """
        logger.debug("Calculating toxicity scores")

        try:
            # Filter out empty texts
            non_empty_texts = [text for text in self.texts if text and text.strip()]

            if not non_empty_texts:
                logger.warning("No valid texts for toxicity scoring, returning zero scores")
                return np.zeros(len(self.texts))

            # Get toxicity scores
            toxicity_scores = np.array(self.detoxifier.toxicity_score(self.texts))

            # Validate scores
            if len(toxicity_scores) != len(self.texts):
                logger.error(f"Toxicity score count mismatch: {len(toxicity_scores)} vs {len(self.texts)}")
                return np.zeros(len(self.texts))

            logger.debug(f"Toxicity score range: {toxicity_scores.min():.3f} to {toxicity_scores.max():.3f}")
            logger.debug(f"High toxicity content: {(toxicity_scores > 0.7).sum()} items")

            return toxicity_scores

        except Exception as e:
            logger.error(f"Toxicity scoring failed: {e}")
            # Return neutral scores on failure
            return np.full(len(self.texts), 0.5)

    def _get_interests_score(self, interests: List[str]) -> np.ndarray:
        """
        Calculate interest alignment scores based on user preferences.

        Computes similarity between content embeddings and interest embeddings
        to measure how well content matches user preferences.

        Args:
            interests (List[str]): List of user interest keywords/topics

        Returns:
            np.ndarray: Normalized interest scores (0-1, higher = better match)

        Raises:
            Exception: If embedding calculation fails
        """
        logger.debug(f"Calculating interest scores for {len(interests)} interests")

        try:
            if not interests:
                logger.debug("No interests provided, returning zero scores")
                return np.zeros(len(self.texts)).reshape(-1, 1)

            # Clean and validate interests
            valid_interests = [interest.strip() for interest in interests if interest and interest.strip()]

            if not valid_interests:
                logger.warning("No valid interests after cleaning, returning zero scores")
                return np.zeros(len(self.texts)).reshape(-1, 1)

            # Generate interest embeddings
            logger.debug(f"Encoding {len(valid_interests)} interest keywords")
            interest_embeddings = self.embeddor.encode(valid_interests)

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(
                self.text_embeddings, interest_embeddings + 1e-6  # Small epsilon for numerical stability
            )

            # Average similarity across all interests
            interest_scores = np.mean(similarity_matrix, axis=1)

            # Normalize scores
            normalized_scores = MinMaxScaler().fit_transform(interest_scores.reshape(-1, 1)).flatten()

            logger.debug(f"Interest score range: {normalized_scores.min():.3f} to {normalized_scores.max():.3f}")
            logger.debug(f"High interest content: {(normalized_scores > 0.7).sum()} items")

            return normalized_scores.reshape(-1, 1)

        except Exception as e:
            logger.error(f"Interest scoring failed: {e}")
            # Return neutral scores on failure
            return np.full(len(self.texts), 0.5).reshape(-1, 1)

    @staticmethod
    def _get_diversity_score(interests_scores: np.ndarray, diversity_strength: float) -> np.ndarray:
        """
        Calculate diversity scores to promote content variety.

        Applies diversity adjustments based on interest scores to prevent
        echo chambers and promote content variety. Uses different adjustments
        for similar, near, and diverse content categories.

        Args:
            interests_scores (np.ndarray): Base interest alignment scores
            diversity_strength (float): Strength of diversity promotion (0-1)

        Returns:
            np.ndarray: Adjusted diversity scores

        Raises:
            Exception: If score calculation fails
        """
        logger.debug(f"Calculating diversity scores with strength {diversity_strength}")

        try:
            scores = interests_scores.copy().flatten()
            original_mean = scores.mean()

            # Similar Posts (high similarity) - reduce to promote diversity
            similar_mask = (scores >= 0.8) & (scores <= 1.0)
            similar_count = similar_mask.sum()
            if similar_count > 0:
                scores[similar_mask] -= SIMILAR_POSTS_ALPHA * diversity_strength * scores[similar_mask]
                logger.debug(f"Applied diversity penalty to {similar_count} similar posts")

            # Near Posts (moderate similarity) - boost moderately
            diversity_strength_near = diversity_strength if diversity_strength <= 0.5 else (1 - diversity_strength / 2)
            near_mask = (scores >= 0.6) & (scores < 0.8)
            near_count = near_mask.sum()
            if near_count > 0:
                scores[near_mask] += diversity_strength_near * (1 - scores[near_mask])
                logger.debug(f"Applied moderate diversity boost to {near_count} near posts")

            # Diverse Posts (low similarity) - boost significantly
            diverse_mask = (scores >= 0.4) & (scores < 0.6)
            diverse_count = diverse_mask.sum()
            if diverse_count > 0:
                scores[diverse_mask] += diversity_strength * (1 - scores[diverse_mask])
                logger.debug(f"Applied diversity boost to {diverse_count} diverse posts")

            final_mean = scores.mean()
            logger.debug(f"Diversity adjustment: mean score {original_mean:.3f} -> {final_mean:.3f}")

            return scores.reshape(-1, 1)

        except Exception as e:
            logger.error(f"Diversity scoring failed: {e}")
            # Return original scores on failure
            return interests_scores

    def get_ranking_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of the ranking process.

        Returns:
            Dict[str, Any]: Dictionary containing ranking statistics and metrics
        """
        try:
            summary = {
                "total_posts": len(self.data),
                "score_statistics": {
                    "mean": float(self.data[DataCols.RECOMMENDATION_SCORE].mean()),
                    "std": float(self.data[DataCols.RECOMMENDATION_SCORE].std()),
                    "min": float(self.data[DataCols.RECOMMENDATION_SCORE].min()),
                    "max": float(self.data[DataCols.RECOMMENDATION_SCORE].max()),
                },
                "component_means": {
                    "uniqueness": float(self.data[DataCols.UNIQUENESS_SCORE].mean()),
                    "freshness": float(self.data[DataCols.FRESHNESS_SCORE].mean()),
                    "toxicity": float(self.data[DataCols.TOXICITY_SCORE].mean()),
                    "interests": float(self.data[DataCols.INTERESTS_SCORE].mean()),
                    "diversity": float(self.data[DataCols.DIVERSITY_SCORE].mean()),
                },
            }

            logger.info(
                f"Ranking summary: {len(self.data)} posts, "
                f"mean score: {float(self.data[DataCols.RECOMMENDATION_SCORE].mean()):.3f}"
            )

            return summary

        except Exception as e:
            logger.error(f"Failed to generate ranking summary: {e}")
            return {}
