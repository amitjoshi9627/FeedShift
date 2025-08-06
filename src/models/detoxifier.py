"""
Toxicity detection module for content filtering and safety.

This module provides toxicity scoring capabilities using both machine learning
models and regex-based fallbacks to identify and filter harmful content in
social media feeds.
"""

import logging
import re
from typing import List, Union

import numpy as np
from detoxify import Detoxify

from src.models.constants import DEFAULT_NON_TOXIC_SCORE, DEFAULT_TOXIC_SCORE

# Configure logger
logger = logging.getLogger(__name__)


class DetoxifyModelError(Exception):
    """
    Custom exception raised when Detoxify model fails to load or predict.

    This exception is used to handle specific failures related to the
    toxicity detection model, allowing for graceful fallback to regex-based
    detection when the ML model is unavailable.
    """

    pass


class FeedShiftDetoxified:
    """
    Advanced toxicity detection system with ML model and regex fallback.

    This class provides comprehensive toxicity scoring for text content using
    a hybrid approach that combines machine learning models with regex-based
    pattern matching. It automatically falls back to regex detection when
    the ML model is unavailable or fails.

    Features:
    - Primary ML-based toxicity detection using Detoxify
    - Regex-based fallback for reliability
    - Configurable toxicity scoring thresholds
    - Batch processing capabilities
    - Robust error handling and logging

    Attributes:
        model (Detoxify): ML toxicity detection model (None if unavailable)
        toxic_words (List[str]): Regex patterns for toxic content detection
        default_toxic_score (float): Score assigned to toxic content
        default_non_toxic_score (float): Score assigned to clean content
    """

    def __init__(self) -> None:
        """
        Initialize the toxicity detection system.

        Attempts to load the Detoxify ML model and sets up regex patterns
        for fallback detection. Configures scoring thresholds and logging.

        Raises:
            DetoxifyModelError: If model loading fails (handled gracefully)
        """
        logger.info("Initializing FeedShiftDetoxified toxicity detector")

        # Initialize ML model with fallback
        try:
            logger.debug("Loading Detoxify ML model")
            self.model = Detoxify("original")
            logger.info("Successfully loaded Detoxify model")
        except Exception as e:
            logger.warning(f"Failed to load Detoxify model: {e}")
            logger.info("Will use regex-based fallback for toxicity detection")
            self.model = None

        # Define toxic word patterns for regex fallback
        self.toxic_words: List[str] = [
            r"\b(fuck|shit|bitch|asshole|bastard|dumb|retard|suck|hate|idiot|kill|die)\b",
            r"\b(stupid|moron|jerk|ugly|loser|trash|racist|sexist|terrorist)\b",
            r"\b(damn|hell|crap|piss|wtf|stfu|gtfo)\b",  # Additional mild profanity
            r"\b(nazi|hitler|genocide|rape|murder|suicide)\b",  # Extreme content
        ]

        # Configure scoring thresholds
        self.default_toxic_score = DEFAULT_TOXIC_SCORE
        self.default_non_toxic_score = DEFAULT_NON_TOXIC_SCORE

        logger.debug(
            f"Configured toxicity scoring: toxic={self.default_toxic_score}, "
            f"non-toxic={self.default_non_toxic_score}"
        )
        logger.debug(f"Loaded {len(self.toxic_words)} regex patterns for fallback detection")

    def _is_toxic_regex(self, text: str) -> float:
        """
        Perform regex-based toxicity detection on text content.

        Uses predefined regex patterns to identify potentially toxic content
        when the ML model is unavailable. This provides a fast, reliable
        fallback method for basic toxicity detection.

        Args:
            text (str): Text content to analyze for toxicity

        Returns:
            float: Toxicity score (default_toxic_score if toxic,
                   default_non_toxic_score if clean)

        Example:
            >>> detector = FeedShiftDetoxified()
            >>> score = detector._is_toxic_regex("This is a good post")
            >>> print(score)  # 0.1 (non-toxic)
        """
        if not isinstance(text, str) or not text.strip():
            logger.debug("Empty or invalid text provided to regex detector")
            return self.default_non_toxic_score

        try:
            # Check against all toxic word patterns
            text_lower = text.lower()
            for pattern in self.toxic_words:
                if re.search(pattern, text_lower):
                    logger.debug(f"Regex detected toxic content: pattern '{pattern}' matched")
                    return self.default_toxic_score

            logger.debug("Regex detected clean content")
            return self.default_non_toxic_score

        except re.error as e:
            logger.error(f"Regex pattern error in toxicity detection: {e}")
            return self.default_non_toxic_score
        except Exception as e:
            logger.error(f"Unexpected error in regex toxicity detection: {e}")
            return self.default_non_toxic_score

    def toxicity_score(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate toxicity scores for single text or batch of texts.

        Primary method for toxicity detection that uses the ML model when
        available, automatically falling back to regex-based detection when
        needed. Handles both single strings and lists of texts efficiently.

        Args:
            texts (Union[str, List[str]]): Single text string or list of texts
                to analyze for toxicity

        Returns:
            np.ndarray: Array of toxicity scores (0.0-1.0, higher = more toxic).
                Shape is (n_texts, 1) for consistency with other scoring functions.

        Raises:
            Exception: If both ML model and regex fallback fail

        Example:
            >>> detector = FeedShiftDetoxified()
            >>> scores = detector.toxicity_score(["Good post", "Bad content"])
            >>> print(scores.shape)  # (2, 1)
        """
        # Normalize input to list format
        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Calculating toxicity scores for {len(texts)} texts")

        try:
            # Attempt ML model prediction first
            if self.model is not None:
                logger.debug("Using Detoxify ML model for toxicity detection")
                try:
                    # Get predictions from model
                    predictions = self.model.predict(texts)

                    # Extract maximum score across all toxicity categories
                    # Detoxify returns: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
                    score_matrix = np.column_stack(list(predictions.values()))
                    scores = np.max(score_matrix, axis=1).reshape(-1, 1)

                    logger.debug(f"ML model toxicity score range: {scores.min():.3f} to {scores.max():.3f}")
                    logger.debug(f"High toxicity items: {(scores > 0.7).sum().item()}")

                    return scores

                except Exception as e:
                    logger.error(f"ML model prediction failed: {e}")
                    logger.info("Falling back to regex-based detection")
                    # Continue to regex fallback

            # Regex fallback detection
            logger.debug("Using regex-based toxicity detection")
            scores = np.array([self._is_toxic_regex(text) for text in texts]).reshape(-1, 1)

            toxic_count = (scores == self.default_toxic_score).sum()
            logger.debug(f"Regex detected {toxic_count} toxic items out of {len(texts)}")

            return scores

        except Exception as e:
            logger.error(f"All toxicity detection methods failed: {e}")
            # Return neutral scores as last resort
            logger.warning("Returning neutral toxicity scores due to detection failure")
            return np.full((len(texts), 1), 0.5)

    def get_toxicity_stats(self, texts: Union[str, List[str]]) -> dict:
        """
        Get detailed toxicity statistics for analysis and debugging.

        Args:
            texts (Union[str, List[str]]): Text(s) to analyze

        Returns:
            dict: Dictionary containing toxicity statistics
        """
        scores = self.toxicity_score(texts)

        return {
            "total_texts": len(scores),
            "mean_toxicity": float(scores.mean()),
            "max_toxicity": float(scores.max()),
            "min_toxicity": float(scores.min()),
            "toxic_items": int((scores > 0.5).sum()),
            "model_available": self.model is not None,
            "detection_method": "ml_model" if self.model is not None else "regex_fallback",
        }

    def is_content_safe(self, text: str, threshold: float = 0.5) -> bool:
        """
        Simple boolean check for content safety.

        Args:
            text (str): Text to check
            threshold (float): Toxicity threshold (0.0-1.0)

        Returns:
            bool: True if content is safe (below threshold)
        """
        score = self.toxicity_score(text)[0, 0]
        return np.any(score < threshold)
