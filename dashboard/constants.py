from dataclasses import dataclass
from enum import StrEnum
from typing import List

from src.config.paths import DATA_DIR
from src.data.constants import RedditDataCols, DataCols


@dataclass(frozen=True)
class UIPath:
    CSV_UPLOAD_PATH = DATA_DIR / "dashboard/feedshift_data.csv"


@dataclass(frozen=True)
class InterestCategories:
    """Available interest categories for content filtering"""

    TECHNOLOGY: str = "Technology"
    SPORTS: str = "Sports"
    ENTERTAINMENT: str = "Entertainment"
    POLITICS: str = "Politics"
    SCIENCE: str = "Science"
    HEALTH: str = "Health"
    BUSINESS: str = "Business"
    TRAVEL: str = "Travel"
    FOOD: str = "Food"
    FASHION: str = "Fashion"

    @classmethod
    def get_all(cls) -> List[str]:
        """Return all available interest categories as a list"""
        return [
            cls.TECHNOLOGY,
            cls.SPORTS,
            cls.ENTERTAINMENT,
            cls.POLITICS,
            cls.SCIENCE,
            cls.HEALTH,
            cls.BUSINESS,
            cls.TRAVEL,
            cls.FOOD,
            cls.FASHION,
        ]


@dataclass(frozen=True)
class UIConstants:
    """UI-related constants"""

    # File upload
    MAX_FILES: int = 1
    UPLOAD_ZONE_MIN_HEIGHT: str = "150px"
    UPLOAD_ZONE_MAX_WIDTH: str = "400px"

    # Processing
    DEBOUNCE_DELAY: float = 0.3  # 300ms debounce
    RESULTS_HEIGHT: str = "400px"
    TOP_POSTS_LIMIT: int = 10

    # Container sizing
    CONTAINER_MAX_WIDTH: str = "1200px"
    CARD_MAX_WIDTH: str = "800px"
    CONTROL_MAX_WIDTH: str = "500px"

    # Logo
    LOGO_HEIGHT: str = "250px"
    LOGO_FILENAME: str = "feedshift_main_logo-removebg.png"

    # Date Time Format for Saving file dynamically
    DATE_TIME_FORMAT: str = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class DefaultValues:
    """Default values for various parameters"""

    TOXICITY_STRICTNESS: float = 0.5
    TOXICITY_MIN: float = 0.0
    TOXICITY_MAX: float = 1.0
    TOXICITY_STEP: float = 0.05


@dataclass(frozen=True)
class ContentFields:
    """Field names for content in different data sources"""

    CONTENT_FIELDS: tuple = (DataCols.PROCESSED_TEXT, RedditDataCols.PROCESSED_TITLE)
    AUTHOR_FIELDS: tuple = (DataCols.AUTHOR, DataCols.USER, DataCols.USERNAME)
    FALLBACK_CONTENT: str = "No content available"
    FALLBACK_AUTHOR: str = "Unknown"


# Color scheme and styling constants
@dataclass(frozen=True)
class ColorScheme:
    """Color scheme constants for consistent theming"""

    PRIMARY: str = "purple"
    SECONDARY: str = "gray"

    # CSS Variables (for reference)
    PURPLE_1: str = "var(--purple-1)"
    PURPLE_2: str = "var(--purple-2)"
    PURPLE_3: str = "var(--purple-3)"
    PURPLE_6: str = "var(--purple-6)"
    PURPLE_7: str = "var(--purple-7)"
    PURPLE_8: str = "var(--purple-8)"
    PURPLE_9: str = "var(--purple-9)"
    PURPLE_11: str = "var(--purple-11)"
    GRAY_9: str = "var(--gray-9)"
    GRAY_10: str = "var(--gray-10)"
    GRAY_11: str = "var(--gray-11)"
    GRAY_12: str = "var(--gray-12)"


class PostSimilarityType(StrEnum):
    """Post Similarity types"""

    SAME = "Same"
    SIMILAR = "Similar"
    NEAR = "Near"
    DIVERSE = "Diverse"
    DIFFERENT = "Different"
