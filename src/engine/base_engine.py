import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """
    Abstract base class for all feed processing engines.

    This class defines the common interface that all platform-specific engines
    must implement. It serves as a contract for feed processing operations
    across different social media platforms and content sources.

    All concrete engine implementations must provide a `run` method that
    processes and returns ranked feed data.
    """

    @abstractmethod
    def run(self, **kwargs: Any) -> pd.DataFrame:
        """
        Main method to process, rerank, or generate the content feed.

        This abstract method must be implemented by all concrete engine classes.
        It should contain the core logic for processing platform-specific data
        and returning a ranked/filtered DataFrame of content.

        Args:
            **kwargs: Flexible keyword arguments that can include:
                - interests (List[str]): User interest keywords
                - toxicity_strictness (float): Toxicity filtering level
                - diversity_strength (float): Content diversity requirements
                - limit (int): Maximum number of results to return
                - Other platform-specific parameters

        Returns:
            pd.DataFrame: Processed and ranked DataFrame containing feed content.
                The structure may vary by platform but should include:
                - Content text/title
                - Timestamp information
                - Ranking scores
                - Platform-specific metadata

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
            Exception: Platform-specific errors during feed processing
        """
        logger.debug(f"run() method called on {self.__class__.__name__}")
        pass
