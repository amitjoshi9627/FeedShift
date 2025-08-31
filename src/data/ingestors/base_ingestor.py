import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """
    Abstract base class for data ingestion from external sources.

    This class defines the common interface that all platform-specific ingestors
    must implement. It provides a contract for fetching data from external APIs
    or sources and managing data persistence in the database.

    All concrete ingestor implementations must provide methods for:
    - Ingesting data from external sources
    - Loading stored data for processing
    """

    @abstractmethod
    def ingest(self, *args: Any, **kwargs: Any) -> pd.DataFrame | None:
        """
        Fetch data from external source and store it in the database.

        This abstract method must be implemented by all concrete ingestor classes.
        It should handle the complete ingestion pipeline including data fetching,
        preprocessing, and database storage.

        Args:
            *args: Variable length argument list for platform-specific parameters
            **kwargs: Arbitrary keyword arguments that may include:
                - limit (int): Maximum number of records to fetch
                - sort_by (str): Sorting criteria for data retrieval
                - time_filter (str): Time-based filtering parameters
                - Other platform-specific parameters

        Returns:
            pd.DataFrame | None: DataFrame containing ingested data,
                or None if ingestion fails or no data is available

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
            Exception: Platform-specific errors during data ingestion

        Example:
            >>> ingestor = SomeConcreteIngestor()
            >>> data = ingestor.ingest("data_source", limit=100)
        """
        logger.debug(f"ingest() method called on {self.__class__.__name__}")
        return None

    @abstractmethod
    def load_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Load previously ingested data from the database as a DataFrame.

        This method retrieves stored data that was previously ingested and
        processed. It's typically used by ranking engines and other downstream
        components that need access to the cleaned and structured data.

        Args:
            limit (int): Maximum number of records to retrieve from database.
                Defaults to 100 to manage memory usage and performance.

        Returns:
            pd.DataFrame: DataFrame containing the requested data with all
                necessary columns for further processing. Structure varies
                by platform but should include:
                - Content text/title
                - Timestamps
                - Platform-specific metadata
                - Processed/cleaned versions of text

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
            Exception: Database connection or query errors

        Example:
            >>> ingestor = SomeConcreteIngestor()
            >>> recent_data = ingestor.load_data(limit=50)
        """
        logger.debug(f"load_data() method called on {self.__class__.__name__} with limit={limit}")
        pass
