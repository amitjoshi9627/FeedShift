from abc import ABC, abstractmethod
import pandas as pd


class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self, *args, **kwargs):
        """
        Fetches data from external source (e.g., Reddit, Twitter)
        and stores into the database.
        """
        pass

    @abstractmethod
    def load_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Loads ingested data from the database as a DataFrame.
        Typically used by the ranking engine.
        """
        pass
