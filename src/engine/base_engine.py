from abc import ABC, abstractmethod
import pandas as pd


class BaseEngine(ABC):
    @abstractmethod
    def run(self, **kwargs) -> pd.DataFrame:
        """
        Main method to rerank or generate the feed.
        """
        pass
