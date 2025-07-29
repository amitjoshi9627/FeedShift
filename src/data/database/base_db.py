from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class BaseDB(ABC):
    @abstractmethod
    def upsert_post(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_posts(self, limit: Optional[int] = None, save_df: bool = True) -> pd.DataFrame:
        pass

    @abstractmethod
    def _create_tables(self):
        pass
