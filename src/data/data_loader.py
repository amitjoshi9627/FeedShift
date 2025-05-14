import pdb
from typing import Optional

import pandas as pd

from src.config.paths import RAW_DATA
from src.data.preprocessing import FeedShiftPreprocessor
from src.utils.tools import save_csv, load_csv


class FeedShiftDataLoader:
    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        if data is not None and not data.empty:
            save_csv(data, RAW_DATA)
        self.raw_data = load_csv(RAW_DATA)
        self.preprocessor = FeedShiftPreprocessor()
        self.data = pd.DataFrame()

    @property
    def processed_data(self) -> pd.DataFrame:
        self.data = self.preprocessor.process_data(self.raw_data)
        return self.data


