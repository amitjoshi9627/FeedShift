import re
from abc import ABC

import pandas as pd

from src.config.paths import PROCESSED_DATA_PATH
from src.data.constants import DataCols
from src.utils.tools import save_csv


class BasePreprocessor(ABC):
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        processed_data = raw_data.copy(deep=True).drop_duplicates()
        return self.post_processing(processed_data)

    @staticmethod
    def post_processing(data: pd.DataFrame, timestamp_col: str = DataCols.TIMESTAMP) -> pd.DataFrame:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], format="mixed", dayfirst=False).dt.strftime(
            "%b %d, %Y at %H:%M"
        )
        data = data.sort_values(by=[timestamp_col])
        save_csv(data, PROCESSED_DATA_PATH)
        return data

    @staticmethod
    def clean_text(text: str) -> str | None:
        if not isinstance(text, str):
            return None

        # 2. Remove URLs (Markdown links and bare URLs)
        # Markdown links: [text](url) -> remove the entire link structure
        text = re.sub(r"\[(.*?)]\((.*?)\)", r"\1", text)  # Keeps the text, removes the URL part
        # Or to remove both text and URL: text = re.sub(r'\[.*?\]\(.*?\)', '', text)

        # Bare URLs: http(s)://... or www....
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 3. Remove Markdown list bullet points and other common formatting
        text = re.sub(r"^\s*[\*\-]\s*", "", text, flags=re.MULTILINE)  # Bullet points at start of line
        text = text.replace("*", "")  # Remove remaining asterisks (e.g., for italics/bold)
        text = text.replace("_", "")  # Remove underscores (for italics)
        text = text.replace("~", "")  # Remove strikethrough tildes
        text = text.replace(":", ": ")  # Add space after :

        # 4. Normalize whitespace: newlines, tabs, multiple spaces
        text = re.sub(r"[\n\r\t]+", " ", text)  # Replace newlines, carriage returns, tabs with single space
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space

        # 5. Remove leading/trailing whitespace
        text = text.strip()

        return text
