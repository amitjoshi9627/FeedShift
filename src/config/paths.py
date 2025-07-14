from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]

DATA_DIR = ROOT_DIR / ".data_cache"
RAW_DATA_PATH = DATA_DIR / "raw/raw_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed/processed_data.csv"
