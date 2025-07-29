from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
CONFIG_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / ".data_cache"
RAW_DATA_PATH = DATA_DIR / "raw/raw_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed/reddit_processed_data.csv"
PRAW_INI_PATH = CONFIG_DIR / "praw.ini"
ENV_FILE_PATH = CONFIG_DIR / ".env"

DATABASE_DIR = DATA_DIR / "database/"
