from functools import lru_cache
from pathlib import Path

from src.engine import BaseEngine
from src.engine.constants import Platform
from src.engine import RedditEngine


@lru_cache(maxsize=5)
def get_engine(platform: str, path: str | Path | None = None):
    print(f"Yeah we are getting engine for {platform} X {path}")
    if platform.lower() == "reddit":
        return RedditEngine(path)
    else:
        return BaseEngine(path)


if __name__ == "__main__":
    engine = get_engine(platform=Platform.REDDIT)
    result = engine.run(interests=["Technology", "Science"]).head()
    print(result)
