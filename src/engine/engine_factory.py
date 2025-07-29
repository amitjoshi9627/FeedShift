from functools import lru_cache

from src.data.constants import SubRedditList
from src.engine.constants import Platform
from src.engine import RedditEngine


@lru_cache(maxsize=5)
def get_engine(platform: str, subreddit: str):
    print(f"Yeah we are getting engine for {platform} X {subreddit}")
    if platform == Platform.REDDIT:
        return RedditEngine(subreddit)
    else:
        raise ValueError(f"Working on getting Engine for {platform}. Sorry :>")


if __name__ == "__main__":
    engine = get_engine(platform=Platform.REDDIT, subreddit=SubRedditList.JOKES)
    result = engine.run().head()
    print(result)
