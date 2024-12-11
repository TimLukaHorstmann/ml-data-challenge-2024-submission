import re

@staticmethod
def filter_tweet(tweet: str) -> str | None:
    """
    Filters tweets based on predefined conditions:
    - Removes tweets containing a URL.
    - Removes retweets.
    - Removes tweets containing usernames.
    
    Returns:
        - `None` if the tweet should be excluded.
        - The original tweet if it passes all filters.
    """
    tweet_lower = tweet.lower()
    if re.search(r"http[s]?\S+", tweet_lower):
        return None
    if re.search(r'rt @?[a-zA-Z0-9_]+:? .*', tweet_lower):
        return None
    if '@' in tweet_lower:
        return None 
    return tweet