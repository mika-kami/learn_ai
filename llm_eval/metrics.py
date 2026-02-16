from . import config


def count_length_score(tokens: int, max_len: int = config.MAX_TOKENS) -> float:
    """
    Returns a normalized length score between 0 and 1.
    """
    if not tokens:
        return 0.0
    score = tokens / max_len
    return min(score, 1.0)


def count_latency_score(latency: float, max_latency: float = config.LATENCY_THRESHOLD) -> float:
    """
    Returns a normalized latency score between 0 and 1.
    """
    if not latency or latency >= max_latency:
        return 0.0
    score = 1 - (latency / max_latency)
    return max(score, 0.0)


def contains_keywords(text: str, keywords: list[str]) -> float:
    """
    Percentage of keywords that appear in the response.
    """
    if not keywords:
        return 0.0

    text_lower = text.lower()
    count = sum(1 for kw in keywords if kw.lower() in text_lower)
    return count / len(keywords)