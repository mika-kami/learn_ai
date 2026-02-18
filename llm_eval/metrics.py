from abc import ABC, abstractmethod
from llm_eval import config


class BaseMetric(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, response_text: str, **kwargs) -> float:
        pass


# =========================
# LENGTH METRIC
# =========================

class LengthMetric(BaseMetric):

    def name(self) -> str:
        return "length_score"

    def compute(self, response_text: str, **kwargs) -> float:

        tokens = kwargs.get("tokens")

        if tokens is None:
            return 0.0

        max_len = config.MAX_TOKENS

        if max_len <= 0:
            return 0.0

        score = tokens / max_len

        return min(score, 1.0)


# =========================
# LATENCY METRIC
# =========================

class LatencyMetric(BaseMetric):

    def name(self) -> str:
        return "latency_score"

    def compute(self, response_text: str, **kwargs) -> float:

        latency = kwargs.get("latency")

        if latency is None:
            return 0.0

        max_latency = config.LATENCY_THRESHOLD

        if latency >= max_latency:
            return 0.0

        score = 1 - (latency / max_latency)

        return max(score, 0.0)


# =========================
# KEYWORD METRIC
# =========================

class KeywordMetric(BaseMetric):

    def __init__(self, keywords=None):

        if keywords is None:
            keywords = config.EVAL_KEYWORDS

        self.keywords = keywords

    def name(self) -> str:
        return "keyword_score"

    def compute(self, response_text: str, **kwargs) -> float:

        if not response_text:
            return 0.0

        if not self.keywords:
            return 0.0

        text_lower = response_text.lower()

        count = sum(
            1 for kw in self.keywords
            if kw.lower() in text_lower
        )

        return count / len(self.keywords)
