from abc import ABC, abstractmethod

from llm_eval import config
from llm_eval.embeddings import EmbeddingClient, cosine_similarity


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
        result = min(score, 1.0)

        return round(result, 3)


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
        result = max(score, 0.0)

        return round(result, 3)


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

        count = sum(1 for kw in self.keywords if kw.lower() in text_lower)
        result = count / len(self.keywords)

        return round(result, 3)


# =========================
# EMBEDDINGS METRIC
# =========================


class SemanticSimilarityMetric(BaseMetric):

    def __init__(self):

        self.embedding_client = EmbeddingClient()

    def name(self) -> str:
        return "semantic_similarity_score"

    def compute(self, response_text: str, **kwargs) -> float:

        expected = kwargs.get("expected")

        if not response_text or not expected:
            return 0.0

        emb1 = self.embedding_client.get_embedding(response_text)
        emb2 = self.embedding_client.get_embedding(expected)

        similarity = cosine_similarity(emb1, emb2)

        return max(0.0, similarity)
    

# =========================
# STABILITY METRIC
# =========================

class StabilityMetric(BaseMetric):
    def __init__(self):

        self.embedding_client = EmbeddingClient()

    def name(self) -> str:
        return "stability_score"

    def compute(self, responses, **kwargs):

        if len(responses) < 2:
            return 1.0

        embeddings = [
            self.embedding_client.get_embedding(r)
            for r in responses
            if r
        ]

        sims = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(
                    cosine_similarity(
                        embeddings[i],
                        embeddings[j]
                    )
                )

        if not sims:
            return 0.0

        return sum(sims) / len(sims)
