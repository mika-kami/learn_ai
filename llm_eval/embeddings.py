import math

from openai import OpenAI

from llm_eval import config


class EmbeddingClient:

    def __init__(self):

        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.EMBEDDING_MODEL_NAME

    def get_embedding(self, text: str):

        response = self.client.embeddings.create(model=self.model, input=text)

        return response.data[0].embedding


# =========================
# COSINE SIMILARITY
# =========================


def cosine_similarity(vec1, vec2):

    dot = sum(a * b for a, b in zip(vec1, vec2))

    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)
