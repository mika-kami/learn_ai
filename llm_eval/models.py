from typing import Optional

from pydantic import BaseModel


class LLMResult(BaseModel):
    prompt: str
    response: str
    latency: Optional[float] = None
    tokens: Optional[int] = None
    model: Optional[str] = None


class Report(BaseModel):
    prompt: str
    response: str

    latency: Optional[float] = None
    tokens: Optional[int] = None

    keyword_score: float = 0.0
    length_score: float = 0.0
    latency_score: float = 0.0
    semantic_similarity_score: float = 0.0
    stability_score: float = 0.0

    final_score: Optional[float] = None
    was_passed: Optional[bool] = None

    error: Optional[str] = None
