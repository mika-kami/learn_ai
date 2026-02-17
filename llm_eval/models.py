from pydantic import BaseModel


class LLMResult(BaseModel):
    prompt: str
    response: str
    latency: float | None = None
    tokens: int | None = None


class Report(BaseModel):
    prompt: str
    response: str
    keyword_score: float
    length_score: float
    latency: float | None = None
    final_score: float | None = None
    was_passed: bool | None = None
