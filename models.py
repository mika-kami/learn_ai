from pydantic import BaseModel

class LLMResult(BaseModel):
    prompt: str
    response: str
    latency: float
    tokens: int | None = None
    