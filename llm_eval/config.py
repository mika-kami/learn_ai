import os

from dotenv import load_dotenv

load_dotenv()

# =========================
# API / MODEL
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TEMPERATURE = 0.3
MAX_TOKENS = 50

# =========================
# EVALUATION SETTINGS
# =========================

# Keywords you want to evaluate against
EVAL_KEYWORDS = [
    "embeddings",
    "vectors",
    "generation",
    "language",
    "processing",
]

# Latency threshold in seconds (for scoring)
LATENCY_THRESHOLD = 5.0

# Pass threshold for final score (0 to 1)
PASS_THRESHOLD = 0.5
