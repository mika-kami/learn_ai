from dotenv import load_dotenv
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# =========================
# API / MODEL
# =========================

API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.3
MAX_TOKENS = 300

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
PASS_THRESHOLD = 0.8
