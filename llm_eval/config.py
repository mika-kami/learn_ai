import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M")

API_KEY = os.getenv("OPENAI_API_KEY")

# Model configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = "0.3"
MAX_TOKENS = "300"

# Naming
REPORTS_DIR = "./reports"
REPORTS_NAME = f"reports_{timestamp}.json"
RESULTS_DIR = "./results"
RESULTS_NAME = f"results_{timestamp}.json"

# Keywords you want to evaluate against
EVAL_KEYWORDS = ["nothing", "nothingness", "void", "emptiness", "null"]

# Latency threshold in seconds (for scoring)
LATENCY_THRESHOLD = 5.0

# Pass threshold for final score (0 to 1)
PASS_THRESHOLD = 0.8
