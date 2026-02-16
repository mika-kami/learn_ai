import json
from rich import print

from .config import EVAL_KEYWORDS, PASS_THRESHOLD, REPORTS_DIR, REPORTS_NAME
from .models import Report
from .metrics import count_latency_score, count_length_score, contains_keywords


class Evaluator:

    def generate_report(self, response):
        # ---- METRICS ----
        keywords_score = contains_keywords(response.response, EVAL_KEYWORDS)
        length_score = count_length_score(response.tokens)
        latency_score = count_latency_score(response.latency)
        final = round((keywords_score + length_score + latency_score) / 3, 3)
        was_passed: bool = final >= PASS_THRESHOLD

        report = Report(
            prompt=response.prompt,
            response=response.response,
            keyword_score=keywords_score,
            length_score=length_score,       
            latency=float(f"{response.latency:.2f}"),
            final_score=final,
            was_passed=was_passed
        )

        print(f"[green]OK[/green] - latency: {response.latency:.3f}s | score: {final}")

        return [report.model_dump()]   # return list of dicts
    

    def evaluate_error(self, prompt, error):
        report = {
            "prompt": prompt,
            "response": None,
            "keyword_score": 0,
            "length_score": 0,     
            "latency": None,
            "final_score": 0,
            "error": str(error)
        }

        print(f"[red]ERROR[/red] - {error}")
        return [report]

    def save_report(self, reports):
        """Writes a list of reports to JSON."""
        with open(f"{REPORTS_DIR}/{REPORTS_NAME}", "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)