import json
from datetime import datetime

from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table

from llm_eval.metrics import LatencyMetric, LengthMetric, KeywordMetric
from llm_eval.config import (
    EVAL_KEYWORDS
)
from llm_eval.client import LLMClient
from llm_eval.evaluator import Evaluator
from llm_eval.models import LLMResult
from llm_eval.reporter import Reporter


# =========================
# METRICS CONFIG
# =========================

metrics = [
    LatencyMetric(),
    LengthMetric(),
    KeywordMetric(EVAL_KEYWORDS)
]

weights = {
    "latency_score": 0.4,
    "length_score": 0.3,
    "keyword_score": 0.3
}

evaluator = Evaluator(metrics, weights)

# =========================
# REPORTER CONFIG
# =========================

reporter = Reporter()

# =========================
# LOAD PROMPTS
# =========================

def load_prompts(path="./data/prompts.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# MAIN RUN
# =========================

def run_evaluation():

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()

    print(f"[bold cyan]Loaded {len(prompts)} prompts.[/bold cyan]")

    client = LLMClient()
    console = Console()

    raw_results = []
    report_results = []

    for idx, prompt in enumerate(prompts, start=1):

        print(f"\n[bold yellow]Running {idx}/{len(prompts)}:[/bold yellow] {prompt}")

        try:

            raw = client.send_prompt(prompt)
            result = LLMResult(**raw)

            # -------- RAW SAVE --------

            raw_dump = result.model_dump()

            if raw_dump.get("latency") is not None:
                raw_dump["latency"] = float(f"{raw_dump['latency']:.2f}")

            raw_results.append(raw_dump)

            # -------- EVALUATION --------

            report_results.extend(
                evaluator.evaluate(result)
            )

            # -------- TABLE OUTPUT --------

            table = Table(title=f"Meta Data #{idx}")

            table.add_column("Key", style="bold")
            table.add_column("Value", style="italic green")

            response_text = result.response or ""

            safe_response = (
                response_text[:120] + "..."
                if len(response_text) > 120 else response_text
            )

            table.add_row("Prompt", prompt)
            table.add_row("Response (trimmed)", safe_response)
            table.add_row("Latency", f"{result.latency:.2f} sec" if result.latency else "N/A")
            table.add_row("Tokens", str(result.tokens))

            console.print(table)

        except Exception as e:

            print(f"[red]ERROR with prompt:[/red] {prompt}")
            print(e)

            raw_results.append({
                "prompt": prompt,
                "response": None,
                "latency": None,
                "tokens": None,
                "error": str(e)
            })

            report_results.extend(
                evaluator.evaluate_error(prompt=prompt, error=e)
            )

    # =========================
    # SAVE RAW
    # =========================
    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M")

    results_path = results_dir / f"results_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    # =========================
    # SAVE REPORT
    # =========================

    reporter.save_report(report_results)

    print(f"\n[bold green]Saved results to {results_path}[/bold green]")


if __name__ == "__main__":
    run_evaluation()
