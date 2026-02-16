import json
from datetime import datetime
from rich import print
from rich.console import Console
from rich.table import Table

from .config import REPORTS_NAME, RESULTS_DIR, RESULTS_NAME
from .client import LLMClient
from .evaluator import Evaluator
from .models import LLMResult


timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M")


def load_prompts(path="./data/prompts.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    prompts = load_prompts()
    print(f"[bold cyan]Loaded {len(prompts)} prompts.[/bold cyan]")

    client = LLMClient()
    evaluator = Evaluator()
    console = Console()

    raw_results = []
    report_results = []

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n[bold yellow]Running {idx}/{len(prompts)}:[/bold yellow] {prompt}")
        
        try:
            raw = client.send_prompt(prompt)
            result = LLMResult(**raw)

            # ---- Save raw result ----
            raw_dump = result.model_dump()
            raw_dump["latency"] = float(f"{raw_dump['latency']:.2f}")
            raw_results.append(raw_dump)

            # ---- Evaluation ----
            report_results += evaluator.generate_report(response=result)

            # ---- TABLE OUTPUT ----
            table = Table(title=f"Meta Data #{idx}")

            table.add_column("Key", style="bold")
            table.add_column("Value", style="italic green")

            safe_response = (
                result.response[:120] + "..."
                if len(result.response) > 120 else result.response
            )

            table.add_row("Prompt", prompt)
            table.add_row("Response (trimmed)", safe_response)
            table.add_row("Latency", f"{result.latency:.2f} sec")
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

            report_results += evaluator.evaluate_error(prompt=prompt, error=e)

    # ---- SAVE RAW RESPONSES ----
    with open(f"{RESULTS_DIR}/{RESULTS_NAME}", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    # ---- SAVE EVALUATION REPORT ----
    evaluator.save_report(reports=report_results)

    print(f"\n[bold green]Saved results to {RESULTS_NAME}[/bold green]")
    print(f"[bold green]Saved evaluation to {REPORTS_NAME}[/bold green]")


if __name__ == "__main__":
    run_evaluation()