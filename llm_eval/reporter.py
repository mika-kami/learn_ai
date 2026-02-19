import json
from datetime import datetime
from pathlib import Path

from rich import print

from llm_eval import config


class Reporter:

    def __init__(self):
        pass

    # =========================
    # SAVE REPORT
    # =========================

    def save_report(self, reports):

        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        path = reports_dir / f"reports_{timestamp}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)

        print(f"[bold green]Saved report to {path}[/bold green]")
