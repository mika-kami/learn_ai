import json
from datetime import datetime
from pathlib import Path
from rich import print
from llm_eval.models import Report
from llm_eval import config


class Reporter:

    def __init__(self):
        pass

    # =========================
    # SAVE RAW RESULTS
    # =========================

    def save_raw(self, raw_results):

        path = Path(config.RESULTS_DIR) / config.RESULTS_NAME

        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)

        return path


    # =========================
    # SAVE REPORT
    # =========================


    def save_report(self, reports):

        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M")
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        path = reports_dir / f"reports_{timestamp}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        
        print(f"[bold green]Saved evaluation to {path}[/bold green]")

