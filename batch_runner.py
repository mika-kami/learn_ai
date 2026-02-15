import json
from rich import print

from client import LLMClient
from models import LLMResult

def load_prompts(path="prompts.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def batch_run():
    prompts = load_prompts()
    print(f"[bold cyan]Loaded {len(prompts)} prompts.[/bold cyan]")

    client = LLMClient()
    results = []

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n[bold yellow]Running {idx}/{len(prompts)}:[/bold yellow] {prompt}")

        try:
            raw = client.send_prompt(prompt)
            result = LLMResult(**raw)
            results.append(result.model_dump())

            print(f"[green]OK[/green] - latency: {result.latency:.3f}s")

        except Exception as e:
            print(f"[red]ERROR with prompt:[/red] {prompt}")
            print(e)

            # добавляем ошибку как пустой ответ, чтобы не падал весь pipeline
            results.append({
                "prompt": prompt,
                "response": None,
                "latency": None,
                "tokens": None,
                "error": str(e)
            })

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[bold green]Saved results to results.json[/bold green]")

if __name__ == "__main__":
    batch_run()