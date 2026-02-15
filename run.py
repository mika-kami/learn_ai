import sys
from rich import print
from rich.console import Console
from rich.table import Table

from client import LLMClient
from models import LLMResult

def main():
    prompt = input("Enter your prompt: ").strip()

    if not prompt:
        print("[red]Error: prompt cannot be empty.[/red]")
        return

    client = LLMClient()
    raw_result = client.send_prompt(prompt)

    result = LLMResult(**raw_result)
    
    table = Table(title="Meta Data")
    table.add_column("Key", style="bold")
    table.add_column("Value", style="italic, green")
    table.add_row("Prompt", result.prompt)
    table.add_row("Response", result.response)
    table.add_row("Latency (sec)", f"{result.latency:.5f}")
    table.add_row("Tokens", str(result.tokens))
    
    console = Console()
    console.print(table)

if __name__ == "__main__":
    main()