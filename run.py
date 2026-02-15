import sys
from rich import print

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

    print("\n[bold yellow]=== RESPONSE ===[/bold yellow]")
    print(result.response)

    print("\n[bold cyan]=== META ===[/bold cyan]")
    print(f"Latency: {result.latency:.3f} sec")
    print(f"Tokens: {result.tokens}")
    print(f"Prompt: {result.prompt}")

if __name__ == "__main__":
    main()