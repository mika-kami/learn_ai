import os
import time
from dotenv import load_dotenv
from openai import OpenAI


def main():
    # 1. Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # 2. Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # 3. Set up prompts and parameters
    system_prompt = "You are a precise technical assistant."
    user_prompt = input("Enter your prompt: ")

    temperature = 0.3
    max_tokens = 300

    # 4. Track latency
    start_time = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        end_time = time.perf_counter()
        latency = end_time - start_time

        # 5. Options to print response and metadata
        answer = response.choices[0].message.content

        print("\n=== RESPONSE ===\n")
        print(answer)

        print("\n=== META ===\n")
        print(f"Latency: {latency:.2f} seconds")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Temperature: {temperature}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
