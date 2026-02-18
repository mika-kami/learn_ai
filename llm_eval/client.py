import time
from timeit import main
from dotenv import load_dotenv
from openai import OpenAI

from llm_eval import config


class LLMClient():
    def __init__(self):

        if not config.API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=config.API_KEY)

        self.model = config.MODEL_NAME
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS

    def send_prompt(self, prompt: str):

        start_time = time.perf_counter()

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are a precise technical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        latency = time.perf_counter() - start_time

        # -------- Extract text safely --------
        answer = ""

        try:
            if hasattr(response, "output_text") and response.output_text:
                answer = response.output_text
            else:
                answer = response.output[0].content[0].text
        except Exception:
            answer = str(response)

        # -------- Tokens safely --------
        tokens = None
        try:
            if response.usage:
                tokens = response.usage.output_tokens
        except Exception:
            pass

        return {
            "prompt": prompt,
            "response": answer,
            "latency": latency,
            "tokens": tokens
        }
