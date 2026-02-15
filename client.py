import os
import time
from timeit import main
from dotenv import load_dotenv
from openai import OpenAI

import config
import models


class LLMClient():
    def __init__(self):
        # 1. Load environment variables from .env file
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        # 2. Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        self.model=config.MODEL_NAME
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS

    def send_prompt(self, prompt: str):
        start_time = time.perf_counter()

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are a precise technical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )

        latency = time.perf_counter() - start_time

        answer = response.output_text
        tokens = response.usage.output_tokens

        return {
            "prompt": prompt,
            "response": answer,
            "latency": latency,
            "tokens": tokens
        }