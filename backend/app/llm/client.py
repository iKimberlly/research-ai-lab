
import ollama

from app.config import settings


class LLMClient:
    def __init__(self):
        self.model = settings.llm_model

        self.client = ollama.Client(
            host=settings.ollama_host
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = self.client.chat(
            model=self.model,
            messages=messages
        )

        return response["message"]["content"]
