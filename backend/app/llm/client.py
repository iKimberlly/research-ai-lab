import ollama

from app.config import settings


class LLMClient:

    def __init__(self):
        self.model = settings.llm_model

        self.client = ollama.Client(
            host=settings.ollama_host
        )

    def generate(self, prompt: str) -> str:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]