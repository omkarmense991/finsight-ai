# src/finsight/rag/llm_client.py

from openai import OpenAI

from src.finsight.config import OPENAI_MODEL
from src.finsight.rag.prompt import SYSTEM_PROMPT


class OpenAILLMClient:
    def __init__(self, model: str = OPENAI_MODEL):
        self.client = OpenAI()
        self.model = model

    def generate_answer(self, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )

        return response.output_text
