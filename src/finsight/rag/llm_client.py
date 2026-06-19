# src/finsight/rag/llm_client.py

from abc import ABC, abstractmethod

from google import genai
from google.genai import types

from src.finsight.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
)
from src.finsight.rag.prompt import SYSTEM_PROMPT


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_answer(
        self,
        user_prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        pass


class GeminiLLMClient(BaseLLMClient):
    def __init__(self, model: str = GEMINI_MODEL):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model

    def generate_answer(
        self,
        user_prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )

        return response.text


def get_llm_client() -> BaseLLMClient:
    if LLM_PROVIDER == "gemini":
        return GeminiLLMClient()

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")