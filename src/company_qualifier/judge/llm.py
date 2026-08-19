import json
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"


class LLMClient(Protocol):
    def complete(self, prompt: str, schema: type[T]) -> T: ...


class OllamaClient:
    def complete(self, prompt: str, schema: type[T]) -> T:
        resp = httpx.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",              # Ollama forțează output JSON valid
            "stream": False,
            "options": {"temperature": 0},
        }, timeout=120.0)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return schema.model_validate_json(content)