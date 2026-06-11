from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import AppConfig


@dataclass
class OpenAICompatibleChatClient:
    config: AppConfig

    @property
    def available(self) -> bool:
        return self.config.has_chat_config

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        if not self.available:
            raise RuntimeError("Chat API is not configured.")
        url = f"{self.config.base_url}/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.chat_model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=120,
        )
        if response.status_code >= 400:
            body = response.text[:800]
            raise RuntimeError(f"Chat API failed: HTTP {response.status_code}: {body}")
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("Chat API returned no choices.")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Chat API returned an empty answer.")
        return content.strip()
