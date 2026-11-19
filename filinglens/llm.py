from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Iterator


class LLMConfigurationError(RuntimeError):
    """Raised when model configuration is missing or invalid."""


class LLMResponseError(RuntimeError):
    """Raised when the model endpoint returns an unusable response."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "").strip(),
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            model=os.getenv("LLM_MODEL", "").strip(),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        )

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", self.base_url),
                ("LLM_API_KEY", self.api_key),
                ("LLM_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            raise LLMConfigurationError(
                f"Missing model configuration: {', '.join(missing)}"
            )


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings) -> None:
        settings.validate()
        self.settings = settings

    def complete(self, *, system: str, user: str) -> str:
        request = self._build_request(system=system, user=user, stream=False)

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMResponseError(
                f"Model request failed (HTTP {exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMResponseError(f"Could not connect to the model: {exc.reason}") from exc

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "The model response does not follow the OpenAI Chat Completions format."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("The model returned an empty response.")
        return content.strip()

    def stream(self, *, system: str, user: str) -> Iterator[str]:
        request = self._build_request(system=system, user=user, stream=True)
        emitted = False
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                        delta = payload["choices"][0].get("delta", {})
                        content = delta.get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if isinstance(content, str) and content:
                        emitted = True
                        yield content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMResponseError(
                f"Model request failed (HTTP {exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMResponseError(f"Could not connect to the model: {exc.reason}") from exc

        if not emitted:
            raise LLMResponseError("The model returned no streaming content.")

    def _build_request(self, *, system: str, user: str, stream: bool):
        endpoint = self.settings.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        payload = {
            "model": self.settings.model,
            "temperature": 0.1,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        return urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FilingLens-Copilot/0.1",
            },
            method="POST",
        )

