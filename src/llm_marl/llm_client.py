# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests


class CardLLMError(RuntimeError):
    pass


@dataclass
class LLMCompletionResult:
    status_code: int
    content: str
    raw_json: dict
    token_usage: dict
    latency_seconds: float = 0.0


class OpenAICompatibleLLMClient:
    def __init__(self, config):
        self.config = config

    def complete_json(self, system_prompt: str, user_prompt: str) -> LLMCompletionResult:
        if getattr(self.config, "provider", "") == "ollama":
            return self._complete_ollama_json(system_prompt, user_prompt)

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if getattr(self.config, "enable_thinking", None) is not None:
            payload["enable_thinking"] = bool(self.config.enable_thinking)
        thinking_mode = (getattr(self.config, "thinking_mode", "") or "").strip().lower()
        is_deepseek = "deepseek" in self.config.base_url.lower()
        if is_deepseek and thinking_mode in {"disabled", "enabled"}:
            payload["thinking"] = {"type": thinking_mode}
        elif is_deepseek and getattr(self.config, "enable_thinking", None) is not None:
            payload["thinking"] = {
                "type": "enabled" if bool(self.config.enable_thinking) else "disabled"
            }

        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project

        payload_variants = []
        response_format = self.config.response_format.strip()
        if self.config.use_response_format and response_format:
            with_format = dict(payload)
            with_format["response_format"] = {"type": response_format}
            payload_variants.append(with_format)
        payload_variants.append(dict(payload))
        compact_payload = dict(payload)
        compact_payload["max_tokens"] = min(self.config.max_tokens, 2000)
        payload_variants.append(compact_payload)

        last_error = None
        started_at = time.perf_counter()
        for variant_index, payload_variant in enumerate(payload_variants):
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload_variant,
                        timeout=self.config.timeout_seconds,
                    )
                    response.raise_for_status()
                    raw_json = response.json()
                    content = self._extract_content(raw_json)
                    return LLMCompletionResult(
                        status_code=response.status_code,
                        content=content,
                        raw_json=raw_json,
                        token_usage=extract_token_usage(
                            raw_json,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            content=content,
                        ),
                        latency_seconds=time.perf_counter() - started_at,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if (
                        variant_index + 1 < len(payload_variants)
                        and "did not include any choices" in str(exc)
                    ):
                        break
                    if attempt >= self.config.max_retries:
                        break
                    time.sleep(self.config.retry_backoff_seconds)

        raise CardLLMError(f"LLM request failed after retries: {last_error}")

    def _complete_ollama_json(
        self, system_prompt: str, user_prompt: str
    ) -> LLMCompletionResult:
        url = self.config.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": getattr(self.config, "ollama_keep_alive", "10m") or "10m",
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "num_ctx": getattr(self.config, "ollama_num_ctx", 4096),
            },
        }
        if getattr(self.config, "enable_thinking", None) is not None:
            payload["think"] = bool(self.config.enable_thinking)

        last_error = None
        started_at = time.perf_counter()
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                raw_json = response.json()
                content = ((raw_json.get("message") or {}).get("content") or "").strip()
                if not content:
                    raise CardLLMError("Ollama response content was empty.")
                return LLMCompletionResult(
                    status_code=response.status_code,
                    content=content,
                    raw_json=raw_json,
                    token_usage=extract_token_usage(
                        raw_json,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        content=content,
                    ),
                    latency_seconds=time.perf_counter() - started_at,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds)

        raise CardLLMError(f"Ollama request failed after retries: {last_error}")

    @staticmethod
    def _extract_content(payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise CardLLMError("LLM response did not include any choices.")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise CardLLMError("LLM response content was empty.")
        return content


def extract_json_payload(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        direct_error = exc

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise CardLLMError(
            f"Could not parse JSON from the LLM response: {direct_error}"
        ) from direct_error
    try:
        return json.loads(text[first : last + 1])
    except json.JSONDecodeError as exc:
        raise CardLLMError(
            f"Could not parse JSON object from the LLM response: {exc}"
        ) from exc


def extract_token_usage(
    payload: dict,
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    content: str = "",
) -> dict:
    raw_usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = _usage_int(
        usage,
        "prompt_tokens",
        "input_tokens",
        "prompt_token_count",
    )
    completion_tokens = _usage_int(
        usage,
        "completion_tokens",
        "output_tokens",
        "completion_token_count",
    )
    total_tokens = _usage_int(
        usage,
        "total_tokens",
        "total_token_count",
    )
    estimated = False

    if prompt_tokens is None:
        prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        estimated = True
    if completion_tokens is None:
        completion_tokens = _estimate_tokens(content)
        estimated = True
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
        estimated = True

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "estimated": bool(estimated),
    }


def _usage_int(usage: dict, *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
