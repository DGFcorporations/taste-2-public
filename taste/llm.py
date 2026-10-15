"""llm — one small, honest chat-JSON client shared by the judge (D2) and hone (F2).

Provider is chosen by ``TASTE_JUDGE_MODEL``. Keys come from the process env, or
are read (read-only, only the named provider keys) from ``~/Dev/.env``.
Nothing is ever written or logged with a key value.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("taste.llm")

_KEY_NAMES = ("OPENAI_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY", "GROQ_API_KEY")
_ENV_FILE = Path(os.environ.get("TASTE_ENV_FILE", Path.home() / "Dev" / ".env"))
TIMEOUT_S = float(os.environ.get("TASTE_LLM_TIMEOUT", "30"))


class LLMUnavailable(RuntimeError):
    """No usable provider/key — callers degrade to their deterministic path."""


def judge_model() -> str | None:
    return os.environ.get("TASTE_JUDGE_MODEL") or None


def _read_env_file(name: str) -> str | None:
    try:
        for line in _ENV_FILE.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == name:
                return v.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def _api_key(model: str) -> str:
    # Prefer the key that matches the provider implied by the model name.
    low = model.lower()
    order = list(_KEY_NAMES)
    if "qwen" in low:
        order = ["QWEN_API_KEY", "DASHSCOPE_API_KEY", *order]
    elif "groq" in low or "gpt-oss" in low:
        order = ["GROQ_API_KEY", *order]
    for name in order:
        val = os.environ.get(name) or _read_env_file(name)
        if val:
            return val
    raise LLMUnavailable("no provider API key in env or ~/Dev/.env")


def _base_url(model: str) -> str:
    if os.environ.get("OPENAI_BASE_URL"):
        return os.environ["OPENAI_BASE_URL"]
    low = model.lower()
    if "qwen" in low:
        return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    if "groq" in low or "gpt-oss" in low:
        return "https://api.groq.com/openai/v1"
    return "https://api.openai.com/v1"


def parse_json(raw: str) -> dict:
    """Parse a JSON object from model output, tolerating ``` fences and prose."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def chat_json(model: str, system: str, prompt: str) -> dict:
    """Single JSON-mode chat completion. Raises on any failure (never swallows)."""
    import httpx

    headers = {"Authorization": f"Bearer {_api_key(model)}",
               "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    with httpx.Client(timeout=TIMEOUT_S) as client:
        r = client.post(f"{_base_url(model).rstrip('/')}/chat/completions",
                        headers=headers, json=payload)
        r.raise_for_status()
        return parse_json(r.json()["choices"][0]["message"]["content"])
