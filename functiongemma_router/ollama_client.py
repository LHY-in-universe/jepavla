"""Ollama HTTP 客户端 — 参数化 model / tools, 可被不同路径复用。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def call_ollama(
    messages: list[dict[str, Any]],
    *,
    url: str,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """发送请求到 Ollama /api/chat, 返回完整 JSON 响应。"""
    payload: dict[str, Any] = {
        "model": model,
        "stream": stream,
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "无法连接到 Ollama。请先确认 `ollama serve` 正在运行。"
        ) from exc
