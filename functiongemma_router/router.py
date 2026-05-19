"""核心路由: 一次 FunctionGemma 调用, tool_calls 同时判定 E/M/H。"""

from __future__ import annotations

import json
from typing import Any

from functiongemma_router.config import RouterConfig
from functiongemma_router.handlers import HandlerResult
from functiongemma_router.ollama_client import call_ollama
from functiongemma_router.tools import execute_tool


def route(
    user_prompt: str,
    config: RouterConfig | None = None,
) -> HandlerResult:
    """调用一次 FunctionGemma, 根据 tool_calls 自动分流。

    E: tool_calls 为空 → 直接返回 (只调了 1 次模型)
    M: 有 tool_calls → 执行 → 回传, 无新 tool_calls (共 2 次)
    H: 回传后还有 tool_calls → 多轮循环 (3~N 次)
    """
    cfg = config or RouterConfig()

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    tool_calls_made: list[dict[str, Any]] = []

    # ── 第 1 次调用: 用不用工具? 多复杂? 一次回答两个问题 ──
    r1 = call_ollama(messages, url=cfg.ollama_url, model=cfg.model, tools=cfg.tools)
    msg1 = r1.get("message", {})
    tcs1 = msg1.get("tool_calls", [])

    # E 路径: 模型认为不需要工具 → 交给独立模型回答
    if not tcs1:
        return _handle_e(cfg, user_prompt)

    # 有工具 → 执行, 追加 assistant 消息
    messages.append({"role": "assistant", "content": msg1.get("content", ""), "tool_calls": tcs1})
    _run_tools(messages, tcs1, tool_calls_made, round_idx=1)

    # ── 第 2 次调用: 回传工具结果, 判断 M or H ──
    r2 = call_ollama(messages, url=cfg.ollama_url, model=cfg.model, tools=cfg.tools)
    msg2 = r2.get("message", {})
    tcs2 = msg2.get("tool_calls", [])

    # M 路径: 一轮工具就够了
    if not tcs2:
        return HandlerResult(
            level="M", final_answer=str(msg2.get("content", "")).strip(),
            rounds=1, tool_calls_made=tool_calls_made,
        )

    # H 路径: 模型还想继续调工具 → 多轮循环
    current_round = 2
    messages.append({"role": "assistant", "content": msg2.get("content", ""), "tool_calls": tcs2})
    _run_tools(messages, tcs2, tool_calls_made, round_idx=current_round)

    for current_round in range(3, cfg.max_rounds + 1):
        rn = call_ollama(messages, url=cfg.ollama_url, model=cfg.model, tools=cfg.tools)
        msgn = rn.get("message", {})
        tcsn = msgn.get("tool_calls", [])

        if not tcsn:
            return HandlerResult(
                level="H", final_answer=str(msgn.get("content", "")).strip(),
                rounds=current_round - 1, tool_calls_made=tool_calls_made,
            )

        messages.append({"role": "assistant", "content": msgn.get("content", ""), "tool_calls": tcsn})
        _run_tools(messages, tcsn, tool_calls_made, round_idx=current_round)

    # 达到最大轮数, 强制获取最终回答
    messages.append({"role": "user", "content": "请根据之前的工具调用结果，给出最终回答。"})
    r_end = call_ollama(messages, url=cfg.ollama_url, model=cfg.model, tools=cfg.tools)
    return HandlerResult(
        level="H",
        final_answer=str(r_end.get("message", {}).get("content", "")).strip(),
        rounds=cfg.max_rounds,
        tool_calls_made=tool_calls_made,
    )


def _run_tools(
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    collector: list[dict[str, Any]],
    round_idx: int,
) -> None:
    """执行 tool_calls, 追加 tool 消息, 记录到 collector。"""
    for tc in tool_calls:
        fn = tc.get("function", {})
        fn_name = fn.get("name", "unknown")
        fn_args: dict[str, Any] = fn.get("arguments", {})

        result = execute_tool(fn_name, fn_args)
        collector.append({"round": round_idx, "name": fn_name, "arguments": fn_args, "result": result})
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})


def _handle_e(cfg: RouterConfig, user_prompt: str) -> HandlerResult:
    """E 路径: 用独立模型回答, 不携带工具定义。"""
    r = call_ollama(
        [{"role": "user", "content": user_prompt}],
        url=cfg.ollama_url,
        model=cfg.e_model,
        tools=None,
    )
    content = str(r.get("message", {}).get("content", "")).strip()
    return HandlerResult(level="E", final_answer=content)
