"""HandlerResult 数据结构 — E/M/H 路径的统一输出。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HandlerResult:
    """单次路由的完整结果。"""

    level: str                              # "E" | "M" | "H"
    final_answer: str                       # 最终自然语言回答
    rounds: int = 0                         # 模型调用次数 (不含首次)
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    # 每条: {round, name, arguments, result}
