"""配置中心 — 模型地址、工具定义、路由参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Ollama 连接 ────────────────────────────────────────────

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "functiongemma"

# ── 工具定义 (OpenAI function-calling schema) ──────────────

TOOLS: list[dict[str, Any]] = [
    # 1. 天气查询
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气，返回温度、湿度、天气状况和风力",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名称，例如 北京、上海"},
                },
                "required": ["city"],
            },
        },
    },
    # 2. 计算器
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学运算。支持加减乘除幂取模",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个操作数"},
                    "b": {"type": "number", "description": "第二个操作数"},
                    "op": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "power", "modulo"],
                        "description": "运算类型",
                    },
                },
                "required": ["a", "b", "op"],
            },
        },
    },
    # 3. 搜索
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {"type": "integer", "description": "最大返回条数，默认 5"},
                    "language": {"type": "string", "enum": ["zh", "en"], "description": "搜索结果语言，默认 zh"},
                },
                "required": ["query"],
            },
        },
    },
    # 4. 获取当前时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，包括星期几和时区信息",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # 5. 翻译
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "将文本翻译为目标语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待翻译文本"},
                    "target_language": {"type": "string", "description": "目标语言，例如 en、zh、ja、fr、de，默认 en"},
                    "source_language": {"type": "string", "description": "源语言，留空则自动检测"},
                },
                "required": ["text", "target_language"],
            },
        },
    },
    # 6. 发送邮件
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送电子邮件",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"},
                    "cc": {"type": "string", "description": "抄送邮箱，多人用逗号分隔"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    # 7. 数据库查询
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "在内部数据库中执行 SQL 查询（仅限只读 SELECT）",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SELECT 查询语句"},
                },
                "required": ["sql"],
            },
        },
    },
]


@dataclass
class RouterConfig:
    """路由系统配置，全部可调。"""

    ollama_url: str = OLLAMA_URL
    model: str = MODEL

    # ── E 路径模型 (无工具直接回答) ──
    e_model: str = "qwen2.5:0.5b"

    max_rounds: int = 5

    @property
    def tools(self) -> list[dict[str, Any]]:
        return TOOLS
