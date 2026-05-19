"""Mock 工具实现 + 统一调度。"""

from __future__ import annotations

import time
from typing import Any

# ── 天气 ───────────────────────────────────────────────────

WEATHER_DB: dict[str, dict[str, object]] = {
    "北京": {"temperature": 26, "humidity": 45, "condition": "晴", "wind": "北风 3 级"},
    "上海": {"temperature": 24, "humidity": 70, "condition": "多云", "wind": "东南风 2 级"},
    "深圳": {"temperature": 29, "humidity": 85, "condition": "小雨", "wind": "南风 4 级"},
    "杭州": {"temperature": 25, "humidity": 60, "condition": "阴", "wind": "东北风 2 级"},
    "成都": {"temperature": 22, "humidity": 55, "condition": "阴转晴", "wind": "无持续风向"},
}


def get_weather(city: str) -> dict[str, Any]:
    return dict(WEATHER_DB.get(city, {"temperature": 20, "humidity": 50, "condition": "未知", "wind": "未知"}))


# ── 计算 ───────────────────────────────────────────────────

def calculate(a: float, b: float, op: str) -> dict[str, Any]:
    try:
        if op == "add":
            result = a + b
        elif op == "subtract":
            result = a - b
        elif op == "multiply":
            result = a * b
        elif op == "divide":
            if b == 0:
                return {"error": "除数不能为零"}
            result = a / b
        elif op == "power":
            result = a**b
        elif op == "modulo":
            if b == 0:
                return {"error": "取模运算中除数不能为零"}
            result = a % b
        else:
            return {"error": f"不支持的运算: {op}"}
        return {"expression": f"{a} {op} {b}", "result": result}
    except Exception as exc:
        return {"error": str(exc)}


# ── 搜索 ───────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5, language: str = "zh") -> dict[str, Any]:
    mock_results = {
        "人工智能": [
            {"title": "人工智能 - 维基百科", "url": "https://zh.wikipedia.org/wiki/人工智能"},
            {"title": "什么是人工智能？| IBM", "url": "https://www.ibm.com/think/topics/artificial-intelligence"},
        ],
        "Python": [
            {"title": "Welcome to Python.org", "url": "https://www.python.org/"},
            {"title": "Python 教程 | 菜鸟教程", "url": "https://www.runoob.com/python/"},
        ],
    }
    results = mock_results.get(query, [{"title": f"搜索结果: {query}", "url": f"https://example.com/?q={query}"}])
    return {"query": query, "results": results[:max_results], "total": len(results)}


# ── 时间 ───────────────────────────────────────────────────

def get_current_time() -> dict[str, Any]:
    t = time.localtime()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return {
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S", t),
        "weekday": weekdays[t.tm_wday],
        "timezone": "CST (UTC+8)",
    }


# ── 翻译 ───────────────────────────────────────────────────

def translate(text: str, target_language: str, source_language: str = "") -> dict[str, Any]:
    mock_translations = {
        ("你好", "en"): "Hello",
        ("谢谢", "en"): "Thank you",
        ("hello", "zh"): "你好",
        ("good morning", "zh"): "早上好",
    }
    translated = mock_translations.get(
        (text, target_language),
        f"[{text} → {target_language} 的翻译结果]",
    )
    return {
        "original": text,
        "translated": translated,
        "target_language": target_language,
        "source_language": source_language or "auto",
    }


# ── 邮件 ───────────────────────────────────────────────────

def send_email(to: str, subject: str, body: str, cc: str = "") -> dict[str, Any]:
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
        "cc": cc if cc else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── 数据库 ─────────────────────────────────────────────────

def query_database(sql: str) -> dict[str, Any]:
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return {"error": "仅允许 SELECT 查询", "sql": sql}
    if any(kw in sql_upper for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]):
        return {"error": "检测到危险操作，查询被拒绝", "sql": sql}
    return {"sql": sql, "result": [{"id": 1, "name": "示例数据"}], "row_count": 1}


# ── 统一调度 ───────────────────────────────────────────────

TOOL_MAP: dict[str, Any] = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web,
    "get_current_time": get_current_time,
    "translate": translate,
    "send_email": send_email,
    "query_database": query_database,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """根据 tool_call 执行工具并返回结果。"""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"参数错误: {exc}"}
