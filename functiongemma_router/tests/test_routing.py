"""验证 E/M/H 路由分流的准确性 (需 live Ollama)."""

from __future__ import annotations

import pytest

from functiongemma_router.config import RouterConfig
from functiongemma_router.router import route

# ── 测试用例 ───────────────────────────────────────────────

# 应走 E 路径 (不调工具)
E_CASES = [
    ("闲聊", "你好"),
    ("常识", "水的沸点是多少度？"),
    ("创作", "写一首关于春天的五言绝句"),
    ("推理", "如果所有A都是B，B都是C，A是C吗？"),
    ("无意义", "asdfghjkl"),
]

# 应走 M 路径 (单轮工具)
M_CASES = [
    ("天气", "北京天气怎么样"),
    ("计算", "123 * 456 等于多少"),
    ("时间", "现在几点了"),
    ("翻译", "把 hello 翻译成中文"),
    ("搜索", "帮我搜索人工智能"),
    ("邮件", "发邮件给 test@qq.com 主题请假 正文明天休息"),
]

# 应走 H 路径 (多轮/复杂)
H_CASES = [
    ("多步比较", "查北京和上海天气，告诉我哪个更热"),
    ("多工具", "现在几点了？帮我搜索一下最新AI新闻"),
]


class TestRouting:
    """端到端路由测试 (需要 Ollama 运行)."""

    @pytest.mark.parametrize("label, prompt", E_CASES)
    def test_e_path(self, label: str, prompt: str) -> None:
        result = route(prompt)
        assert result.level == "E", f"[{label}] 期望 E, 实际 {result.level}"
        assert result.final_answer, "回答不应为空"

    @pytest.mark.parametrize("label, prompt", M_CASES)
    def test_m_path(self, label: str, prompt: str) -> None:
        result = route(prompt)
        assert result.level in ("M", "H"), f"[{label}] 期望 M/H (调工具), 实际 {result.level}"
        assert result.final_answer, "回答不应为空"

    @pytest.mark.parametrize("label, prompt", H_CASES)
    def test_h_path(self, label: str, prompt: str) -> None:
        result = route(prompt)
        # H 用例至少应该调了工具 (M 或 H)
        assert result.level in ("M", "H"), f"[{label}] 期望调工具"
        assert len(result.tool_calls_made) > 0, f"[{label}] 应该有工具调用"
        assert result.final_answer, "回答不应为空"


class TestRouterConfig:
    """配置和工具定义测试 (不需要 Ollama)."""

    def test_default_config(self) -> None:
        cfg = RouterConfig()
        assert cfg.model == "functiongemma"
        assert cfg.max_rounds == 5
        assert len(cfg.tools) == 7

    def test_custom_config(self) -> None:
        cfg = RouterConfig(model="custom-model", max_rounds=3)
        assert cfg.model == "custom-model"
        assert cfg.max_rounds == 3

    def test_tool_names_unique(self) -> None:
        cfg = RouterConfig()
        names = [t["function"]["name"] for t in cfg.tools]
        assert len(names) == len(set(names)), "工具名重复"

    def test_all_tools_have_required_fields(self) -> None:
        cfg = RouterConfig()
        for tool in cfg.tools:
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"
