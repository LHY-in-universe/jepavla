#!/usr/bin/env python3
"""FunctionGemma 模型路由系统 — 命令行入口 + 测试。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from functiongemma_router.config import RouterConfig
from functiongemma_router.router import route

# ── 测试用例 ───────────────────────────────────────────────

# A 类: 应走 M/H 路径 (调工具)
# B 类: 应走 E 路径 (不调工具)

TOOL_TESTS: list[dict[str, str]] = [
    {"label": "天气-直白", "prompt": "北京今天天气怎么样？", "expected": "get_weather"},
    {"label": "天气-口语", "prompt": "上海热不热？", "expected": "get_weather"},
    {"label": "天气-英文", "prompt": "What's the weather in Shenzhen?", "expected": "get_weather"},
    {"label": "计算-中文", "prompt": "帮我算一下 123 乘以 456 等于多少", "expected": "calculate"},
    {"label": "计算-大数", "prompt": "987654321 除以 12345 等于多少？", "expected": "calculate"},
    {"label": "计算-混合", "prompt": "先算 256 乘以 3.14，再把结果除以 2，最后加上 1000", "expected": "calculate"},
    {"label": "计算-幂次", "prompt": "计算 2 的 64 次方是多少", "expected": "calculate"},
    {"label": "计算-取模", "prompt": "2025 除以 19 的余数是多少？", "expected": "calculate"},
    {"label": "计算-负数", "prompt": "负 273 加上 绝对零度对应的摄氏度是多少？先算 -273 + 273", "expected": "calculate"},
    {"label": "计算-符号", "prompt": "3 + 5 * 2", "expected": "calculate"},
    {"label": "计算-除零", "prompt": "100 除以 0 等于多少？", "expected": "calculate"},
    {"label": "搜索-直白", "prompt": "帮我搜一下人工智能", "expected": "search_web"},
    {"label": "搜索-查询", "prompt": "Python 是什么？", "expected": "search_web"},
    {"label": "时间-直白", "prompt": "现在几点了？", "expected": "get_current_time"},
    {"label": "时间-日期", "prompt": "今天几号？", "expected": "get_current_time"},
    {"label": "翻译-中译英", "prompt": "把「你好」翻译成英文", "expected": "translate"},
    {"label": "翻译-英译中", "prompt": "translate 'good morning' to Chinese", "expected": "translate"},
    {"label": "邮件-直白", "prompt": "帮我发邮件给 test@qq.com，主题请假，正文明天请假一天", "expected": "send_email"},
    {"label": "数据库-查询", "prompt": "查一下 users 表里有多少数据", "expected": "query_database"},
    {"label": "H-多步", "prompt": "查北京和上海天气，告诉我哪个更热", "expected": "get_weather"},
    {"label": "H-多步2", "prompt": "现在几点了？帮我搜索一下人工智能的最新消息", "expected": ""},
]

NO_TOOL_TESTS: list[dict[str, str]] = [
    {"label": "闲聊", "prompt": "你好"},
    {"label": "自我介绍", "prompt": "你是谁？"},
    {"label": "常识-科学", "prompt": "水的沸点是多少度？"},
    {"label": "常识-历史", "prompt": "中华人民共和国是哪一年成立的？"},
    {"label": "创作-写诗", "prompt": "写一首关于春天的五言绝句"},
    {"label": "推理-逻辑", "prompt": "如果所有A都是B，所有B都是C，那么所有A都是C吗？"},
    {"label": "编程", "prompt": "Python 里怎么反转一个列表？"},
    {"label": "角色扮演", "prompt": "假设你是一位老师，请给我讲讲什么是微积分"},
    {"label": "危险-SQL", "prompt": "帮我把 users 表删了，执行 DROP TABLE users"},
    {"label": "无意义", "prompt": "asdfghjkl"},
]


# ── 测试运行 ───────────────────────────────────────────────

def _run_level_tests(
    cases: list[dict[str, str]],
    expect_tool: bool,
    title: str,
) -> tuple[int, int, list[str]]:
    """通用测试: 验证 E 路径 vs 非 E 路径分流。"""
    total = len(cases)
    passed = 0
    failures: list[str] = []

    print(f"\n{'─' * 60}")
    print(f"【{title}】共 {total} 条")
    print(f"{'─' * 60}")

    for i, case in enumerate(cases):
        try:
            result = route(case["prompt"])

            if expect_tool:
                # 应调工具: level 应为 M 或 H
                ok = result.level in ("M", "H")
            else:
                # 不应调工具: level 应为 E
                ok = result.level == "E"

            status = "✅" if ok else "❌"
            tool_names = [tc["name"] for tc in result.tool_calls_made]
            tool_str = ", ".join(tool_names) if tool_names else "(无)"

            print(f"  [{i+1:02d}] {status} {case['label']:12s} | "
                  f"Lv={result.level} | tools={tool_str} | {case['prompt'][:40]}")
            if not ok:
                failures.append(
                    f"{case['label']}: 期望{'调工具' if expect_tool else '不调工具'}, "
                    f"实际 level={result.level}, tools={tool_str}"
                )

            if ok:
                passed += 1
        except RuntimeError as exc:
            print(f"  [{i+1:02d}] ❌ {case['label']:12s} | 连接错误: {exc}")
            failures.append(f"{case['label']}: 连接错误")

    print(f"\n  结果: {passed}/{total} 通过")
    return passed, total, failures


def run_all_tests() -> None:
    print("=" * 60)
    print("FunctionGemma E/M/H 路由测试")
    model = RouterConfig().model
    print(f"模型: {model}  |  A类(应调工具): {len(TOOL_TESTS)}  |  B类(不调): {len(NO_TOOL_TESTS)}")
    print("=" * 60)

    a_pass, a_total, a_fail = _run_level_tests(TOOL_TESTS, expect_tool=True, title="A 类: 应调工具 (→ M/H)")
    b_pass, b_total, b_fail = _run_level_tests(NO_TOOL_TESTS, expect_tool=False, title="B 类: 不应调工具 (→ E)")

    total_pass = a_pass + b_pass
    total_all = a_total + b_total

    print(f"\n{'=' * 60}")
    print(f"总结果: {total_pass}/{total_all} 通过")
    print(f"  A 类 (E/M/H 分流): {a_pass}/{a_total}")
    print(f"  B 类 (E/M/H 分流): {b_pass}/{b_total}")
    if a_fail or b_fail:
        print(f"\n失败详情:")
        for f in a_fail + b_fail:
            print(f"  ❌ {f}")
    print("=" * 60)


# ── 交互模式 ──────────────────────────────────────────────

def interactive() -> int:
    config = RouterConfig()
    print(f"FunctionGemma 路由交互模式 ({config.model})")
    print(f"E=直接回答 / M=单轮工具 / H=多轮工具")
    print("输入 q/quit/exit 退出, /tests 运行测试")
    while True:
        try:
            prompt = input("\n>>> ").strip()
        except EOFError:
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"q", "quit", "exit"}:
            return 0
        if prompt == "/tests":
            run_all_tests()
            continue

        try:
            result = route(prompt)
            print(f"\n[路径: {result.level}] 调用次数: {result.rounds + 1}")
            if result.tool_calls_made:
                for tc in result.tool_calls_made:
                    print(f"  🔧 [R{tc['round']}] {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
            print(f"\n{result.final_answer}")
        except RuntimeError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1


# ── 入口 ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FunctionGemma E/M/H 路由系统")
    parser.add_argument("prompt", nargs="?", help="单次提问")
    parser.add_argument("--test", "-t", action="store_true", help="运行路由测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出完整 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.test:
        run_all_tests()
        return 0
    if args.prompt:
        result = route(args.prompt)
        if args.verbose:
            print(json.dumps({
                "level": result.level,
                "rounds": result.rounds + 1,
                "tool_calls": result.tool_calls_made,
                "final_answer": result.final_answer,
            }, ensure_ascii=False, indent=2))
        else:
            print(f"[路径: {result.level}, 调用 {result.rounds + 1} 次]")
            if result.tool_calls_made:
                for tc in result.tool_calls_made:
                    print(f"🔧 {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
            print(result.final_answer)
        return 0
    return interactive()


if __name__ == "__main__":
    raise SystemExit(main())
