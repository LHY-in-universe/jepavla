"""FunctionGemma 模型路由系统 — 一次调用, 两个维度同时判断 (E/M/H)."""

from functiongemma_router.config import RouterConfig
from functiongemma_router.handlers import HandlerResult
from functiongemma_router.router import route

__all__ = ["route", "RouterConfig", "HandlerResult"]
