# -*- coding: utf-8 -*-
"""
===================================
数据源策略层 - 包初始化
===================================

本包实现策略模式管理多个数据源，实现：
1. 统一的数据获取接口
2. 自动故障切换
3. 防封禁流控策略

数据源优先级（2026-01 调整）：
【配置了 TUSHARE_TOKEN 时】
1. TushareFetcher (Priority 0) - 🔥 最高优先级（稳定、专业）
2. BaostockFetcher (Priority 2) - 证券宝（免费、稳定）
3. AkshareFetcher (Priority 3) - 东方财富（GitHub Actions 环境不稳定）
4. EfinanceFetcher (Priority 4) - 东方财富（GitHub Actions 环境不稳定）
5. YfinanceFetcher (Priority 5) - Yahoo Finance（兜底）

【未配置 TUSHARE_TOKEN 时】
1. BaostockFetcher (Priority 2) - 证券宝（免费、稳定）
2. AkshareFetcher (Priority 3) - 东方财富
3. EfinanceFetcher (Priority 4) - 东方财富
4. YfinanceFetcher (Priority 5) - Yahoo Finance
5. TushareFetcher (Priority 99) - 不可用

提示：优先级数字越小越优先；东方财富接口在 GitHub Actions 中易被封禁，已降级
"""

from .base import BaseFetcher, DataFetcherManager
from .efinance_fetcher import EfinanceFetcher
from .akshare_fetcher import AkshareFetcher
from .tushare_fetcher import TushareFetcher
from .baostock_fetcher import BaostockFetcher
from .yfinance_fetcher import YfinanceFetcher

__all__ = [
    'BaseFetcher',
    'DataFetcherManager',
    'EfinanceFetcher',
    'AkshareFetcher',
    'TushareFetcher',
    'BaostockFetcher',
    'YfinanceFetcher',
]
