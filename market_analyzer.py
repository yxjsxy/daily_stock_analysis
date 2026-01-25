# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告

数据源优先级：
1. Tushare Pro（稳定、专业）
2. Akshare（备选，东方财富接口不稳定）
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd

from config import get_config
from search_service import SearchService

logger = logging.getLogger(__name__)

# User-Agent 列表（防封禁）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    north_flow: float = 0.0             # 北向资金净流入（亿元）
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    # 主要指数代码
    MAIN_INDICES = {
        'sh000001': '上证指数',
        'sz399001': '深证成指',
        'sz399006': '创业板指',
        'sh000688': '科创50',
        'sh000016': '上证50',
        'sh000300': '沪深300',
    }
    
    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        """
        初始化大盘分析器
        
        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        
        # 初始化 Tushare API（优先数据源）
        self._tushare_api = None
        self._init_tushare()
        
    def _init_tushare(self) -> None:
        """初始化 Tushare API"""
        try:
            if self.config.tushare_token:
                import tushare as ts
                ts.set_token(self.config.tushare_token)
                self._tushare_api = ts.pro_api()
                logger.info("[大盘] Tushare API 初始化成功")
            else:
                logger.warning("[大盘] Tushare Token 未配置，将使用 Akshare 作为备选")
        except Exception as e:
            logger.warning(f"[大盘] Tushare API 初始化失败: {e}，将使用 Akshare 作为备选")
            self._tushare_api = None

    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情
        overview.indices = self._get_main_indices()
        
        # 2. 获取涨跌统计
        self._get_market_statistics(overview)
        
        # 3. 获取板块涨跌榜
        self._get_sector_rankings(overview)
        
        # 4. 获取北向资金（可选）
        # self._get_north_flow(overview)
        
        return overview

    def _set_random_user_agent(self) -> None:
        """设置随机 User-Agent（防封禁）"""
        try:
            import requests
            # 尝试设置 akshare 内部使用的 session
            random_ua = random.choice(USER_AGENTS)
            # 通过环境变量影响某些库的行为
            import os
            os.environ['USER_AGENT'] = random_ua
            logger.debug(f"[大盘] 设置 User-Agent: {random_ua[:50]}...")
        except Exception as e:
            logger.debug(f"[大盘] 设置 User-Agent 失败: {e}")

    def _call_akshare_with_retry(self, fn, name: str, attempts: int = 3):
        """带重试和防封禁策略的 akshare 调用"""
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                # 防封禁：设置随机 User-Agent
                self._set_random_user_agent()
                
                # 防封禁：请求前随机休眠 1-3 秒
                sleep_time = random.uniform(1.0, 3.0)
                logger.debug(f"[大盘] 请求前休眠 {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(f"[大盘] {name} 获取失败 (attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    # 指数退避
                    backoff_time = min(2 ** attempt + random.uniform(0, 2), 10)
                    logger.info(f"[大盘] 等待 {backoff_time:.1f}s 后重试...")
                    time.sleep(backoff_time)
        logger.error(f"[大盘] {name} 最终失败: {last_error}")
        return None
    
    # Tushare 指数代码映射
    TUSHARE_INDEX_MAP = {
        'sh000001': '000001.SH',  # 上证指数
        'sz399001': '399001.SZ',  # 深证成指
        'sz399006': '399006.SZ',  # 创业板指
        'sh000688': '000688.SH',  # 科创50
        'sh000016': '000016.SH',  # 上证50
        'sh000300': '000300.SH',  # 沪深300
    }

    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情（优先使用 Tushare）"""
        indices = []
        
        # 方案1: 优先使用 Tushare
        if self._tushare_api:
            indices = self._get_indices_from_tushare()
            if indices:
                return indices
            logger.warning("[大盘] Tushare 获取指数失败，尝试 Akshare...")
        
        # 方案2: 备选使用 Akshare
        indices = self._get_indices_from_akshare()
        return indices

    def _get_indices_from_tushare(self) -> List[MarketIndex]:
        """使用 Tushare 获取指数行情"""
        indices = []
        today = datetime.now().strftime('%Y%m%d')
        
        try:
            logger.info("[大盘] 使用 Tushare 获取指数行情...")
            
            for ak_code, name in self.MAIN_INDICES.items():
                ts_code = self.TUSHARE_INDEX_MAP.get(ak_code)
                if not ts_code:
                    continue
                
                try:
                    # 获取指数日线数据
                    df = self._tushare_api.index_daily(
                        ts_code=ts_code,
                        start_date=today,
                        end_date=today
                    )
                    
                    if df is not None and not df.empty:
                        row = df.iloc[0]
                        index = MarketIndex(
                            code=ak_code,
                            name=name,
                            current=float(row.get('close', 0) or 0),
                            change=float(row.get('change', 0) or 0),
                            change_pct=float(row.get('pct_chg', 0) or 0),
                            open=float(row.get('open', 0) or 0),
                            high=float(row.get('high', 0) or 0),
                            low=float(row.get('low', 0) or 0),
                            prev_close=float(row.get('pre_close', 0) or 0),
                            volume=float(row.get('vol', 0) or 0) * 100,  # Tushare 单位是手
                            amount=float(row.get('amount', 0) or 0) * 1000,  # Tushare 单位是千元
                        )
                        # 计算振幅
                        if index.prev_close > 0:
                            index.amplitude = (index.high - index.low) / index.prev_close * 100
                        indices.append(index)
                        
                except Exception as e:
                    error_msg = str(e)
                    # 检测到权限不足，立即放弃 Tushare，节省时间
                    if '没有接口访问权限' in error_msg or '权限' in error_msg:
                        logger.warning(f"[大盘] Tushare 积分不足，跳过指数接口")
                        return []  # 立即返回空列表，触发回退到 Akshare
                    logger.warning(f"[大盘] Tushare 获取 {name} 失败: {e}")
                    continue
            
            if indices:
                logger.info(f"[大盘] Tushare 获取到 {len(indices)} 个指数行情")
            
        except Exception as e:
            logger.error(f"[大盘] Tushare 获取指数行情失败: {e}")
        
        return indices

    def _get_indices_from_akshare(self) -> List[MarketIndex]:
        """使用 Akshare 获取指数行情（备选方案）"""
        indices = []
        
        try:
            logger.info("[大盘] 使用 Akshare 获取指数行情...")
            
            # 使用 akshare 获取指数行情（新浪财经接口，包含深市指数）
            df = self._call_akshare_with_retry(ak.stock_zh_index_spot_sina, "指数行情", attempts=2)
            
            if df is not None and not df.empty:
                for code, name in self.MAIN_INDICES.items():
                    # 查找对应指数
                    row = df[df['代码'] == code]
                    if row.empty:
                        # 尝试带前缀查找
                        row = df[df['代码'].str.contains(code)]
                    
                    if not row.empty:
                        row = row.iloc[0]
                        index = MarketIndex(
                            code=code,
                            name=name,
                            current=float(row.get('最新价', 0) or 0),
                            change=float(row.get('涨跌额', 0) or 0),
                            change_pct=float(row.get('涨跌幅', 0) or 0),
                            open=float(row.get('今开', 0) or 0),
                            high=float(row.get('最高', 0) or 0),
                            low=float(row.get('最低', 0) or 0),
                            prev_close=float(row.get('昨收', 0) or 0),
                            volume=float(row.get('成交量', 0) or 0),
                            amount=float(row.get('成交额', 0) or 0),
                        )
                        # 计算振幅
                        if index.prev_close > 0:
                            index.amplitude = (index.high - index.low) / index.prev_close * 100
                        indices.append(index)
                        
                logger.info(f"[大盘] Akshare 获取到 {len(indices)} 个指数行情")
                
        except Exception as e:
            logger.error(f"[大盘] Akshare 获取指数行情失败: {e}")
        
        return indices
    
    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计（优先使用 Tushare）"""
        
        # 方案1: 优先使用 Tushare 获取涨跌停统计
        if self._tushare_api:
            if self._get_statistics_from_tushare(overview):
                return
            logger.warning("[大盘] Tushare 获取统计失败，尝试 Akshare...")
        
        # 方案2: 备选使用 Akshare
        self._get_statistics_from_akshare(overview)

    def _get_statistics_from_tushare(self, overview: MarketOverview) -> bool:
        """使用 Tushare 获取市场统计数据"""
        today = datetime.now().strftime('%Y%m%d')
        
        try:
            logger.info("[大盘] 使用 Tushare 获取涨跌停统计...")
            
            # 获取涨停列表
            try:
                limit_up_df = self._tushare_api.limit_list(
                    trade_date=today,
                    limit_type='U'  # 涨停
                )
                if limit_up_df is not None and not limit_up_df.empty:
                    overview.limit_up_count = len(limit_up_df)
            except Exception as e:
                error_msg = str(e)
                if '没有接口访问权限' in error_msg or '权限' in error_msg:
                    logger.warning(f"[大盘] Tushare 积分不足，跳过涨跌停接口")
                    return False
                logger.debug(f"[大盘] Tushare 获取涨停列表失败: {e}")
            
            # 获取跌停列表
            try:
                limit_down_df = self._tushare_api.limit_list(
                    trade_date=today,
                    limit_type='D'  # 跌停
                )
                if limit_down_df is not None and not limit_down_df.empty:
                    overview.limit_down_count = len(limit_down_df)
            except Exception as e:
                error_msg = str(e)
                if '没有接口访问权限' in error_msg or '权限' in error_msg:
                    logger.debug(f"[大盘] Tushare 跌停接口无权限，跳过")
                else:
                    logger.debug(f"[大盘] Tushare 获取跌停列表失败: {e}")
            
            # 如果成功获取到涨跌停数据，认为成功
            if overview.limit_up_count > 0 or overview.limit_down_count > 0:
                logger.info(f"[大盘] Tushare 涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count}")
                # 其他数据（上涨下跌家数、成交额）仍需从 Akshare 补充
                self._supplement_from_akshare(overview)
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"[大盘] Tushare 获取统计失败: {e}")
            return False

    def _supplement_from_akshare(self, overview: MarketOverview):
        """从 Akshare 补充缺失的统计数据（优先新浪接口）"""
        try:
            df = None
            
            # 方案1: 优先使用新浪接口（更稳定）
            try:
                df = self._call_akshare_with_retry(ak.stock_zh_a_spot, "A股实时行情-新浪", attempts=2)
                if df is not None and not df.empty:
                    logger.info("[大盘] 使用新浪接口补充统计数据")
            except Exception as e:
                logger.debug(f"[大盘] 新浪接口获取失败: {e}")
            
            # 方案2: 尝试复用东方财富缓存
            if df is None or df.empty:
                try:
                    from data_provider.akshare_fetcher import _realtime_cache
                    if _realtime_cache.get('data') is not None and not _realtime_cache['data'].empty:
                        df = _realtime_cache['data']
                        logger.debug("[大盘] 复用东方财富缓存数据补充统计")
                except:
                    pass
            
            # 方案3: 直接调用东方财富接口
            if df is None or df.empty:
                df = self._call_akshare_with_retry(ak.stock_zh_a_spot_em, "A股实时行情-东方财富", attempts=2)
            
            if df is not None and not df.empty:
                # 新浪接口列名可能不同，需要兼容处理
                change_col = '涨跌幅' if '涨跌幅' in df.columns else 'changepercent'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    if overview.up_count == 0:
                        overview.up_count = len(df[df[change_col] > 0])
                    if overview.down_count == 0:
                        overview.down_count = len(df[df[change_col] < 0])
                    if overview.flat_count == 0:
                        overview.flat_count = len(df[df[change_col] == 0])
                
                # 成交额列名兼容
                amount_col = '成交额' if '成交额' in df.columns else 'amount'
                if amount_col in df.columns and overview.total_amount == 0:
                    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                    overview.total_amount = df[amount_col].sum() / 1e8
                
                logger.info(f"[大盘] 补充统计: 涨:{overview.up_count} 跌:{overview.down_count} 成交额:{overview.total_amount:.0f}亿")
                
        except Exception as e:
            logger.debug(f"[大盘] 补充统计数据失败: {e}")

    def _get_statistics_from_akshare(self, overview: MarketOverview):
        """使用 Akshare 获取市场统计数据（优先新浪接口）"""
        try:
            logger.info("[大盘] 使用 Akshare 获取市场涨跌统计...")
            
            df = None
            data_source = ""
            
            # 方案1: 优先使用新浪接口（更稳定，不易被封）
            try:
                df = self._call_akshare_with_retry(ak.stock_zh_a_spot, "A股实时行情-新浪", attempts=2)
                if df is not None and not df.empty:
                    data_source = "新浪"
                    logger.info("[大盘] 使用新浪接口获取涨跌统计")
            except Exception as e:
                logger.debug(f"[大盘] 新浪接口获取失败: {e}")
            
            # 方案2: 尝试复用东方财富缓存
            if df is None or df.empty:
                try:
                    from data_provider.akshare_fetcher import _realtime_cache
                    if _realtime_cache.get('data') is not None and not _realtime_cache['data'].empty:
                        df = _realtime_cache['data']
                        data_source = "东方财富(缓存)"
                        logger.info("[大盘] 复用东方财富缓存数据")
                except Exception as e:
                    logger.debug(f"[大盘] 无法复用缓存: {e}")
            
            # 方案3: 直接调用东方财富接口
            if df is None or df.empty:
                df = self._call_akshare_with_retry(ak.stock_zh_a_spot_em, "A股实时行情-东方财富", attempts=2)
                if df is not None and not df.empty:
                    data_source = "东方财富"
            
            if df is not None and not df.empty:
                # 涨跌统计 - 兼容新浪和东方财富的列名
                change_col = '涨跌幅' if '涨跌幅' in df.columns else 'changepercent'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    overview.up_count = len(df[df[change_col] > 0])
                    overview.down_count = len(df[df[change_col] < 0])
                    overview.flat_count = len(df[df[change_col] == 0])
                    
                    # 涨停跌停统计（涨跌幅 >= 9.9% 或 <= -9.9%）
                    overview.limit_up_count = len(df[df[change_col] >= 9.9])
                    overview.limit_down_count = len(df[df[change_col] <= -9.9])
                
                # 两市成交额 - 兼容不同列名
                amount_col = '成交额' if '成交额' in df.columns else 'amount'
                if amount_col in df.columns:
                    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                    overview.total_amount = df[amount_col].sum() / 1e8  # 转为亿元
                
                logger.info(f"[大盘] {data_source} 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿")
            else:
                logger.warning("[大盘] 无法获取涨跌统计数据，将使用默认值")
                
        except Exception as e:
            logger.error(f"[大盘] Akshare 获取涨跌统计失败: {e}")
    
    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜（优先使用 Tushare）"""
        
        # 方案1: 优先使用 Tushare
        if self._tushare_api:
            if self._get_sectors_from_tushare(overview):
                return
            logger.warning("[大盘] Tushare 获取板块失败，尝试 Akshare...")
        
        # 方案2: 备选使用 Akshare
        self._get_sectors_from_akshare(overview)

    def _get_sectors_from_tushare(self, overview: MarketOverview) -> bool:
        """使用 Tushare 获取板块涨跌榜"""
        today = datetime.now().strftime('%Y%m%d')
        
        try:
            logger.info("[大盘] 使用 Tushare 获取板块涨跌榜...")
            
            # 获取同花顺行业指数日线行情
            df = self._tushare_api.ths_daily(
                trade_date=today,
                fields='ts_code,name,close,pct_change'
            )
            
            if df is not None and not df.empty:
                # 过滤掉非行业指数（保留行业板块）
                df = df[df['ts_code'].str.startswith('88')]  # 同花顺行业指数以88开头
                
                if not df.empty:
                    df['pct_change'] = pd.to_numeric(df['pct_change'], errors='coerce')
                    df = df.dropna(subset=['pct_change'])
                    
                    # 涨幅前5
                    top = df.nlargest(5, 'pct_change')
                    overview.top_sectors = [
                        {'name': row['name'], 'change_pct': row['pct_change']}
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5
                    bottom = df.nsmallest(5, 'pct_change')
                    overview.bottom_sectors = [
                        {'name': row['name'], 'change_pct': row['pct_change']}
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] Tushare 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                    logger.info(f"[大盘] Tushare 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
                    return True
            
            return False
            
        except Exception as e:
            error_msg = str(e)
            # 检测到权限不足，快速返回
            if '没有接口访问权限' in error_msg or '权限' in error_msg:
                logger.warning(f"[大盘] Tushare 积分不足，跳过板块接口")
                return False
            logger.warning(f"[大盘] Tushare 获取板块失败: {e}")
            return False

    def _get_sectors_from_akshare(self, overview: MarketOverview):
        """使用 Akshare 获取板块涨跌榜（备选方案）"""
        try:
            logger.info("[大盘] 使用 Akshare 获取板块涨跌榜...")
            
            # 获取行业板块行情
            df = self._call_akshare_with_retry(ak.stock_board_industry_name_em, "行业板块行情", attempts=2)
            
            if df is not None and not df.empty:
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    df = df.dropna(subset=[change_col])
                    
                    # 涨幅前5
                    top = df.nlargest(5, change_col)
                    overview.top_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5
                    bottom = df.nsmallest(5, change_col)
                    overview.bottom_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] Akshare 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                    logger.info(f"[大盘] Akshare 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
                    
        except Exception as e:
            logger.error(f"[大盘] Akshare 获取板块涨跌榜失败: {e}")
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
            
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
                    
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
                
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []
        today = datetime.now()
        month_str = f"{today.year}年{today.month}月"
        
        # 多维度搜索
        search_queries = [
            f"A股 大盘 复盘 {month_str}",
            f"股市 行情 分析 今日 {month_str}",
            f"A股 市场 热点 板块 {month_str}",
        ]
        
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
            
            for query in search_queries:
                # 使用 search_stock_news 方法，传入"大盘"作为股票名
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name="大盘",
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        
        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        try:
            logger.info("[大盘] 调用大模型生成复盘报告...")
            
            generation_config = {
                'temperature': 0.7,
                'max_output_tokens': 2048,
            }
            
            # 根据 analyzer 使用的 API 类型调用
            if self.analyzer._use_openai:
                # 使用 OpenAI 兼容 API
                review = self.analyzer._call_openai_api(prompt, generation_config)
            else:
                # 使用 Gemini API
                response = self.analyzer._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                review = response.text.strip() if response and response.text else None
            
            if review:
                logger.info(f"[大盘] 复盘报告生成成功，长度: {len(review)} 字符")
                return review
            else:
                logger.warning("[大盘] 大模型返回为空")
                return self._generate_template_review(overview, news)
                
        except Exception as e:
            logger.error(f"[大盘] 大模型生成复盘报告失败: {e}")
            return self._generate_template_review(overview, news)
    
    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        prompt = f"""你是一位专业的A股市场分析师，请根据以下数据生成一份简洁的大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_text}

## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元
- 北向资金: {overview.north_flow:+.2f} 亿元

## 板块表现
领涨: {top_sectors_text}
领跌: {bottom_sectors_text}

## 市场新闻
{news_text if news_text else "暂无相关新闻"}

---

# 输出格式模板（请严格按此格式输出）

## 📊 {overview.date} 大盘复盘

### 一、市场总结
（2-3句话概括今日市场整体表现，包括指数涨跌、成交量变化）

### 二、指数点评
（分析上证、深证、创业板等各指数走势特点）

### 三、资金动向
（解读成交额和北向资金流向的含义）

### 四、热点解读
（分析领涨领跌板块背后的逻辑和驱动因素）

### 五、后市展望
（结合当前走势和新闻，给出明日市场预判）

### 六、风险提示
（需要关注的风险点）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
        return prompt
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        
        # 判断市场走势
        sh_index = next((idx for idx in overview.indices if idx.code == '000001'), None)
        if sh_index:
            if sh_index.change_pct > 1:
                market_mood = "强势上涨"
            elif sh_index.change_pct > 0:
                market_mood = "小幅上涨"
            elif sh_index.change_pct > -1:
                market_mood = "小幅下跌"
            else:
                market_mood = "明显下跌"
        else:
            market_mood = "震荡整理"
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_text = "、".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = "、".join([s['name'] for s in overview.bottom_sectors[:3]])
        
        report = f"""## 📊 {overview.date} 大盘复盘

### 一、市场总结
今日A股市场整体呈现**{market_mood}**态势。

### 二、主要指数
{indices_text}

### 三、涨跌统计
| 指标 | 数值 |
|------|------|
| 上涨家数 | {overview.up_count} |
| 下跌家数 | {overview.down_count} |
| 涨停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 两市成交额 | {overview.total_amount:.0f}亿 |
| 北向资金 | {overview.north_flow:+.2f}亿 |

### 四、板块表现
- **领涨**: {top_text}
- **领跌**: {bottom_text}

### 五、风险提示
市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
        return report
    
    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程
        
        Returns:
            复盘报告文本
        """
        logger.info("========== 开始大盘复盘分析 ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview()
        
        # 2. 搜索市场新闻
        news = self.search_market_news()
        
        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        
        logger.info("========== 大盘复盘分析完成 ==========")
        
        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
