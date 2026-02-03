# -*- coding: utf-8 -*-
"""
===================================
缠论分析器 - 缠中说禅技术分析
===================================

缠论核心概念：
1. 分型：顶分型（中间K线最高）、底分型（中间K线最低）
2. 笔：连接相邻分型，顶→底为下跌笔，底→顶为上涨笔
3. 线段：至少3笔构成的更高级别走势
4. 中枢：至少3笔重叠区域（ZG中枢上沿、ZD中枢下沿）
5. 背驰：MACD辅助判断趋势力度衰减
6. 买卖点：
   - 一买：下跌趋势背驰后的第一个买点
   - 二买：回踩不破一买低点
   - 三买：离开中枢后回踩不进中枢
   - 一卖、二卖、三卖同理

使用方式：
    analyzer = ChanAnalyzer()
    result = analyzer.analyze(df, '000001')
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FenXingType(Enum):
    """分型类型"""
    TOP = "顶分型"      # 中间K线高点最高
    BOTTOM = "底分型"   # 中间K线低点最低
    NONE = "无分型"


class BiDirection(Enum):
    """笔方向"""
    UP = "上升笔"       # 底分型 → 顶分型
    DOWN = "下降笔"     # 顶分型 → 底分型


class XianDuanDirection(Enum):
    """线段方向"""
    UP = "上升线段"
    DOWN = "下降线段"


class TrendType(Enum):
    """走势类型"""
    UP_TREND = "上涨趋势"      # 中枢依次升高
    DOWN_TREND = "下跌趋势"    # 中枢依次降低
    CONSOLIDATION = "盘整"    # 只有一个中枢


class BeiChiType(Enum):
    """背驰类型"""
    TREND_BEICHI = "趋势背驰"     # 两个同向走势段对比
    PAN_ZHENG_BEICHI = "盘整背驰" # 同一中枢内对比
    NONE = "无背驰"


class BuySellPoint(Enum):
    """买卖点类型"""
    BUY_1 = "一买"    # 趋势背驰后
    BUY_2 = "二买"    # 回踩不破一买低点
    BUY_3 = "三买"    # 离开中枢后回踩不进中枢
    SELL_1 = "一卖"   # 趋势背驰后
    SELL_2 = "二卖"   # 反弹不破一卖高点
    SELL_3 = "三卖"   # 离开中枢后反弹不进中枢
    NONE = "无买卖点"


@dataclass
class FenXing:
    """分型数据类"""
    index: int              # 在DataFrame中的索引
    type: FenXingType       # 分型类型
    high: float             # 分型高点（顶分型取最高）
    low: float              # 分型低点（底分型取最低）
    date: str               # 日期
    fx_value: float = 0.0   # 分型值（顶分型=high，底分型=low）
    
    def __post_init__(self):
        if self.type == FenXingType.TOP:
            self.fx_value = self.high
        elif self.type == FenXingType.BOTTOM:
            self.fx_value = self.low


@dataclass
class Bi:
    """笔数据类"""
    start_fx: FenXing       # 起始分型
    end_fx: FenXing         # 结束分型
    direction: BiDirection  # 笔方向
    high: float = 0.0       # 笔的最高点
    low: float = 0.0        # 笔的最低点
    power: float = 0.0      # 笔的力度（MACD面积）
    
    def __post_init__(self):
        self.high = max(self.start_fx.high, self.end_fx.high)
        self.low = min(self.start_fx.low, self.end_fx.low)


@dataclass
class ZhongShu:
    """中枢数据类"""
    bis: List[Bi]           # 组成中枢的笔
    zg: float               # 中枢上沿（ZG）
    zd: float               # 中枢下沿（ZD）
    gg: float               # 中枢高高点
    dd: float               # 中枢低低点
    direction: Optional[BiDirection] = None  # 中枢形成方向
    
    @property
    def range(self) -> float:
        """中枢区间"""
        return self.zg - self.zd
    
    @property
    def center(self) -> float:
        """中枢中心点"""
        return (self.zg + self.zd) / 2


@dataclass
class XianDuan:
    """线段数据类"""
    bis: List[Bi]           # 组成线段的笔
    direction: XianDuanDirection  # 线段方向
    high: float = 0.0       # 线段最高点
    low: float = 0.0        # 线段最低点
    
    def __post_init__(self):
        if self.bis:
            self.high = max(bi.high for bi in self.bis)
            self.low = min(bi.low for bi in self.bis)


@dataclass
class ChanAnalysisResult:
    """缠论分析结果"""
    code: str
    
    # === 分型信息 ===
    fenxings: List[FenXing] = field(default_factory=list)
    last_fenxing: Optional[FenXing] = None    # 最近的分型
    fenxing_summary: str = ""                  # 分型摘要
    
    # === 笔信息 ===
    bis: List[Bi] = field(default_factory=list)
    last_bi: Optional[Bi] = None              # 最近的笔
    bi_summary: str = ""                       # 笔摘要
    current_bi_direction: str = ""            # 当前笔方向
    
    # === 线段信息 ===
    xianduans: List[XianDuan] = field(default_factory=list)
    last_xianduan: Optional[XianDuan] = None  # 最近的线段
    xianduan_summary: str = ""                 # 线段摘要
    
    # === 中枢信息 ===
    zhongshus: List[ZhongShu] = field(default_factory=list)
    current_zhongshu: Optional[ZhongShu] = None  # 当前中枢
    zhongshu_summary: str = ""                   # 中枢摘要
    price_position: str = ""                     # 价格相对中枢位置
    
    # === 背驰信息 ===
    beichi_type: BeiChiType = BeiChiType.NONE
    beichi_summary: str = ""
    macd_divergence: bool = False              # MACD是否背离
    
    # === 买卖点 ===
    buy_sell_point: BuySellPoint = BuySellPoint.NONE
    buy_sell_reason: str = ""
    
    # === 趋势判断 ===
    trend_type: TrendType = TrendType.CONSOLIDATION
    trend_summary: str = ""
    
    # === 综合分析 ===
    chan_score: int = 50                       # 缠论评分 0-100
    operation_suggestion: str = ""             # 操作建议
    key_levels: Dict[str, float] = field(default_factory=dict)  # 关键点位
    analysis_summary: str = ""                 # 综合分析摘要
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            # 分型
            'fenxing_count': len(self.fenxings),
            'last_fenxing': self.last_fenxing.type.value if self.last_fenxing else '无',
            'fenxing_summary': self.fenxing_summary,
            # 笔
            'bi_count': len(self.bis),
            'current_bi_direction': self.current_bi_direction,
            'bi_summary': self.bi_summary,
            # 线段
            'xianduan_count': len(self.xianduans),
            'xianduan_summary': self.xianduan_summary,
            # 中枢
            'zhongshu_count': len(self.zhongshus),
            'zhongshu_summary': self.zhongshu_summary,
            'price_position': self.price_position,
            'current_zg': self.current_zhongshu.zg if self.current_zhongshu else 0,
            'current_zd': self.current_zhongshu.zd if self.current_zhongshu else 0,
            # 背驰
            'beichi_type': self.beichi_type.value,
            'beichi_summary': self.beichi_summary,
            'macd_divergence': self.macd_divergence,
            # 买卖点
            'buy_sell_point': self.buy_sell_point.value,
            'buy_sell_reason': self.buy_sell_reason,
            # 趋势
            'trend_type': self.trend_type.value,
            'trend_summary': self.trend_summary,
            # 综合
            'chan_score': self.chan_score,
            'operation_suggestion': self.operation_suggestion,
            'key_levels': self.key_levels,
            'analysis_summary': self.analysis_summary,
        }


class ChanAnalyzer:
    """
    缠论分析器
    
    基于缠中说禅理论进行技术分析：
    1. 识别分型（顶分型、底分型）
    2. 构建笔（连接分型）
    3. 识别线段
    4. 定位中枢
    5. 判断背驰
    6. 确定买卖点
    """
    
    # 分型之间最少K线数（标准缠论要求至少1根独立K线）
    MIN_K_BETWEEN_FX = 4  # 使用4作为标准（包含分型本身3根）
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化分析器
        
        Args:
            strict_mode: 是否使用严格模式（更严格的分型确认）
        """
        self.strict_mode = strict_mode
    
    def analyze(self, df: pd.DataFrame, code: str) -> ChanAnalysisResult:
        """
        分析股票的缠论形态
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            code: 股票代码
            
        Returns:
            ChanAnalysisResult 分析结果
        """
        result = ChanAnalysisResult(code=code)
        
        if df is None or df.empty or len(df) < 10:
            logger.warning(f"[{code}] 数据不足，无法进行缠论分析")
            result.analysis_summary = "数据不足，无法完成缠论分析"
            return result
        
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 1. K线包含处理（合并包含关系）
        df_processed = self._process_include(df)
        
        # 2. 计算MACD（用于背驰判断）
        df_processed = self._calculate_macd(df_processed)
        
        # 3. 识别分型
        fenxings = self._identify_fenxing(df_processed)
        result.fenxings = fenxings
        if fenxings:
            result.last_fenxing = fenxings[-1]
            result.fenxing_summary = self._summarize_fenxings(fenxings)
        
        # 4. 构建笔
        bis = self._build_bi(fenxings, df_processed)
        result.bis = bis
        if bis:
            result.last_bi = bis[-1]
            result.current_bi_direction = bis[-1].direction.value
            result.bi_summary = self._summarize_bis(bis)
        
        # 5. 识别线段
        xianduans = self._build_xianduan(bis)
        result.xianduans = xianduans
        if xianduans:
            result.last_xianduan = xianduans[-1]
            result.xianduan_summary = self._summarize_xianduans(xianduans)
        
        # 6. 定位中枢
        zhongshus = self._identify_zhongshu(bis)
        result.zhongshus = zhongshus
        if zhongshus:
            result.current_zhongshu = zhongshus[-1]
            current_price = float(df.iloc[-1]['close'])
            result.price_position = self._get_price_position(current_price, zhongshus[-1])
            result.zhongshu_summary = self._summarize_zhongshu(zhongshus[-1], current_price)
        
        # 7. 判断背驰
        beichi_type, beichi_summary, macd_div = self._check_beichi(bis, df_processed)
        result.beichi_type = beichi_type
        result.beichi_summary = beichi_summary
        result.macd_divergence = macd_div
        
        # 8. 判断趋势
        result.trend_type, result.trend_summary = self._analyze_trend(zhongshus, bis)
        
        # 9. 确定买卖点
        result.buy_sell_point, result.buy_sell_reason = self._identify_buy_sell_point(
            result, df_processed
        )
        
        # 10. 计算关键点位
        result.key_levels = self._calculate_key_levels(result, df_processed)
        
        # 11. 综合评分和建议
        result.chan_score = self._calculate_score(result)
        result.operation_suggestion = self._generate_suggestion(result)
        result.analysis_summary = self._generate_summary(result)
        
        return result
    
    def _process_include(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        K线包含处理
        
        包含关系：当两根K线高低点存在包含关系时，合并为一根
        - 上涨中取高高、低高
        - 下跌中取低低、高低
        """
        df = df.copy()
        
        # 添加处理后的高低点列
        df['high_p'] = df['high'].astype(float)
        df['low_p'] = df['low'].astype(float)
        
        # 处理包含关系
        i = 1
        while i < len(df):
            prev_high = df.loc[df.index[i-1], 'high_p']
            prev_low = df.loc[df.index[i-1], 'low_p']
            curr_high = df.loc[df.index[i], 'high_p']
            curr_low = df.loc[df.index[i], 'low_p']
            
            # 判断包含关系
            is_include = (
                (prev_high >= curr_high and prev_low <= curr_low) or
                (curr_high >= prev_high and curr_low <= prev_low)
            )
            
            if is_include:
                # 判断方向（通过前两根K线的趋势）
                if i >= 2:
                    prev2_high = df.loc[df.index[i-2], 'high_p']
                    is_up = prev_high > prev2_high
                else:
                    is_up = curr_high > prev_high
                
                if is_up:
                    # 上涨中取高高、低高
                    new_high = max(prev_high, curr_high)
                    new_low = max(prev_low, curr_low)
                else:
                    # 下跌中取低低、高低
                    new_high = min(prev_high, curr_high)
                    new_low = min(prev_low, curr_low)
                
                df.loc[df.index[i], 'high_p'] = new_high
                df.loc[df.index[i], 'low_p'] = new_low
            
            i += 1
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        df = df.copy()
        
        close = df['close'].astype(float)
        
        # EMA计算
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        # DIF
        df['macd_dif'] = ema12 - ema26
        
        # DEA (信号线)
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        
        # MACD柱
        df['macd_bar'] = 2 * (df['macd_dif'] - df['macd_dea'])
        
        return df
    
    def _identify_fenxing(self, df: pd.DataFrame) -> List[FenXing]:
        """
        识别分型
        
        顶分型：中间K线高点比前后都高
        底分型：中间K线低点比前后都低
        """
        fenxings = []
        
        for i in range(1, len(df) - 1):
            prev_high = df.loc[df.index[i-1], 'high_p']
            curr_high = df.loc[df.index[i], 'high_p']
            next_high = df.loc[df.index[i+1], 'high_p']
            
            prev_low = df.loc[df.index[i-1], 'low_p']
            curr_low = df.loc[df.index[i], 'low_p']
            next_low = df.loc[df.index[i+1], 'low_p']
            
            date_val = df.loc[df.index[i], 'date']
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)
            
            # 顶分型判断
            if curr_high > prev_high and curr_high > next_high:
                fx = FenXing(
                    index=i,
                    type=FenXingType.TOP,
                    high=float(curr_high),
                    low=float(curr_low),
                    date=date_str
                )
                fenxings.append(fx)
            
            # 底分型判断
            elif curr_low < prev_low and curr_low < next_low:
                fx = FenXing(
                    index=i,
                    type=FenXingType.BOTTOM,
                    high=float(curr_high),
                    low=float(curr_low),
                    date=date_str
                )
                fenxings.append(fx)
        
        # 过滤：相邻分型必须是顶底交替
        filtered = self._filter_fenxing(fenxings)
        
        return filtered
    
    def _filter_fenxing(self, fenxings: List[FenXing]) -> List[FenXing]:
        """
        过滤分型，确保顶底交替
        
        如果出现连续顶分型，取最高的
        如果出现连续底分型，取最低的
        """
        if not fenxings:
            return []
        
        filtered = [fenxings[0]]
        
        for fx in fenxings[1:]:
            last = filtered[-1]
            
            if fx.type == last.type:
                # 同类型分型，取极值
                if fx.type == FenXingType.TOP:
                    if fx.high > last.high:
                        filtered[-1] = fx
                else:  # BOTTOM
                    if fx.low < last.low:
                        filtered[-1] = fx
            else:
                # 不同类型，检查是否满足最小K线间隔
                if fx.index - last.index >= self.MIN_K_BETWEEN_FX:
                    filtered.append(fx)
                else:
                    # 间隔太近，保留更极端的
                    if fx.type == FenXingType.TOP and fx.high > last.low:
                        # 顶分型高于前底分型低点，可能形成有效笔
                        pass  # 暂时不处理
        
        return filtered
    
    def _build_bi(self, fenxings: List[FenXing], df: pd.DataFrame) -> List[Bi]:
        """
        构建笔
        
        连接相邻的顶底分型形成笔
        """
        bis = []
        
        if len(fenxings) < 2:
            return bis
        
        for i in range(len(fenxings) - 1):
            start_fx = fenxings[i]
            end_fx = fenxings[i + 1]
            
            # 确定笔方向
            if start_fx.type == FenXingType.BOTTOM and end_fx.type == FenXingType.TOP:
                direction = BiDirection.UP
            elif start_fx.type == FenXingType.TOP and end_fx.type == FenXingType.BOTTOM:
                direction = BiDirection.DOWN
            else:
                continue  # 无效的分型组合
            
            # 计算笔的MACD力度（面积）
            power = self._calculate_bi_power(start_fx.index, end_fx.index, df)
            
            bi = Bi(
                start_fx=start_fx,
                end_fx=end_fx,
                direction=direction,
                power=power
            )
            bis.append(bi)
        
        return bis
    
    def _calculate_bi_power(self, start_idx: int, end_idx: int, df: pd.DataFrame) -> float:
        """计算笔的MACD力度（面积）"""
        if 'macd_bar' not in df.columns:
            return 0.0
        
        try:
            macd_slice = df.loc[start_idx:end_idx, 'macd_bar']
            return abs(macd_slice.sum())
        except:
            return 0.0
    
    def _build_xianduan(self, bis: List[Bi]) -> List[XianDuan]:
        """
        构建线段
        
        线段由至少3笔构成，且要满足特征序列的破坏
        简化版：每3笔构成一个线段
        """
        xianduans = []
        
        if len(bis) < 3:
            return xianduans
        
        i = 0
        while i < len(bis) - 2:
            # 取连续3笔
            segment_bis = bis[i:i+3]
            
            # 确定线段方向（以第一笔方向为准）
            first_bi = segment_bis[0]
            if first_bi.direction == BiDirection.UP:
                direction = XianDuanDirection.UP
            else:
                direction = XianDuanDirection.DOWN
            
            xd = XianDuan(
                bis=segment_bis,
                direction=direction
            )
            xianduans.append(xd)
            
            i += 2  # 步进2，允许线段有重叠
        
        return xianduans
    
    def _identify_zhongshu(self, bis: List[Bi]) -> List[ZhongShu]:
        """
        识别中枢
        
        中枢定义：至少3笔的重叠区域
        ZG = min(各笔高点)
        ZD = max(各笔低点)
        如果 ZG > ZD，则形成有效中枢
        """
        zhongshus = []
        
        if len(bis) < 3:
            return zhongshus
        
        i = 0
        while i < len(bis) - 2:
            # 尝试从当前位置构建中枢
            zg = min(bis[i].high, bis[i+1].high, bis[i+2].high)
            zd = max(bis[i].low, bis[i+1].low, bis[i+2].low)
            
            if zg > zd:  # 有效中枢
                # 计算中枢的完整范围
                zs_bis = [bis[i], bis[i+1], bis[i+2]]
                gg = max(bi.high for bi in zs_bis)
                dd = min(bi.low for bi in zs_bis)
                
                # 尝试扩展中枢（加入后续满足条件的笔）
                j = i + 3
                while j < len(bis):
                    new_zg = min(zg, bis[j].high)
                    new_zd = max(zd, bis[j].low)
                    
                    if new_zg > new_zd:
                        # 可以扩展
                        zg = new_zg
                        zd = new_zd
                        zs_bis.append(bis[j])
                        gg = max(gg, bis[j].high)
                        dd = min(dd, bis[j].low)
                        j += 1
                    else:
                        break
                
                zs = ZhongShu(
                    bis=zs_bis,
                    zg=zg,
                    zd=zd,
                    gg=gg,
                    dd=dd,
                    direction=bis[i].direction
                )
                zhongshus.append(zs)
                
                i = j  # 跳过已构建中枢的笔
            else:
                i += 1
        
        return zhongshus
    
    def _get_price_position(self, price: float, zs: ZhongShu) -> str:
        """获取价格相对中枢的位置"""
        if price > zs.zg:
            pct = (price - zs.zg) / zs.zg * 100
            return f"中枢上方 (+{pct:.1f}%)"
        elif price < zs.zd:
            pct = (zs.zd - price) / zs.zd * 100
            return f"中枢下方 (-{pct:.1f}%)"
        else:
            # 在中枢内，计算位置
            range_pct = (price - zs.zd) / (zs.zg - zs.zd) * 100
            return f"中枢内 ({range_pct:.0f}%位置)"
    
    def _check_beichi(
        self, 
        bis: List[Bi], 
        df: pd.DataFrame
    ) -> Tuple[BeiChiType, str, bool]:
        """
        判断背驰
        
        背驰类型：
        1. 趋势背驰：两段同向走势，后一段力度弱于前一段
        2. 盘整背驰：同一中枢内，后一段力度弱于前一段
        
        通过MACD面积对比判断
        """
        if len(bis) < 5:
            return BeiChiType.NONE, "笔数不足，无法判断背驰", False
        
        # 取最后5笔进行分析
        recent_bis = bis[-5:]
        
        # 检查同向笔的力度对比
        # 找到最近两段同向笔
        last_bi = recent_bis[-1]
        
        # 找前一段同向笔
        prev_same_dir_bi = None
        for bi in reversed(recent_bis[:-1]):
            if bi.direction == last_bi.direction:
                prev_same_dir_bi = bi
                break
        
        if prev_same_dir_bi is None:
            return BeiChiType.NONE, "未找到同向笔，无法判断背驰", False
        
        # 计算力度对比
        power_ratio = last_bi.power / prev_same_dir_bi.power if prev_same_dir_bi.power > 0 else 1
        
        macd_div = False
        
        if power_ratio < 0.618:  # 黄金分割点
            macd_div = True
            if last_bi.direction == BiDirection.DOWN:
                # 下跌背驰 = 底背驰 = 买入机会
                return (
                    BeiChiType.TREND_BEICHI,
                    f"底背驰：下跌力度减弱至前段的{power_ratio:.1%}，多头即将反攻",
                    macd_div
                )
            else:
                # 上涨背驰 = 顶背驰 = 卖出信号
                return (
                    BeiChiType.TREND_BEICHI,
                    f"顶背驰：上涨力度减弱至前段的{power_ratio:.1%}，空头即将反攻",
                    macd_div
                )
        
        return BeiChiType.NONE, f"未出现背驰，力度比{power_ratio:.1%}", False
    
    def _analyze_trend(
        self, 
        zhongshus: List[ZhongShu], 
        bis: List[Bi]
    ) -> Tuple[TrendType, str]:
        """分析趋势类型"""
        if not zhongshus:
            if bis:
                # 没有中枢，看笔的方向
                last_bi = bis[-1]
                if last_bi.direction == BiDirection.UP:
                    return TrendType.UP_TREND, "无中枢，当前处于上升笔中"
                else:
                    return TrendType.DOWN_TREND, "无中枢，当前处于下降笔中"
            return TrendType.CONSOLIDATION, "数据不足，无法判断趋势"
        
        if len(zhongshus) == 1:
            return TrendType.CONSOLIDATION, f"形成一个中枢，处于盘整走势"
        
        # 多个中枢，比较中枢位置
        last_zs = zhongshus[-1]
        prev_zs = zhongshus[-2]
        
        if last_zs.zd > prev_zs.zg:
            return TrendType.UP_TREND, f"中枢依次抬升，上涨趋势明确"
        elif last_zs.zg < prev_zs.zd:
            return TrendType.DOWN_TREND, f"中枢依次下降，下跌趋势明确"
        else:
            return TrendType.CONSOLIDATION, f"中枢重叠，大级别盘整"
    
    def _identify_buy_sell_point(
        self, 
        result: ChanAnalysisResult,
        df: pd.DataFrame
    ) -> Tuple[BuySellPoint, str]:
        """
        识别买卖点
        
        一买：下跌趋势背驰后
        二买：一买后回踩不破低点
        三买：离开中枢后回踩不进中枢
        """
        current_price = float(df.iloc[-1]['close'])
        
        # 检查是否有背驰
        if result.beichi_type != BeiChiType.NONE:
            if result.last_bi and result.last_bi.direction == BiDirection.DOWN:
                # 底背驰 = 一买
                return BuySellPoint.BUY_1, "出现底背驰，形成第一类买点"
            elif result.last_bi and result.last_bi.direction == BiDirection.UP:
                # 顶背驰 = 一卖
                return BuySellPoint.SELL_1, "出现顶背驰，形成第一类卖点"
        
        # 检查三买三卖
        if result.current_zhongshu:
            zs = result.current_zhongshu
            
            # 价格在中枢上方，且最近是回踩
            if current_price > zs.zg:
                if result.last_bi and result.last_bi.direction == BiDirection.DOWN:
                    if result.last_bi.low > zs.zg:
                        return BuySellPoint.BUY_3, f"离开中枢后回踩，低点{result.last_bi.low:.2f}在中枢上沿{zs.zg:.2f}上方，形成三买"
            
            # 价格在中枢下方，且最近是反弹
            elif current_price < zs.zd:
                if result.last_bi and result.last_bi.direction == BiDirection.UP:
                    if result.last_bi.high < zs.zd:
                        return BuySellPoint.SELL_3, f"离开中枢后反弹，高点{result.last_bi.high:.2f}在中枢下沿{zs.zd:.2f}下方，形成三卖"
        
        # 检查二买二卖（需要参考前一次买卖点）
        if len(result.bis) >= 4 and result.last_fenxing:
            # 简化版：如果最近底分型不创新低，可能是二买
            if result.last_fenxing.type == FenXingType.BOTTOM:
                prev_bottoms = [fx for fx in result.fenxings if fx.type == FenXingType.BOTTOM]
                if len(prev_bottoms) >= 2:
                    if result.last_fenxing.low > prev_bottoms[-2].low:
                        return BuySellPoint.BUY_2, f"回踩低点{result.last_fenxing.low:.2f}未破前低{prev_bottoms[-2].low:.2f}，形成二买"
            
            elif result.last_fenxing.type == FenXingType.TOP:
                prev_tops = [fx for fx in result.fenxings if fx.type == FenXingType.TOP]
                if len(prev_tops) >= 2:
                    if result.last_fenxing.high < prev_tops[-2].high:
                        return BuySellPoint.SELL_2, f"反弹高点{result.last_fenxing.high:.2f}未破前高{prev_tops[-2].high:.2f}，形成二卖"
        
        return BuySellPoint.NONE, "当前无明确买卖点"
    
    def _calculate_key_levels(
        self, 
        result: ChanAnalysisResult,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """计算关键点位"""
        levels = {}
        current_price = float(df.iloc[-1]['close'])
        levels['current_price'] = current_price
        
        # 中枢点位
        if result.current_zhongshu:
            zs = result.current_zhongshu
            levels['zhongshu_zg'] = zs.zg
            levels['zhongshu_zd'] = zs.zd
            levels['zhongshu_gg'] = zs.gg
            levels['zhongshu_dd'] = zs.dd
        
        # 最近分型点位
        if result.fenxings:
            recent_tops = [fx for fx in result.fenxings[-10:] if fx.type == FenXingType.TOP]
            recent_bottoms = [fx for fx in result.fenxings[-10:] if fx.type == FenXingType.BOTTOM]
            
            if recent_tops:
                levels['recent_top'] = max(fx.high for fx in recent_tops)
            if recent_bottoms:
                levels['recent_bottom'] = min(fx.low for fx in recent_bottoms)
        
        # 建议点位
        if result.buy_sell_point in [BuySellPoint.BUY_1, BuySellPoint.BUY_2, BuySellPoint.BUY_3]:
            # 买入建议
            if result.current_zhongshu:
                levels['stop_loss'] = result.current_zhongshu.zd * 0.97  # 中枢下沿下方3%
            elif result.last_bi:
                levels['stop_loss'] = result.last_bi.low * 0.97
            
            if 'recent_top' in levels:
                levels['target'] = levels['recent_top']
        
        elif result.buy_sell_point in [BuySellPoint.SELL_1, BuySellPoint.SELL_2, BuySellPoint.SELL_3]:
            # 卖出建议
            if result.current_zhongshu:
                levels['stop_loss'] = result.current_zhongshu.zg * 1.03
            elif result.last_bi:
                levels['stop_loss'] = result.last_bi.high * 1.03
            
            if 'recent_bottom' in levels:
                levels['target'] = levels['recent_bottom']
        
        return levels
    
    def _calculate_score(self, result: ChanAnalysisResult) -> int:
        """
        计算缠论评分
        
        评分维度：
        1. 趋势方向（30分）
        2. 买卖点信号（30分）
        3. 背驰信号（20分）
        4. 中枢位置（20分）
        """
        score = 50  # 基础分
        
        # 1. 趋势方向（30分）
        if result.trend_type == TrendType.UP_TREND:
            score += 15
        elif result.trend_type == TrendType.DOWN_TREND:
            score -= 15
        
        # 2. 买卖点信号（30分）
        if result.buy_sell_point in [BuySellPoint.BUY_1, BuySellPoint.BUY_2, BuySellPoint.BUY_3]:
            if result.buy_sell_point == BuySellPoint.BUY_1:
                score += 25  # 一买最强
            elif result.buy_sell_point == BuySellPoint.BUY_3:
                score += 20  # 三买次之
            else:
                score += 15  # 二买
        elif result.buy_sell_point in [BuySellPoint.SELL_1, BuySellPoint.SELL_2, BuySellPoint.SELL_3]:
            if result.buy_sell_point == BuySellPoint.SELL_1:
                score -= 25
            elif result.buy_sell_point == BuySellPoint.SELL_3:
                score -= 20
            else:
                score -= 15
        
        # 3. 背驰信号（20分）
        if result.beichi_type != BeiChiType.NONE:
            if result.last_bi and result.last_bi.direction == BiDirection.DOWN:
                score += 15  # 底背驰加分
            else:
                score -= 15  # 顶背驰减分
        
        # 4. 中枢位置（20分）
        if result.current_zhongshu and result.key_levels.get('current_price'):
            price = result.key_levels['current_price']
            zs = result.current_zhongshu
            
            if price > zs.zg:
                score += 10  # 在中枢上方
            elif price < zs.zd:
                score -= 10  # 在中枢下方
        
        return max(0, min(100, score))
    
    def _generate_suggestion(self, result: ChanAnalysisResult) -> str:
        """生成操作建议"""
        if result.chan_score >= 80:
            return "强烈买入：缠论多重信号共振，趋势向上"
        elif result.chan_score >= 65:
            return "买入：缠论信号积极，可逢低布局"
        elif result.chan_score >= 50:
            return "观望：中枢震荡中，等待方向明确"
        elif result.chan_score >= 35:
            return "减仓：缠论信号转弱，注意风险"
        else:
            return "卖出：缠论信号看空，建议离场"
    
    def _summarize_fenxings(self, fenxings: List[FenXing]) -> str:
        """分型摘要"""
        if not fenxings:
            return "无有效分型"
        
        tops = sum(1 for fx in fenxings if fx.type == FenXingType.TOP)
        bottoms = sum(1 for fx in fenxings if fx.type == FenXingType.BOTTOM)
        last_fx = fenxings[-1]
        
        return f"共{len(fenxings)}个分型（顶{tops}/底{bottoms}），最近为{last_fx.type.value}（{last_fx.date}）"
    
    def _summarize_bis(self, bis: List[Bi]) -> str:
        """笔摘要"""
        if not bis:
            return "无有效笔"
        
        up_bis = sum(1 for bi in bis if bi.direction == BiDirection.UP)
        down_bis = len(bis) - up_bis
        last_bi = bis[-1]
        
        return f"共{len(bis)}笔（上升{up_bis}/下降{down_bis}），当前{last_bi.direction.value}"
    
    def _summarize_xianduans(self, xianduans: List[XianDuan]) -> str:
        """线段摘要"""
        if not xianduans:
            return "无有效线段"
        
        last_xd = xianduans[-1]
        return f"共{len(xianduans)}段，当前{last_xd.direction.value}（高{last_xd.high:.2f}/低{last_xd.low:.2f}）"
    
    def _summarize_zhongshu(self, zs: ZhongShu, price: float) -> str:
        """中枢摘要"""
        position = self._get_price_position(price, zs)
        return f"中枢区间 [{zs.zd:.2f}, {zs.zg:.2f}]，当前价格{position}"
    
    def _generate_summary(self, result: ChanAnalysisResult) -> str:
        """生成综合分析摘要"""
        parts = []
        
        # 趋势判断
        parts.append(f"【趋势】{result.trend_type.value}：{result.trend_summary}")
        
        # 中枢位置
        if result.current_zhongshu:
            parts.append(f"【中枢】{result.zhongshu_summary}")
        
        # 背驰信号
        if result.beichi_type != BeiChiType.NONE:
            parts.append(f"【背驰】{result.beichi_summary}")
        
        # 买卖点
        if result.buy_sell_point != BuySellPoint.NONE:
            parts.append(f"【信号】{result.buy_sell_point.value}：{result.buy_sell_reason}")
        
        # 关键点位
        levels = result.key_levels
        if 'zhongshu_zg' in levels:
            parts.append(f"【点位】中枢上沿{levels['zhongshu_zg']:.2f}，中枢下沿{levels['zhongshu_zd']:.2f}")
        if 'stop_loss' in levels:
            parts.append(f"【止损】{levels['stop_loss']:.2f}")
        
        return "\n".join(parts)
    
    def format_analysis(self, result: ChanAnalysisResult) -> str:
        """
        格式化分析结果为文本
        
        Args:
            result: 分析结果
            
        Returns:
            格式化的分析文本
        """
        lines = [
            f"=== {result.code} 缠论分析 ===",
            f"",
            f"📊 缠论评分: {result.chan_score}/100",
            f"🎯 操作建议: {result.operation_suggestion}",
            f"",
            f"📈 趋势判断: {result.trend_type.value}",
            f"   {result.trend_summary}",
            f"",
            f"🔍 分型: {result.fenxing_summary}",
            f"📏 笔: {result.bi_summary}",
            f"📐 线段: {result.xianduan_summary}",
            f"",
        ]
        
        if result.current_zhongshu:
            lines.extend([
                f"🎯 中枢分析:",
                f"   {result.zhongshu_summary}",
                f"   价格位置: {result.price_position}",
                f"",
            ])
        
        if result.beichi_type != BeiChiType.NONE:
            lines.extend([
                f"⚡ 背驰信号:",
                f"   类型: {result.beichi_type.value}",
                f"   {result.beichi_summary}",
                f"",
            ])
        
        if result.buy_sell_point != BuySellPoint.NONE:
            lines.extend([
                f"💡 买卖点:",
                f"   信号: {result.buy_sell_point.value}",
                f"   {result.buy_sell_reason}",
                f"",
            ])
        
        # 关键点位
        levels = result.key_levels
        if levels:
            lines.append(f"📍 关键点位:")
            if 'current_price' in levels:
                lines.append(f"   当前价格: {levels['current_price']:.2f}")
            if 'zhongshu_zg' in levels:
                lines.append(f"   中枢上沿: {levels['zhongshu_zg']:.2f}")
            if 'zhongshu_zd' in levels:
                lines.append(f"   中枢下沿: {levels['zhongshu_zd']:.2f}")
            if 'stop_loss' in levels:
                lines.append(f"   建议止损: {levels['stop_loss']:.2f}")
            if 'target' in levels:
                lines.append(f"   目标位: {levels['target']:.2f}")
        
        return "\n".join(lines)


def analyze_chan(df: pd.DataFrame, code: str) -> ChanAnalysisResult:
    """
    便捷函数：缠论分析
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        code: 股票代码
        
    Returns:
        ChanAnalysisResult 分析结果
    """
    analyzer = ChanAnalyzer()
    return analyzer.analyze(df, code)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据测试
    import numpy as np
    
    dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
    np.random.seed(42)
    
    # 模拟一个有波动的行情
    base_price = 10.0
    prices = [base_price]
    for i in range(59):
        # 添加趋势和波动
        trend = 0.001 * np.sin(i / 10)  # 周期性趋势
        noise = np.random.randn() * 0.03
        change = trend + noise
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'date': dates,
        'open': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.03)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.03)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices],
    })
    
    analyzer = ChanAnalyzer()
    result = analyzer.analyze(df, '000001')
    print(analyzer.format_analysis(result))
    print("\n" + "="*50)
    print("Result Dict:")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
