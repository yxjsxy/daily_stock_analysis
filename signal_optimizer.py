# -*- coding: utf-8 -*-
"""
===================================
信号优化器 - 预测算法优化层
===================================

功能：
1. P0 - 硬规则过滤：乖离率/RSI/连涨天数等硬性约束
2. P0 - 反转预警：检测追高/杀跌风险
3. P1 - 复牌股处理：停牌/复牌特殊逻辑
4. P1 - 信号置信度衰减：避免连续同向信号
5. P2 - 历史准确率反馈：记录和追踪预测结果

Created: 2026-02-07
Author: 牧牧 for Karl
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# ========== P0: 硬规则过滤 ==========

@dataclass
class HardRuleResult:
    """硬规则检查结果"""
    passed: bool  # 是否通过
    original_signal: str  # 原始信号
    adjusted_signal: str  # 调整后信号
    blocked_reasons: List[str] = field(default_factory=list)  # 被阻止的原因
    warnings: List[str] = field(default_factory=list)  # 警告信息


class HardRuleFilter:
    """
    硬规则过滤器
    
    在 LLM 信号之上增加硬性约束，避免追高杀跌
    """
    
    # 禁止买入的条件
    NO_BUY_RULES = [
        {
            'name': '乖离率过高',
            'condition': lambda d: d.get('bias_ma5', 0) > 5,
            'message': '乖离率 {bias_ma5:.1f}% > 5%，追高风险',
        },
        {
            'name': '当日大涨追高',
            'condition': lambda d: d.get('pct_chg', 0) >= 7,
            'message': '当日涨幅 {pct_chg:.1f}%，次日追高风险极大',
        },
        {
            'name': '连续大涨',
            'condition': lambda d: d.get('consecutive_up_days', 0) >= 4,
            'message': '连涨 {consecutive_up_days} 日，回调风险',
        },
        {
            'name': '涨停次日',
            'condition': lambda d: d.get('prev_limit_up', False),
            'message': '前日涨停，分歧风险大',
        },
        {
            'name': 'RSI超买',
            'condition': lambda d: d.get('rsi', 50) > 80,
            'message': 'RSI={rsi:.0f} 超买区',
        },
        {
            'name': '放量滞涨',
            'condition': lambda d: (
                d.get('volume_ratio', 1) > 2 and 
                abs(d.get('pct_chg', 0)) < 1
            ),
            'message': '放量滞涨，主力出货嫌疑',
        },
    ]
    
    # 禁止卖出的条件
    NO_SELL_RULES = [
        {
            'name': '乖离率过低',
            'condition': lambda d: d.get('bias_ma5', 0) < -5,
            'message': '乖离率 {bias_ma5:.1f}% < -5%，超跌反弹概率大',
        },
        {
            'name': '连续大跌',
            'condition': lambda d: d.get('consecutive_down_days', 0) >= 4,
            'message': '连跌 {consecutive_down_days} 日，反弹概率增加',
        },
        {
            'name': '跌停次日',
            'condition': lambda d: d.get('prev_limit_down', False),
            'message': '前日跌停，恐慌释放后可能反弹',
        },
        {
            'name': 'RSI超卖',
            'condition': lambda d: d.get('rsi', 50) < 20,
            'message': 'RSI={rsi:.0f} 超卖区',
        },
    ]
    
    # 警告条件（不阻止，但提示风险）
    WARNING_RULES = [
        {
            'name': '乖离率偏高',
            'condition': lambda d: 3 < d.get('bias_ma5', 0) <= 5,
            'message': '⚠️ 乖离率 {bias_ma5:.1f}%，注意回调风险',
            'applies_to': ['买入', '强烈买入', '加仓'],
        },
        {
            'name': '量价背离',
            'condition': lambda d: (
                d.get('pct_chg', 0) > 0 and 
                d.get('volume_ratio', 1) < 0.7
            ),
            'message': '⚠️ 缩量上涨，后继乏力',
            'applies_to': ['买入', '强烈买入', '加仓'],
        },
        {
            'name': '大股东减持',
            # [2026-02-14 优化] 减持不再一刀切警告，改为由 NewsHedgeModel 量化对冲
            # 仅当减持比例>5%且无强利好对冲时才警告
            'condition': lambda d: (
                d.get('has_reduction_plan', False) and 
                d.get('reduction_pct', 100) > 5 and
                not d.get('has_strong_positive', False)
            ),
            'message': '⚠️ 大股东减持>5%且无强利好对冲',
            'applies_to': ['买入', '强烈买入', '加仓'],
        },
    ]
    
    def check(self, signal: str, indicators: Dict[str, Any]) -> HardRuleResult:
        """
        检查硬规则
        
        Args:
            signal: LLM 给出的信号 (买入/卖出/观望等)
            indicators: 技术指标数据
            
        Returns:
            HardRuleResult 包含是否通过和原因
        """
        result = HardRuleResult(
            passed=True,
            original_signal=signal,
            adjusted_signal=signal,
        )
        
        # 检查买入信号
        if signal in ['买入', '强烈买入', '加仓']:
            for rule in self.NO_BUY_RULES:
                try:
                    if rule['condition'](indicators):
                        result.passed = False
                        result.adjusted_signal = '观望'
                        reason = rule['message'].format(**indicators)
                        result.blocked_reasons.append(f"[禁买] {rule['name']}: {reason}")
                except Exception as e:
                    logger.warning(f"规则检查异常 {rule['name']}: {e}")
        
        # 检查卖出信号
        elif signal in ['卖出', '强烈卖出', '减仓']:
            for rule in self.NO_SELL_RULES:
                try:
                    if rule['condition'](indicators):
                        result.passed = False
                        result.adjusted_signal = '观望'
                        reason = rule['message'].format(**indicators)
                        result.blocked_reasons.append(f"[禁卖] {rule['name']}: {reason}")
                except Exception as e:
                    logger.warning(f"规则检查异常 {rule['name']}: {e}")
        
        # 检查警告规则
        for rule in self.WARNING_RULES:
            if signal in rule.get('applies_to', []):
                try:
                    if rule['condition'](indicators):
                        warning = rule['message'].format(**indicators)
                        result.warnings.append(warning)
                except Exception as e:
                    pass
        
        if result.blocked_reasons:
            logger.info(f"硬规则过滤: {signal} → {result.adjusted_signal}, 原因: {result.blocked_reasons}")
        
        return result


# ========== P0: 反转预警 ==========

@dataclass
class ReversalWarning:
    """反转预警结果"""
    has_risk: bool
    risk_level: str  # high/medium/low
    risk_factors: List[str]
    suggested_action: str


class ReversalDetector:
    """
    反转风险检测器
    
    检测潜在的趋势反转信号
    """
    
    def detect(self, indicators: Dict[str, Any], current_signal: str) -> ReversalWarning:
        """
        检测反转风险
        
        Args:
            indicators: 技术指标
            current_signal: 当前信号方向
            
        Returns:
            ReversalWarning 反转预警结果
        """
        risk_factors = []
        risk_score = 0
        
        # === 上涨反转风险 (针对买入信号) ===
        if current_signal in ['买入', '强烈买入', '加仓']:
            
            # 1. 顶背离检测
            if indicators.get('macd_divergence') == 'top':
                risk_factors.append("MACD 顶背离")
                risk_score += 30
            
            # 2. 高位放量滞涨
            if (indicators.get('volume_ratio', 1) > 1.5 and 
                indicators.get('pct_chg', 0) < 2 and
                indicators.get('bias_ma5', 0) > 3):
                risk_factors.append("高位放量滞涨")
                risk_score += 25
            
            # 3. 长上影线
            upper_shadow_ratio = indicators.get('upper_shadow_ratio', 0)
            if upper_shadow_ratio > 0.5:  # 上影线占比>50%
                risk_factors.append(f"长上影线 ({upper_shadow_ratio:.0%})")
                risk_score += 20
            
            # 4. 连续上涨后动能衰减
            if (indicators.get('consecutive_up_days', 0) >= 3 and
                indicators.get('pct_chg', 0) < indicators.get('prev_pct_chg', 0)):
                risk_factors.append("连涨后动能衰减")
                risk_score += 15
            
            # 5. 突破后回落 (假突破)
            if (indicators.get('broke_resistance', False) and
                indicators.get('close', 0) < indicators.get('resistance', float('inf'))):
                risk_factors.append("突破后回落，假突破风险")
                risk_score += 25
        
        # === 下跌反转风险 (针对卖出信号) ===
        elif current_signal in ['卖出', '强烈卖出', '减仓']:
            
            # 1. 底背离检测
            if indicators.get('macd_divergence') == 'bottom':
                risk_factors.append("MACD 底背离")
                risk_score += 30
            
            # 2. 低位缩量企稳
            if (indicators.get('volume_ratio', 1) < 0.5 and
                abs(indicators.get('pct_chg', 0)) < 1 and
                indicators.get('bias_ma5', 0) < -3):
                risk_factors.append("低位缩量企稳")
                risk_score += 25
            
            # 3. 长下影线
            lower_shadow_ratio = indicators.get('lower_shadow_ratio', 0)
            if lower_shadow_ratio > 0.5:
                risk_factors.append(f"长下影线 ({lower_shadow_ratio:.0%})")
                risk_score += 20
            
            # 4. 连续下跌后跌幅收窄
            if (indicators.get('consecutive_down_days', 0) >= 3 and
                abs(indicators.get('pct_chg', 0)) < abs(indicators.get('prev_pct_chg', 0))):
                risk_factors.append("连跌后跌幅收窄")
                risk_score += 15
        
        # 确定风险等级
        if risk_score >= 50:
            risk_level = 'high'
            suggested_action = '观望'
        elif risk_score >= 25:
            risk_level = 'medium'
            suggested_action = '降低仓位'
        else:
            risk_level = 'low'
            suggested_action = None
        
        return ReversalWarning(
            has_risk=risk_score > 0,
            risk_level=risk_level,
            risk_factors=risk_factors,
            suggested_action=suggested_action,
        )


# ========== P1: 复牌股处理 ==========

@dataclass
class ResumeTradingResult:
    """复牌股处理结果"""
    is_special: bool  # 是否需要特殊处理
    signal: str
    reason: str
    confidence: float


class ResumeTradingHandler:
    """
    复牌股特殊处理器
    """
    
    def handle(self, stock_info: Dict[str, Any]) -> Optional[ResumeTradingResult]:
        """
        处理复牌股
        
        Args:
            stock_info: 股票信息，包含停牌/复牌状态
            
        Returns:
            ResumeTradingResult 或 None (正常处理)
        """
        # 检查是否停牌中
        if stock_info.get('is_suspended', False):
            return ResumeTradingResult(
                is_special=True,
                signal='观望',
                reason='停牌中，暂不操作',
                confidence=0.0,
            )
        
        # 检查是否复牌首日
        if stock_info.get('just_resumed', False):
            resume_reason = stock_info.get('resume_reason', '')
            suspend_days = stock_info.get('suspend_days', 0)
            
            # 重组复牌
            if '重组' in resume_reason or '收购' in resume_reason:
                return ResumeTradingResult(
                    is_special=True,
                    signal='观望',
                    reason=f'重组复牌首日，等待价格发现 (停牌{suspend_days}日)',
                    confidence=0.2,
                )
            
            # 长期停牌 (>10日)
            if suspend_days > 10:
                return ResumeTradingResult(
                    is_special=True,
                    signal='观望',
                    reason=f'长期停牌后复牌 ({suspend_days}日)，观察成交情况',
                    confidence=0.3,
                )
            
            # 短期停牌
            return ResumeTradingResult(
                is_special=True,
                signal='观望',
                reason='复牌首日，观察开盘走势',
                confidence=0.5,
            )
        
        # 复牌次日
        if stock_info.get('resumed_yesterday', False):
            prev_resume_change = stock_info.get('prev_resume_change', 0)
            
            # 复牌首日涨停
            if prev_resume_change >= 9.8:
                return ResumeTradingResult(
                    is_special=True,
                    signal='观望',
                    reason='复牌首日涨停，次日分歧风险大',
                    confidence=0.4,
                )
            
            # 复牌首日跌停
            if prev_resume_change <= -9.8:
                return ResumeTradingResult(
                    is_special=True,
                    signal='观望',
                    reason='复牌首日跌停，恐慌可能延续',
                    confidence=0.4,
                )
        
        return None  # 正常处理


# ========== P1: 信号置信度衰减 ==========

class SignalConfidenceAdjuster:
    """
    信号置信度调整器
    
    根据历史信号和市场状态调整信号强度
    """
    
    def __init__(self, history_manager=None):
        self.history_manager = history_manager
    
    def adjust(
        self, 
        signal: str, 
        confidence: float,
        context: Dict[str, Any]
    ) -> Tuple[str, float, List[str]]:
        """
        调整信号置信度
        
        Args:
            signal: 原始信号
            confidence: 原始置信度 (0-1)
            context: 上下文信息
            
        Returns:
            (调整后信号, 调整后置信度, 调整原因列表)
        """
        adjustments = []
        
        # 1. 连续同向信号衰减
        prev_signal = context.get('prev_signal', '')
        if self._same_direction(signal, prev_signal):
            confidence *= 0.75
            adjustments.append(f"连续{self._get_direction(signal)}信号，置信度×0.75")
        
        # 2. 极端行情后衰减
        prev_change = abs(context.get('prev_pct_chg', 0))
        if prev_change > 7:  # 涨跌幅>7%
            confidence *= 0.6
            adjustments.append(f"前日大幅波动({prev_change:.1f}%)，置信度×0.6")
        elif prev_change > 5:
            confidence *= 0.8
            adjustments.append(f"前日波动较大({prev_change:.1f}%)，置信度×0.8")
        
        # 3. 缠论与均线矛盾时衰减
        chan_bullish = context.get('chan_bullish', None)
        ma_bullish = context.get('ma_bullish', None)
        if chan_bullish is not None and ma_bullish is not None:
            if chan_bullish != ma_bullish:
                confidence *= 0.6
                adjustments.append("缠论与均线信号矛盾，置信度×0.6")
        
        # 4. 量能不配合衰减
        volume_support = context.get('volume_support', True)
        if not volume_support:
            confidence *= 0.85
            adjustments.append("量能不配合，置信度×0.85")
        
        # 5. 根据历史准确率调整
        if self.history_manager:
            historical_accuracy = self.history_manager.get_signal_accuracy(signal)
            if historical_accuracy is not None and historical_accuracy < 0.4:
                confidence *= 0.7
                adjustments.append(f"该类信号历史准确率低({historical_accuracy:.0%})，置信度×0.7")
        
        # 置信度过低时降级信号
        adjusted_signal = signal
        if confidence < 0.3:
            adjusted_signal = '观望'
            adjustments.append(f"置信度过低({confidence:.0%})，降级为观望")
        elif confidence < 0.5:
            if signal in ['强烈买入', '强烈卖出']:
                adjusted_signal = signal.replace('强烈', '')
                adjustments.append(f"置信度较低({confidence:.0%})，信号降级")
        
        return adjusted_signal, confidence, adjustments
    
    def _same_direction(self, signal1: str, signal2: str) -> bool:
        """判断两个信号是否同向"""
        bullish = ['买入', '强烈买入', '加仓']
        bearish = ['卖出', '强烈卖出', '减仓']
        return (signal1 in bullish and signal2 in bullish) or \
               (signal1 in bearish and signal2 in bearish)
    
    def _get_direction(self, signal: str) -> str:
        """获取信号方向"""
        if signal in ['买入', '强烈买入', '加仓']:
            return '多'
        elif signal in ['卖出', '强烈卖出', '减仓']:
            return '空'
        return '中性'


# ========== P2: 历史准确率反馈 ==========

class PredictionHistoryManager:
    """
    预测历史管理器
    
    记录每次预测及结果，用于统计准确率和优化
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / 'data' / 'predictions.db'
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT,
                signal TEXT NOT NULL,
                confidence REAL,
                price_at_signal REAL,
                target_price REAL,
                stop_loss REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, code)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                result_date TEXT NOT NULL,
                actual_price REAL,
                pct_change REAL,
                is_correct INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_date 
            ON predictions(date)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_code 
            ON predictions(code)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(
        self, 
        date: str, 
        code: str, 
        name: str,
        signal: str, 
        confidence: float,
        price: float,
        target: float = None,
        stop_loss: float = None
    ):
        """记录预测"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (date, code, name, signal, confidence, price_at_signal, target_price, stop_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, code, name, signal, confidence, price, target, stop_loss))
            conn.commit()
            logger.debug(f"记录预测: {date} {code} {signal}")
        except Exception as e:
            logger.error(f"记录预测失败: {e}")
        finally:
            conn.close()
    
    def log_result(
        self, 
        pred_date: str, 
        code: str, 
        result_date: str,
        actual_price: float,
        notes: str = None
    ):
        """记录实际结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 查找对应的预测
            cursor.execute('''
                SELECT id, signal, price_at_signal 
                FROM predictions 
                WHERE date = ? AND code = ?
            ''', (pred_date, code))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"未找到预测记录: {pred_date} {code}")
                return
            
            pred_id, signal, price_at_signal = row
            
            # 计算涨跌幅
            pct_change = (actual_price - price_at_signal) / price_at_signal * 100
            
            # 判断是否正确
            is_correct = self._evaluate_prediction(signal, pct_change)
            
            cursor.execute('''
                INSERT INTO results 
                (prediction_id, result_date, actual_price, pct_change, is_correct, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (pred_id, result_date, actual_price, pct_change, is_correct, notes))
            
            conn.commit()
            logger.debug(f"记录结果: {code} {pct_change:+.2f}% {'✓' if is_correct else '✗'}")
            
        except Exception as e:
            logger.error(f"记录结果失败: {e}")
        finally:
            conn.close()
    
    def _evaluate_prediction(self, signal: str, pct_change: float) -> int:
        """
        评估预测是否正确
        
        买入信号: 次日涨 = 正确
        卖出信号: 次日跌 = 正确
        观望信号: 涨跌<2% = 正确
        """
        if signal in ['买入', '强烈买入', '加仓']:
            return 1 if pct_change > 0 else 0
        elif signal in ['卖出', '强烈卖出', '减仓']:
            return 1 if pct_change < 0 else 0
        else:  # 观望
            return 1 if abs(pct_change) < 2 else 0
    
    def get_accuracy(self, code: str = None, days: int = 30) -> Optional[float]:
        """
        获取准确率
        
        Args:
            code: 股票代码，None 表示全部
            days: 统计天数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            if code:
                cursor.execute('''
                    SELECT AVG(r.is_correct) 
                    FROM results r
                    JOIN predictions p ON r.prediction_id = p.id
                    WHERE p.code = ? AND p.date >= ?
                ''', (code, cutoff_date))
            else:
                cursor.execute('''
                    SELECT AVG(r.is_correct) 
                    FROM results r
                    JOIN predictions p ON r.prediction_id = p.id
                    WHERE p.date >= ?
                ''', (cutoff_date,))
            
            result = cursor.fetchone()[0]
            return result if result is not None else None
            
        finally:
            conn.close()
    
    def get_signal_accuracy(self, signal: str, days: int = 30) -> Optional[float]:
        """
        获取特定信号类型的准确率
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT AVG(r.is_correct), COUNT(*) 
                FROM results r
                JOIN predictions p ON r.prediction_id = p.id
                WHERE p.signal = ? AND p.date >= ?
            ''', (signal, cutoff_date))
            
            result, count = cursor.fetchone()
            
            # 样本太少时不返回
            if count is None or count < 5:
                return None
                
            return result
            
        finally:
            conn.close()
    
    def get_summary(self, days: int = 30) -> Dict[str, Any]:
        """获取准确率汇总"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 总体准确率
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(r.is_correct) as correct,
                    AVG(r.is_correct) as accuracy
                FROM results r
                JOIN predictions p ON r.prediction_id = p.id
                WHERE p.date >= ?
            ''', (cutoff_date,))
            
            total, correct, accuracy = cursor.fetchone()
            
            # 分信号类型统计
            cursor.execute('''
                SELECT 
                    p.signal,
                    COUNT(*) as count,
                    SUM(r.is_correct) as correct,
                    AVG(r.is_correct) as accuracy
                FROM results r
                JOIN predictions p ON r.prediction_id = p.id
                WHERE p.date >= ?
                GROUP BY p.signal
            ''', (cutoff_date,))
            
            by_signal = {}
            for row in cursor.fetchall():
                signal, count, correct, acc = row
                by_signal[signal] = {
                    'count': count,
                    'correct': correct or 0,
                    'accuracy': acc or 0,
                }
            
            return {
                'total': total or 0,
                'correct': correct or 0,
                'accuracy': accuracy or 0,
                'by_signal': by_signal,
                'days': days,
            }
            
        finally:
            conn.close()


# ========== [2026-02-14 新增] 利好vs利空量化对冲模型 ==========

class NewsHedgeModel:
    """
    利好 vs 利空量化对冲模型
    
    [优化点5] 取代简单一票否决制，建立量化对冲机制
    [优化点1] 减持利空权重动态调整：减持≤3%且有强利好时，权重降至 -10~-15
    
    权重体系：
    - 利空：减持(-10~-30)、业绩预亏(-25)、监管处罚(-20)、行业利空(-15)、大额解禁(-20)
    - 利好：并购重组(+25)、业绩预增(+20)、政策利好(+15)、重大合同(+15)、回购(+10)
    - 净值 < -20 → 观望（重大利空无法对冲）
    - 净值 -20~0 → 降低仓位
    - 净值 > 0 → 利好占优，正常操作
    """
    
    # 利空权重映射
    NEGATIVE_WEIGHTS = {
        'reduction_small': -12,     # 减持≤3%（[优化点1] 从-30降至-12）
        'reduction_medium': -20,    # 减持3-5%
        'reduction_large': -30,     # 减持>5%
        'earnings_loss': -25,       # 业绩预亏
        'regulatory_penalty': -20,  # 监管处罚
        'sector_negative': -15,     # 行业利空
        'large_unlock': -20,        # 大额解禁
    }
    
    # 利好权重映射
    POSITIVE_WEIGHTS = {
        'merger_acquisition': 25,   # 并购重组
        'earnings_increase': 20,    # 业绩预增
        'policy_positive': 15,      # 政策利好
        'major_contract': 15,       # 重大合同
        'buyback': 10,              # 回购
        'institutional_buy': 10,    # 机构增持
    }
    
    def evaluate(self, news_factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        量化评估利好利空对冲后的净影响
        
        Args:
            news_factors: {
                'negatives': [{'type': 'reduction_small', 'detail': '...'}],
                'positives': [{'type': 'merger_acquisition', 'detail': '...'}],
                'reduction_pct': 3.0,  # 减持比例
            }
        
        Returns:
            {
                'net_score': int,  # 净得分
                'negative_total': int,
                'positive_total': int,
                'should_veto': bool,  # 是否一票否决（仅净分<-20）
                'position_adjust': float,  # 仓位调整系数 0.0-1.0
                'details': [str],
            }
        """
        negative_total = 0
        positive_total = 0
        details = []
        
        # 计算利空得分
        for neg in news_factors.get('negatives', []):
            neg_type = neg.get('type', '')
            
            # [优化点1] 减持权重动态计算
            if neg_type.startswith('reduction'):
                reduction_pct = news_factors.get('reduction_pct', 5)
                has_strong_positive = len(news_factors.get('positives', [])) > 0 and \
                    any(p.get('type') in ('merger_acquisition', 'earnings_increase') 
                        for p in news_factors.get('positives', []))
                
                if reduction_pct <= 3 and has_strong_positive:
                    weight = -12  # 有强利好对冲，减持权重降至-12
                    details.append(f"减持{reduction_pct}%+强利好对冲: {weight}分")
                elif reduction_pct <= 3:
                    weight = -15
                    details.append(f"减持{reduction_pct}%(小比例): {weight}分")
                elif reduction_pct <= 5:
                    weight = -20
                    details.append(f"减持{reduction_pct}%(中等): {weight}分")
                else:
                    weight = -30
                    details.append(f"减持{reduction_pct}%(大比例): {weight}分")
                negative_total += weight
            else:
                weight = self.NEGATIVE_WEIGHTS.get(neg_type, -10)
                negative_total += weight
                details.append(f"利空[{neg_type}]: {weight}分")
        
        # 计算利好得分
        for pos in news_factors.get('positives', []):
            pos_type = pos.get('type', '')
            weight = self.POSITIVE_WEIGHTS.get(pos_type, 5)
            positive_total += weight
            details.append(f"利好[{pos_type}]: +{weight}分")
        
        net_score = positive_total + negative_total
        
        # 判断是否一票否决（仅当净分极低时）
        should_veto = net_score < -20
        
        # 仓位调整系数
        if net_score < -20:
            position_adjust = 0.0  # 不建议操作
        elif net_score < -10:
            position_adjust = 0.3
        elif net_score < 0:
            position_adjust = 0.6
        else:
            position_adjust = 1.0
        
        return {
            'net_score': net_score,
            'negative_total': negative_total,
            'positive_total': positive_total,
            'should_veto': should_veto,
            'position_adjust': position_adjust,
            'details': details,
        }


# ========== [2026-02-14 新增] 趋势惯性因子 (Momentum Tracker) ==========

class MomentumTracker:
    """
    趋势惯性因子
    
    [优化点2] 解决连续三天信号翻转问题（2/10看多→2/11看空→2/12看多）
    
    使用 3-5 日信号方向的加权平均作为 momentum_score:
    - momentum_score > 0.3 → 偏多惯性，空信号需更强证据
    - momentum_score < -0.3 → 偏空惯性，多信号需更强证据
    - -0.3 ~ 0.3 → 无明显惯性
    
    权重占总评分的 15-20%
    """
    
    # 信号到数值的映射
    SIGNAL_VALUES = {
        '强烈买入': 1.0,
        '买入': 0.7,
        '加仓': 0.7,
        '持有': 0.3,
        '观望': 0.0,
        '减仓': -0.5,
        '卖出': -0.7,
        '强烈卖出': -1.0,
    }
    
    # 日权重：近日权重更高（index 0 = 最近一天）
    DAY_WEIGHTS = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    def calculate_momentum(self, recent_signals: List[str]) -> Dict[str, Any]:
        """
        计算趋势惯性得分
        
        Args:
            recent_signals: 最近 3-5 天的信号列表，[最近一天, 前一天, ...]
        
        Returns:
            {
                'momentum_score': float (-1 ~ 1),
                'direction': str ('bullish'/'bearish'/'neutral'),
                'signal_stability': float (0-1, 信号稳定度),
                'flip_count': int (翻转次数),
                'adjustment': str (建议调整),
            }
        """
        if not recent_signals:
            return {
                'momentum_score': 0.0,
                'direction': 'neutral',
                'signal_stability': 0.5,
                'flip_count': 0,
                'adjustment': '无历史数据',
            }
        
        # 转换信号为数值
        values = [self.SIGNAL_VALUES.get(s, 0.0) for s in recent_signals[:5]]
        
        # 加权平均
        weights = self.DAY_WEIGHTS[:len(values)]
        weight_sum = sum(weights)
        momentum_score = sum(v * w for v, w in zip(values, weights)) / weight_sum
        
        # 计算翻转次数
        flip_count = 0
        for i in range(1, len(values)):
            if (values[i] > 0 and values[i-1] < 0) or (values[i] < 0 and values[i-1] > 0):
                flip_count += 1
        
        # 信号稳定度：翻转越多越不稳定
        signal_stability = max(0.0, 1.0 - flip_count * 0.25)
        
        # 方向判断
        if momentum_score > 0.3:
            direction = 'bullish'
        elif momentum_score < -0.3:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # 调整建议
        if flip_count >= 2 and len(values) <= 3:
            adjustment = '信号频繁翻转，建议降低仓位或观望'
        elif direction == 'bullish':
            adjustment = '多头惯性，空信号需更强证据才能翻转'
        elif direction == 'bearish':
            adjustment = '空头惯性，多信号需更强证据才能翻转'
        else:
            adjustment = '无明显惯性'
        
        return {
            'momentum_score': round(momentum_score, 3),
            'direction': direction,
            'signal_stability': round(signal_stability, 2),
            'flip_count': flip_count,
            'adjustment': adjustment,
        }
    
    def apply_momentum_filter(
        self, 
        current_signal: str, 
        momentum_result: Dict[str, Any],
        current_score: int
    ) -> Tuple[str, int, List[str]]:
        """
        用惯性因子过滤当前信号
        
        [优化点2] 权重 15-20%，防止过度敏感
        
        Returns:
            (adjusted_signal, adjusted_score, reasons)
        """
        momentum_score = momentum_result['momentum_score']
        direction = momentum_result['direction']
        stability = momentum_result['signal_stability']
        flip_count = momentum_result['flip_count']
        
        adjusted_signal = current_signal
        adjusted_score = current_score
        reasons = []
        
        # 惯性加分/减分（权重约15-20分，满分100）
        momentum_bonus = int(momentum_score * 18)  # ±18分范围
        adjusted_score += momentum_bonus
        
        if momentum_bonus != 0:
            reasons.append(f"[惯性因子] momentum={momentum_score:.2f}, 调整{momentum_bonus:+d}分")
        
        # 翻转惩罚：连续翻转降低置信度
        if flip_count >= 2:
            penalty = -8
            adjusted_score += penalty
            reasons.append(f"[翻转惩罚] {flip_count}次翻转, {penalty}分")
        
        # 惯性阻力：当信号与惯性方向相反时，需要更强的信号
        current_value = self.SIGNAL_VALUES.get(current_signal, 0.0)
        if direction == 'bullish' and current_value < -0.3:
            # 多头惯性中出现空信号 → 信号降级
            if abs(current_value) < abs(momentum_score):
                adjusted_signal = '观望'
                reasons.append(f"[惯性阻力] 多头惯性中弱空信号→观望")
        elif direction == 'bearish' and current_value > 0.3:
            # 空头惯性中出现多信号 → 信号降级
            if abs(current_value) < abs(momentum_score):
                adjusted_signal = '观望'
                reasons.append(f"[惯性阻力] 空头惯性中弱多信号→观望")
        
        # 确保分数在合理范围
        adjusted_score = max(0, min(100, adjusted_score))
        
        return adjusted_signal, adjusted_score, reasons


# ========== [2026-02-14 新增] 量价突破信号检测 ==========

class VolumeBreakthroughDetector:
    """
    量价突破信号检测器
    
    [优化点3] 量比>1.5 + 收盘突破前高 = 强制看多，覆盖弱空信号
    
    这是极强的技术信号，应该有最高优先级覆盖弱空信号
    """
    
    def detect(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测量价突破信号
        
        Args:
            indicators: {
                'volume_ratio': 量比,
                'close': 收盘价,
                'prev_high_20d': 近20日最高价,
                'prev_high_5d': 近5日最高价,
                'pct_chg': 涨跌幅,
            }
        
        Returns:
            {
                'is_breakthrough': bool,
                'strength': str ('strong'/'moderate'/'none'),
                'override_bearish': bool,  # 是否应覆盖弱空信号
                'forced_signal': str or None,
                'reasons': [str],
            }
        """
        volume_ratio = indicators.get('volume_ratio', 1.0)
        close = indicators.get('close', 0)
        prev_high_20d = indicators.get('prev_high_20d', float('inf'))
        prev_high_5d = indicators.get('prev_high_5d', float('inf'))
        pct_chg = indicators.get('pct_chg', 0)
        
        reasons = []
        is_breakthrough = False
        strength = 'none'
        override_bearish = False
        forced_signal = None
        
        # 强突破：量比>1.5 + 收盘突破20日前高
        if volume_ratio > 1.5 and close > prev_high_20d and pct_chg > 0:
            is_breakthrough = True
            strength = 'strong'
            override_bearish = True
            forced_signal = '买入'
            reasons.append(
                f"🚀 强势量价突破: 量比{volume_ratio:.2f}>1.5, "
                f"收盘{close:.2f}突破20日前高{prev_high_20d:.2f}"
            )
        
        # 中等突破：量比>1.3 + 收盘突破5日前高
        elif volume_ratio > 1.3 and close > prev_high_5d and pct_chg > 1:
            is_breakthrough = True
            strength = 'moderate'
            override_bearish = True  # 仍覆盖弱空
            forced_signal = '买入'
            reasons.append(
                f"📈 量价突破: 量比{volume_ratio:.2f}>1.3, "
                f"收盘{close:.2f}突破5日前高{prev_high_5d:.2f}"
            )
        
        # 弱突破信号（不强制覆盖，仅加分）
        elif volume_ratio > 1.5 and pct_chg > 2:
            is_breakthrough = True
            strength = 'moderate'
            override_bearish = False
            reasons.append(f"📊 放量上涨: 量比{volume_ratio:.2f}, 涨幅{pct_chg:.1f}%")
        
        return {
            'is_breakthrough': is_breakthrough,
            'strength': strength,
            'override_bearish': override_bearish,
            'forced_signal': forced_signal,
            'reasons': reasons,
        }


# ========== [2026-02-14 新增] 缠论状态机 ==========

class ChanStateMachine:
    """
    缠论跨日状态机
    
    [优化点4] 确保笔-段-中枢判断跨日连贯
    
    状态转移规则：
    - 上升笔 → 只能转为 顶分型确认 → 下降笔
    - 下降笔 → 只能转为 底分型确认 → 上升笔
    - 不允许直接 上升笔 → 下降笔（中间必须经过分型确认）
    
    持久化到 JSON 文件，确保跨日一致
    """
    
    VALID_TRANSITIONS = {
        '上升笔': ['顶分型待确认', '上升笔延续'],
        '顶分型待确认': ['下降笔', '上升笔延续'],  # 确认失败回到上升
        '下降笔': ['底分型待确认', '下降笔延续'],
        '底分型待确认': ['上升笔', '下降笔延续'],
        '上升笔延续': ['顶分型待确认', '上升笔延续'],
        '下降笔延续': ['底分型待确认', '下降笔延续'],
        '未知': ['上升笔', '下降笔', '未知'],
    }
    
    def __init__(self, state_file: str = None):
        if state_file is None:
            state_file = str(Path(__file__).parent / 'data' / 'chan_state.json')
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.states = self._load_states()
    
    def _load_states(self) -> Dict[str, Dict]:
        """加载状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_states(self):
        """保存状态"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.states, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缠论状态失败: {e}")
    
    def get_state(self, code: str) -> Dict[str, Any]:
        """获取某股票的缠论状态"""
        return self.states.get(code, {
            'current_bi': '未知',
            'zhongshu_low': None,
            'zhongshu_high': None,
            'last_update': None,
            'bi_history': [],
        })
    
    def validate_transition(self, code: str, new_bi: str, date: str) -> Dict[str, Any]:
        """
        验证缠论状态转移是否合法
        
        Args:
            code: 股票代码
            new_bi: 新的笔状态
            date: 日期
        
        Returns:
            {
                'valid': bool,
                'current_state': str,
                'new_state': str,
                'warning': str or None,
                'corrected_state': str,  # 如果不合法，给出修正建议
            }
        """
        state = self.get_state(code)
        current_bi = state.get('current_bi', '未知')
        
        # 简化新笔状态的映射
        bi_mapping = {
            '离开中枢向上笔': '上升笔',
            '向上笔': '上升笔',
            '上涨笔': '上升笔',
            '一卖后向下笔': '下降笔',
            '向下笔': '下降笔',
            '下跌笔': '下降笔',
        }
        normalized_new = bi_mapping.get(new_bi, new_bi)
        normalized_current = bi_mapping.get(current_bi, current_bi)
        
        # 检查转移是否合法
        valid_targets = self.VALID_TRANSITIONS.get(normalized_current, ['未知'])
        
        # 直接从上升笔到下降笔是不合法的（需经过顶分型确认）
        if normalized_current == '上升笔' and normalized_new == '下降笔':
            return {
                'valid': False,
                'current_state': current_bi,
                'new_state': new_bi,
                'warning': f'状态矛盾：{current_bi}→{new_bi}，缺少顶分型确认过渡',
                'corrected_state': '顶分型待确认',
            }
        
        if normalized_current == '下降笔' and normalized_new == '上升笔':
            return {
                'valid': False,
                'current_state': current_bi,
                'new_state': new_bi,
                'warning': f'状态矛盾：{current_bi}→{new_bi}，缺少底分型确认过渡',
                'corrected_state': '底分型待确认',
            }
        
        return {
            'valid': True,
            'current_state': current_bi,
            'new_state': new_bi,
            'warning': None,
            'corrected_state': normalized_new,
        }
    
    def update_state(self, code: str, new_bi: str, date: str, 
                     zhongshu_low: float = None, zhongshu_high: float = None):
        """更新缠论状态"""
        state = self.get_state(code)
        
        # 记录历史
        if 'bi_history' not in state:
            state['bi_history'] = []
        state['bi_history'].append({
            'date': date,
            'bi': state.get('current_bi', '未知'),
        })
        # 只保留最近10条
        state['bi_history'] = state['bi_history'][-10:]
        
        state['current_bi'] = new_bi
        state['last_update'] = date
        if zhongshu_low is not None:
            state['zhongshu_low'] = zhongshu_low
        if zhongshu_high is not None:
            state['zhongshu_high'] = zhongshu_high
        
        self.states[code] = state
        self._save_states()


# ========== 综合优化器 ==========

class SignalOptimizer:
    """
    信号优化器 - 整合所有优化逻辑
    """
    
    def __init__(self, db_path: str = None):
        self.hard_rule_filter = HardRuleFilter()
        self.reversal_detector = ReversalDetector()
        self.resume_handler = ResumeTradingHandler()
        self.history_manager = PredictionHistoryManager(db_path)
        self.confidence_adjuster = SignalConfidenceAdjuster(self.history_manager)
        # [2026-02-14 新增] 三个优化模块
        self.news_hedge = NewsHedgeModel()
        self.momentum_tracker = MomentumTracker()
        self.volume_breakthrough = VolumeBreakthroughDetector()
        self.chan_state_machine = ChanStateMachine()
    
    def optimize(
        self, 
        signal: str, 
        confidence: float,
        indicators: Dict[str, Any],
        stock_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        优化信号
        
        Args:
            signal: LLM 给出的原始信号
            confidence: 原始置信度
            indicators: 技术指标
            stock_info: 股票基本信息
            context: 上下文信息
            
        Returns:
            优化结果字典
        """
        result = {
            'original_signal': signal,
            'original_confidence': confidence,
            'final_signal': signal,
            'final_confidence': confidence,
            'adjustments': [],
            'warnings': [],
            'blocked': False,
        }
        
        # 1. 复牌股处理 (优先级最高)
        resume_result = self.resume_handler.handle(stock_info)
        if resume_result and resume_result.is_special:
            result['final_signal'] = resume_result.signal
            result['final_confidence'] = resume_result.confidence
            result['adjustments'].append(f"[复牌处理] {resume_result.reason}")
            return result
        
        # 2. 硬规则过滤
        hard_rule_result = self.hard_rule_filter.check(signal, indicators)
        if not hard_rule_result.passed:
            result['final_signal'] = hard_rule_result.adjusted_signal
            result['blocked'] = True
            result['adjustments'].extend(hard_rule_result.blocked_reasons)
        result['warnings'].extend(hard_rule_result.warnings)
        
        # 3. [2026-02-14 优化点3] 量价突破检测 — 最高优先级
        breakthrough = self.volume_breakthrough.detect(indicators)
        if breakthrough['is_breakthrough']:
            result['warnings'].extend(breakthrough['reasons'])
            if breakthrough['override_bearish'] and breakthrough['forced_signal']:
                # 量价突破覆盖弱空信号
                current_signal = result['final_signal']
                if current_signal in ['观望', '减仓', '卖出'] and not result.get('_strong_bearish'):
                    result['final_signal'] = breakthrough['forced_signal']
                    result['blocked'] = False
                    result['adjustments'].append(
                        f"[量价突破] {breakthrough['strength']}突破信号覆盖弱空→{breakthrough['forced_signal']}"
                    )
        
        # 4. 反转预警
        reversal_warning = self.reversal_detector.detect(indicators, signal)
        if reversal_warning.has_risk:
            result['warnings'].extend([f"[反转风险] {f}" for f in reversal_warning.risk_factors])
            if reversal_warning.risk_level == 'high' and not result['blocked']:
                result['final_signal'] = reversal_warning.suggested_action or '观望'
                result['adjustments'].append(f"[反转预警] 高风险，信号降级")
        
        # 5. [2026-02-14 优化点5] 利好vs利空量化对冲
        news_factors = context.get('news_factors', {})
        if news_factors.get('negatives') or news_factors.get('positives'):
            hedge_result = self.news_hedge.evaluate(news_factors)
            result['hedge_result'] = hedge_result
            if hedge_result['should_veto']:
                result['final_signal'] = '观望'
                result['blocked'] = True
                result['adjustments'].append(
                    f"[利空对冲] 净分{hedge_result['net_score']}(<-20)，利空无法对冲→观望"
                )
            elif hedge_result['position_adjust'] < 1.0:
                result['adjustments'].append(
                    f"[利空对冲] 净分{hedge_result['net_score']}，建议仓位×{hedge_result['position_adjust']}"
                )
            result['adjustments'].extend(hedge_result['details'])
        
        # 6. [2026-02-14 优化点2] 趋势惯性因子
        recent_signals = context.get('recent_signals', [])
        if recent_signals:
            momentum_result = self.momentum_tracker.calculate_momentum(recent_signals)
            result['momentum'] = momentum_result
            
            current_score = context.get('sentiment_score', 50)
            adj_signal, adj_score, mom_reasons = self.momentum_tracker.apply_momentum_filter(
                result['final_signal'], momentum_result, current_score
            )
            if adj_signal != result['final_signal']:
                result['final_signal'] = adj_signal
            result['adjustments'].extend(mom_reasons)
        
        # 7. [2026-02-14 优化点4] 缠论状态机验证
        chan_bi = context.get('chan_current_bi', '')
        code = context.get('code', '')
        date = context.get('date', '')
        if chan_bi and code:
            transition = self.chan_state_machine.validate_transition(code, chan_bi, date)
            if not transition['valid']:
                result['warnings'].append(f"[缠论状态机] {transition['warning']}")
                result['adjustments'].append(
                    f"[缠论修正] {transition['current_state']}→{transition['corrected_state']}"
                )
            # 更新状态
            corrected = transition['corrected_state'] if not transition['valid'] else chan_bi
            self.chan_state_machine.update_state(
                code, corrected, date,
                context.get('zhongshu_low'), context.get('zhongshu_high')
            )
        
        # 8. 置信度调整
        if not result['blocked']:
            adjusted_signal, adjusted_conf, adj_reasons = self.confidence_adjuster.adjust(
                result['final_signal'],
                result['final_confidence'],
                context
            )
            result['final_signal'] = adjusted_signal
            result['final_confidence'] = adjusted_conf
            result['adjustments'].extend(adj_reasons)
        
        return result
    
    def log_prediction(self, date: str, code: str, name: str, signal: str, 
                       confidence: float, price: float, target: float = None, 
                       stop_loss: float = None):
        """记录预测到历史库"""
        self.history_manager.log_prediction(
            date, code, name, signal, confidence, price, target, stop_loss
        )
    
    def log_result(self, pred_date: str, code: str, result_date: str, 
                   actual_price: float, notes: str = None):
        """记录实际结果"""
        self.history_manager.log_result(
            pred_date, code, result_date, actual_price, notes
        )
    
    def get_accuracy_summary(self, days: int = 30) -> Dict[str, Any]:
        """获取准确率汇总"""
        return self.history_manager.get_summary(days)


# 便捷函数
_optimizer_instance = None

def get_optimizer() -> SignalOptimizer:
    """获取信号优化器单例"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = SignalOptimizer()
    return _optimizer_instance


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    optimizer = SignalOptimizer()
    
    # 测试硬规则过滤
    print("=== 测试硬规则过滤 ===")
    test_indicators = {
        'bias_ma5': 6.5,  # 乖离率过高
        'consecutive_up_days': 3,
        'rsi': 75,
    }
    
    result = optimizer.optimize(
        signal='强烈买入',
        confidence=0.8,
        indicators=test_indicators,
        stock_info={},
        context={}
    )
    
    print(f"原始信号: {result['original_signal']}")
    print(f"优化后信号: {result['final_signal']}")
    print(f"调整原因: {result['adjustments']}")
    print(f"警告: {result['warnings']}")
