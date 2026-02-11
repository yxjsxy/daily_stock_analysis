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
            'condition': lambda d: d.get('has_reduction_plan', False),
            'message': '⚠️ 存在大股东减持计划',
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
        
        # 3. 反转预警
        reversal_warning = self.reversal_detector.detect(indicators, signal)
        if reversal_warning.has_risk:
            result['warnings'].extend([f"[反转风险] {f}" for f in reversal_warning.risk_factors])
            if reversal_warning.risk_level == 'high' and not result['blocked']:
                result['final_signal'] = reversal_warning.suggested_action or '观望'
                result['adjustments'].append(f"[反转预警] 高风险，信号降级")
        
        # 4. 置信度调整
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
