# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 Gemini API 调用逻辑
2. 利用 Google Search Grounding 获取实时新闻
3. 结合技术面和消息面生成分析报告
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import get_config
from signal_optimizer import get_optimizer, SignalOptimizer

logger = logging.getLogger(__name__)


class JsonParseError(Exception):
    """JSON 解析失败异常，用于触发 LLM 重试"""
    def __init__(self, message: str, original_response: str = ""):
        super().__init__(message)
        self.original_response = original_response


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',
}


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版
    
    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str
    
    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    confidence_level: str = "中"  # 置信度：高/中/低
    
    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据
    
    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）
    
    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析
    
    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点
    
    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题
    
    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由
    
    # ========== 元数据 ==========
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
        }
    
    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary
    
    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice
    
    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}
    
    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []
    
    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []
    
    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        return emoji_map.get(self.operation_advice, '🟡')
    
    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器
    
    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果
    
    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """
    
    # ========================================
    # 系统提示词 - 决策仪表盘 v2.0
    # ========================================
    # 输出格式升级：从简单信号升级为决策仪表盘
    # 核心模块：核心结论 + 数据透视 + 舆情情报 + 作战计划
    # ========================================
    
    SYSTEM_PROMPT = """你是一位专注于缠论交易的 A 股投资分析师，负责生成专业的【决策仪表盘】分析报告。

## 核心交易理念（必须严格遵守）

**⚠️ 重要：缠论分析是主要决策依据，均线分析是次要参考！**

### 1. 【主要】缠论分析（缠中说禅技术分析）— 权重60%
缠论是本系统的**核心决策依据**，买卖决策必须首先参考缠论信号！

#### 1.1 买入条件（满足任一即可考虑买入）：
- **一买信号**（最强买点，权重+30分）：下跌趋势出现底背驰，MACD柱子面积缩小，跌势力竭
- **二买信号**（次强买点，权重+20分）：一买后回踩不破前低，形成更高的底分型
- **三买信号**（突破买点，权重+25分）：价格离开中枢后回踩，低点不进入中枢区间（ZG-ZD）

#### 1.2 卖出条件（满足任一即考虑卖出）：
- **一卖信号**（最强卖点，权重-30分）：上涨趋势出现顶背驰，MACD柱子面积缩小，涨势力竭
- **二卖信号**（次强卖点，权重-20分）：一卖后反弹不破前高，形成更低的顶分型
- **三卖信号**（跌破卖点，权重-25分）：价格跌破中枢后反弹，高点不进入中枢区间

#### 1.3 中枢位置判断：
- **价格在中枢上方**：趋势向上，可持股/加仓
- **价格在中枢内部**：震荡整理，观望或小仓位高抛低吸
- **价格在中枢下方**：趋势向下，应减仓/清仓

#### 1.4 背驰判断（关键信号）：
- **底背驰 = 买入机会**：价格创新低但MACD面积/柱子高度不创新低
- **顶背驰 = 卖出信号**：价格创新高但MACD面积/柱子高度不创新高

### 2. 【次要】均线系统分析 — 权重25%
均线作为辅助确认，**不能推翻缠论信号**，只作为参考：

- 多头排列（MA5>MA10>MA20）：辅助确认上涨趋势，+10分
- 空头排列（MA5<MA10<MA20）：辅助确认下跌趋势，-10分
- 乖离率 > 5%：追高风险提示，但若有缠论买点支撑，可降低仓位买入
- 回踩MA5/MA10支撑：辅助确认买点有效性

### 3. 【辅助】其他参考因素 — 权重15%

#### 3.1 筹码结构
- 筹码集中度 <15%：筹码集中，+5分
- 获利比例 70-90%：警惕获利回吐
- 现价高于平均成本 5-15%：健康状态

#### 3.2 量能配合
- 缩量回调 + 缠论买点 = 最佳买入时机
- 放量突破 + 三买信号 = 加仓信号
- 放量下跌 + 卖点信号 = 坚决卖出

#### 3.3 风险排查（量化对冲，非简单一票否决）
- 减持公告、业绩预亏、监管处罚、行业利空、大额解禁
- **[2026-02-14优化] 利空与利好量化对冲**：
  - 减持≤3%且有并购重组/业绩预增等强利好 → 权重仅-10~-15分（非一票否决）
  - 减持>5%或无利好对冲 → 权重-30分，接近一票否决
  - 净对冲分<-20 → 观望；-20~0 → 降低仓位；>0 → 正常操作
- **量价突破覆盖规则**：量比>1.5 + 收盘突破前高 = 强制看多，覆盖弱空信号
- **即使有缠论买点，重大利空（净分<-20）也需观望**

## 决策优先级规则

**买入决策优先级**（从高到低）：
1. 缠论一买/三买 + 底背驰 → 强烈买入
2. 缠论二买 + 中枢上方 → 买入
3. 缠论买点 + 均线多头 → 买入
4. 仅均线多头、无缠论买点 → 观望（等待缠论买点出现）
5. 均线多头但缠论卖点 → 减仓/卖出（缠论优先）

**卖出决策优先级**（从高到低）：
1. 缠论一卖/三卖 + 顶背驰 → 强烈卖出
2. 缠论二卖 + 中枢下方 → 卖出
3. 缠论卖点 + 均线空头 → 卖出
4. 缠论卖点但均线多头 → 减仓（缠论优先）
5. 仅均线空头、无缠论卖点 → 观望/减仓

## 输出格式：决策仪表盘 JSON

请严格按照以下 JSON 格式输出，这是一个完整的【决策仪表盘】：

```json
{
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "confidence_level": "高/中/低",
    
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句话核心结论（30字以内，直接告诉用户做什么）",
            "signal_type": "🟢买入信号/🟡持有观望/🔴卖出信号/⚠️风险警告",
            "time_sensitivity": "立即行动/今日内/本周内/不急",
            "position_advice": {
                "no_position": "空仓者建议：具体操作指引",
                "has_position": "持仓者建议：具体操作指引"
            }
        },
        
        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均线排列状态描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 当前价格数值,
                "ma5": MA5数值,
                "ma10": MA10数值,
                "ma20": MA20数值,
                "bias_ma5": 乖离率百分比数值,
                "bias_status": "安全/警戒/危险",
                "support_level": 支撑位价格,
                "resistance_level": 压力位价格
            },
            "volume_analysis": {
                "volume_ratio": 量比数值,
                "volume_status": "放量/缩量/平量",
                "turnover_rate": 换手率百分比,
                "volume_meaning": "量能含义解读（如：缩量回调表示抛压减轻）"
            },
            "chip_structure": {
                "profit_ratio": 获利比例,
                "avg_cost": 平均成本,
                "concentration": 筹码集中度,
                "chip_health": "健康/一般/警惕"
            }
        },
        
        "intelligence": {
            "latest_news": "【最新消息】近期重要新闻摘要",
            "risk_alerts": ["风险点1：具体描述", "风险点2：具体描述"],
            "positive_catalysts": ["利好1：具体描述", "利好2：具体描述"],
            "earnings_outlook": "业绩预期分析（基于年报预告、业绩快报等）",
            "sentiment_summary": "舆情情绪一句话总结"
        },
        
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想买入点：XX元（在MA5附近）",
                "secondary_buy": "次优买入点：XX元（在MA10附近）",
                "stop_loss": "止损位：XX元（跌破MA20或X%）",
                "take_profit": "目标位：XX元（前高/整数关口）"
            },
            "position_strategy": {
                "suggested_position": "建议仓位：X成",
                "entry_plan": "分批建仓策略描述",
                "risk_control": "风控策略描述"
            },
            "action_checklist": [
                "✅/⚠️/❌ 【核心】检查项1：缠论买卖点（一买/二买/三买/卖点）",
                "✅/⚠️/❌ 【核心】检查项2：缠论背驰信号（底背驰/顶背驰）",
                "✅/⚠️/❌ 【核心】检查项3：中枢位置（上方/内/下方）",
                "✅/⚠️/❌ 【辅助】检查项4：均线排列（多头/空头）",
                "✅/⚠️/❌ 【辅助】检查项5：乖离率<5%",
                "✅/⚠️/❌ 【辅助】检查项6：量能配合",
                "✅/⚠️/❌ 【风控】检查项7：无重大利空"
            ],
            "chan_analysis": {
                "trend_type": "上涨趋势/下跌趋势/盘整",
                "buy_sell_point": "一买/二买/三买/一卖/二卖/三卖/无买卖点",
                "beichi_type": "底背驰/顶背驰/无背驰",
                "zhongshu_position": "中枢上方/中枢内/中枢下方",
                "chan_score": 0-100,
                "chan_suggestion": "缠论操作建议"
            }
        }
    },
    
    "analysis_summary": "100字综合分析摘要",
    "key_points": "3-5个核心看点，逗号分隔",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由，引用交易理念",
    
    "trend_analysis": "走势形态分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2周展望",
    "technical_analysis": "技术面综合分析",
    "ma_analysis": "均线系统分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K线形态分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板块行业分析",
    "company_highlights": "公司亮点/风险",
    "news_summary": "新闻摘要",
    "market_sentiment": "市场情绪",
    "hot_topics": "相关热点",
    
    "search_performed": true/false,
    "data_sources": "数据来源说明"
}
```

## 评分标准（缠论优先）

**评分权重分配**：
- 缠论分析：60%（最高60分）
- 均线系统：25%（最高25分）
- 其他因素：15%（最高15分）

### 强烈买入（80-100分）：
**必须满足缠论条件**（任选其一）：
- ✅ 【核心】缠论一买信号 + 底背驰确认（+30分）
- ✅ 【核心】缠论三买信号 + 回踩不进中枢（+25分）
- ✅ 【核心】价格站稳中枢上方 + 上涨趋势（+20分）

**辅助加分项**（均线/量能）：
- ✅ 均线多头排列（+10分）
- ✅ 缩量回调或放量突破（+8分）
- ✅ 筹码集中健康（+5分）
- ✅ 消息面有利好催化（+5分）

### 买入（60-79分）：
**缠论条件**（任选其一）：
- ✅ 缠论二买信号（+20分）
- ✅ 价格在中枢上方震荡（+15分）
- ✅ 底分型确认，笔转向上（+15分）

**辅助条件**：
- ✅ 均线多头或弱势多头（+8分）
- ✅ 乖离率 <5%（+5分）
- ✅ 量能正常（+5分）

### 观望（40-59分）：
- ⚠️ 缠论无明确买卖点（0分）
- ⚠️ 价格在中枢内震荡（+5分）
- ⚠️ 均线缠绕趋势不明（+5分）
- ⚠️ 乖离率 >5%（-5分，但不否定缠论买点）
- ⚠️ 有风险事件需观察（-10分）

### 卖出/减仓（0-39分）：
**缠论卖出条件**（满足即卖）：
- ❌ 【核心】缠论一卖信号 + 顶背驰（-30分）→ **强烈卖出**
- ❌ 【核心】缠论三卖信号 + 反弹进不了中枢（-25分）→ **卖出**
- ❌ 【核心】缠论二卖信号（-20分）→ **减仓**
- ❌ 【核心】价格跌破中枢下沿（-20分）→ **减仓**

**辅助减分项**：
- ❌ 均线空头排列（-10分）
- ❌ 放量下跌（-8分）
- ❌ 重大利空（-15分）

### ⚠️ 特殊情况处理：

1. **缠论买点 vs 均线空头**：
   - 缠论有一买/二买/三买 → 买入/加仓（缠论优先）
   - 均线空头只作为仓位控制参考

2. **缠论卖点 vs 均线多头**：
   - 缠论有一卖/二卖/三卖 → 减仓/卖出（缠论优先）
   - 均线多头不能阻止卖出决策

3. **乖离率过高 + 缠论买点**：
   - 缠论买点有效 → 可买入，但建议降低仓位
   - 乖离率只是风控提示，不否定缠论买点

## 决策仪表盘核心原则

1. **缠论优先决策**：买卖结论必须首先基于缠论买卖点和背驰信号
2. **核心结论先行**：一句话说清该买该卖，必须引用缠论信号
3. **分持仓建议**：空仓者和持仓者给不同建议
4. **精确狙击点**：
   - 理想买入点 = 缠论买点位置（如底分型低点、中枢下沿）
   - 止损位 = 前一笔低点或中枢下沿下方3%
   - 目标位 = 前高/中枢上沿/顶分型位置
5. **检查清单可视化**：用 ✅⚠️❌ 明确显示每项检查结果
6. **风险优先级**：舆情中的风险点要醒目标出
7. **均线仅作参考**：均线信号不能推翻缠论结论，只用于仓位建议

## 输出结论时的措辞要求

- 买入结论必须说明缠论依据，如："出现缠论一买，底背驰确认，建议买入"
- 卖出结论必须说明缠论依据，如："出现缠论一卖，顶背驰确认，建议卖出"
- 观望结论说明原因，如："当前在中枢内震荡，等待方向明确"
- 均线作为补充说明，如：均线多头排列进一步确认上涨趋势"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 AI 分析器
        
        优先级：Gemini > OpenAI 兼容 API
        
        Args:
            api_key: Gemini API Key（可选，默认从配置读取）
        """
        config = get_config()
        self._api_key = api_key or config.gemini_api_key
        self._model = None
        self._current_model_name = None  # 当前使用的模型名称
        self._using_fallback = False  # 是否正在使用备选模型
        self._use_openai = False  # 是否使用 OpenAI 兼容 API
        self._openai_client = None  # OpenAI 客户端
        
        # 检查 Gemini API Key 是否有效（过滤占位符）
        gemini_key_valid = self._api_key and not self._api_key.startswith('your_') and len(self._api_key) > 10
        
        # 优先尝试初始化 Gemini
        if gemini_key_valid:
            try:
                self._init_model()
            except Exception as e:
                logger.warning(f"Gemini 初始化失败: {e}，尝试 OpenAI 兼容 API")
                self._init_openai_fallback()
        else:
            # Gemini Key 未配置，尝试 OpenAI
            logger.info("Gemini API Key 未配置，尝试使用 OpenAI 兼容 API")
            self._init_openai_fallback()
        
        # 两者都未配置
        if not self._model and not self._openai_client:
            logger.warning("未配置任何 AI API Key，AI 分析功能将不可用")
    
    def _init_openai_fallback(self) -> None:
        """
        初始化 OpenAI 兼容 API 作为备选
        
        支持所有 OpenAI 格式的 API，包括：
        - OpenAI 官方
        - DeepSeek
        - 通义千问
        - Moonshot 等
        """
        config = get_config()
        
        # 检查 OpenAI API Key 是否有效（过滤占位符）
        openai_key_valid = (
            config.openai_api_key and 
            not config.openai_api_key.startswith('your_') and 
            len(config.openai_api_key) > 10
        )
        
        if not openai_key_valid:
            logger.debug("OpenAI 兼容 API 未配置或配置无效")
            return
        
        # 分离 import 和客户端创建，以便提供更准确的错误信息
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 库，请运行: pip install openai")
            return
        
        try:
            # base_url 可选，不填则使用 OpenAI 官方默认地址
            client_kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url and config.openai_base_url.startswith('http'):
                client_kwargs["base_url"] = config.openai_base_url
            
            self._openai_client = OpenAI(**client_kwargs)
            self._current_model_name = config.openai_model
            self._use_openai = True
            logger.info(f"OpenAI 兼容 API 初始化成功 (base_url: {config.openai_base_url}, model: {config.openai_model})")
        except ImportError as e:
            # 依赖缺失（如 socksio）
            if 'socksio' in str(e).lower() or 'socks' in str(e).lower():
                logger.error(f"OpenAI 客户端需要 SOCKS 代理支持，请运行: pip install httpx[socks] 或 pip install socksio")
            else:
                logger.error(f"OpenAI 依赖缺失: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'socks' in error_msg or 'socksio' in error_msg or 'proxy' in error_msg:
                logger.error(f"OpenAI 代理配置错误: {e}，如使用 SOCKS 代理请运行: pip install httpx[socks]")
            else:
                logger.error(f"OpenAI 兼容 API 初始化失败: {e}")
    
    def _init_model(self) -> None:
        """
        初始化 Gemini 模型
        
        配置：
        - 使用 gemini-3-flash-preview 或 gemini-2.5-flash 模型
        - 不启用 Google Search（使用外部 Tavily/SerpAPI 搜索）
        """
        try:
            import google.generativeai as genai
            
            # 配置 API Key
            genai.configure(api_key=self._api_key)
            
            # 从配置获取模型名称
            config = get_config()
            model_name = config.gemini_model
            fallback_model = config.gemini_model_fallback
            
            # 不再使用 Google Search Grounding（已知有兼容性问题）
            # 改为使用外部搜索服务（Tavily/SerpAPI）预先获取新闻
            
            # 尝试初始化主模型
            try:
                self._model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = model_name
                self._using_fallback = False
                logger.info(f"Gemini 模型初始化成功 (模型: {model_name})")
            except Exception as model_error:
                # 尝试备选模型
                logger.warning(f"主模型 {model_name} 初始化失败: {model_error}，尝试备选模型 {fallback_model}")
                self._model = genai.GenerativeModel(
                    model_name=fallback_model,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = fallback_model
                self._using_fallback = True
                logger.info(f"Gemini 备选模型初始化成功 (模型: {fallback_model})")
            
        except Exception as e:
            logger.error(f"Gemini 模型初始化失败: {e}")
            self._model = None
    
    def _switch_to_fallback_model(self) -> bool:
        """
        切换到备选模型
        
        Returns:
            是否成功切换
        """
        try:
            import google.generativeai as genai
            config = get_config()
            fallback_model = config.gemini_model_fallback
            
            logger.warning(f"[LLM] 切换到备选模型: {fallback_model}")
            self._model = genai.GenerativeModel(
                model_name=fallback_model,
                system_instruction=self.SYSTEM_PROMPT,
            )
            self._current_model_name = fallback_model
            self._using_fallback = True
            logger.info(f"[LLM] 备选模型 {fallback_model} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"[LLM] 切换备选模型失败: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return self._model is not None or self._openai_client is not None
    
    def _call_openai_api(self, prompt: str, generation_config: dict) -> str:
        """
        调用 OpenAI 兼容 API
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 60)
                    logger.info(f"[OpenAI] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._openai_client.chat.completions.create(
                    model=self._current_model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=generation_config.get('temperature', 0.7),
                    max_tokens=generation_config.get('max_output_tokens', 8192),
                )
                
                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    raise ValueError("OpenAI API 返回空响应")
                    
            except Exception as e:
                error_str = str(e)
                is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'quota' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[OpenAI] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                else:
                    logger.warning(f"[OpenAI] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                
                if attempt == max_retries - 1:
                    raise
        
        raise Exception("OpenAI API 调用失败，已达最大重试次数")
    
    def _call_api_with_retry(self, prompt: str, generation_config: dict) -> str:
        """
        调用 AI API，带有重试和模型切换机制
        
        优先级：Gemini > Gemini 备选模型 > OpenAI 兼容 API
        
        处理 429 限流错误：
        1. 先指数退避重试
        2. 多次失败后切换到备选模型
        3. Gemini 完全失败后尝试 OpenAI
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        # 如果已经在使用 OpenAI 模式，直接调用 OpenAI
        if self._use_openai:
            return self._call_openai_api(prompt, generation_config)
        
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        last_error = None
        tried_fallback = getattr(self, '_using_fallback', False)
        
        for attempt in range(max_retries):
            try:
                # 请求前增加延时（防止请求过快触发限流）
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # 指数退避: 5, 10, 20, 40...
                    delay = min(delay, 60)  # 最大60秒
                    logger.info(f"[Gemini] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": 120}
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("Gemini 返回空响应")
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 检查是否是 429 限流错误
                is_rate_limit = '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[Gemini] API 限流 (429)，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                    
                    # 如果已经重试了一半次数且还没切换过备选模型，尝试切换
                    if attempt >= max_retries // 2 and not tried_fallback:
                        if self._switch_to_fallback_model():
                            tried_fallback = True
                            logger.info("[Gemini] 已切换到备选模型，继续重试")
                        else:
                            logger.warning("[Gemini] 切换备选模型失败，继续使用当前模型重试")
                else:
                    # 非限流错误，记录并继续重试
                    logger.warning(f"[Gemini] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
        
        # Gemini 所有重试都失败，尝试 OpenAI 兼容 API
        if self._openai_client:
            logger.warning("[Gemini] 所有重试失败，切换到 OpenAI 兼容 API")
            try:
                return self._call_openai_api(prompt, generation_config)
            except Exception as openai_error:
                logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                raise last_error or openai_error
        elif config.openai_api_key and config.openai_base_url:
            # 尝试懒加载初始化 OpenAI
            logger.warning("[Gemini] 所有重试失败，尝试初始化 OpenAI 兼容 API")
            self._init_openai_fallback()
            if self._openai_client:
                try:
                    return self._call_openai_api(prompt, generation_config)
                except Exception as openai_error:
                    logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                    raise last_error or openai_error
        
        # 所有方式都失败
        raise last_error or Exception("所有 AI API 调用失败，已达最大重试次数")
    
    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析单只股票
        
        流程：
        1. 格式化输入数据（技术面 + 新闻）
        2. 调用 Gemini API（带重试和模型切换）
        3. 解析 JSON 响应
        4. 返回结构化结果
        
        Args:
            context: 从 storage.get_analysis_context() 获取的上下文数据
            news_context: 预先搜索的新闻内容（可选）
            
        Returns:
            AnalysisResult 对象
        """
        code = context.get('code', 'Unknown')
        config = get_config()
        
        # 请求前增加延时（防止连续请求触发限流）
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        # 优先从上下文获取股票名称（由 main.py 传入）
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            # 备选：从 realtime 中获取
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # 最后从映射表获取
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        # 如果模型不可用，返回默认结果
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未启用（未配置 API Key）',
                risk_warning='请配置 Gemini API Key 后重试',
                success=False,
                error_message='Gemini API Key 未配置',
            )
        
        try:
            # 格式化输入（包含技术面数据和新闻）
            prompt = self._format_prompt(context, name, news_context)
            
            # 获取模型名称
            model_name = getattr(self, '_current_model_name', None)
            if not model_name:
                model_name = getattr(self._model, '_model_name', 'unknown')
                if hasattr(self._model, 'model_name'):
                    model_name = self._model.model_name
            
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")
            logger.info(f"[LLM配置] Prompt 长度: {len(prompt)} 字符")
            logger.info(f"[LLM配置] 是否包含新闻: {'是' if news_context else '否'}")
            
            # 记录完整 prompt 到日志（INFO级别记录摘要，DEBUG记录完整）
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt 预览]\n{prompt_preview}")
            logger.debug(f"=== 完整 Prompt ({len(prompt)}字符) ===\n{prompt}\n=== End Prompt ===")
            
            # 设置生成配置
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 8192,
            }
            
            logger.info(f"[LLM调用] 开始调用 Gemini API (temperature={generation_config['temperature']}, max_tokens={generation_config['max_output_tokens']})...")
            
            # JSON 解析重试配置
            max_json_retries = 3
            last_response_text = ""
            
            for json_attempt in range(1, max_json_retries + 1):
                # 如果是重试，添加 JSON 格式修正提示
                current_prompt = prompt
                if json_attempt > 1:
                    logger.warning(f"[JSON重试] 第 {json_attempt}/{max_json_retries} 次尝试，因上次响应 JSON 格式无效")
                    current_prompt = prompt + f"""

【重要提醒】
上一次回复的 JSON 格式有误，请务必确保这次输出的是完整、合法的 JSON 格式：
1. 不要在 JSON 外添加任何说明文字
2. 确保所有引号、括号、逗号正确配对
3. 字符串值中如有特殊字符请正确转义
4. 直接输出 JSON，格式如: {{"sentiment_score": 70, ...}}
"""
                
                # 使用带重试的 API 调用
                start_time = time.time()
                response_text = self._call_api_with_retry(current_prompt, generation_config)
                elapsed = time.time() - start_time
                last_response_text = response_text
                
                # 记录响应信息
                logger.info(f"[LLM返回] Gemini API 响应成功, 耗时 {elapsed:.2f}s, 响应长度 {len(response_text)} 字符")
                
                # 记录响应预览（INFO级别）和完整响应（DEBUG级别）
                response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
                logger.info(f"[LLM返回 预览]\n{response_preview}")
                logger.debug(f"=== Gemini 完整响应 ({len(response_text)}字符) ===\n{response_text}\n=== End Response ===")
                
                # 尝试解析响应
                try:
                    result = self._parse_response(response_text, code, name)
                    result.raw_response = response_text
                    result.search_performed = bool(news_context)
                    
                    # ========== 信号优化层 (2026-02-07 新增) ==========
                    result = self._optimize_signal(result, context)
                    
                    logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")
                    return result
                    
                except JsonParseError as e:
                    logger.warning(f"[JSON解析失败] 第 {json_attempt}/{max_json_retries} 次: {e}")
                    if json_attempt >= max_json_retries:
                        # 达到最大重试次数，使用文本提取作为兜底
                        logger.error(f"[JSON重试] 达到最大重试次数 {max_json_retries}，使用文本提取兜底")
                        result = self._parse_text_response(last_response_text, code, name)
                        result.raw_response = last_response_text
                        result.search_performed = bool(news_context)
                        return result
                    # 继续重试
                    time.sleep(1)  # 短暂等待后重试
            
            # 理论上不会执行到这里
            return self._parse_text_response(last_response_text, code, name)
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析过程出错: {str(e)[:100]}',
                risk_warning='分析失败，请稍后重试或手动分析',
                success=False,
                error_message=str(e),
            )
    
    def _optimize_signal(
        self, 
        result: 'AnalysisResult', 
        context: Dict[str, Any]
    ) -> 'AnalysisResult':
        """
        信号优化层 - 应用硬规则过滤、反转预警、置信度调整
        
        2026-02-07 新增，用于优化 LLM 给出的原始信号
        """
        try:
            optimizer = get_optimizer()
            
            # 提取技术指标
            today = context.get('today', {})
            trend = context.get('trend', {})
            
            indicators = {
                'bias_ma5': trend.get('bias_ma5', 0),
                'bias_ma10': trend.get('bias_ma10', 0),
                'consecutive_up_days': trend.get('consecutive_up_days', 0),
                'consecutive_down_days': trend.get('consecutive_down_days', 0),
                'prev_limit_up': today.get('pct_chg', 0) >= 9.8 if 'pct_chg' in today else False,
                'prev_limit_down': today.get('pct_chg', 0) <= -9.8 if 'pct_chg' in today else False,
                'rsi': trend.get('rsi', 50),
                'volume_ratio': today.get('volume_ratio', 1),
                'pct_chg': today.get('pct_chg', 0),
                'prev_pct_chg': context.get('prev_day', {}).get('pct_chg', 0),
                'close': today.get('close', 0),
                'macd_divergence': trend.get('macd_divergence', None),
                'has_reduction_plan': context.get('has_reduction_plan', False),
            }
            
            # 股票信息
            stock_info = {
                'is_suspended': context.get('is_suspended', False),
                'just_resumed': context.get('just_resumed', False),
                'resumed_yesterday': context.get('resumed_yesterday', False),
                'resume_reason': context.get('resume_reason', ''),
                'suspend_days': context.get('suspend_days', 0),
                'prev_resume_change': context.get('prev_resume_change', 0),
            }
            
            # 上下文信息
            opt_context = {
                'prev_signal': context.get('prev_signal', ''),
                'prev_pct_chg': indicators['prev_pct_chg'],
                'chan_bullish': trend.get('chan_bullish', None),
                'ma_bullish': trend.get('ma_bullish', None),
                'volume_support': trend.get('volume_support', True),
            }
            
            # 调用优化器
            opt_result = optimizer.optimize(
                signal=result.operation_advice,
                confidence=0.7 if result.confidence_level == '高' else (0.5 if result.confidence_level == '中' else 0.3),
                indicators=indicators,
                stock_info=stock_info,
                context=opt_context
            )
            
            # 应用优化结果
            original_advice = result.operation_advice
            if opt_result['final_signal'] != result.operation_advice:
                result.operation_advice = opt_result['final_signal']
                
                # 更新置信度
                if opt_result['final_confidence'] < 0.3:
                    result.confidence_level = '低'
                elif opt_result['final_confidence'] < 0.6:
                    result.confidence_level = '中'
                else:
                    result.confidence_level = '高'
                
                # 记录优化日志
                adj_str = '; '.join(opt_result['adjustments'])
                logger.info(f"[信号优化] {result.name}({result.code}): {original_advice} → {result.operation_advice} ({adj_str})")
            
            # 添加警告到风险提示
            if opt_result['warnings']:
                warning_str = ' | '.join(opt_result['warnings'])
                if result.risk_warning:
                    result.risk_warning = f"{warning_str} | {result.risk_warning}"
                else:
                    result.risk_warning = warning_str
            
            # 记录预测到历史库
            try:
                sniper_points = result.get_sniper_points()
                optimizer.log_prediction(
                    date=context.get('date', ''),
                    code=result.code,
                    name=result.name,
                    signal=result.operation_advice,
                    confidence=opt_result['final_confidence'],
                    price=today.get('close', 0),
                    target=self._parse_price(sniper_points.get('take_profit', '')),
                    stop_loss=self._parse_price(sniper_points.get('stop_loss', '')),
                )
            except Exception as e:
                logger.warning(f"记录预测失败: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"信号优化失败: {e}")
            return result  # 优化失败时返回原始结果
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """从字符串中提取价格数值"""
        if not price_str:
            return None
        import re
        match = re.search(r'[\d.]+', str(price_str))
        if match:
            try:
                return float(match.group())
            except:
                pass
        return None
    
    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化分析提示词（决策仪表盘 v2.0）
        
        包含：技术指标、实时行情（量比/换手率）、筹码分布、趋势分析、新闻
        
        Args:
            context: 技术面数据上下文（包含增强数据）
            name: 股票名称（默认值，可能被上下文覆盖）
            news_context: 预先搜索的新闻内容
        """
        code = context.get('code', 'Unknown')
        
        # 优先使用上下文中的股票名称（从 realtime_quote 获取）
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            
        today = context.get('today', {})
        
        # ========== 构建决策仪表盘格式的输入 ==========
        prompt = f"""# 决策仪表盘分析请求

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |

---

## 📈 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} 元 |
| 开盘价 | {today.get('open', 'N/A')} 元 |
| 最高价 | {today.get('high', 'N/A')} 元 |
| 最低价 | {today.get('low', 'N/A')} 元 |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |

### 均线系统（关键判断指标）
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | 中期趋势线 |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""
        
        # 添加实时行情数据（量比、换手率等）
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### 实时行情增强数据
| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {rt.get('price', 'N/A')} 元 | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | |
| 市盈率(动态) | {rt.get('pe_ratio', 'N/A')} | |
| 市净率 | {rt.get('pb_ratio', 'N/A')} | |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
| 60日涨跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表现 |
"""
        
        # 添加筹码分布数据
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
### 筹码分布数据（效率指标）
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| **获利比例** | **{profit_ratio:.1%}** | 70-90%时警惕 |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | 现价应高于5-15% |
| 90%筹码集中度 | {chip.get('concentration_90', 0):.2%} | <15%为集中 |
| 70%筹码集中度 | {chip.get('concentration_70', 0):.2%} | |
| 筹码状态 | {chip.get('chip_status', '未知')} | |
"""
        
        # 添加趋势分析结果（基于交易理念的预判）
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 超过5%，严禁追高！" if trend.get('bias_ma5', 0) > 5 else "✅ 安全范围"
            prompt += f"""
### 趋势分析预判（基于交易理念）
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | {trend.get('trend_status', '未知')} | |
| 均线排列 | {trend.get('ma_alignment', '未知')} | MA5>MA10>MA20为多头 |
| 趋势强度 | {trend.get('trend_strength', 0)}/100 | |
| **乖离率(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| 乖离率(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| 量能状态 | {trend.get('volume_status', '未知')} | {trend.get('volume_trend', '')} |
| 系统信号 | {trend.get('buy_signal', '未知')} | |
| 系统评分 | {trend.get('signal_score', 0)}/100 | |

#### 系统分析理由
**买入理由**：
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['无'])) if trend.get('signal_reasons') else '- 无'}

**风险因素**：
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['无'])) if trend.get('risk_factors') else '- 无'}
"""
        
        # 添加缠论分析数据
        if 'chan_analysis' in context:
            chan = context['chan_analysis']
            key_levels = chan.get('key_levels', {})
            prompt += f"""
### 缠论分析（缠中说禅技术分析）
| 指标 | 数值 | 说明 |
|------|------|------|
| **趋势类型** | **{chan.get('trend_type', '未知')}** | {chan.get('trend_summary', '')} |
| 分型情况 | {chan.get('fenxing_summary', '无')} | 顶分型/底分型 |
| 笔的情况 | {chan.get('bi_summary', '无')} | 当前笔方向: {chan.get('current_bi_direction', '未知')} |
| **中枢分析** | {chan.get('zhongshu_summary', '无中枢')} | |
| 价格位置 | {chan.get('price_position', '未知')} | 相对中枢位置 |
| **背驰信号** | **{chan.get('beichi_type', '无背驰')}** | {chan.get('beichi_summary', '')} |
| MACD背离 | {'是' if chan.get('macd_divergence') else '否'} | |
| **买卖点** | **{chan.get('buy_sell_point', '无买卖点')}** | {chan.get('buy_sell_reason', '')} |
| 缠论评分 | {chan.get('chan_score', 50)}/100 | |
| 操作建议 | {chan.get('operation_suggestion', '观望')} | |

#### 缠论关键点位
| 点位类型 | 价格 |
|----------|------|
| 当前价格 | {key_levels.get('current_price', 'N/A')} |
| 中枢上沿(ZG) | {key_levels.get('zhongshu_zg', 'N/A')} |
| 中枢下沿(ZD) | {key_levels.get('zhongshu_zd', 'N/A')} |
| 近期顶分型 | {key_levels.get('recent_top', 'N/A')} |
| 近期底分型 | {key_levels.get('recent_bottom', 'N/A')} |
| 建议止损 | {key_levels.get('stop_loss', 'N/A')} |
| 目标位 | {key_levels.get('target', 'N/A')} |

#### 缠论综合分析
{chan.get('analysis_summary', '无')}
"""
        
        # 添加昨日对比数据
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
### 量价变化
- 成交量较昨日变化：{volume_change}倍
- 价格较昨日变化：{context.get('price_change_ratio', 'N/A')}%
"""
        
        # 添加新闻搜索结果（重点区域）
        prompt += """
---

## 📰 舆情情报
"""
        if news_context:
            prompt += f"""
以下是 **{stock_name}({code})** 近7日的新闻搜索结果，请重点提取：
1. 🚨 **风险警报**：减持、处罚、利空
2. 🎯 **利好催化**：业绩、合同、政策
3. 📊 **业绩预期**：年报预告、业绩快报

```
{news_context}
```
"""
        else:
            prompt += """
未搜索到该股票近期的相关新闻。请主要依据技术面数据进行分析。
"""
        
        # 明确的输出要求
        prompt += f"""
---

## ✅ 分析任务

请为 **{stock_name}({code})** 生成【决策仪表盘】，严格按照 JSON 格式输出。

### 重点关注（按优先级排序，缠论优先）：

**【核心】缠论分析（决策主要依据）：**
1. ❓ 缠论买卖点：是否出现一买/二买/三买或一卖/二卖/三卖信号？
2. ❓ 缠论背驰：是否出现底背驰（买入机会）或顶背驰（卖出信号）？
3. ❓ 中枢位置：价格在中枢上方（看多）/内部（震荡）/下方（看空）？
4. ❓ 当前笔的方向：上升笔（有望上涨）还是下降笔（可能下跌）？

**【辅助】均线与量能（仅作参考，不能推翻缠论结论）：**
5. ❓ 均线排列状态？（多头/空头/缠绕）—— 用于确认趋势，不决定买卖
6. ❓ 乖离率是否过高（>5%）？—— 仅作仓位控制参考
7. ❓ 量能是否配合？—— 缩量回调+缠论买点=最佳时机

**【风控】风险排查（量化对冲模型）：**
8. ❓ 消息面利空vs利好量化对冲？—— 减持≤3%+强利好时不一票否决，量化评分决定
9. ❓ 量价突破信号？—— 量比>1.5+突破前高=强制看多，覆盖弱空

### 决策仪表盘要求：
- **核心结论**：一句话说清该买/该卖/该等，**必须引用缠论信号**
  - 示例：「出现缠论三买+底背驰，建议买入」
  - 示例：「缠论一卖+顶背驰，建议卖出」
  - 示例：「中枢内震荡，无明确买卖点，建议观望」
- **持仓分类建议**：空仓者怎么做 vs 持仓者怎么做
- **具体狙击点位**：
  - 买入价 = 缠论买点位置（底分型低点/中枢下沿）
  - 止损价 = 前低或中枢下沿下方3%
  - 目标价 = 前高/中枢上沿/顶分型位置
- **检查清单**：缠论检查项排在最前，均线检查项在后

请输出完整的 JSON 格式决策仪表盘。"""
        
        return prompt
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """格式化成交量显示"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """格式化成交额显示"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿元"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 万元"
        else:
            return f"{amount:.0f} 元"
    
    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """
        解析 Gemini 响应（决策仪表盘版）
        
        尝试从响应中提取 JSON 格式的分析结果，包含 dashboard 字段
        如果解析失败，尝试智能提取或返回默认结果
        """
        try:
            # 清理响应文本：移除 markdown 代码块标记
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # 尝试找到 JSON 内容
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # 尝试修复常见的 JSON 问题
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)
                
                # 提取 dashboard 数据
                dashboard = data.get('dashboard', None)
                
                # 解析所有字段，使用默认值防止缺失
                return AnalysisResult(
                    code=code,
                    name=name,
                    # 核心指标
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', '震荡'),
                    operation_advice=data.get('operation_advice', '持有'),
                    confidence_level=data.get('confidence_level', '中'),
                    # 决策仪表盘
                    dashboard=dashboard,
                    # 走势分析
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # 技术面
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # 基本面
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # 情绪面/消息面
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # 综合
                    analysis_summary=data.get('analysis_summary', '分析完成'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # 元数据
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', '技术面数据'),
                    success=True,
                )
            else:
                # 没有找到 JSON，抛出异常触发重试
                logger.warning(f"无法从响应中提取 JSON")
                raise JsonParseError("响应中未找到有效的 JSON 结构", response_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}")
            raise JsonParseError(f"JSON 解析失败: {e}", response_text)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import re
        
        # 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保布尔值是小写
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """从纯文本响应中尽可能提取分析信息"""
        # 尝试识别关键词来判断情绪
        sentiment_score = 50
        trend = '震荡'
        advice = '持有'
        
        text_lower = response_text.lower()
        
        # 简单的情绪识别
        positive_keywords = ['看多', '买入', '上涨', '突破', '强势', '利好', '加仓', 'bullish', 'buy']
        negative_keywords = ['看空', '卖出', '下跌', '跌破', '弱势', '利空', '减仓', 'bearish', 'sell']
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = '看多'
            advice = '买入'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = '看空'
            advice = '卖出'
        
        # 截取前500字符作为摘要
        summary = response_text[:500] if response_text else '无分析结果'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            confidence_level='低',
            analysis_summary=summary,
            key_points='JSON解析失败，仅供参考',
            risk_warning='分析结果可能不准确，建议结合其他信息判断',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        注意：为避免 API 速率限制，每次分析之间会有延迟
        
        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    """获取 Gemini 分析器实例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟上下文数据
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': '多头排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析测试 ===")
        result = analyzer.analyze(test_context)
        print(f"分析结果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳过测试")
