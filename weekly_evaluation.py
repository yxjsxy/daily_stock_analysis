#!/usr/bin/env python3
"""
Weekly Stock Prediction Evaluation System v2.0
每周五评估股票预测准确性，模拟交易盈亏，形成闭环迭代

功能：
1. 收集本周预测数据
2. 获取实盘结果
3. 模拟1000股交易盈亏
4. 评估预测准确性
5. 生成改进建议
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List
import subprocess

# 配置
PROJECT_DIR = Path(__file__).parent
REPORTS_DIR = PROJECT_DIR / "reports"
DB_PATH = PROJECT_DIR / "data" / "stock_analysis.db"
EVAL_DIR = PROJECT_DIR / "evaluations"
EVAL_DIR.mkdir(exist_ok=True)

# 模拟交易配置
SHARES_PER_STOCK = 1000  # 每只股票假定持仓

# Gemini API
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyCwMjBNpGvdURI1NJJB30AvwEWn9NzFw5Q")


@dataclass
class Prediction:
    """单次预测记录"""
    date: str
    stock_code: str
    stock_name: str
    sentiment_score: int
    trend_prediction: str  # 看多/看空/震荡
    operation_advice: str  # 买入/卖出/观望
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    close_price: Optional[float] = None
    
    def get_action(self) -> str:
        """解析操作建议为标准动作"""
        advice_lower = self.operation_advice.lower()
        if any(x in advice_lower for x in ['买入', '加仓', 'buy', '建仓']):
            return 'BUY'
        elif any(x in advice_lower for x in ['卖出', '减仓', 'sell', '清仓']):
            return 'SELL'
        else:
            return 'HOLD'


@dataclass 
class ActualResult:
    """实盘结果"""
    stock_code: str
    start_date: str
    end_date: str
    start_price: float
    end_price: float
    high_price: float
    low_price: float
    change_pct: float
    daily_data: List[tuple] = field(default_factory=list)  # [(date, open, high, low, close), ...]


@dataclass
class TradeSimulation:
    """模拟交易结果"""
    stock_code: str
    stock_name: str
    initial_shares: int
    initial_value: float
    final_shares: int
    final_value: float
    realized_pnl: float  # 已实现盈亏
    unrealized_pnl: float  # 未实现盈亏
    total_pnl: float
    total_pnl_pct: float
    trades: List[dict] = field(default_factory=list)  # 交易记录


@dataclass
class EvaluationResult:
    """评估结果"""
    stock_code: str
    stock_name: str
    predictions: list
    actual: ActualResult
    trade_sim: Optional[TradeSimulation]
    direction_correct: bool
    target_hit: bool
    stop_hit: bool
    avg_score: float
    actual_change: float
    evaluation_notes: str


def parse_report(report_path: Path) -> list[Prediction]:
    """解析单个报告文件，提取预测数据"""
    predictions = []
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    date_match = re.search(r'# 🎯 (\d{4}-\d{2}-\d{2})', content)
    if not date_match:
        return predictions
    report_date = date_match.group(1)
    
    stock_sections = re.split(r'## [⚪🟠🔴🟢] ', content)[1:]
    
    for section in stock_sections:
        header_match = re.match(r'(.+?) \((\d{6})\)', section)
        if not header_match:
            continue
        
        stock_name = header_match.group(1)
        stock_code = header_match.group(2)
        
        score_match = re.search(r'sentiment_score["\s:]+(\d+)', section)
        score = int(score_match.group(1)) if score_match else 50
        
        trend_match = re.search(r'trend_prediction["\s:]+["\']?([^"\'}\n]+)', section)
        trend = trend_match.group(1).strip() if trend_match else "震荡"
        
        advice_match = re.search(r'operation_advice["\s:]+["\']?([^"\'}\n]+)', section)
        advice = advice_match.group(1).strip() if advice_match else "观望"
        
        price_match = re.search(r'当前价[^\d]*(\d+\.?\d*)', section)
        close_price = float(price_match.group(1)) if price_match else None
        
        target_match = re.search(r'目标位[^\d]*(\d+\.?\d*)', section)
        target_price = float(target_match.group(1)) if target_match else None
        
        stop_match = re.search(r'止损位[^\d]*(\d+\.?\d*)', section)
        stop_loss = float(stop_match.group(1)) if stop_match else None
        
        predictions.append(Prediction(
            date=report_date,
            stock_code=stock_code,
            stock_name=stock_name,
            sentiment_score=score,
            trend_prediction=trend,
            operation_advice=advice,
            target_price=target_price,
            stop_loss=stop_loss,
            close_price=close_price
        ))
    
    return predictions


def get_weekly_predictions(weeks_ago: int = 0) -> dict[str, list[Prediction]]:
    """获取本周（或指定周）的预测数据"""
    today = datetime.now()
    
    # 本周一到今天（周五）
    if weeks_ago == 0:
        # 找到本周一
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = today
    else:
        start_of_week = today - timedelta(days=today.weekday() + 7 * weeks_ago)
        end_of_week = start_of_week + timedelta(days=4)  # 周一到周五
    
    predictions_by_stock = {}
    
    current_date = start_of_week
    while current_date <= end_of_week and current_date <= today:
        report_file = REPORTS_DIR / f"report_{current_date.strftime('%Y%m%d')}.md"
        
        if report_file.exists():
            daily_predictions = parse_report(report_file)
            for pred in daily_predictions:
                if pred.stock_code not in predictions_by_stock:
                    predictions_by_stock[pred.stock_code] = []
                predictions_by_stock[pred.stock_code].append(pred)
        
        current_date += timedelta(days=1)
    
    return predictions_by_stock


def get_actual_results(stock_code: str, start_date: str, end_date: str) -> Optional[ActualResult]:
    """获取实盘结果"""
    try:
        if DB_PATH.exists():
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT date, open, high, low, close 
                    FROM stock_daily 
                    WHERE code = ? AND date >= ? AND date <= ?
                    ORDER BY date
                ''', (stock_code, start_date, end_date))
                rows = cursor.fetchall()
                
                if rows and len(rows) >= 1:
                    daily_data = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
                    start_price = rows[0][4]  # close of first day
                    end_price = rows[-1][4]   # close of last day
                    high_price = max(r[2] for r in rows if r[2])
                    low_price = min(r[3] for r in rows if r[3])
                    change_pct = (end_price - start_price) / start_price * 100 if start_price else 0
                    
                    return ActualResult(
                        stock_code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        start_price=start_price,
                        end_price=end_price,
                        high_price=high_price,
                        low_price=low_price,
                        change_pct=change_pct,
                        daily_data=daily_data
                    )
        
        # Fallback to akshare if DB doesn't have data
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                start_date=start_date.replace('-', ''), 
                                end_date=end_date.replace('-', ''),
                                adjust="qfq")
        
        if df is not None and len(df) >= 1:
            daily_data = [(row['日期'], row['开盘'], row['最高'], row['最低'], row['收盘']) 
                         for _, row in df.iterrows()]
            start_price = df.iloc[0]['收盘']
            end_price = df.iloc[-1]['收盘']
            high_price = df['最高'].max()
            low_price = df['最低'].min()
            change_pct = (end_price - start_price) / start_price * 100
            
            return ActualResult(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                start_price=start_price,
                end_price=end_price,
                high_price=high_price,
                low_price=low_price,
                change_pct=change_pct,
                daily_data=daily_data
            )
    
    except Exception as e:
        print(f"获取 {stock_code} 实盘数据失败: {e}")
    
    return None


def simulate_trading(predictions: List[Prediction], actual: ActualResult) -> TradeSimulation:
    """
    模拟交易：假定周初持有1000股，根据每日预测执行操作
    
    规则：
    - 买入信号：如果空仓则全仓买入，已有仓位则持有
    - 卖出信号：如果有仓位则全部卖出
    - 观望信号：维持现状
    """
    if not predictions or not actual:
        return None
    
    stock_name = predictions[0].stock_name
    stock_code = predictions[0].stock_code
    
    # 初始状态：持有1000股
    shares = SHARES_PER_STOCK
    initial_price = actual.start_price
    initial_value = shares * initial_price
    
    cash = 0  # 卖出后的现金
    realized_pnl = 0
    trades = []
    
    # 按日期排序预测
    sorted_preds = sorted(predictions, key=lambda x: x.date)
    
    # 构建日期到价格的映射
    price_map = {d[0]: d[4] for d in actual.daily_data}  # date -> close
    
    for pred in sorted_preds:
        action = pred.get_action()
        current_price = price_map.get(pred.date, pred.close_price)
        
        if not current_price:
            continue
        
        if action == 'SELL' and shares > 0:
            # 卖出
            sell_value = shares * current_price
            realized_pnl += sell_value - (shares * initial_price)
            cash += sell_value
            trades.append({
                'date': pred.date,
                'action': 'SELL',
                'shares': shares,
                'price': current_price,
                'value': sell_value,
                'reason': pred.operation_advice
            })
            shares = 0
            
        elif action == 'BUY' and shares == 0 and cash > 0:
            # 买入
            shares = int(cash / current_price)
            buy_value = shares * current_price
            cash -= buy_value
            trades.append({
                'date': pred.date,
                'action': 'BUY',
                'shares': shares,
                'price': current_price,
                'value': buy_value,
                'reason': pred.operation_advice
            })
    
    # 计算最终价值
    final_price = actual.end_price
    final_shares_value = shares * final_price if shares > 0 else 0
    final_value = final_shares_value + cash
    
    unrealized_pnl = (final_price - initial_price) * shares if shares > 0 else 0
    total_pnl = final_value - initial_value
    total_pnl_pct = (total_pnl / initial_value) * 100 if initial_value > 0 else 0
    
    return TradeSimulation(
        stock_code=stock_code,
        stock_name=stock_name,
        initial_shares=SHARES_PER_STOCK,
        initial_value=initial_value,
        final_shares=shares,
        final_value=final_value,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        trades=trades
    )


def evaluate_predictions(predictions: list[Prediction], actual: ActualResult) -> EvaluationResult:
    """评估预测准确性"""
    if not predictions:
        return None
    
    avg_score = sum(p.sentiment_score for p in predictions) / len(predictions)
    
    # 方向判断
    predicted_direction = "多" if avg_score > 60 else ("空" if avg_score < 40 else "震荡")
    actual_direction = "多" if actual.change_pct > 2 else ("空" if actual.change_pct < -2 else "震荡")
    direction_correct = (predicted_direction == actual_direction) or \
                        (predicted_direction == "震荡" and abs(actual.change_pct) < 5)
    
    # 目标位/止损位检查
    target_hit = False
    stop_hit = False
    notes = []
    
    if direction_correct:
        notes.append("✅ 方向预测正确")
    else:
        notes.append(f"❌ 方向预测错误 (预测: {predicted_direction}, 实际: {actual_direction})")
    
    for pred in predictions:
        if pred.target_price and actual.high_price >= pred.target_price:
            target_hit = True
            notes.append(f"🎯 目标位 {pred.target_price} 已触及")
        if pred.stop_loss and actual.low_price <= pred.stop_loss:
            stop_hit = True
            notes.append(f"🛑 止损位 {pred.stop_loss} 已触发")
    
    # 模拟交易
    trade_sim = simulate_trading(predictions, actual)
    
    if trade_sim:
        pnl_emoji = "📈" if trade_sim.total_pnl >= 0 else "📉"
        notes.append(f"{pnl_emoji} 模拟盈亏: ¥{trade_sim.total_pnl:,.2f} ({trade_sim.total_pnl_pct:+.2f}%)")
    
    return EvaluationResult(
        stock_code=predictions[0].stock_code,
        stock_name=predictions[0].stock_name,
        predictions=predictions,
        actual=actual,
        trade_sim=trade_sim,
        direction_correct=direction_correct,
        target_hit=target_hit,
        stop_hit=stop_hit,
        avg_score=avg_score,
        actual_change=actual.change_pct,
        evaluation_notes="\n".join(notes)
    )


def generate_improvement_suggestions(evaluations: list[EvaluationResult]) -> str:
    """使用 Gemini 生成改进建议"""
    summary_data = []
    total_pnl = 0
    
    for eval in evaluations:
        entry = {
            "stock": f"{eval.stock_name}({eval.stock_code})",
            "avg_score": eval.avg_score,
            "actual_change": f"{eval.actual_change:.2f}%",
            "direction_correct": eval.direction_correct,
            "target_hit": eval.target_hit,
            "stop_hit": eval.stop_hit,
        }
        if eval.trade_sim:
            entry["simulated_pnl"] = f"¥{eval.trade_sim.total_pnl:,.2f}"
            entry["simulated_pnl_pct"] = f"{eval.trade_sim.total_pnl_pct:+.2f}%"
            entry["trades"] = eval.trade_sim.trades
            total_pnl += eval.trade_sim.total_pnl
        summary_data.append(entry)
    
    correct_count = sum(1 for e in evaluations if e.direction_correct)
    accuracy = correct_count / len(evaluations) * 100 if evaluations else 0
    
    prompt = f"""作为股票预测系统优化专家，分析以下一周评估结果并给出具体改进建议：

## 本周评估数据
{json.dumps(summary_data, ensure_ascii=False, indent=2)}

## 统计摘要
- 评估股票数: {len(evaluations)}
- 方向正确率: {accuracy:.1f}%
- 目标位触及: {sum(1 for e in evaluations if e.target_hit)}
- 止损触发: {sum(1 for e in evaluations if e.stop_hit)}
- **模拟总盈亏: ¥{total_pnl:,.2f}**

请分析并给出：

## 1. 本周表现总结
- 预测系统的优势和不足
- 盈亏原因分析

## 2. 具体改进建议
针对以下方面给出可执行的代码级改进建议：
- 技术指标权重调整
- 评分算法优化
- 买卖信号触发条件
- 风险控制参数

## 3. 下周优化重点
- 优先修改的模块
- 建议测试的参数变化

请用中文回答，格式清晰，建议要具体可执行。"""

    try:
        result = subprocess.run(
            ['gemini', prompt],
            capture_output=True,
            text=True,
            timeout=90
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Gemini 调用失败: {e}")
    
    return "无法生成改进建议（Gemini API 调用失败）"


def run_weekly_evaluation(weeks_ago: int = 0) -> str:
    """运行每周评估"""
    print(f"🔄 开始本周评估...")
    
    today = datetime.now()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = today
    
    start_date = start_of_week.strftime('%Y-%m-%d')
    end_date = end_of_week.strftime('%Y-%m-%d')
    
    # 1. 获取预测数据
    predictions_by_stock = get_weekly_predictions(weeks_ago)
    
    if not predictions_by_stock:
        return "❌ 本周没有预测数据"
    
    print(f"📊 找到 {len(predictions_by_stock)} 只股票的预测数据")
    
    # 2. 评估每只股票
    evaluations = []
    total_pnl = 0
    
    for stock_code, predictions in predictions_by_stock.items():
        print(f"  评估 {stock_code}...")
        
        # 获取实盘数据
        actual = get_actual_results(stock_code, start_date, end_date)
        
        if actual:
            eval_result = evaluate_predictions(predictions, actual)
            if eval_result:
                evaluations.append(eval_result)
                if eval_result.trade_sim:
                    total_pnl += eval_result.trade_sim.total_pnl
                    pnl_str = f"¥{eval_result.trade_sim.total_pnl:+,.2f}"
                else:
                    pnl_str = "N/A"
                print(f"    {eval_result.stock_name}: {'✅' if eval_result.direction_correct else '❌'} {eval_result.actual_change:+.2f}% | 盈亏: {pnl_str}")
        else:
            print(f"    ⚠️ 无法获取实盘数据")
    
    if not evaluations:
        return "❌ 没有可评估的数据"
    
    # 3. 生成统计
    correct_count = sum(1 for e in evaluations if e.direction_correct)
    accuracy = correct_count / len(evaluations) * 100
    
    # 4. 生成改进建议
    print("🤖 生成 AI 改进建议...")
    suggestions = generate_improvement_suggestions(evaluations)
    
    # 5. 生成报告
    report = f"""# 📈 股票预测周度评估报告 v2.0

**评估周期**: {start_date} ~ {end_date}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**模拟初始持仓**: 每只股票 {SHARES_PER_STOCK} 股

---

## 💰 本周模拟交易汇总

| 指标 | 数值 |
|------|------|
| 评估股票数 | {len(evaluations)} |
| 方向正确 | {correct_count} |
| 方向准确率 | **{accuracy:.1f}%** |
| 目标位触及 | {sum(1 for e in evaluations if e.target_hit)} |
| 止损触发 | {sum(1 for e in evaluations if e.stop_hit)} |
| **总模拟盈亏** | **¥{total_pnl:+,.2f}** |

---

## 📋 各股票详细评估

"""
    
    for eval in evaluations:
        status = "✅" if eval.direction_correct else "❌"
        pnl_status = "📈" if (eval.trade_sim and eval.trade_sim.total_pnl >= 0) else "📉"
        
        report += f"""### {status} {eval.stock_name} ({eval.stock_code})

**预测表现**
| 指标 | 数值 |
|------|------|
| 平均评分 | {eval.avg_score:.0f} |
| 实际涨跌 | {eval.actual_change:+.2f}% |
| 最高价 | {eval.actual.high_price:.2f} |
| 最低价 | {eval.actual.low_price:.2f} |
| 方向正确 | {'是' if eval.direction_correct else '否'} |
| 目标位触及 | {'是' if eval.target_hit else '否'} |
| 止损触发 | {'是' if eval.stop_hit else '否'} |

"""
        
        if eval.trade_sim:
            report += f"""**{pnl_status} 模拟交易**
| 指标 | 数值 |
|------|------|
| 初始持仓 | {eval.trade_sim.initial_shares} 股 |
| 初始价值 | ¥{eval.trade_sim.initial_value:,.2f} |
| 最终持仓 | {eval.trade_sim.final_shares} 股 |
| 最终价值 | ¥{eval.trade_sim.final_value:,.2f} |
| **盈亏** | **¥{eval.trade_sim.total_pnl:+,.2f} ({eval.trade_sim.total_pnl_pct:+.2f}%)** |

"""
            if eval.trade_sim.trades:
                report += "**交易记录**\n"
                for trade in eval.trade_sim.trades:
                    report += f"- {trade['date']}: {trade['action']} {trade['shares']}股 @ ¥{trade['price']:.2f} ({trade['reason']})\n"
                report += "\n"
        
        report += f"**评估备注**: {eval.evaluation_notes}\n\n---\n\n"
    
    report += f"""## 🤖 AI 改进建议

{suggestions}

---

## 🔧 本周 Vibe Coding 优化任务

基于以上评估结果，今晚 Night Vibe Coding 应重点修改：
1. 根据 AI 建议调整 `analyzer.py` 中的评分权重
2. 优化 `config.py` 中的买卖信号触发阈值
3. 测试新参数并记录变更

---

*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 6. 保存报告
    report_filename = f"weekly_eval_{start_date.replace('-', '')}_{end_date.replace('-', '')}.md"
    report_path = EVAL_DIR / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 评估报告已保存: {report_path}")
    print(f"💰 本周模拟总盈亏: ¥{total_pnl:+,.2f}")
    
    return report


if __name__ == "__main__":
    import sys
    weeks_ago = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    report = run_weekly_evaluation(weeks_ago)
    print("\n" + "="*60)
    print(report[:3000] + "..." if len(report) > 3000 else report)
