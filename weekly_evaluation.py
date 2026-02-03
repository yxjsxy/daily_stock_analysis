#!/usr/bin/env python3
"""
Weekly Stock Prediction Evaluation System
每周评估股票预测准确性，形成闭环验证

功能：
1. 收集过去一周的预测数据
2. 获取实盘结果
3. 对比预测 vs 实际
4. 计算准确率
5. 生成改进建议
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import subprocess

# 配置
PROJECT_DIR = Path(__file__).parent
REPORTS_DIR = PROJECT_DIR / "reports"
DB_PATH = PROJECT_DIR / "data" / "stock_analysis.db"
EVAL_DIR = PROJECT_DIR / "evaluations"
EVAL_DIR.mkdir(exist_ok=True)

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
    change_pct: float  # 涨跌幅


@dataclass
class EvaluationResult:
    """评估结果"""
    stock_code: str
    stock_name: str
    predictions: list  # List[Prediction]
    actual: ActualResult
    direction_correct: bool  # 方向是否正确
    avg_score: float
    actual_change: float
    evaluation_notes: str


def parse_report(report_path: Path) -> list[Prediction]:
    """解析单个报告文件，提取预测数据"""
    predictions = []
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取日期
    date_match = re.search(r'# 🎯 (\d{4}-\d{2}-\d{2})', content)
    if not date_match:
        return predictions
    report_date = date_match.group(1)
    
    # 分割每只股票的分析
    stock_sections = re.split(r'## [⚪🟠🔴🟢] ', content)[1:]
    
    for section in stock_sections:
        # 提取股票信息
        header_match = re.match(r'(.+?) \((\d{6})\)', section)
        if not header_match:
            continue
        
        stock_name = header_match.group(1)
        stock_code = header_match.group(2)
        
        # 提取评分
        score_match = re.search(r'sentiment_score["\s:]+(\d+)', section)
        score = int(score_match.group(1)) if score_match else 50
        
        # 提取趋势预测
        trend_match = re.search(r'trend_prediction["\s:]+["\']?([^"\'}\n]+)', section)
        trend = trend_match.group(1).strip() if trend_match else "震荡"
        
        # 提取操作建议
        advice_match = re.search(r'operation_advice["\s:]+["\']?([^"\'}\n]+)', section)
        advice = advice_match.group(1).strip() if advice_match else "观望"
        
        # 提取当前价格
        price_match = re.search(r'当前价[^\d]*(\d+\.?\d*)', section)
        close_price = float(price_match.group(1)) if price_match else None
        
        # 提取目标位
        target_match = re.search(r'目标位[^\d]*(\d+\.?\d*)', section)
        target_price = float(target_match.group(1)) if target_match else None
        
        # 提取止损位
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


def get_weekly_predictions(weeks_ago: int = 1) -> dict[str, list[Prediction]]:
    """获取过去一周的预测数据"""
    today = datetime.now()
    
    # 计算上周的日期范围
    if weeks_ago == 0:
        # 本周 (周一到今天)
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = today
    else:
        # 上周
        start_of_week = today - timedelta(days=today.weekday() + 7 * weeks_ago)
        end_of_week = start_of_week + timedelta(days=6)
    
    predictions_by_stock = {}
    
    # 遍历日期范围内的报告
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
    """获取实盘结果 (从数据库或API)"""
    try:
        # 尝试从数据库获取
        if DB_PATH.exists():
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                # 数据库日期格式可能是 YYYY-MM-DD 或 YYYYMMDD
                cursor.execute('''
                    SELECT date, close, high, low 
                    FROM stock_daily 
                    WHERE code = ? AND date >= ? AND date <= ?
                    ORDER BY date
                ''', (stock_code, start_date, end_date))
                rows = cursor.fetchall()
                
                # 如果没有数据，尝试不带横杠的格式
                if not rows:
                    start_date_nodash = start_date.replace('-', '')
                    end_date_nodash = end_date.replace('-', '')
                    cursor.execute('''
                        SELECT date, close, high, low 
                        FROM stock_daily 
                        WHERE code = ? AND date >= ? AND date <= ?
                        ORDER BY date
                    ''', (stock_code, start_date_nodash, end_date_nodash))
                    rows = cursor.fetchall()
                
                print(f"    [DB] 找到 {len(rows)} 条数据")
                if rows and len(rows) >= 1:
                    start_price = rows[0][1]
                    end_price = rows[-1][1]
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
                        change_pct=change_pct
                    )
        
        # 尝试使用 akshare 获取
        import akshare as ak
        
        # 格式化股票代码
        if stock_code.startswith('6'):
            full_code = f"sh{stock_code}"
        else:
            full_code = f"sz{stock_code}"
        
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                start_date=start_date.replace('-', ''), 
                                end_date=end_date.replace('-', ''),
                                adjust="qfq")
        
        if df is not None and len(df) >= 2:
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
                change_pct=change_pct
            )
    
    except Exception as e:
        print(f"获取 {stock_code} 实盘数据失败: {e}")
    
    return None


def evaluate_predictions(predictions: list[Prediction], actual: ActualResult) -> EvaluationResult:
    """评估预测准确性"""
    if not predictions:
        return None
    
    # 计算平均评分
    avg_score = sum(p.sentiment_score for p in predictions) / len(predictions)
    
    # 判断方向是否正确
    # 评分 > 60 = 看多, < 40 = 看空, 40-60 = 震荡
    predicted_direction = "多" if avg_score > 60 else ("空" if avg_score < 40 else "震荡")
    actual_direction = "多" if actual.change_pct > 2 else ("空" if actual.change_pct < -2 else "震荡")
    
    direction_correct = (predicted_direction == actual_direction) or \
                        (predicted_direction == "震荡" and abs(actual.change_pct) < 5)
    
    # 生成评估备注
    notes = []
    if direction_correct:
        notes.append("✅ 方向预测正确")
    else:
        notes.append(f"❌ 方向预测错误 (预测: {predicted_direction}, 实际: {actual_direction})")
    
    # 检查目标位是否触及
    for pred in predictions:
        if pred.target_price and actual.high_price >= pred.target_price:
            notes.append(f"🎯 目标位 {pred.target_price} 已触及 (最高 {actual.high_price})")
        if pred.stop_loss and actual.low_price <= pred.stop_loss:
            notes.append(f"🛑 止损位 {pred.stop_loss} 已触发 (最低 {actual.low_price})")
    
    return EvaluationResult(
        stock_code=predictions[0].stock_code,
        stock_name=predictions[0].stock_name,
        predictions=predictions,
        actual=actual,
        direction_correct=direction_correct,
        avg_score=avg_score,
        actual_change=actual.change_pct,
        evaluation_notes="\n".join(notes)
    )


def generate_improvement_suggestions(evaluations: list[EvaluationResult]) -> str:
    """使用 Gemini 生成改进建议"""
    # 准备数据摘要
    summary_data = []
    for eval in evaluations:
        summary_data.append({
            "stock": f"{eval.stock_name}({eval.stock_code})",
            "avg_score": eval.avg_score,
            "actual_change": f"{eval.actual_change:.2f}%",
            "direction_correct": eval.direction_correct,
            "notes": eval.evaluation_notes
        })
    
    prompt = f"""分析以下股票预测评估结果，并给出改进建议：

## 评估数据
{json.dumps(summary_data, ensure_ascii=False, indent=2)}

## 统计摘要
- 总预测数: {len(evaluations)}
- 方向正确率: {sum(1 for e in evaluations if e.direction_correct) / len(evaluations) * 100:.1f}%

请分析：
1. 预测系统的优势和不足
2. 哪些类型的股票预测更准确
3. 具体的改进建议（技术指标、情绪分析、风险控制等方面）
4. 建议调整的参数或策略

请用中文回答，格式清晰。"""

    try:
        # 调用 gemini CLI
        result = subprocess.run(
            ['gemini', prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Gemini 调用失败: {e}")
    
    return "无法生成改进建议"


def run_weekly_evaluation(weeks_ago: int = 1) -> str:
    """运行每周评估"""
    print(f"🔄 开始第 {weeks_ago} 周评估...")
    
    # 1. 获取预测数据
    predictions_by_stock = get_weekly_predictions(weeks_ago)
    
    if not predictions_by_stock:
        return "❌ 没有找到预测数据"
    
    print(f"📊 找到 {len(predictions_by_stock)} 只股票的预测数据")
    
    # 2. 获取日期范围
    all_dates = []
    for preds in predictions_by_stock.values():
        all_dates.extend([p.date for p in preds])
    
    if not all_dates:
        return "❌ 没有有效的预测日期"
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # 3. 评估每只股票
    evaluations = []
    for stock_code, predictions in predictions_by_stock.items():
        print(f"  评估 {stock_code}...")
        
        # 获取实盘数据
        actual = get_actual_results(stock_code, start_date, end_date)
        
        if actual:
            eval_result = evaluate_predictions(predictions, actual)
            if eval_result:
                evaluations.append(eval_result)
                print(f"    {eval_result.stock_name}: {'✅' if eval_result.direction_correct else '❌'} {eval_result.actual_change:+.2f}%")
        else:
            print(f"    ⚠️ 无法获取实盘数据")
    
    if not evaluations:
        return "❌ 没有可评估的数据"
    
    # 4. 生成统计
    correct_count = sum(1 for e in evaluations if e.direction_correct)
    accuracy = correct_count / len(evaluations) * 100
    
    # 5. 生成改进建议
    print("🤖 生成 AI 改进建议...")
    suggestions = generate_improvement_suggestions(evaluations)
    
    # 6. 生成报告
    report = f"""# 📈 股票预测周度评估报告

**评估周期**: {start_date} ~ {end_date}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 📊 总体表现

| 指标 | 数值 |
|------|------|
| 评估股票数 | {len(evaluations)} |
| 方向正确 | {correct_count} |
| 方向错误 | {len(evaluations) - correct_count} |
| **准确率** | **{accuracy:.1f}%** |

---

## 📋 详细评估

"""
    
    for eval in evaluations:
        status = "✅" if eval.direction_correct else "❌"
        report += f"""### {status} {eval.stock_name} ({eval.stock_code})

| 预测 | 实际 |
|------|------|
| 平均评分 | {eval.avg_score:.0f} |
| 周涨跌幅 | {eval.actual_change:+.2f}% |
| 最高价 | {eval.actual.high_price:.2f} |
| 最低价 | {eval.actual.low_price:.2f} |

**评估**: {eval.evaluation_notes}

---

"""
    
    report += f"""## 🤖 AI 改进建议

{suggestions}

---

*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 7. 保存报告
    report_filename = f"weekly_eval_{start_date.replace('-', '')}_{end_date.replace('-', '')}.md"
    report_path = EVAL_DIR / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 评估报告已保存: {report_path}")
    
    return report


if __name__ == "__main__":
    import sys
    weeks_ago = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    report = run_weekly_evaluation(weeks_ago)
    print("\n" + "="*50)
    print(report[:2000] + "..." if len(report) > 2000 else report)
