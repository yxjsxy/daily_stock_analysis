#!/usr/bin/env python3
"""
本周回测分析 (2026-02-03 ~ 2026-02-06)
模拟资金: ¥10,000 (虚拟)
每只股票: 1000股基准单位
"""

import json
from datetime import datetime

# 实盘数据
prices = {
    "300751": {  # 迈为股份
        "2026-02-02": 304.20,
        "2026-02-03": 335.00,
        "2026-02-04": 328.60,
        "2026-02-05": 294.21,
        "2026-02-06": 302.15,
    },
    "002300": {  # 太阳电缆
        "2026-02-02": 8.70,
        "2026-02-03": 8.69,
        "2026-02-04": 9.56,
        "2026-02-05": 9.32,
        "2026-02-06": 9.25,
    },
    "300666": {  # 江丰电子 (停牌到2/6)
        "2026-02-02": 113.48,
        "2026-02-06": 123.61,
    },
}

# 预测信号 (来自每日报告)
# 报告在收盘后生成，信号用于次日操作
predictions = {
    "2026-02-03": {  # 2/3晚报告，用于2/4操作
        "300751": {"signal": "强烈买入", "price": 335.0, "target": 360.0, "stop": 300.0, "position": 0.7},
        "002300": {"signal": "卖出", "price": 8.69, "target": 8.98, "stop": 8.40, "position": 0},
        "300666": {"signal": "卖出", "price": 113.48, "target": 116.8, "stop": 110.0, "position": 0},
    },
    "2026-02-04": {  # 2/4晚报告，用于2/5操作
        "300751": {"signal": "观望", "price": 328.6, "target": 340.0, "stop": 305.0, "position": 0.3},
        "002300": {"signal": "买入", "price": 9.56, "target": 11.5, "stop": 8.64, "position": 0.4},
        "300666": {"signal": "观望", "price": 113.48, "note": "停牌", "position": 0},
    },
    "2026-02-05": {  # 2/5晚报告，用于2/6操作
        "300751": {"signal": "卖出", "price": 294.21, "target": 308.0, "stop": 278.0, "position": 0.1},
        "002300": {"signal": "观望", "price": 9.32, "target": 9.9, "stop": 8.68, "position": 0.2},
        "300666": {"signal": "减仓", "price": 113.48, "note": "停牌", "position": 0},
    },
}

def simulate_trades():
    """模拟交易"""
    
    # 初始状态：假设2/2收盘时空仓
    positions = {
        "300751": {"shares": 0, "avg_cost": 0},
        "002300": {"shares": 0, "avg_cost": 0},
        "300666": {"shares": 0, "avg_cost": 0},
    }
    
    base_shares = 1000  # 基准股数
    cash = 10000  # 虚拟现金 (仅做标记，不限制交易)
    trade_log = []
    
    # 模拟交易序列
    trades = [
        # 2/4 开盘操作 (基于2/3报告)
        {
            "date": "2026-02-04",
            "actions": [
                {"code": "300751", "action": "buy", "reason": "强烈买入信号", "target_pos": 0.7},
                {"code": "002300", "action": "sell", "reason": "卖出信号", "target_pos": 0},
                {"code": "300666", "action": "hold", "reason": "停牌"},
            ]
        },
        # 2/5 开盘操作 (基于2/4报告)
        {
            "date": "2026-02-05",
            "actions": [
                {"code": "300751", "action": "reduce", "reason": "观望/减仓", "target_pos": 0.3},
                {"code": "002300", "action": "buy", "reason": "买入信号", "target_pos": 0.4},
                {"code": "300666", "action": "hold", "reason": "停牌"},
            ]
        },
        # 2/6 开盘操作 (基于2/5报告)
        {
            "date": "2026-02-06",
            "actions": [
                {"code": "300751", "action": "sell", "reason": "卖出信号", "target_pos": 0.1},
                {"code": "002300", "action": "hold", "reason": "观望"},
                {"code": "300666", "action": "sell", "reason": "减仓/复牌卖出", "target_pos": 0},
            ]
        },
    ]
    
    pnl_details = []
    
    for day_trades in trades:
        date = day_trades["date"]
        print(f"\n{'='*50}")
        print(f"📅 {date} 交易执行")
        print('='*50)
        
        for act in day_trades["actions"]:
            code = act["code"]
            action = act["action"]
            
            # 获取当日开盘价 (实际交易价)
            if date not in prices.get(code, {}):
                print(f"  {code}: 停牌，跳过")
                continue
                
            # 使用前一日收盘价模拟 (简化)
            prev_dates = sorted([d for d in prices[code].keys() if d < date])
            if not prev_dates:
                continue
            prev_close = prices[code][prev_dates[-1]]
            today_close = prices[code][date]
            
            current_shares = positions[code]["shares"]
            target_shares = int(base_shares * act.get("target_pos", 0))
            
            if action == "buy" and current_shares < target_shares:
                buy_shares = target_shares - current_shares
                buy_cost = buy_shares * prev_close
                positions[code]["shares"] = target_shares
                positions[code]["avg_cost"] = prev_close
                print(f"  🟢 {code} 买入 {buy_shares}股 @ {prev_close:.2f} = ¥{buy_cost:,.0f}")
                trade_log.append({"date": date, "code": code, "action": "买入", "shares": buy_shares, "price": prev_close})
                
            elif action == "sell" and current_shares > target_shares:
                sell_shares = current_shares - target_shares
                sell_value = sell_shares * prev_close
                pnl = (prev_close - positions[code]["avg_cost"]) * sell_shares if positions[code]["avg_cost"] > 0 else 0
                positions[code]["shares"] = target_shares
                print(f"  🔴 {code} 卖出 {sell_shares}股 @ {prev_close:.2f} = ¥{sell_value:,.0f} (盈亏: ¥{pnl:+,.0f})")
                trade_log.append({"date": date, "code": code, "action": "卖出", "shares": sell_shares, "price": prev_close, "pnl": pnl})
                pnl_details.append({"code": code, "pnl": pnl})
                
            elif action == "reduce" and current_shares > target_shares:
                sell_shares = current_shares - target_shares
                sell_value = sell_shares * prev_close
                pnl = (prev_close - positions[code]["avg_cost"]) * sell_shares if positions[code]["avg_cost"] > 0 else 0
                positions[code]["shares"] = target_shares
                print(f"  🟡 {code} 减仓 {sell_shares}股 @ {prev_close:.2f} = ¥{sell_value:,.0f} (盈亏: ¥{pnl:+,.0f})")
                trade_log.append({"date": date, "code": code, "action": "减仓", "shares": sell_shares, "price": prev_close, "pnl": pnl})
                pnl_details.append({"code": code, "pnl": pnl})
                
            else:
                print(f"  ⚪ {code} 持仓不变 ({current_shares}股)")
    
    # 计算最终持仓市值 (以2/6收盘价计算)
    print(f"\n{'='*50}")
    print("📊 期末持仓 (2026-02-06 收盘)")
    print('='*50)
    
    total_value = 0
    total_cost = 0
    for code, pos in positions.items():
        if pos["shares"] > 0:
            if "2026-02-06" in prices[code]:
                close_price = prices[code]["2026-02-06"]
            else:
                close_price = list(prices[code].values())[-1]
            market_value = pos["shares"] * close_price
            cost_value = pos["shares"] * pos["avg_cost"]
            unrealized_pnl = market_value - cost_value
            total_value += market_value
            total_cost += cost_value
            print(f"  {code}: {pos['shares']}股 @ {close_price:.2f} = ¥{market_value:,.0f} (浮盈: ¥{unrealized_pnl:+,.0f})")
    
    realized_pnl = sum(p["pnl"] for p in pnl_details)
    unrealized_pnl = total_value - total_cost
    total_pnl = realized_pnl + unrealized_pnl
    
    print(f"\n{'='*50}")
    print("💰 盈亏汇总")
    print('='*50)
    print(f"  已实现盈亏: ¥{realized_pnl:+,.0f}")
    print(f"  未实现盈亏: ¥{unrealized_pnl:+,.0f}")
    print(f"  总盈亏: ¥{total_pnl:+,.0f}")
    
    return {
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "positions": positions,
        "trade_log": trade_log,
    }

def analyze_prediction_accuracy():
    """分析预测准确性"""
    
    print(f"\n{'='*50}")
    print("🎯 预测准确性分析")
    print('='*50)
    
    analysis = []
    
    # 2/3 预测 vs 2/4 实际
    print("\n📌 2/3 预测 → 2/4 验证:")
    
    # 300751: 预测买入 @ 335, 2/4 收盘 328.6 (-1.9%)
    pred = predictions["2026-02-03"]["300751"]
    actual_change = (prices["300751"]["2026-02-04"] - prices["300751"]["2026-02-03"]) / prices["300751"]["2026-02-03"] * 100
    result = "❌ 错误" if actual_change < 0 else "✅ 正确"
    print(f"  300751: 预测{pred['signal']} → 实际{actual_change:+.1f}% {result}")
    analysis.append({"date": "2026-02-03", "code": "300751", "pred": pred["signal"], "actual": actual_change, "correct": actual_change > 0})
    
    # 002300: 预测卖出 @ 8.69, 2/4 收盘 9.56 (+10%)
    actual_change = (prices["002300"]["2026-02-04"] - prices["002300"]["2026-02-03"]) / prices["002300"]["2026-02-03"] * 100
    result = "✅ 正确" if actual_change < 0 else "❌ 错误"
    print(f"  002300: 预测卖出 → 实际{actual_change:+.1f}% {result}")
    analysis.append({"date": "2026-02-03", "code": "002300", "pred": "卖出", "actual": actual_change, "correct": actual_change < 0})
    
    # 2/4 预测 vs 2/5 实际
    print("\n📌 2/4 预测 → 2/5 验证:")
    
    # 300751: 预测观望/减仓, 2/5 收盘 294.21 (-10.5%)
    actual_change = (prices["300751"]["2026-02-05"] - prices["300751"]["2026-02-04"]) / prices["300751"]["2026-02-04"] * 100
    result = "✅ 正确 (预判了下跌风险)" if actual_change < 0 else "❌ 错误"
    print(f"  300751: 预测观望/减仓 → 实际{actual_change:+.1f}% {result}")
    analysis.append({"date": "2026-02-04", "code": "300751", "pred": "观望/减仓", "actual": actual_change, "correct": True})
    
    # 002300: 预测买入 @ 9.56, 2/5 收盘 9.32 (-2.5%)
    actual_change = (prices["002300"]["2026-02-05"] - prices["002300"]["2026-02-04"]) / prices["002300"]["2026-02-04"] * 100
    result = "❌ 错误" if actual_change < 0 else "✅ 正确"
    print(f"  002300: 预测买入 → 实际{actual_change:+.1f}% {result}")
    analysis.append({"date": "2026-02-04", "code": "002300", "pred": "买入", "actual": actual_change, "correct": actual_change > 0})
    
    # 2/5 预测 vs 2/6 实际
    print("\n📌 2/5 预测 → 2/6 验证:")
    
    # 300751: 预测卖出 @ 294.21, 2/6 收盘 302.15 (+2.7%)
    actual_change = (prices["300751"]["2026-02-06"] - prices["300751"]["2026-02-05"]) / prices["300751"]["2026-02-05"] * 100
    result = "✅ 正确" if actual_change < 0 else "❌ 错误 (反弹了)"
    print(f"  300751: 预测卖出 → 实际{actual_change:+.1f}% {result}")
    analysis.append({"date": "2026-02-05", "code": "300751", "pred": "卖出", "actual": actual_change, "correct": actual_change < 0})
    
    # 002300: 预测观望
    actual_change = (prices["002300"]["2026-02-06"] - prices["002300"]["2026-02-05"]) / prices["002300"]["2026-02-05"] * 100
    print(f"  002300: 预测观望 → 实际{actual_change:+.1f}% ⚪ 观望正确")
    
    # 300666: 预测减仓, 复牌涨8.9%
    actual_change = (prices["300666"]["2026-02-06"] - 113.48) / 113.48 * 100
    result = "❌ 错误 (复牌大涨)"
    print(f"  300666: 预测减仓 → 实际{actual_change:+.1f}% {result}")
    analysis.append({"date": "2026-02-05", "code": "300666", "pred": "减仓", "actual": actual_change, "correct": actual_change < 0})
    
    # 统计准确率
    correct = sum(1 for a in analysis if a["correct"])
    total = len(analysis)
    accuracy = correct / total * 100
    
    print(f"\n📈 准确率: {correct}/{total} = {accuracy:.1f}%")
    
    return analysis

if __name__ == "__main__":
    print("🔄 本周回测分析 (2026-02-03 ~ 2026-02-06)")
    print("=" * 50)
    
    # 模拟交易
    results = simulate_trades()
    
    # 分析预测准确性
    accuracy = analyze_prediction_accuracy()
    
    # 问题总结
    print(f"\n{'='*50}")
    print("🔍 发现的问题")
    print('='*50)
    print("""
1. 【追涨杀跌】2/3预测强烈买入300751 @ 335高点，次日即下跌
2. 【错失行情】2/3预测卖出002300，错过次日+10%涨停
3. 【复牌误判】300666预测减仓，实际复牌大涨8.9%
4. 【信号滞后】报告在收盘后生成，次日开盘价已变化

优化建议:
- 增加乖离率过滤：乖离率>5%时降低买入信号强度
- 加入涨停/跌停后冷却期：不在大涨次日追高
- 重组复牌单独处理：停牌股不给卖出信号
- 增加开盘价预判：考虑隔夜消息影响
""")
