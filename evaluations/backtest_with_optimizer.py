#!/usr/bin/env python3
"""
使用信号优化器重新回测本周数据
对比优化前后的效果
"""

import sys
sys.path.insert(0, '..')

from signal_optimizer import SignalOptimizer

# 实盘数据
PRICES = {
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

# 原始LLM预测 (来自报告)
ORIGINAL_PREDICTIONS = {
    "2026-02-03": {  # 报告在2/3晚生成，用于2/4操作
        "300751": {
            "signal": "强烈买入", 
            "price": 335.0,
            "indicators": {
                "bias_ma5": 3.54,
                "consecutive_up_days": 2,
                "pct_chg": 10.12,  # 当日涨10%!
                "rsi": 70,
                "volume_ratio": 1.21,
            }
        },
        "002300": {
            "signal": "卖出", 
            "price": 8.69,
            "indicators": {
                "bias_ma5": 1.52,
                "consecutive_up_days": 0,
                "consecutive_down_days": 0,
                "pct_chg": -0.11,
                "rsi": 40,
                "volume_ratio": 0.78,
            }
        },
        "300666": {
            "signal": "卖出", 
            "price": 113.48,
            "stock_info": {"is_suspended": True},
        },
    },
    "2026-02-04": {
        "300751": {
            "signal": "观望",  # LLM已经谨慎了
            "price": 328.6,
            "indicators": {
                "bias_ma5": 2.43,
                "consecutive_up_days": 0,
                "pct_chg": -1.91,
                "rsi": 55,
                "volume_ratio": 1.04,
            }
        },
        "002300": {
            "signal": "买入",
            "price": 9.56,
            "indicators": {
                "bias_ma5": 9.0,  # 乖离率高!
                "pct_chg": 10.01,  # 涨停!
                "rsi": 65,
                "volume_ratio": 2.18,
            }
        },
        "300666": {
            "signal": "观望",
            "price": 113.48,
            "stock_info": {"is_suspended": True},
        },
    },
    "2026-02-05": {
        "300751": {
            "signal": "卖出",
            "price": 294.21,
            "indicators": {
                "bias_ma5": -6.45,  # 超跌
                "consecutive_down_days": 1,
                "pct_chg": -10.47,  # 大跌
                "rsi": 30,
                "volume_ratio": 0.96,
            }
        },
        "002300": {
            "signal": "观望",
            "price": 9.32,
            "indicators": {
                "bias_ma5": 4.13,
                "pct_chg": -2.51,
                "rsi": 50,
            }
        },
        "300666": {
            "signal": "减仓",
            "price": 113.48,
            "stock_info": {
                "just_resumed": True,
                "resume_reason": "资产重组",
                "suspend_days": 7,
            },
        },
    },
}


def apply_optimizer(date: str, predictions: dict) -> dict:
    """对预测应用优化器"""
    optimizer = SignalOptimizer()
    optimized = {}
    
    for code, pred in predictions.items():
        indicators = pred.get('indicators', {})
        stock_info = pred.get('stock_info', {})
        
        result = optimizer.optimize(
            signal=pred['signal'],
            confidence=0.7,
            indicators=indicators,
            stock_info=stock_info,
            context={'prev_signal': '', 'prev_pct_chg': 0}
        )
        
        optimized[code] = {
            'original_signal': pred['signal'],
            'optimized_signal': result['final_signal'],
            'blocked': result['blocked'],
            'adjustments': result['adjustments'],
            'price': pred['price'],
        }
    
    return optimized


def calculate_accuracy(predictions: dict, date: str) -> dict:
    """计算预测准确率"""
    
    # 获取次日价格变化
    next_dates = {
        "2026-02-03": "2026-02-04",
        "2026-02-04": "2026-02-05",
        "2026-02-05": "2026-02-06",
    }
    
    next_date = next_dates.get(date)
    if not next_date:
        return {}
    
    results = {}
    for code, pred in predictions.items():
        if code not in PRICES:
            continue
        if date not in PRICES[code] or next_date not in PRICES[code]:
            continue
        
        price_today = PRICES[code][date]
        price_next = PRICES[code][next_date]
        pct_change = (price_next - price_today) / price_today * 100
        
        signal = pred.get('optimized_signal', pred.get('signal'))
        
        # 判断是否正确
        if signal in ['买入', '强烈买入', '加仓']:
            correct = pct_change > 0
        elif signal in ['卖出', '强烈卖出', '减仓']:
            correct = pct_change < 0
        else:  # 观望
            correct = abs(pct_change) < 3  # 观望时小幅波动算对
        
        results[code] = {
            'signal': signal,
            'pct_change': pct_change,
            'correct': correct,
        }
    
    return results


def main():
    print("=" * 60)
    print("📊 信号优化器回测对比 (2026-02-03 ~ 2026-02-06)")
    print("=" * 60)
    
    total_original_correct = 0
    total_optimized_correct = 0
    total_count = 0
    
    for date, preds in ORIGINAL_PREDICTIONS.items():
        print(f"\n{'=' * 60}")
        print(f"📅 {date} 预测 → 次日验证")
        print("=" * 60)
        
        # 应用优化器
        optimized = apply_optimizer(date, preds)
        
        # 计算原始准确率
        original_results = {}
        for code, pred in preds.items():
            if code in PRICES:
                next_date = {"2026-02-03": "2026-02-04", "2026-02-04": "2026-02-05", "2026-02-05": "2026-02-06"}.get(date)
                if next_date and date in PRICES[code] and next_date in PRICES[code]:
                    pct_change = (PRICES[code][next_date] - PRICES[code][date]) / PRICES[code][date] * 100
                    signal = pred['signal']
                    if signal in ['买入', '强烈买入', '加仓']:
                        correct = pct_change > 0
                    elif signal in ['卖出', '强烈卖出', '减仓']:
                        correct = pct_change < 0
                    else:
                        correct = abs(pct_change) < 3
                    original_results[code] = {'signal': signal, 'pct_change': pct_change, 'correct': correct}
        
        # 计算优化后准确率
        optimized_preds = {code: {'optimized_signal': opt['optimized_signal']} for code, opt in optimized.items()}
        optimized_results = calculate_accuracy(optimized_preds, date)
        
        # 显示对比
        for code in preds:
            orig = original_results.get(code, {})
            opt = optimized.get(code, {})
            opt_result = optimized_results.get(code, {})
            
            orig_signal = preds[code]['signal']
            opt_signal = opt.get('optimized_signal', orig_signal)
            pct_change = orig.get('pct_change', 0)
            
            orig_correct = orig.get('correct', None)
            opt_correct = opt_result.get('correct', None)
            
            # 统计
            if orig_correct is not None:
                total_count += 1
                if orig_correct:
                    total_original_correct += 1
                if opt_correct:
                    total_optimized_correct += 1
            
            # 显示
            orig_mark = "✅" if orig_correct else "❌" if orig_correct is not None else "⚪"
            opt_mark = "✅" if opt_correct else "❌" if opt_correct is not None else "⚪"
            
            changed = "🔄" if orig_signal != opt_signal else "  "
            
            adjustments = opt.get('adjustments', [])
            adj_str = adjustments[0] if adjustments else ""
            
            print(f"\n  {code}:")
            print(f"    原始信号: {orig_signal:8} → 实际{pct_change:+.1f}% {orig_mark}")
            print(f"    优化信号: {opt_signal:8} → 实际{pct_change:+.1f}% {opt_mark} {changed}")
            if adj_str:
                print(f"    调整原因: {adj_str}")
    
    # 汇总
    print(f"\n{'=' * 60}")
    print("📈 准确率对比")
    print("=" * 60)
    
    orig_acc = total_original_correct / total_count * 100 if total_count > 0 else 0
    opt_acc = total_optimized_correct / total_count * 100 if total_count > 0 else 0
    improvement = opt_acc - orig_acc
    
    print(f"  原始准确率: {total_original_correct}/{total_count} = {orig_acc:.1f}%")
    print(f"  优化后准确率: {total_optimized_correct}/{total_count} = {opt_acc:.1f}%")
    print(f"  提升: {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"\n  🎉 优化有效！准确率提升 {improvement:.1f}%")
    elif improvement < 0:
        print(f"\n  ⚠️ 优化后准确率下降，需要调整规则")
    else:
        print(f"\n  ➡️ 准确率持平")


if __name__ == "__main__":
    main()
