# 📈 Daily Stock Analysis — A股智能分析系统

AI-powered A-share stock analysis system with technical analysis, news grounding, and automated reporting.

## ✨ Features

- 📊 **技术分析** — K 线形态、缠论分析（chan_analyzer）、信号优化
- 🔍 **消息面分析** — Google Search Grounding 实时新闻整合
- 🤖 **AI 研报** — Gemini API 生成综合分析报告
- 📱 **飞书推送** — 分析结果自动推送到飞书文档
- 🌐 **Web UI** — 可视化分析界面
- ⏰ **定时调度** — 自动每日分析（scheduler）
- 📈 **信号优化** — 买卖信号优化器（signal_optimizer）
- 📋 **周度评估** — 自动周度策略评估（weekly_evaluation）
- 🐳 **Docker 部署** — docker-compose 一键部署

## 🛠️ Tech Stack

- Python 3.10+
- Gemini API + Google Search Grounding
- 缠论（Chan Theory）分析引擎
- FastAPI (Web UI)
- Docker / docker-compose
- 飞书 API

## 📂 Key Files

```
main.py               # 主入口
analyzer.py            # AI 分析层（Gemini API）
chan_analyzer.py        # 缠论分析器
stock_analyzer.py       # 股票数据分析
market_analyzer.py      # 市场整体分析
signal_optimizer.py     # 信号优化
scheduler.py           # 定时调度
webui.py               # Web 界面
notification.py        # 通知推送
feishu_doc.py          # 飞书文档集成
weekly_evaluation.py   # 周度评估
config.py              # 配置管理
```

## 🚀 Usage

```bash
# 直接运行
python main.py

# Docker 部署
docker-compose up -d

# Web UI
python webui.py
```
