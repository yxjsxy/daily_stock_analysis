# A股智能分析系统 测试指南

## 🔧 环境准备

### 本地环境
```bash
cd ~/Documents/vibe_coding/daily_stock_analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 环境变量
```bash
export GEMINI_API_KEY="your-key"
export TELEGRAM_BOT_TOKEN="your-token"  # 可选
export TELEGRAM_CHAT_ID="your-chat-id"  # 可选
```

---

## 🧪 功能测试

### 1. 数据获取测试
```bash
# 测试 AkShare 数据源
python -c "
import akshare as ak
df = ak.stock_zh_a_spot_em()
print(f'获取到 {len(df)} 只股票数据')
print(df.head())
"
```
**预期**: 输出 A 股实时行情列表

### 2. AI 分析测试
```bash
# 测试 Gemini API
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content('你好')
print(response.text)
"
```
**预期**: Gemini 返回响应

### 3. 单只股票分析
```bash
# 分析贵州茅台
python main.py --stock 600519 --no-push
```
**预期**: 生成分析报告，不推送

### 4. 完整运行测试
```bash
# 运行完整分析（使用测试股票池）
python main.py --test
```
**预期**: 分析所有自选股，生成报告

---

## 📤 推送测试

### Telegram 推送
```bash
python main.py --stock 600519 --telegram
```
**预期**: 收到 Telegram 消息

### 企业微信推送
```bash
python main.py --stock 600519 --wechat
```
**预期**: 收到企业微信消息

### 邮件推送
```bash
python main.py --stock 600519 --email
```
**预期**: 收到邮件

---

## 🔄 GitHub Actions 测试

### 手动触发
1. 进入 GitHub 仓库 → Actions
2. 选择 "Daily Stock Analysis"
3. 点击 "Run workflow"
4. 检查运行日志

### Secrets 验证
```yaml
# 检查必要的 Secrets
GEMINI_API_KEY: ✓/✗
TELEGRAM_BOT_TOKEN: ✓/✗
TELEGRAM_CHAT_ID: ✓/✗
```

### 定时任务
- **运行时间**: 每天 15:30 (UTC+8)
- **验证**: 检查 Actions 历史记录

---

## 📊 输出验证

### 报告格式检查
- [ ] 包含"决策仪表盘"
- [ ] 包含买入/止损/目标价位
- [ ] 包含检查清单 (✅⚠️❌)
- [ ] 包含技术指标分析
- [ ] 包含舆情分析

### 数据准确性
- [ ] 股价数据与东方财富一致
- [ ] MA 均线计算正确
- [ ] 成交量数据正确

---

## 🐛 常见问题排查

### AkShare 获取失败
```bash
# 检查网络
curl -I https://www.eastmoney.com

# 更新 AkShare
pip install --upgrade akshare
```

### Gemini API 限流
- 检查配额: https://aistudio.google.com
- 切换到 OpenAI 兼容 API

### 推送失败
```bash
# 测试 Telegram Bot
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe"
```

---

## 🚀 性能测试

| 场景 | 预期时间 | 实际时间 |
|------|----------|----------|
| 单只股票分析 | < 30s | |
| 5只股票完整分析 | < 3min | |
| 报告生成 | < 5s | |
| Telegram 推送 | < 2s | |

---

## ✅ 发布 Checklist

- [ ] 本地运行测试通过
- [ ] API 密钥有效
- [ ] GitHub Actions 运行成功
- [ ] 推送渠道正常
- [ ] 数据准确性验证
- [ ] 定时任务配置正确
