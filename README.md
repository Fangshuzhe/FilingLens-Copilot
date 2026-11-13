# FilingLens Copilot

Evidence-first annual report research copilot.

FilingLens Copilot 将上市公司年报转化为有出处、可追问、可导出的研究结论。

核心原则：

> 每个关键结论都能回到年报原文，每个数字都由程序计算。

## 当前能力

- PDF 年报上传与分页文本解析
- SQLite 本地存储，无需 MySQL
- 中文友好的 BM25 证据检索
- 兼容 OpenAI Chat Completions 接口的大模型问答
- 结构化研究底稿
- 结论与原文页码绑定
- 引用校验与证据不足拒答
- 研究板收藏与 Markdown 简报导出

## 运行

建议使用 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

设置模型配置：

```powershell
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_API_KEY = "your-api-key"
$env:LLM_MODEL = "gpt-4.1-mini"
```

启动应用：

```powershell
streamlit run app.py
```

也可以复制 `.env.example` 为 `.env` 后填写模型配置。

## 测试

```powershell
python -m pytest
```

## 项目结构

```text
app.py                    Streamlit 界面
filinglens/
  db.py                   SQLite 数据层
  llm.py                  OpenAI 兼容模型客户端
  models.py               核心数据模型
  parser.py               PDF 解析与切分
  reporting.py            Markdown 报告导出
  retrieval.py            中文 BM25 检索
  service.py              问答、证据校验与拒答
tests/                    自动化测试
data/                     本地数据库、上传文件和运行时数据
```

## 产品文档

- [产品方案](企业年报研究Copilot项目方案.md)

## 产品边界

本项目提供企业研究辅助，不提供股票买卖建议。
