# FilingLens Copilot

![FilingLens logo](assets/filinglens-logo.png)

An evidence-first annual report research copilot.

FilingLens turns lengthy public-company reports into traceable, searchable, comparable, and exportable research findings.

> Every key conclusion should trace back to the source report. Every number should be calculated by code.

## Features

- Clean chat-first interface with a compact sidebar
- General conversation without uploading a report
- Streaming responses in General Chat
- Local SQLite conversation history that survives page refreshes
- Conversation history with reopen and delete actions
- PDF annual-report upload and page-aware text parsing
- SQLite storage with no MySQL requirement
- BM25 evidence retrieval for English and Chinese documents
- OpenAI Chat Completions-compatible model integration
- Structured company research briefs
- Evidence IDs linked to source pages
- Citation validation and insufficient-evidence refusal
- PDF page preview with attempted source-text highlighting
- Financial fact discovery and source-table extraction
- Year-over-year annual-report change analysis
- Two-company comparison
- Research Board with Markdown and HTML export

The model answers in the same language as the user's latest question. English questions receive English answers.

## Current Limitations

- Scanned PDFs without a text layer are not yet processed with OCR.
- Financial facts are source-location aids; values with different definitions are not combined automatically.
- Streamlit Community Cloud uses ephemeral storage unless an external database is connected.

## Run Locally

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Configure a model:

```powershell
$env:LLM_BASE_URL = "https://openrouter.ai/api/v1"
$env:LLM_API_KEY = "your-openrouter-key"
$env:LLM_MODEL = "baidu/ernie-5.5-flash-preview:free"
```

Start the application:

```powershell
streamlit run app.py
```

You can also copy `.env.example` to `.env` and fill in the values.

## Streamlit Community Cloud

Deployment settings:

- Repository: `Fangshuzhe/FilingLens-Copilot`
- Branch: `main`
- Main file path: `app.py`
- Python: `3.12`

Add the following values under App settings → Secrets:

```toml
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_API_KEY = "replace-with-your-openrouter-key"
LLM_MODEL = "baidu/ernie-5.5-flash-preview:free"
LLM_TIMEOUT_SECONDS = "120"
FILINGLENS_DATA_DIR = "/tmp/filinglens"
```

Community Cloud storage is temporary. Conversations and uploaded reports may be cleared after an instance restart. Use a managed database such as Supabase, Turso, or PostgreSQL when durable storage is required.

## Tests

```powershell
python -m pytest
```

## Project Structure

```text
app.py                    Streamlit application
filinglens/
  db.py                   SQLite data layer
  financial.py            Financial fact discovery
  llm.py                  OpenAI-compatible model client
  models.py               Core data models
  parser.py               PDF parsing and chunking
  pdf_preview.py          PDF page rendering and highlighting
  reporting.py            Markdown and HTML export
  retrieval.py            BM25 retrieval
  service.py              Q&A, change analysis, comparison, validation
  tables.py               PDF table extraction
tests/                    Automated tests
data/                     Local runtime data
```

## Product Documentation

- [Product plan](PRODUCT_PLAN.md)

## Brand Assets

- `assets/filinglens-mark.svg`: vector application icon
- `assets/filinglens-logo.svg`: horizontal light-background logo
- `assets/filinglens-logo-inverse.svg`: dark-background logo
- `assets/*.png`: preview exports

## Scope

FilingLens provides company research assistance. It does not provide buy, sell, or investment recommendations.

---

# FilingLens Copilot 中文说明

FilingLens 是一个证据优先的企业年报研究工作台，可将长篇上市公司年报转化为可追溯、可追问、可比较和可导出的研究结论。

核心能力：

- 无需上传年报即可直接对话
- 流式回答、会话持久化、历史会话重开与删除
- PDF 年报解析、页码保留和证据检索
- 带页码与原文引用的年报问答
- 结构化公司研究底稿
- 财务事实定位和原始表格提取
- 连续年度变化分析
- 双公司对比
- 研究板以及 Markdown/HTML 报告导出
- 英文提问时使用英文回答

本地运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
streamlit run app.py
```

Streamlit Community Cloud 的文件系统为临时存储。若需长期保存对话和年报，应接入 Supabase、Turso 或 PostgreSQL。

本项目提供企业研究辅助，不提供股票买卖建议。
