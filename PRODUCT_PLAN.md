# FilingLens Copilot Product Plan

Version: v0.3

Positioning: an evidence-first annual-report research workspace for company analysis.

## 1. Product Definition

FilingLens helps users transform lengthy public-company annual reports into traceable, searchable, comparable, and exportable research findings.

It addresses three concrete problems:

1. Important information is scattered across long reports.
2. General-purpose language models can produce unsupported conclusions.
3. Ad hoc Q&A does not naturally become a reusable research report.

Core promise:

> Every key conclusion should trace back to source text. Every number should be calculated by code.

FilingLens assists with company research but does not provide investment recommendations.

## 2. Users and Core Scenarios

Primary users:

- Independent researchers who need to understand a public company quickly
- Small research teams performing initial company screening and due diligence
- Strategy, consulting, and investment-research professionals tracking annual-report changes

Core jobs:

- Understand how a company makes money
- Produce a structured company and risk overview
- Verify the source of a conclusion
- Compare two companies
- Identify changes in management narrative across years
- Turn research into a shareable brief

## 3. Product Loop

```text
Create a research conversation
    ↓
Ask general business questions or upload annual reports
    ↓
Parse the source and build an evidence index
    ↓
Generate a structured research brief
    ↓
Continue with evidence Q&A or analysis tools
    ↓
Save findings to the Research Board
    ↓
Export a Markdown or HTML brief
```

FilingLens is a research workflow, not only a chat interface.

## 4. Information Architecture

### General Chat

- Works without a report upload
- Streams answers progressively
- Persists conversation history in SQLite
- Stores the active conversation ID in the URL
- Supports reopening and deleting old conversations
- Uses the same language as the user's latest question

### Current Document

Users select a company report as the context for:

- Research Brief
- Evidence Q&A
- Financial Facts
- Year-over-Year
- Company Comparison
- Research Board

### Research Brief

The system generates a structured overview covering:

- Business overview
- Revenue model
- Customer and market exposure
- Competitive advantages
- Growth drivers
- Operating risks
- Management priorities

Every factual claim includes evidence references.

### Evidence Q&A

- Retrieves relevant report passages
- Produces structured claims
- Binds each claim to evidence IDs and page numbers
- Allows users to open the original PDF page
- Refuses unsupported questions

### Financial Facts

- Detects company financial metrics in source text
- Extracts reported numbers
- Shows the source context and page
- Extracts visible source tables
- Avoids combining incompatible definitions automatically

### Year-over-Year Analysis

Requires two reports from the same company.

The system compares:

- Business descriptions
- Revenue mix
- Market and competition narrative
- Cost and profitability discussion
- Risk disclosures

### Company Comparison

Compares two companies across:

- Business model
- Revenue mix
- Customers and markets
- Growth drivers
- Competitive advantages
- Key risks

### Research Board

- Saves conclusions and source evidence
- Preserves report, page, and evidence metadata
- Exports Markdown or HTML research briefs

## 5. Functional Priorities

| Priority | Capability | User Value |
|---|---|---|
| P0 | Chat and local history | Immediate research without setup |
| P0 | PDF parsing and page retention | Reliable source context |
| P0 | Evidence retrieval and Q&A | Traceable answers |
| P0 | Citation validation | Reduced hallucination |
| P0 | Research brief | Fast company orientation |
| P0 | Research Board and export | Reusable output |
| P1 | Financial facts and tables | Faster numerical verification |
| P1 | Year-over-year analysis | Management narrative tracking |
| P1 | Company comparison | Competitive research |
| P2 | OCR for scanned reports | Broader document coverage |
| P2 | External durable storage | Cross-instance data retention |
| P2 | Watchlists and alerts | Continuous monitoring |

## 6. Model and Evidence Design

### Response Pipeline

```text
Identify intent and time range
    ↓
Retrieve passages by section, page, and keyword
    ↓
Generate structured claims
    ↓
Validate evidence IDs
    ↓
Display the answer or request a correction
```

### Information Labels

FilingLens distinguishes:

- Annual report fact: directly disclosed in the source
- Program result: produced by a deterministic rule or calculation
- Model judgment: interpretation based on supplied facts

Model judgment must never be presented as source text.

### Anti-Hallucination Rules

- Factual claims require evidence IDs
- Numbers must come from source evidence or deterministic extraction
- Unsupported questions return `insufficient_evidence`
- Invalid citations trigger a controlled retry
- Retrieved document instructions are treated as untrusted

### Language Behavior

- General Chat answers in the language of the user's latest question
- English questions receive English answers
- Structured analysis prompts produce English reports
- Identity and developer answers are handled deterministically

## 7. Interface Design

The interface follows a clean chat-first layout:

```text
Left: navigation, conversations, document controls
Center: chat, briefs, questions, or analysis
Right side within the conversation: user questions and avatars
```

Key interactions:

- User questions align to the right
- Model answers align to the left with the FilingLens avatar
- Evidence opens the source PDF page
- Findings can be saved to the Research Board
- Reports export as Markdown or HTML
- Collapsed navigation remains recoverable from the top-left expand control

## 8. Technology

```text
PDF
 ↓
pypdf parsing
 ↓
Page-aware chunking
 ↓
SQLite document and conversation storage
 ↓
BM25 evidence retrieval
 ↓
OpenAI-compatible streaming model API
 ↓
Citation validation and structured rendering
 ↓
Streamlit interface and export tools
```

Core data objects:

- Workspace
- Company
- Report
- Page
- Chunk
- Evidence
- Insight
- Conversation
- Report draft

Technical constraints:

- SQLite for local persistence
- No vector database in the first version
- No unrestricted model-generated Python
- Fixed formulas for supported financial calculations
- Streamlit Community Cloud deployment support

## 9. Deployment

Target platform: Streamlit Community Cloud.

Required settings:

- Main file: `app.py`
- Python: `3.12`
- Secrets: model endpoint, API key, model name
- Runtime data directory: `/tmp/filinglens`

Community Cloud uses ephemeral storage. Durable deployment requires an external managed database.

## 10. Quality Targets

- Text-PDF page extraction success rate: at least 95%
- Citation page accuracy: at least 95%
- Factual conclusions with evidence coverage: 100%
- Unsupported-question refusal accuracy: at least 90%
- At least 90% of a 30-question golden test set answered correctly

## 11. Delivery Phases

### Phase 1: Trusted Q&A

- PDF parsing and page retention
- SQLite schema
- Evidence retrieval
- Citation validation and refusal

### Phase 2: Complete Research Workflow

- Structured research brief
- Research Board
- Markdown and HTML export
- Chat persistence and history management

### Phase 3: Differentiated Analysis

- Financial facts and source tables
- Year-over-year change analysis
- Company comparison
- Cloud deployment

## 12. Success Criteria

FilingLens succeeds when users can:

1. Understand a company faster.
2. Verify conclusions against source evidence.
3. Turn fragmented analysis into a report.
4. Trust the system to refuse unsupported questions.
