from __future__ import annotations

import base64
import hashlib
import html
import os
import re
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from filinglens.db import Database
from filinglens.financial import FinancialFact, extract_financial_facts
from filinglens.llm import (
    LLMConfigurationError,
    LLMResponseError,
    LLMSettings,
    OpenAICompatibleLLM,
)
from filinglens.models import (
    Claim,
    Evidence,
    Report,
    ResearchAnswer,
)
from filinglens.parser import (
    PdfExtractionError,
    chunk_pages,
    extraction_quality,
    parse_pdf,
)
from filinglens.pdf_preview import render_pdf_page
from filinglens.reporting import (
    change_report_to_markdown,
    company_comparison_to_markdown,
    company_report_to_markdown,
    markdown_to_html,
)
from filinglens.service import FilingLensService
from filinglens.tables import extract_tables


load_dotenv()


def apply_streamlit_secrets() -> None:
    keys = (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "FILINGLENS_DATA_DIR",
    )
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in keys:
        if key not in os.environ and key in secrets:
            os.environ[key] = str(secrets[key])


apply_streamlit_secrets()

DATA_DIR = Path(os.getenv("FILINGLENS_DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "filinglens.db"
LOGO_MARK_PATH = Path(__file__).parent / "assets" / "filinglens-mark.svg"

st.set_page_config(
    page_title="FilingLens Copilot",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"], #MainMenu, footer {
        visibility: hidden;
    }
    [data-testid="stExpandSidebarButton"] {
        display: block !important;
        visibility: visible !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        z-index: 1000000 !important;
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
    }
    .block-container {
        max-width: 900px;
        padding-top: 2.8rem;
        padding-bottom: 7rem;
    }
    [data-testid="stSidebar"] {
        background: #F7F8FA;
        border-right: 1px solid #ECEEF2;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }
    .fl-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0 0 1.4rem 0;
        color: #111827;
        font-size: 1.08rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .fl-logo {
        width: 28px;
        height: 28px;
        display: block;
    }
    .fl-label {
        margin: 1.1rem 0 0.35rem 0;
        color: #9CA3AF;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .fl-welcome {
        display: flex;
        min-height: 52vh;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .fl-welcome-mark {
        width: 52px;
        height: 52px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(77, 107, 254, 0.18);
        border-radius: 16px;
    }
    .fl-welcome h1 {
        margin: 0;
        color: #111827;
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.03em;
    }
    .fl-welcome p {
        max-width: 560px;
        margin: 12px 0 0 0;
        color: #8A93A3;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.35rem 0;
        width: 100%;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 0.96rem;
        line-height: 1.8;
    }
    [data-testid="stChatInput"] {
        max-width: 840px;
        margin: 0 auto;
    }
    .stButton button {
        border-radius: 10px;
    }
    .fl-user-row {
        display: flex;
        align-items: flex-start;
        justify-content: flex-end;
        gap: 12px;
        width: 100%;
        margin: 10px 0 18px 0;
    }
    .fl-user-text {
        max-width: 72%;
        color: #1F2937;
        font-size: 0.96rem;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }
    .fl-user-avatar {
        display: flex;
        flex: 0 0 38px;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: #FF5A5F;
    }
    .fl-user-avatar svg {
        width: 24px;
        height: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database() -> Database:
    return Database(DB_PATH)


@st.cache_resource
def get_service() -> FilingLensService:
    settings = LLMSettings.from_env()
    return FilingLensService(
        database=get_database(),
        llm=OpenAICompatibleLLM(settings),
    )


@st.cache_data(show_spinner=False, max_entries=60)
def get_page_image(
    file_path: str,
    modified_time: float,
    page_number: int,
    quote: str,
) -> bytes:
    del modified_time
    return render_pdf_page(
        file_path,
        page_number=page_number,
        quote=quote,
    )


def get_logo_data_uri() -> str:
    encoded = base64.b64encode(LOGO_MARK_PATH.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render_evidence(
    evidence: Evidence,
    *,
    reports_by_id: dict[int, Report],
) -> None:
    source_report = reports_by_id.get(evidence.report_id)
    source = (
        f"{evidence.company or source_report.company} | "
        f"{evidence.year or source_report.year} | Page {evidence.page_number}"
        if source_report
        else f"Page {evidence.page_number}"
    )
    st.markdown(f"**{evidence.evidence_id}｜{source}**")
    st.markdown(f"> {evidence.text.replace(chr(10), '  ')}")

    if source_report and Path(source_report.file_path).exists():
        with st.popover("View PDF page"):
            image = get_page_image(
                source_report.file_path,
                Path(source_report.file_path).stat().st_mtime,
                evidence.page_number,
                evidence.text,
            )
            st.image(image, use_container_width=True)


def render_claim(
    claim: Claim,
    *,
    evidence: list[Evidence],
    anchor_report: Report,
    reports_by_id: dict[int, Report],
    key_prefix: str,
) -> None:
    citations = " ".join(f"`{item}`" for item in claim.evidence_ids)
    label = "Annual report fact" if claim.kind == "annual_report_fact" else "Model judgment"
    selected_evidence = [
        item for item in evidence if item.evidence_id in claim.evidence_ids
    ]

    left, right = st.columns([6, 1])
    with left:
        st.markdown(f"**{label}**　{claim.text}　{citations}")
        for item in selected_evidence:
            if (
                item.company
                and anchor_report.company
                and item.company != anchor_report.company
            ):
                st.caption(f"{item.evidence_id} | {item.company} | Page {item.page_number}")
            else:
                st.caption(f"{item.evidence_id} | Page {item.page_number}")
    with right:
        if st.button(
            "Save",
            key=(
                f"{key_prefix}-save-"
                f"{_text_digest(claim.text)}"
            ),
        ):
            get_database().add_insight(
                report_id=anchor_report.id,
                text=claim.text,
                evidence=selected_evidence,
                kind=claim.kind,
            )
            st.toast("Saved to Research Board")


def render_answer(
    *,
    answer: ResearchAnswer,
    anchor_report: Report,
    reports_by_id: dict[int, Report],
    key_prefix: str,
) -> None:
    if answer.status == "insufficient_evidence":
        st.warning(answer.summary)
        for item in answer.missing:
            st.caption(f"Missing: {item}")
        return

    if answer.status != "answered":
        st.error(answer.summary)
        for item in answer.missing:
            st.caption(item)
        return

    if answer.summary:
        st.markdown(f"### {answer.summary}")

    if answer.sections:
        for section_index, section in enumerate(answer.sections):
            with st.container(border=True):
                st.markdown(f"#### {section.title}")
                for claim_index, claim in enumerate(section.claims):
                    render_claim(
                        claim,
                        evidence=answer.evidence,
                        anchor_report=anchor_report,
                        reports_by_id=reports_by_id,
                        key_prefix=(
                            f"{key_prefix}-{section_index}-{claim_index}"
                        ),
                    )
    else:
        for claim_index, claim in enumerate(answer.claims):
            render_claim(
                claim,
                evidence=answer.evidence,
                anchor_report=anchor_report,
                reports_by_id=reports_by_id,
                key_prefix=f"{key_prefix}-{claim_index}",
            )

    with st.expander("View all evidence and PDF pages"):
        for item in answer.evidence:
            render_evidence(item, reports_by_id=reports_by_id)


def render_download_pair(
    *,
    markdown_text: str,
    title: str,
    file_stem: str,
    key: str,
) -> None:
    markdown_column, html_column = st.columns(2)
    with markdown_column:
        st.download_button(
            "Download Markdown",
            data=markdown_text,
            file_name=f"{file_stem}.md",
            mime="text/markdown",
            key=f"{key}-md",
        )
    with html_column:
        st.download_button(
            "Download HTML",
            data=markdown_to_html(markdown_text, title=title),
            file_name=f"{file_stem}.html",
            mime="text/html",
            key=f"{key}-html",
        )


def render_general_chat(
    service: FilingLensService,
    *,
    conversation_id: str,
) -> None:
    database = get_database()
    messages = database.list_chat_messages(conversation_id)

    if not messages:
        logo_uri = get_logo_data_uri()
        st.markdown(
            f"""
            <div class="fl-welcome">
              <img class="fl-welcome-mark" src="{logo_uri}" alt="FilingLens">
              <h1>What can I help you research?</h1>
              <p>Discuss business models, financial metrics, industries, or data analysis. Upload an annual report to unlock evidence Q&A and research tools.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in messages:
        _render_chat_message(message.role, message.content)

    question = st.chat_input("Message FilingLens")
    if question:
        prior_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        _render_chat_message("user", question)
        with st.chat_message("assistant", avatar=str(LOGO_MARK_PATH)):
            try:
                response = st.write_stream(
                    service.general_chat_stream(
                        question=question,
                        history=prior_messages,
                    )
                )
            except (LLMConfigurationError, LLMResponseError, ValueError) as exc:
                st.error(str(exc))
                return
        if isinstance(response, list):
            response = "".join(str(item) for item in response)
        response = str(response)
        database.add_chat_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        database.add_chat_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response,
        )
        st.rerun()


def _render_chat_message(role: str, content: str) -> None:
    if role == "assistant":
        with st.chat_message(role, avatar=str(LOGO_MARK_PATH)):
            st.markdown(content)
    else:
        safe_content = html.escape(content).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="fl-user-row">
              <div class="fl-user-text">{safe_content}</div>
              <div class="fl-user-avatar" aria-label="User avatar">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <circle cx="12" cy="8" r="4" fill="none" stroke="#FFFFFF" stroke-width="2"/>
                  <path d="M4.8 20c.8-4 3.5-6 7.2-6s6.4 2 7.2 6" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
                </svg>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_brief(
    service: FilingLensService,
    report: Report,
    reports_by_id: dict[int, Report],
) -> None:
    brief_key = f"brief-{report.id}"
    if st.button("Generate Research Brief", type="primary"):
        with st.spinner("Retrieving evidence and generating the research brief..."):
            try:
                st.session_state[brief_key] = service.generate_company_brief(
                    report_id=report.id
                )
            except (LLMConfigurationError, LLMResponseError, ValueError) as exc:
                st.error(str(exc))

    answer = st.session_state.get(brief_key)
    if answer:
        render_answer(
            answer=answer,
            anchor_report=report,
            reports_by_id=reports_by_id,
            key_prefix=f"brief-{report.id}",
        )
        if answer.is_answered:
            insights = get_database().list_insights(report.id)
            markdown_text = company_report_to_markdown(
                report,
                answer=answer,
                insights=insights,
            )
            render_download_pair(
                markdown_text=markdown_text,
                title=f"{report.company} {report.year} Company Research Brief",
                file_stem=f"{report.company}-{report.year}-research-brief",
                key=f"brief-download-{report.id}",
            )
    else:
        st.info("Generate a structured brief covering business, revenue, customers, moats, growth, and risk.")


def render_research_chat(
    service: FilingLensService,
    report: Report,
    reports_by_id: dict[int, Report],
) -> None:
    answer_key = f"qa-answer-{report.id}"
    question_key = f"qa-question-{report.id}"

    with st.form(f"qa-form-{report.id}"):
        question = st.text_area(
            "Research question",
            placeholder="For example: Where did revenue growth come from?",
            key=question_key,
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a research question.")
        else:
            with st.spinner("Retrieving evidence and generating an answer..."):
                try:
                    st.session_state[answer_key] = service.ask(
                        report_id=report.id,
                        question=question,
                    )
                except (LLMConfigurationError, LLMResponseError, ValueError) as exc:
                    st.error(str(exc))

    answer = st.session_state.get(answer_key)
    if answer:
        render_answer(
            answer=answer,
            anchor_report=report,
            reports_by_id=reports_by_id,
            key_prefix=f"qa-{report.id}",
        )
    else:
        st.caption(
            "Suggested questions: How does the company make money? What drives growth? What are the main risks?"
        )


def render_financial_facts(
    report: Report,
    reports_by_id: dict[int, Report],
) -> None:
    pages = get_database().list_pages(report.id)
    facts = extract_financial_facts(pages)
    st.caption(
        "Financial facts are extracted from the source with keyword and number rules "
        "to support fast verification. Values with different definitions are not combined."
    )
    if not facts:
        st.info("No financial facts with explicit figures were detected.")
        return

    st.dataframe(
        [
            {
                "Metric": item.metric,
                "Reported value": ", ".join(item.values),
                "Page": item.page_number,
            }
            for item in facts
        ],
        use_container_width=True,
        hide_index=True,
    )

    for index, fact in enumerate(facts):
        with st.expander(
            f"{fact.metric} | {', '.join(fact.values)} | Page {fact.page_number}"
        ):
            st.write(fact.context)
            evidence = _financial_evidence(report, fact, index)
            render_evidence(evidence, reports_by_id=reports_by_id)
            if st.button(
                "Save to Research Board",
                key=f"financial-save-{report.id}-{index}",
            ):
                get_database().add_insight(
                    report_id=report.id,
                    text=f"{fact.metric}：{'、'.join(fact.values)}",
                    evidence=[evidence],
                    kind="annual_report_fact",
                )
                st.toast("Financial fact saved to Research Board")

    pages_with_facts = sorted({item.page_number for item in facts})
    tables = extract_tables(
        report.file_path,
        page_numbers=pages_with_facts,
    )
    if tables:
        st.markdown("### Related Source Tables")
        for table_index, table in enumerate(tables):
            with st.expander(
                f"Table {table_index + 1} | Page {table.page_number}"
            ):
                st.dataframe(
                    table.rows,
                    use_container_width=True,
                    hide_index=True,
                )


def render_change_radar(
    service: FilingLensService,
    report: Report,
    reports: list[Report],
    reports_by_id: dict[int, Report],
) -> None:
    previous_reports = [
        item
        for item in reports
        if item.company.strip() == report.company.strip()
        and item.year < report.year
    ]
    if not previous_reports:
        st.info("Upload an earlier annual report for the same company to compare changes.")
        return

    labels = {
        item.id: f"{item.year} | {item.filename}"
        for item in previous_reports
    }
    previous_id = st.selectbox(
        "Select previous annual report",
        options=list(labels),
        format_func=lambda item_id: labels[item_id],
    )
    state_key = f"change-{report.id}-{previous_id}"
    if st.button("Analyze Year-over-Year Changes", type="primary"):
        with st.spinner("Comparing the two annual reports..."):
            try:
                st.session_state[state_key] = service.compare_annual_reports(
                    current_report_id=report.id,
                    previous_report_id=previous_id,
                )
            except (LLMConfigurationError, LLMResponseError, ValueError) as exc:
                st.error(str(exc))

    change_report = st.session_state.get(state_key)
    if not change_report:
        st.caption("Compares business disclosures, revenue mix, competition, and risk factors.")
        return
    if change_report.status != "answered":
        st.warning(change_report.summary)
        for item in change_report.missing:
            st.caption(item)
        return

    st.markdown(f"### {change_report.summary}")
    for index, item in enumerate(change_report.changes):
        with st.container(border=True):
            st.markdown(f"#### {item.dimension}｜{item.change_type}")
            previous_column, current_column = st.columns(2)
            previous_column.markdown("**Previous year**")
            previous_column.write(item.previous)
            current_column.markdown("**Current year**")
            current_column.write(item.current)
            st.caption("Evidence: " + " ".join(item.evidence_ids))

            selected = [
                evidence
                for evidence in change_report.evidence
                if evidence.evidence_id in item.evidence_ids
            ]
            if st.button(
                "Save to Research Board",
                key=f"change-save-{report.id}-{previous_id}-{index}",
            ):
                get_database().add_insight(
                    report_id=report.id,
                    text=f"{item.dimension}：{item.change_type}",
                    evidence=selected,
                    kind="model_judgment",
                )
                st.toast("Change finding saved to Research Board")

    with st.expander("View change-analysis evidence"):
        for evidence in change_report.evidence:
            render_evidence(evidence, reports_by_id=reports_by_id)

    markdown_text = change_report_to_markdown(
        report,
        reports_by_id[previous_id],
        change_report,
    )
    render_download_pair(
        markdown_text=markdown_text,
        title=f"{report.company} Annual Report Change Radar",
        file_stem=f"{report.company}-{report.year}-report-change",
        key=f"change-download-{report.id}-{previous_id}",
    )


def render_company_comparison(
    service: FilingLensService,
    report: Report,
    reports: list[Report],
    reports_by_id: dict[int, Report],
) -> None:
    other_reports = [
        item
        for item in reports
        if item.company.strip() != report.company.strip()
    ]
    if not other_reports:
        st.info("Upload at least one annual report from another company.")
        return

    labels = {
        item.id: f"{item.company} | {item.year} | {item.filename}"
        for item in other_reports
    }
    other_id = st.selectbox(
        "Select comparison company report",
        options=list(labels),
        format_func=lambda item_id: labels[item_id],
    )
    state_key = f"comparison-{report.id}-{other_id}"
    if st.button("Generate Company Comparison", type="primary"):
        with st.spinner("Retrieving evidence from both companies and comparing..."):
            try:
                st.session_state[state_key] = service.compare_companies(
                    report_a_id=report.id,
                    report_b_id=other_id,
                )
            except (LLMConfigurationError, LLMResponseError, ValueError) as exc:
                st.error(str(exc))

    answer = st.session_state.get(state_key)
    if not answer:
        st.caption("Compares business model, revenue mix, growth, moats, and risk.")
        return
    render_answer(
        answer=answer,
        anchor_report=report,
        reports_by_id=reports_by_id,
        key_prefix=f"comparison-{report.id}-{other_id}",
    )
    if answer.is_answered:
        markdown_text = company_comparison_to_markdown(
            report,
            reports_by_id[other_id],
            answer,
        )
        render_download_pair(
            markdown_text=markdown_text,
            title=f"{report.company} vs {reports_by_id[other_id].company}",
            file_stem=f"{report.company}-vs-{reports_by_id[other_id].company}",
            key=f"comparison-download-{report.id}-{other_id}",
        )


def render_research_board(
    report: Report,
    reports_by_id: dict[int, Report],
) -> None:
    insights = get_database().list_insights(report.id)
    if not insights:
        st.info("The Research Board is empty. Save conclusions from Q&A, briefs, or change analysis.")
        return

    for insight in insights:
        label = "Annual report fact" if insight.kind == "annual_report_fact" else "Model judgment"
        with st.container(border=True):
            st.markdown(f"**{label}**　{insight.text}")
            for evidence in insight.evidence:
                st.caption(
                    f"{evidence.evidence_id}｜{evidence.company or report.company}"
                    f" | Page {evidence.page_number}"
                )
                with st.expander(f"View evidence {evidence.evidence_id}"):
                    render_evidence(evidence, reports_by_id=reports_by_id)

    markdown_text = company_report_to_markdown(
        report,
        insights=insights,
    )
    render_download_pair(
        markdown_text=markdown_text,
        title=f"{report.company} Research Board Brief",
        file_stem=f"{report.company}-{report.year}-research-board-brief",
        key=f"board-download-{report.id}",
    )


def render_upload(*, expanded: bool) -> None:
    with st.sidebar.expander("Add Annual Report", expanded=expanded):
        with st.form("upload-report"):
            company = st.text_input("Company name")
            year = st.number_input(
                "Report year",
                min_value=1990,
                max_value=2100,
                value=2025,
                step=1,
            )
            uploaded_file = st.file_uploader("Select PDF", type=["pdf"])
            submitted = st.form_submit_button("Parse and Index")

        if not submitted:
            return
        if not uploaded_file or not company.strip():
            st.warning("Enter a company name and select a PDF.")
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(uploaded_file.name).name
        stored_path = UPLOAD_DIR / f"{uuid4().hex}-{safe_name}"
        stored_path.write_bytes(uploaded_file.getbuffer())

        try:
            with st.spinner("Parsing the annual report and building the search index..."):
                pages = parse_pdf(stored_path)
                chunks = chunk_pages(pages)
                report_id = get_database().add_report(
                    company=company,
                    year=int(year),
                    filename=safe_name,
                    file_path=stored_path,
                    pages=pages,
                    chunks=chunks,
                )
        except (PdfExtractionError, OSError, ValueError) as exc:
            stored_path.unlink(missing_ok=True)
            st.error(str(exc))
            return

        quality = extraction_quality(pages)
        st.success(
            f"Parsed {quality['page_count']} pages and indexed "
            f"{len(chunks)} search passages."
        )
        st.session_state["selected_report_id"] = report_id


def render_sidebar_brand() -> None:
    logo_uri = get_logo_data_uri()
    st.sidebar.markdown(
        f"""
        <div class="fl-brand">
          <img class="fl-logo" src="{logo_uri}" alt="FilingLens">
          <span>FilingLens</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("New Conversation", use_container_width=True):
        st.query_params["chat"] = uuid4().hex
        st.session_state["main-view"] = "General Chat"
        st.rerun()


def render_conversation_history(current_conversation_id: str) -> None:
    conversations = get_database().list_conversations()
    if not conversations:
        return

    st.sidebar.markdown(
        '<div class="fl-label">Conversations</div>',
        unsafe_allow_html=True,
    )
    for conversation in conversations[:20]:
        title = " ".join(conversation.title.split())
        if len(title) > 24:
            title = f"{title[:24]}…"
        if conversation.conversation_id == current_conversation_id:
            label = f"Current | {title}"
        else:
            label = title
        label_column, menu_column = st.sidebar.columns([5, 1])
        with label_column:
            if st.button(
                label,
                key=f"conversation-{conversation.conversation_id}",
                use_container_width=True,
            ):
                st.query_params["chat"] = conversation.conversation_id
                st.session_state["main-view"] = "General Chat"
                st.rerun()
        with menu_column:
            with st.popover("⋯", use_container_width=True):
                st.caption(title)
                if st.button(
                    "Delete conversation",
                    key=f"delete-{conversation.conversation_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    get_database().delete_conversation(
                        conversation.conversation_id
                    )
                    if (
                        conversation.conversation_id
                        == current_conversation_id
                    ):
                        st.query_params["chat"] = uuid4().hex
                    st.rerun()


def render_sidebar_navigation(
    *,
    reports: list[Report],
) -> tuple[Report, str]:
    reports_by_id = {item.id: item for item in reports}
    report_labels = {
        item.id: f"{item.company} | {item.year}"
        for item in reports
    }
    report_ids = list(report_labels)

    st.sidebar.markdown(
        '<div class="fl-label">Current document</div>',
        unsafe_allow_html=True,
    )
    selected_id = st.sidebar.selectbox(
        "Current document",
        options=report_ids,
        format_func=lambda item_id: report_labels[item_id],
        index=_selected_report_index(report_ids),
        label_visibility="collapsed",
    )
    st.session_state["selected_report_id"] = selected_id

    st.sidebar.markdown(
        '<div class="fl-label">Research tools</div>',
        unsafe_allow_html=True,
    )
    view = st.sidebar.radio(
        "Research tools",
        options=[
            "General Chat",
            "Research Brief",
            "Evidence Q&A",
            "Financial Facts",
            "Year-over-Year",
            "Company Comparison",
            "Research Board",
        ],
        key="main-view",
        label_visibility="collapsed",
    )
    return reports_by_id[selected_id], view


def main() -> None:
    database = get_database()
    existing_reports = database.list_reports()

    render_sidebar_brand()
    conversation_id = get_current_conversation_id()
    render_conversation_history(conversation_id)
    render_upload(expanded=not existing_reports)
    reports = database.list_reports()

    try:
        service = get_service()
    except LLMConfigurationError as exc:
        st.error(f"{exc}. Configure the model first.")
        service = None
    except Exception as exc:
        st.error(f"Model initialization failed: {exc}")
        service = None

    if not reports:
        st.sidebar.caption("Upload an annual report to unlock evidence Q&A and research tools.")
        if service is None:
            st.error("General Chat requires a configured model.")
        else:
            render_general_chat(
                service,
                conversation_id=conversation_id,
            )
        return

    report, view = render_sidebar_navigation(reports=reports)
    reports_by_id = {item.id: item for item in reports}
    st.sidebar.caption(
        f"{report.filename} | {report.page_count} pages"
    )

    if view == "General Chat":
        if service is None:
            st.error("General Chat requires a configured model.")
        else:
            render_general_chat(
                service,
                conversation_id=conversation_id,
            )
    elif view == "Research Brief":
        st.subheader("Research Brief")
        if service is None:
            st.info("Configure a model to generate a structured research brief.")
        else:
            render_brief(service, report, reports_by_id)
    elif view == "Evidence Q&A":
        st.subheader("Evidence Q&A")
        if service is None:
            st.info("Configure a model to ask questions with evidence citations.")
        else:
            render_research_chat(service, report, reports_by_id)
    elif view == "Financial Facts":
        st.subheader("Financial Facts")
        render_financial_facts(report, reports_by_id)
    elif view == "Year-over-Year":
        st.subheader("Year-over-Year Changes")
        if service is None:
            st.info("Configure a model to analyze changes across annual reports.")
        else:
            render_change_radar(
                service,
                report,
                reports,
                reports_by_id,
            )
    elif view == "Company Comparison":
        st.subheader("Company Comparison")
        if service is None:
            st.info("Configure a model to compare two companies.")
        else:
            render_company_comparison(
                service,
                report,
                reports,
                reports_by_id,
            )
    elif view == "Research Board":
        st.subheader("Research Board")
        render_research_board(report, reports_by_id)


def _financial_evidence(
    report: Report,
    fact: FinancialFact,
    index: int,
) -> Evidence:
    return Evidence(
        evidence_id=f"F{index + 1}",
        chunk_id=0,
        report_id=report.id,
        page_number=fact.page_number,
        text=fact.context,
        score=0.0,
        company=report.company,
        year=report.year,
        filename=report.filename,
    )


def _selected_report_index(report_ids: list[int]) -> int:
    selected_id = st.session_state.get("selected_report_id")
    if selected_id in report_ids:
        return report_ids.index(selected_id)
    return 0


def get_current_conversation_id() -> str:
    value = st.query_params.get("chat")
    if isinstance(value, list):
        value = value[0] if value else ""
    value = str(value or "").strip()
    if re.fullmatch(r"[a-f0-9]{32}", value):
        return value

    conversation_id = uuid4().hex
    st.query_params["chat"] = conversation_id
    return conversation_id


def _text_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    main()

