from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from filinglens.db import Database
from filinglens.llm import (
    LLMConfigurationError,
    LLMResponseError,
    LLMSettings,
    OpenAICompatibleLLM,
)
from filinglens.models import Report, ResearchAnswer
from filinglens.parser import (
    PdfExtractionError,
    chunk_pages,
    extraction_quality,
    parse_pdf,
)
from filinglens.reporting import insights_to_markdown
from filinglens.service import FilingLensService


load_dotenv()

DATA_DIR = Path(os.getenv("FILINGLENS_DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "filinglens.db"

st.set_page_config(
    page_title="FilingLens Copilot",
    page_icon="🔎",
    layout="wide",
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


def render_answer(
    *,
    answer: ResearchAnswer,
    report: Report,
    key_prefix: str,
) -> None:
    database = get_database()
    if answer.status == "insufficient_evidence":
        st.warning(answer.summary)
        for item in answer.missing:
            st.caption(f"缺失：{item}")
        return

    if answer.status != "answered":
        st.error(answer.summary)
        for item in answer.missing:
            st.caption(item)
        return

    if answer.summary:
        st.markdown(f"### {answer.summary}")

    for position, claim in enumerate(answer.claims):
        citations = " ".join(f"`{item}`" for item in claim.evidence_ids)
        label = "年报事实" if claim.kind == "annual_report_fact" else "模型判断"
        st.markdown(f"**{label}**　{claim.text}　{citations}")
        selected_evidence = [
            item
            for item in answer.evidence
            if item.evidence_id in claim.evidence_ids
        ]
        if st.button(
            "加入研究板",
            key=f"{key_prefix}-save-{position}-{_text_digest(claim.text)}",
        ):
            database.add_insight(
                report_id=report.id,
                text=claim.text,
                evidence=selected_evidence,
                kind=claim.kind,
            )
            st.success("已加入研究板")

    with st.expander("查看全部原文证据"):
        for item in answer.evidence:
            st.markdown(
                f"**{item.evidence_id}｜第 {item.page_number} 页**"
            )
            st.markdown(f"> {item.text.replace(chr(10), '  ')}")


def render_brief(service: FilingLensService, report: Report) -> None:
    brief_key = f"brief-{report.id}"
    if st.button("生成公司研究底稿", type="primary"):
        with st.spinner("正在检索年报并生成研究底稿..."):
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
            report=report,
            key_prefix=f"brief-{report.id}",
        )
    else:
        st.info("点击上方按钮，基于年报生成固定结构的公司研究底稿。")


def render_research_chat(service: FilingLensService, report: Report) -> None:
    answer_key = f"qa-answer-{report.id}"
    question_key = f"qa-question-{report.id}"

    with st.form(f"qa-form-{report.id}"):
        question = st.text_area(
            "研究问题",
            placeholder="例如：公司收入增长主要来自哪里？",
            key=question_key,
        )
        submitted = st.form_submit_button("开始分析", type="primary")

    if submitted:
        if not question.strip():
            st.warning("请输入研究问题。")
        else:
            with st.spinner("正在检索证据并生成回答..."):
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
            report=report,
            key_prefix=f"qa-{report.id}",
        )
    else:
        st.caption(
            "建议问题：公司靠什么赚钱？增长来自哪里？主要风险是什么？"
        )


def render_research_board(report: Report) -> None:
    database = get_database()
    insights = database.list_insights(report.id)
    if not insights:
        st.info("研究板为空。问答或研究底稿中的结论可以加入这里。")
        return

    for insight in insights:
        label = "年报事实" if insight.kind == "annual_report_fact" else "模型判断"
        with st.container(border=True):
            st.markdown(f"**{label}**　{insight.text}")
            for evidence in insight.evidence:
                st.caption(
                    f"{evidence.evidence_id}｜第 {evidence.page_number} 页"
                )

    markdown = insights_to_markdown(report, insights)
    st.download_button(
        "导出研究简报",
        data=markdown,
        file_name=f"{report.company}-{report.year}-研究简报.md",
        mime="text/markdown",
    )


def render_upload() -> None:
    with st.sidebar.form("upload-report"):
        st.markdown("### 添加年报")
        company = st.text_input("公司名称")
        year = st.number_input(
            "年报年份",
            min_value=1990,
            max_value=2100,
            value=2025,
            step=1,
        )
        uploaded_file = st.file_uploader("选择 PDF", type=["pdf"])
        submitted = st.form_submit_button("解析并入库")

    if not submitted:
        return
    if not uploaded_file or not company.strip():
        st.sidebar.warning("请填写公司名称并选择 PDF。")
        return

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    stored_path = UPLOAD_DIR / f"{uuid4().hex}-{safe_name}"
    stored_path.write_bytes(uploaded_file.getbuffer())

    try:
        with st.spinner("正在解析年报并建立检索索引..."):
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
        st.sidebar.error(str(exc))
        return

    quality = extraction_quality(pages)
    st.sidebar.success(
        f"解析完成：{quality['page_count']} 页，"
        f"{len(chunks)} 个检索片段。"
    )
    st.session_state["selected_report_id"] = report_id


def main() -> None:
    st.title("FilingLens Copilot")
    st.caption("证据优先的企业年报研究工作台")

    database = get_database()
    render_upload()

    reports = database.list_reports()
    if not reports:
        st.info("请先在左侧上传一份年报 PDF。")
        return

    report_labels = {
        item.id: f"{item.company} · {item.year} · {item.filename}"
        for item in reports
    }
    report_ids = list(report_labels)
    selected_id = st.sidebar.selectbox(
        "当前年报",
        options=report_ids,
        format_func=lambda item_id: report_labels[item_id],
        index=_selected_report_index(report_ids),
    )
    st.session_state["selected_report_id"] = selected_id
    report = database.get_report(selected_id)
    if report is None:
        st.error("年报记录不存在。")
        return

    chunks = database.list_chunks(report.id)
    columns = st.columns(4)
    columns[0].metric("公司", report.company)
    columns[1].metric("年份", report.year)
    columns[2].metric("页数", report.page_count)
    columns[3].metric("检索片段", len(chunks))

    try:
        service = get_service()
    except LLMConfigurationError as exc:
        st.warning(f"{exc}。上传和解析已可用，模型分析尚未启用。")
        service = None
    except Exception as exc:
        st.error(f"模型初始化失败：{exc}")
        service = None

    tab_brief, tab_qa, tab_board = st.tabs(
        ["公司研究底稿", "证据问答", "研究板"]
    )
    with tab_brief:
        if service is None:
            st.info("配置 LLM_BASE_URL、LLM_API_KEY 和 LLM_MODEL 后可生成研究底稿。")
        else:
            render_brief(service, report)

    with tab_qa:
        if service is None:
            st.info("配置模型后可使用带证据引用的问答。")
        else:
            render_research_chat(service, report)

    with tab_board:
        render_research_board(report)

    with st.sidebar.expander("解析说明"):
        text_chunks = len(chunks)
        st.write(f"当前年报包含 {report.page_count} 页、{text_chunks} 个片段。")
        st.caption("引用格式：证据编号对应原文页码和检索片段。")


def _selected_report_index(report_ids: list[int]) -> int:
    selected_id = st.session_state.get("selected_report_id")
    if selected_id in report_ids:
        return report_ids.index(selected_id)
    return 0


def _text_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    main()

