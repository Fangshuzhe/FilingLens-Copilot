from __future__ import annotations

import hashlib
import os
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


def render_evidence(
    evidence: Evidence,
    *,
    reports_by_id: dict[int, Report],
) -> None:
    source_report = reports_by_id.get(evidence.report_id)
    source = (
        f"{evidence.company or source_report.company} · "
        f"{evidence.year or source_report.year} · 第 {evidence.page_number} 页"
        if source_report
        else f"第 {evidence.page_number} 页"
    )
    st.markdown(f"**{evidence.evidence_id}｜{source}**")
    st.markdown(f"> {evidence.text.replace(chr(10), '  ')}")

    if source_report and Path(source_report.file_path).exists():
        with st.popover("查看 PDF 原页"):
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
    label = "年报事实" if claim.kind == "annual_report_fact" else "模型判断"
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
                st.caption(f"{item.evidence_id} · {item.company} · 第 {item.page_number} 页")
            else:
                st.caption(f"{item.evidence_id} · 第 {item.page_number} 页")
    with right:
        if st.button(
            "收藏",
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
            st.toast("已加入研究板")


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
            st.caption(f"缺失：{item}")
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

    with st.expander("查看全部原文证据与 PDF 原页"):
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
            "下载 Markdown",
            data=markdown_text,
            file_name=f"{file_stem}.md",
            mime="text/markdown",
            key=f"{key}-md",
        )
    with html_column:
        st.download_button(
            "下载 HTML",
            data=markdown_to_html(markdown_text, title=title),
            file_name=f"{file_stem}.html",
            mime="text/html",
            key=f"{key}-html",
        )


def render_brief(
    service: FilingLensService,
    report: Report,
    reports_by_id: dict[int, Report],
) -> None:
    brief_key = f"brief-{report.id}"
    if st.button("生成公司研究底稿", type="primary"):
        with st.spinner("正在按八个维度检索年报并生成研究底稿..."):
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
                title=f"{report.company} {report.year} 公司研究简报",
                file_stem=f"{report.company}-{report.year}-公司研究简报",
                key=f"brief-download-{report.id}",
            )
    else:
        st.info("系统会从业务、收入、客户、壁垒、增长、风险等维度生成结构化底稿。")


def render_research_chat(
    service: FilingLensService,
    report: Report,
    reports_by_id: dict[int, Report],
) -> None:
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
            anchor_report=report,
            reports_by_id=reports_by_id,
            key_prefix=f"qa-{report.id}",
        )
    else:
        st.caption(
            "建议问题：公司靠什么赚钱？增长来自哪里？主要风险是什么？"
        )


def render_financial_facts(
    report: Report,
    reports_by_id: dict[int, Report],
) -> None:
    pages = get_database().list_pages(report.id)
    facts = extract_financial_facts(pages)
    st.caption(
        "以下内容按关键词和数字规则从原文中提取，仅用于快速定位，"
        "不对不同口径的数据自动相加或计算。"
    )
    if not facts:
        st.info("没有识别到包含明确数字的财务事实。")
        return

    st.dataframe(
        [
            {
                "指标": item.metric,
                "原文数值": "、".join(item.values),
                "页码": item.page_number,
            }
            for item in facts
        ],
        use_container_width=True,
        hide_index=True,
    )

    for index, fact in enumerate(facts):
        with st.expander(
            f"{fact.metric}｜{ '、'.join(fact.values) }｜第 {fact.page_number} 页"
        ):
            st.write(fact.context)
            evidence = _financial_evidence(report, fact, index)
            render_evidence(evidence, reports_by_id=reports_by_id)
            if st.button(
                "加入研究板",
                key=f"financial-save-{report.id}-{index}",
            ):
                get_database().add_insight(
                    report_id=report.id,
                    text=f"{fact.metric}：{'、'.join(fact.values)}",
                    evidence=[evidence],
                    kind="annual_report_fact",
                )
                st.toast("财务事实已加入研究板")

    pages_with_facts = sorted({item.page_number for item in facts})
    tables = extract_tables(
        report.file_path,
        page_numbers=pages_with_facts,
    )
    if tables:
        st.markdown("### 相关原文表格")
        for table_index, table in enumerate(tables):
            with st.expander(
                f"表格 {table_index + 1}｜第 {table.page_number} 页"
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
        st.info("请先上传同一家公司较早年份的年报，才能生成变化雷达。")
        return

    labels = {
        item.id: f"{item.year} · {item.filename}"
        for item in previous_reports
    }
    previous_id = st.selectbox(
        "选择较早年报",
        options=list(labels),
        format_func=lambda item_id: labels[item_id],
    )
    state_key = f"change-{report.id}-{previous_id}"
    if st.button("分析年度变化", type="primary"):
        with st.spinner("正在比较两个年度年报..."):
            try:
                st.session_state[state_key] = service.compare_annual_reports(
                    current_report_id=report.id,
                    previous_report_id=previous_id,
                )
            except (LLMConfigurationError, LLMResponseError, ValueError) as exc:
                st.error(str(exc))

    change_report = st.session_state.get(state_key)
    if not change_report:
        st.caption("重点比较业务表述、收入结构、市场竞争和风险因素的变化。")
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
            previous_column.markdown("**上一年度**")
            previous_column.write(item.previous)
            current_column.markdown("**本年度**")
            current_column.write(item.current)
            st.caption("证据：" + " ".join(item.evidence_ids))

            selected = [
                evidence
                for evidence in change_report.evidence
                if evidence.evidence_id in item.evidence_ids
            ]
            if st.button(
                "加入研究板",
                key=f"change-save-{report.id}-{previous_id}-{index}",
            ):
                get_database().add_insight(
                    report_id=report.id,
                    text=f"{item.dimension}：{item.change_type}",
                    evidence=selected,
                    kind="model_judgment",
                )
                st.toast("变化结论已加入研究板")

    with st.expander("查看变化分析证据"):
        for evidence in change_report.evidence:
            render_evidence(evidence, reports_by_id=reports_by_id)

    markdown_text = change_report_to_markdown(
        report,
        reports_by_id[previous_id],
        change_report,
    )
    render_download_pair(
        markdown_text=markdown_text,
        title=f"{report.company} 年报变化雷达",
        file_stem=f"{report.company}-{report.year}-年报变化",
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
        st.info("请先上传至少另一家公司的年报。")
        return

    labels = {
        item.id: f"{item.company} · {item.year} · {item.filename}"
        for item in other_reports
    }
    other_id = st.selectbox(
        "选择对比公司年报",
        options=list(labels),
        format_func=lambda item_id: labels[item_id],
    )
    state_key = f"comparison-{report.id}-{other_id}"
    if st.button("生成公司对比", type="primary"):
        with st.spinner("正在从两家公司年报中检索并对比..."):
            try:
                st.session_state[state_key] = service.compare_companies(
                    report_a_id=report.id,
                    report_b_id=other_id,
                )
            except (LLMConfigurationError, LLMResponseError, ValueError) as exc:
                st.error(str(exc))

    answer = st.session_state.get(state_key)
    if not answer:
        st.caption("按商业模式、收入结构、增长、壁垒和风险生成对比矩阵。")
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
        st.info("研究板为空。问答、研究底稿和变化分析中的结论可以加入这里。")
        return

    for insight in insights:
        label = "年报事实" if insight.kind == "annual_report_fact" else "模型判断"
        with st.container(border=True):
            st.markdown(f"**{label}**　{insight.text}")
            for evidence in insight.evidence:
                st.caption(
                    f"{evidence.evidence_id}｜{evidence.company or report.company}"
                    f"｜第 {evidence.page_number} 页"
                )
                with st.expander(f"查看证据 {evidence.evidence_id}"):
                    render_evidence(evidence, reports_by_id=reports_by_id)

    markdown_text = company_report_to_markdown(
        report,
        insights=insights,
    )
    render_download_pair(
        markdown_text=markdown_text,
        title=f"{report.company} 研究板简报",
        file_stem=f"{report.company}-{report.year}-研究板简报",
        key=f"board-download-{report.id}",
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

    reports_by_id = {item.id: item for item in reports}
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
    report = reports_by_id[selected_id]

    chunks = database.list_chunks(report.id)
    columns = st.columns(4)
    columns[0].metric("公司", report.company)
    columns[1].metric("年份", report.year)
    columns[2].metric("页数", report.page_count)
    columns[3].metric("检索片段", len(chunks))

    try:
        service = get_service()
    except LLMConfigurationError as exc:
        st.warning(f"{exc}。上传、解析和财务事实提取仍可使用。")
        service = None
    except Exception as exc:
        st.error(f"模型初始化失败：{exc}")
        service = None

    tabs = st.tabs(
        [
            "公司研究底稿",
            "证据问答",
            "财务事实",
            "年报变化",
            "公司对比",
            "研究板",
        ]
    )
    with tabs[0]:
        if service is None:
            st.info("配置模型后可生成结构化研究底稿。")
        else:
            render_brief(service, report, reports_by_id)
    with tabs[1]:
        if service is None:
            st.info("配置模型后可进行带证据引用的问答。")
        else:
            render_research_chat(service, report, reports_by_id)
    with tabs[2]:
        render_financial_facts(report, reports_by_id)
    with tabs[3]:
        if service is None:
            st.info("配置模型后可分析连续年度年报变化。")
        else:
            render_change_radar(
                service,
                report,
                reports,
                reports_by_id,
            )
    with tabs[4]:
        if service is None:
            st.info("配置模型后可生成双公司对比。")
        else:
            render_company_comparison(
                service,
                report,
                reports,
                reports_by_id,
            )
    with tabs[5]:
        render_research_board(report, reports_by_id)

    with st.sidebar.expander("当前数据"):
        st.write(f"{report.page_count} 页，{len(chunks)} 个检索片段")
        st.caption("点击证据编号旁的“查看 PDF 原页”可定位并高亮原文。")


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


def _text_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    main()

