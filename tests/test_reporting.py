from filinglens.models import (
    AnswerSection,
    Claim,
    Evidence,
    Insight,
    Report,
    ResearchAnswer,
)
from filinglens.reporting import (
    company_report_to_markdown,
    insights_to_markdown,
    markdown_to_html,
)


def test_insights_export_includes_evidence():
    report = Report(
        id=1,
        company="示例公司",
        year=2025,
        filename="sample.pdf",
        file_path="sample.pdf",
        page_count=10,
        created_at="2026-11-13",
    )
    insight = Insight(
        id=1,
        report_id=1,
        text="海外市场是主要增长来源。",
        evidence=[
            Evidence(
                evidence_id="E1",
                chunk_id=1,
                report_id=1,
                page_number=3,
                text="公司收入增长主要来自海外市场扩张。",
                score=1.0,
            )
        ],
        kind="annual_report_fact",
        created_at="2026-11-13",
    )

    markdown = insights_to_markdown(report, [insight])

    assert "# 示例公司（2025）研究简报" in markdown
    assert "第 3 页" in markdown
    assert "海外市场扩张" in markdown


def test_company_report_export_includes_sections_and_html():
    report = Report(
        id=1,
        company="示例公司",
        year=2025,
        filename="sample.pdf",
        file_path="sample.pdf",
        page_count=10,
        created_at="2026-11-13",
    )
    answer = ResearchAnswer(
        status="answered",
        summary="公司通过软件订阅获得收入。",
        claims=[],
        evidence=[],
        sections=[
            AnswerSection(
                title="商业模式",
                claims=[
                    Claim(
                        text="收入主要来自软件订阅。",
                        evidence_ids=["E1"],
                    )
                ],
            )
        ],
    )

    markdown = company_report_to_markdown(report, answer=answer)
    html = markdown_to_html(markdown, title="示例公司研究简报")

    assert "## 商业模式" in markdown
    assert "收入主要来自软件订阅" in markdown
    assert "<html" in html
    assert "示例公司研究简报" in html

