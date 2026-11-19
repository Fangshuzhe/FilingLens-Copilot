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
        company="Example Corp",
        year=2025,
        filename="sample.pdf",
        file_path="sample.pdf",
        page_count=10,
        created_at="2026-11-13",
    )
    insight = Insight(
        id=1,
        report_id=1,
        text="Overseas markets are the primary growth driver.",
        evidence=[
            Evidence(
                evidence_id="E1",
                chunk_id=1,
                report_id=1,
                page_number=3,
                text="Revenue growth was driven mainly by overseas expansion.",
                score=1.0,
            )
        ],
        kind="annual_report_fact",
        created_at="2026-11-13",
    )

    markdown = insights_to_markdown(report, [insight])

    assert "# Example Corp (2025) Research Brief" in markdown
    assert "Page 3" in markdown
    assert "overseas expansion" in markdown


def test_company_report_export_includes_sections_and_html():
    report = Report(
        id=1,
        company="Example Corp",
        year=2025,
        filename="sample.pdf",
        file_path="sample.pdf",
        page_count=10,
        created_at="2026-11-13",
    )
    answer = ResearchAnswer(
        status="answered",
        summary="The company earns revenue from software subscriptions.",
        claims=[],
        evidence=[],
        sections=[
            AnswerSection(
                title="Business model",
                claims=[
                    Claim(
                        text="Revenue comes primarily from software subscriptions.",
                        evidence_ids=["E1"],
                    )
                ],
            )
        ],
    )

    markdown = company_report_to_markdown(report, answer=answer)
    html = markdown_to_html(markdown, title="Example Corp Research Brief")

    assert "## Business model" in markdown
    assert "Revenue comes primarily from software subscriptions" in markdown
    assert "<html" in html
    assert "Example Corp Research Brief" in html

