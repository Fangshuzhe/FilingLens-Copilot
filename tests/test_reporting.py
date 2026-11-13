from filinglens.models import Evidence, Insight, Report
from filinglens.reporting import insights_to_markdown


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

