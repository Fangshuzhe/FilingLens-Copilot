from filinglens.db import Database
from filinglens.models import Evidence, PageText, TextChunk


def test_database_round_trip(tmp_path):
    database = Database(tmp_path / "test.db")
    report_id = database.add_report(
        company="示例公司",
        year=2025,
        filename="sample.pdf",
        file_path=tmp_path / "sample.pdf",
        pages=[PageText(page_number=1, text="公司的主要业务是软件服务。")],
        chunks=[
            TextChunk(
                chunk_index=0,
                page_number=1,
                text="公司的主要业务是软件服务。",
            )
        ],
    )

    report = database.get_report(report_id)
    assert report is not None
    assert report.company == "示例公司"
    assert report.page_count == 1
    assert len(database.list_chunks(report_id)) == 1

    evidence = Evidence(
        evidence_id="E1",
        chunk_id=database.list_chunks(report_id)[0].id,
        report_id=report_id,
        page_number=1,
        text="公司的主要业务是软件服务。",
        score=1.5,
    )
    insight_id = database.add_insight(
        report_id=report_id,
        text="公司通过软件服务获得收入。",
        evidence=[evidence],
        kind="annual_report_fact",
    )

    insights = database.list_insights(report_id)
    assert insights[0].id == insight_id
    assert insights[0].evidence[0].text == evidence.text

