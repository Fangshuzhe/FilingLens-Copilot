from filinglens.db import Database
from filinglens.models import PageText, TextChunk
from filinglens.service import FilingLensService


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, system, user):
        self.calls.append({"system": system, "user": user})
        return self.responses.pop(0)


def build_service(tmp_path, responses):
    database = Database(tmp_path / "test.db")
    report_id = database.add_report(
        company="示例公司",
        year=2025,
        filename="sample.pdf",
        file_path=tmp_path / "sample.pdf",
        pages=[PageText(page_number=3, text="公司收入增长主要来自海外市场扩张。")],
        chunks=[
            TextChunk(
                chunk_index=0,
                page_number=3,
                text="公司收入增长主要来自海外市场扩张。",
            )
        ],
    )
    llm = SequenceLLM(responses)
    return FilingLensService(database=database, llm=llm), report_id, llm


def test_answer_requires_valid_evidence_ids(tmp_path):
    service, report_id, llm = build_service(
        tmp_path,
        [
            """
            {
              "status": "answered",
              "summary": "海外扩张推动增长。",
              "claims": [
                {"text": "收入增长来自海外市场。", "evidence_ids": ["E9"], "kind": "annual_report_fact"}
              ],
              "missing": []
            }
            """,
            """
            {
              "status": "answered",
              "summary": "海外扩张推动增长。",
              "claims": [
                {"text": "收入增长来自海外市场。", "evidence_ids": ["E1"], "kind": "annual_report_fact"}
              ],
              "missing": []
            }
            """,
        ],
    )

    answer = service.ask(report_id=report_id, question="收入增长来自哪里？")

    assert answer.is_answered
    assert answer.claims[0].evidence_ids == ["E1"]
    assert len(llm.calls) == 2
    assert "上一次输出未通过校验" in llm.calls[1]["user"]


def test_insufficient_evidence_skips_llm(tmp_path):
    service, report_id, llm = build_service(
        tmp_path,
        [],
    )

    answer = service.ask(report_id=report_id, question="董事长电话号码是多少？")

    assert answer.status == "insufficient_evidence"
    assert not llm.calls


def test_generate_structured_company_brief(tmp_path):
    service, report_id, _ = build_service(
        tmp_path,
        [
            """
            {
              "status": "answered",
              "summary": "公司通过海外市场软件服务增长。",
              "sections": [
                {
                  "title": "业务定位",
                  "claims": [
                    {
                      "text": "公司收入增长主要来自海外市场扩张。",
                      "evidence_ids": ["E1"],
                      "kind": "annual_report_fact"
                    }
                  ]
                }
              ],
              "missing": []
            }
            """
        ],
    )

    answer = service.generate_company_brief(report_id=report_id)

    assert answer.is_answered
    assert answer.sections[0].title == "业务定位"
    assert answer.claims[0].text == "公司收入增长主要来自海外市场扩张。"


def test_compare_annual_reports_uses_current_and_previous_evidence(tmp_path):
    database = Database(tmp_path / "test.db")
    previous_id = database.add_report(
        company="示例公司",
        year=2024,
        filename="2024.pdf",
        file_path=tmp_path / "2024.pdf",
        pages=[PageText(page_number=2, text="公司主营业务为软件服务。")],
        chunks=[TextChunk(0, 2, "公司主营业务为软件服务。")],
    )
    current_id = database.add_report(
        company="示例公司",
        year=2025,
        filename="2025.pdf",
        file_path=tmp_path / "2025.pdf",
        pages=[PageText(page_number=5, text="公司新增海外软件订阅业务。")],
        chunks=[TextChunk(0, 5, "公司新增海外软件订阅业务。")],
    )
    llm = SequenceLLM(
        [
            """
            {
              "status": "answered",
              "summary": "业务范围扩展到海外。",
              "changes": [
                {
                  "dimension": "业务与收入",
                  "previous": "主营业务为软件服务。",
                  "current": "新增海外软件订阅业务。",
                  "change_type": "新增",
                  "evidence_ids": ["C1", "P1"]
                }
              ],
              "missing": []
            }
            """
        ]
    )
    service = FilingLensService(database=database, llm=llm)

    report = service.compare_annual_reports(
        current_report_id=current_id,
        previous_report_id=previous_id,
    )

    assert report.is_answered
    assert report.changes[0].change_type == "新增"
    assert set(report.changes[0].evidence_ids) == {"C1", "P1"}


def test_compare_companies_requires_evidence_from_both(tmp_path):
    database = Database(tmp_path / "test.db")
    report_a_id = database.add_report(
        company="A 公司",
        year=2025,
        filename="a.pdf",
        file_path=tmp_path / "a.pdf",
        pages=[PageText(page_number=1, text="A 公司主营企业软件。")],
        chunks=[TextChunk(0, 1, "A 公司主营企业软件。")],
    )
    report_b_id = database.add_report(
        company="B 公司",
        year=2025,
        filename="b.pdf",
        file_path=tmp_path / "b.pdf",
        pages=[PageText(page_number=2, text="B 公司主营消费硬件。")],
        chunks=[TextChunk(0, 2, "B 公司主营消费硬件。")],
    )
    llm = SequenceLLM(
        [
            """
            {
              "status": "answered",
              "summary": "两家公司分别服务企业软件和消费硬件市场。",
              "sections": [
                {
                  "title": "商业模式",
                  "claims": [
                    {"text": "A 公司主营企业软件。", "evidence_ids": ["A1"], "kind": "annual_report_fact"},
                    {"text": "B 公司主营消费硬件。", "evidence_ids": ["B1"], "kind": "annual_report_fact"}
                  ]
                }
              ],
              "missing": []
            }
            """
        ]
    )
    service = FilingLensService(database=database, llm=llm)

    answer = service.compare_companies(
        report_a_id=report_a_id,
        report_b_id=report_b_id,
    )

    assert answer.is_answered
    assert {claim.evidence_ids[0] for claim in answer.claims} == {"A1", "B1"}

