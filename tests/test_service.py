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

