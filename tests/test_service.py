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

    def stream(self, *, system, user):
        self.calls.append({"system": system, "user": user})
        response = self.responses.pop(0)
        midpoint = len(response) // 2
        yield response[:midpoint]
        yield response[midpoint:]


def build_service(tmp_path, responses):
    database = Database(tmp_path / "test.db")
    report_id = database.add_report(
        company="Example Corp",
        year=2025,
        filename="sample.pdf",
        file_path=tmp_path / "sample.pdf",
        pages=[
            PageText(
                page_number=3,
                text="Revenue growth was driven mainly by overseas expansion.",
            )
        ],
        chunks=[
            TextChunk(
                chunk_index=0,
                page_number=3,
                text="Revenue growth was driven mainly by overseas expansion.",
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
              "summary": "Overseas expansion drove growth.",
              "claims": [
                {"text": "Revenue growth came from overseas markets.", "evidence_ids": ["E9"], "kind": "annual_report_fact"}
              ],
              "missing": []
            }
            """,
            """
            {
              "status": "answered",
              "summary": "Overseas expansion drove growth.",
              "claims": [
                {"text": "Revenue growth came from overseas markets.", "evidence_ids": ["E1"], "kind": "annual_report_fact"}
              ],
              "missing": []
            }
            """,
        ],
    )

    answer = service.ask(
        report_id=report_id,
        question="What drove revenue growth?",
    )

    assert answer.is_answered
    assert answer.claims[0].evidence_ids == ["E1"]
    assert len(llm.calls) == 2
    assert "previous output failed validation" in llm.calls[1]["user"]


def test_insufficient_evidence_skips_llm(tmp_path):
    service, report_id, llm = build_service(tmp_path, [])

    answer = service.ask(
        report_id=report_id,
        question="What is the CEO's phone number?",
    )

    assert answer.status == "insufficient_evidence"
    assert not llm.calls


def test_general_chat_works_without_report_context(tmp_path):
    service, _, llm = build_service(
        tmp_path,
        ["Gross margin equals gross profit divided by revenue."],
    )

    answer = service.general_chat(
        question="How do I calculate gross margin?",
        history=[
            {"role": "user", "content": "I am researching a company."},
            {"role": "assistant", "content": "Start with the business model."},
        ],
    )

    assert answer == "Gross margin equals gross profit divided by revenue."
    assert "Current user question: How do I calculate gross margin?" in llm.calls[0]["user"]
    assert "I am researching a company" in llm.calls[0]["user"]


def test_identity_question_returns_product_description_without_llm(tmp_path):
    service, _, llm = build_service(tmp_path, [])

    answer = service.general_chat(question="Who are you?")

    assert "FilingLens" in answer
    assert "built on a large language model" in answer
    assert "page-level citations" in answer
    assert not llm.calls


def test_developer_question_returns_required_names_without_llm(tmp_path):
    service, _, llm = build_service(tmp_path, [])

    answer = service.general_chat(question="Who developed FilingLens?")

    assert "Fang Shuzhe" in answer
    assert "Wu Diwen" in answer
    assert not llm.calls


def test_general_chat_stream_yields_model_chunks(tmp_path):
    service, _, llm = build_service(
        tmp_path,
        ["First section. Second section."],
    )

    chunks = list(
        service.general_chat_stream(
            question="Explain business models.",
            history=[],
        )
    )

    assert "".join(chunks) == "First section. Second section."
    assert len(chunks) == 2
    assert llm.calls[0]["system"]


def test_generate_structured_company_brief(tmp_path):
    service, report_id, _ = build_service(
        tmp_path,
        [
            """
            {
              "status": "answered",
              "summary": "The company grows through overseas software services.",
              "sections": [
                {
                  "title": "Business overview",
                  "claims": [
                    {
                      "text": "Revenue growth was driven mainly by overseas expansion.",
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
    assert answer.sections[0].title == "Business overview"
    assert answer.claims[0].text == "Revenue growth was driven mainly by overseas expansion."


def test_compare_annual_reports_uses_current_and_previous_evidence(tmp_path):
    database = Database(tmp_path / "test.db")
    previous_id = database.add_report(
        company="Example Corp",
        year=2024,
        filename="2024.pdf",
        file_path=tmp_path / "2024.pdf",
        pages=[
            PageText(
                page_number=2,
                text="The company's business model provides software services.",
            )
        ],
        chunks=[
            TextChunk(
                0,
                2,
                "The company's business model provides software services.",
            )
        ],
    )
    current_id = database.add_report(
        company="Example Corp",
        year=2025,
        filename="2025.pdf",
        file_path=tmp_path / "2025.pdf",
        pages=[
            PageText(
                page_number=5,
                text="The company's business model added overseas software subscriptions.",
            )
        ],
        chunks=[
            TextChunk(
                0,
                5,
                "The company's business model added overseas software subscriptions.",
            )
        ],
    )
    llm = SequenceLLM(
        [
            """
            {
              "status": "answered",
              "summary": "The business expanded internationally.",
              "changes": [
                {
                  "dimension": "Business and revenue",
                  "previous": "The company provided software services.",
                  "current": "The company added overseas software subscriptions.",
                  "change_type": "added",
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
    assert report.changes[0].change_type == "added"
    assert set(report.changes[0].evidence_ids) == {"C1", "P1"}


def test_compare_companies_requires_evidence_from_both(tmp_path):
    database = Database(tmp_path / "test.db")
    report_a_id = database.add_report(
        company="Company A",
        year=2025,
        filename="a.pdf",
        file_path=tmp_path / "a.pdf",
        pages=[
            PageText(
                page_number=1,
                text="Company A's business model provides enterprise software.",
            )
        ],
        chunks=[
            TextChunk(
                0,
                1,
                "Company A's business model provides enterprise software.",
            )
        ],
    )
    report_b_id = database.add_report(
        company="Company B",
        year=2025,
        filename="b.pdf",
        file_path=tmp_path / "b.pdf",
        pages=[
            PageText(
                page_number=2,
                text="Company B's business model sells consumer hardware.",
            )
        ],
        chunks=[
            TextChunk(
                0,
                2,
                "Company B's business model sells consumer hardware.",
            )
        ],
    )
    llm = SequenceLLM(
        [
            """
            {
              "status": "answered",
              "summary": "The companies target different markets.",
              "sections": [
                {
                  "title": "Business model",
                  "claims": [
                    {"text": "Company A provides enterprise software.", "evidence_ids": ["A1"], "kind": "annual_report_fact"},
                    {"text": "Company B sells consumer hardware.", "evidence_ids": ["B1"], "kind": "annual_report_fact"}
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
