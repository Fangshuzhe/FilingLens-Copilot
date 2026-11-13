from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int
    text: str


@dataclass(frozen=True)
class Report:
    id: int
    company: str
    year: int
    filename: str
    file_path: str
    page_count: int
    created_at: str


@dataclass(frozen=True)
class StoredChunk:
    id: int
    report_id: int
    chunk_index: int
    page_number: int
    text: str


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    chunk_id: int
    report_id: int
    page_number: int
    text: str
    score: float
    company: str = ""
    year: int | None = None
    filename: str = ""


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    kind: str = "annual_report_fact"


@dataclass(frozen=True)
class AnswerSection:
    title: str
    claims: list[Claim] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchAnswer:
    status: str
    summary: str
    claims: list[Claim]
    evidence: list[Evidence]
    missing: list[str] = field(default_factory=list)
    raw_response: str = ""
    sections: list[AnswerSection] = field(default_factory=list)

    @property
    def is_answered(self) -> bool:
        return self.status == "answered"


@dataclass(frozen=True)
class ChangeItem:
    dimension: str
    previous: str
    current: str
    change_type: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class ChangeReport:
    status: str
    summary: str
    changes: list[ChangeItem]
    evidence: list[Evidence]
    missing: list[str] = field(default_factory=list)
    raw_response: str = ""

    @property
    def is_answered(self) -> bool:
        return self.status == "answered"


@dataclass(frozen=True)
class Insight:
    id: int
    report_id: int
    text: str
    evidence: list[Evidence]
    kind: str
    created_at: str

