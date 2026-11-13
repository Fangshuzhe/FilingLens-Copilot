from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Sequence

from filinglens.models import Evidence, Insight, PageText, Report, StoredChunk, TextChunk


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    year INTEGER NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(report_id, page_number)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(report_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_report_id ON chunks(report_id);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def add_report(
        self,
        *,
        company: str,
        year: int,
        filename: str,
        file_path: str | Path,
        pages: Sequence[PageText],
        chunks: Sequence[TextChunk],
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO reports(company, year, filename, file_path, page_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (company.strip(), year, filename, str(file_path), len(pages)),
            )
            report_id = int(cursor.lastrowid)

            connection.executemany(
                "INSERT INTO pages(report_id, page_number, text) VALUES (?, ?, ?)",
                [
                    (report_id, page.page_number, page.text)
                    for page in pages
                ],
            )
            connection.executemany(
                """
                INSERT INTO chunks(report_id, chunk_index, page_number, text)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        report_id,
                        chunk.chunk_index,
                        chunk.page_number,
                        chunk.text,
                    )
                    for chunk in chunks
                ],
            )
            return report_id

    def list_reports(self) -> list[Report]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reports ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_report(row) for row in rows]

    def get_report(self, report_id: int) -> Report | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM reports WHERE id = ?",
                (report_id,),
            ).fetchone()
        return self._row_to_report(row) if row else None

    def list_chunks(self, report_id: int) -> list[StoredChunk]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, report_id, chunk_index, page_number, text
                FROM chunks
                WHERE report_id = ?
                ORDER BY chunk_index
                """,
                (report_id,),
            ).fetchall()
        return [
            StoredChunk(
                id=int(row["id"]),
                report_id=int(row["report_id"]),
                chunk_index=int(row["chunk_index"]),
                page_number=int(row["page_number"]),
                text=str(row["text"]),
            )
            for row in rows
        ]

    def list_pages(self, report_id: int) -> list[PageText]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT page_number, text
                FROM pages
                WHERE report_id = ?
                ORDER BY page_number
                """,
                (report_id,),
            ).fetchall()
        return [
            PageText(
                page_number=int(row["page_number"]),
                text=str(row["text"]),
            )
            for row in rows
        ]

    def add_insight(
        self,
        *,
        report_id: int,
        text: str,
        evidence: Sequence[Evidence],
        kind: str,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO insights(report_id, text, evidence_json, kind)
                VALUES (?, ?, ?, ?)
                """,
                (
                    report_id,
                    text.strip(),
                    json.dumps(
                        [asdict(item) for item in evidence],
                        ensure_ascii=False,
                    ),
                    kind,
                ),
            )
            return int(cursor.lastrowid)

    def list_insights(self, report_id: int) -> list[Insight]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, report_id, text, evidence_json, kind, created_at
                FROM insights
                WHERE report_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (report_id,),
            ).fetchall()
        return [
            Insight(
                id=int(row["id"]),
                report_id=int(row["report_id"]),
                text=str(row["text"]),
                evidence=[
                    Evidence(**item)
                    for item in json.loads(row["evidence_json"])
                ],
                kind=str(row["kind"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> Report:
        return Report(
            id=int(row["id"]),
            company=str(row["company"]),
            year=int(row["year"]),
            filename=str(row["filename"]),
            file_path=str(row["file_path"]),
            page_count=int(row["page_count"]),
            created_at=str(row["created_at"]),
        )

