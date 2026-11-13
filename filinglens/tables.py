from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedTable:
    page_number: int
    rows: list[list[str]]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)


def extract_tables(
    file_path: str | Path,
    *,
    page_numbers: list[int] | None = None,
    max_tables_per_page: int = 2,
    max_rows: int = 50,
    max_columns: int = 20,
) -> list[ExtractedTable]:
    """Extract visible tables from selected pages using PyMuPDF."""
    import fitz

    with fitz.open(str(file_path)) as document:
        selected = page_numbers or list(range(1, len(document) + 1))
        tables: list[ExtractedTable] = []
        for page_number in sorted(set(selected)):
            if page_number < 1 or page_number > len(document):
                continue
            page = document.load_page(page_number - 1)
            finder = page.find_tables()
            for table in finder.tables[:max_tables_per_page]:
                raw_rows = table.extract()
                rows = [
                    [
                        (_clean_cell(cell) if cell is not None else "")
                        for cell in row[:max_columns]
                    ]
                    for row in raw_rows[:max_rows]
                ]
                rows = [row for row in rows if any(cell for cell in row)]
                if len(rows) >= 2 and max((len(row) for row in rows), default=0) >= 2:
                    tables.append(
                        ExtractedTable(
                            page_number=page_number,
                            rows=rows,
                        )
                    )
        return tables


def _clean_cell(value) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()

