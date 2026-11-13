from pathlib import Path

from filinglens.tables import extract_tables


def test_extract_tables_returns_empty_for_blank_page(tmp_path):
    from pypdf import PdfWriter

    path = Path(tmp_path / "blank.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with path.open("wb") as handle:
        writer.write(handle)

    assert extract_tables(path) == []

