import pytest

from filinglens.models import PageText
from filinglens.parser import (
    PdfExtractionError,
    chunk_pages,
    extraction_quality,
    parse_pdf,
)


def test_chunk_pages_preserves_page_number():
    pages = [
        PageText(page_number=1, text="The first page describes the business. " * 30),
        PageText(page_number=2, text="The second page describes key risks. " * 30),
    ]

    chunks = chunk_pages(pages, target_chars=300, overlap_chars=40)

    assert chunks
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_extraction_quality_counts_empty_pages():
    quality = extraction_quality(
        [
            PageText(page_number=1, text="This is sufficiently long extracted text. " * 3),
            PageText(page_number=2, text=""),
        ]
    )

    assert quality["page_count"] == 2
    assert quality["text_pages"] == 1
    assert quality["empty_pages"] == 1


def test_parse_pdf_rejects_scanned_pdf_without_text(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(PdfExtractionError):
        parse_pdf(path)

