from pathlib import Path

import pytest

from filinglens.pdf_preview import render_pdf_page


def test_render_pdf_page_rejects_invalid_page(tmp_path):
    from pypdf import PdfWriter

    path = Path(tmp_path / "blank.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ValueError):
        render_pdf_page(path, page_number=2)


def test_render_pdf_page_returns_png(tmp_path):
    from pypdf import PdfWriter

    path = Path(tmp_path / "blank.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    image = render_pdf_page(path, page_number=1)

    assert image.startswith(b"\x89PNG")

