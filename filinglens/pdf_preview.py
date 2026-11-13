from __future__ import annotations

from pathlib import Path


def render_pdf_page(
    file_path: str | Path,
    *,
    page_number: int,
    quote: str = "",
    zoom: float = 1.6,
) -> bytes:
    """Render one PDF page as PNG and highlight a matching quote when possible."""
    import fitz

    if page_number < 1:
        raise ValueError("page_number must be positive")

    with fitz.open(str(file_path)) as document:
        if page_number > len(document):
            raise ValueError(f"PDF only has {len(document)} pages")
        page = document.load_page(page_number - 1)

        search_terms = _search_terms(quote)
        highlighted = False
        for term in search_terms:
            rectangles = page.search_for(term)
            if not rectangles:
                continue
            for rectangle in rectangles:
                annotation = page.add_highlight_annot(rectangle)
                annotation.update()
            highlighted = True
            break

        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = pixmap.tobytes("png")

    return image


def _search_terms(quote: str) -> list[str]:
    normalized = " ".join(quote.split())
    if not normalized:
        return []

    terms = [normalized]
    if len(normalized) > 80:
        terms.append(normalized[:80])
    if len(normalized) > 40:
        terms.append(normalized[:40])
    return list(dict.fromkeys(term for term in terms if len(term) >= 4))

