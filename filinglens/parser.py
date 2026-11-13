from __future__ import annotations

import re
from pathlib import Path

from filinglens.models import PageText, TextChunk


class PdfExtractionError(ValueError):
    """Raised when a PDF has no usable text layer."""


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: str | Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        pages.append(PageText(page_number=page_number, text=normalize_text(extracted)))

    text_characters = sum(len(page.text) for page in pages)
    if not pages or text_characters < max(50, len(pages) * 10):
        raise PdfExtractionError(
            "PDF 中没有足够的可提取文本，可能是扫描版或受保护文件。"
        )
    return pages


def extraction_quality(pages: list[PageText]) -> dict[str, int | float]:
    page_count = len(pages)
    text_pages = sum(1 for page in pages if len(page.text.strip()) >= 20)
    empty_pages = page_count - text_pages
    return {
        "page_count": page_count,
        "text_pages": text_pages,
        "empty_pages": empty_pages,
        "text_page_ratio": text_pages / page_count if page_count else 0.0,
    }


def chunk_pages(
    pages: list[PageText],
    *,
    target_chars: int = 900,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    if target_chars < 200:
        raise ValueError("target_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than target_chars")

    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue

        units = _split_units(text, target_chars)
        buffer = ""
        for unit in units:
            if buffer and len(buffer) + len(unit) + 1 > target_chars:
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        text=buffer.strip(),
                    )
                )
                chunk_index += 1
                buffer = _overlap_tail(buffer, overlap_chars)

            buffer = f"{buffer}\n{unit}".strip() if buffer else unit

        if buffer.strip():
            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    text=buffer.strip(),
                )
            )
            chunk_index += 1

    return chunks


def _split_units(text: str, target_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？!?；;\n])", text)
    units: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= target_chars:
            units.append(sentence)
            continue
        for start in range(0, len(sentence), target_chars):
            units.append(sentence[start : start + target_chars])
    return units


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars == 0:
        return ""
    return text[-overlap_chars:]

