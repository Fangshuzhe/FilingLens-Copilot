from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from filinglens.models import Evidence, StoredChunk


LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._%-]*")
CHINESE_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens = [token for token in LATIN_TOKEN_RE.findall(normalized) if len(token) > 1]

    for sequence in CHINESE_SEQUENCE_RE.findall(normalized):
        if len(sequence) == 1:
            tokens.append(sequence)
        elif len(sequence) == 2:
            tokens.append(sequence)
        else:
            tokens.extend(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
            )
    return tokens


@dataclass(frozen=True)
class _IndexedChunk:
    chunk: StoredChunk
    term_frequencies: Counter[str]
    length: int


class BM25Retriever:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def search(
        self,
        query: str,
        chunks: list[StoredChunk],
        *,
        top_k: int = 8,
    ) -> list[Evidence]:
        if not chunks or top_k <= 0:
            return []

        indexed = []
        for chunk in chunks:
            chunk_tokens = tokenize(chunk.text)
            indexed.append(
                _IndexedChunk(
                    chunk=chunk,
                    term_frequencies=Counter(chunk_tokens),
                    length=max(1, len(chunk_tokens)),
                )
            )
        average_length = sum(item.length for item in indexed) / len(indexed)
        document_frequency: Counter[str] = Counter()
        for item in indexed:
            document_frequency.update(item.term_frequencies.keys())

        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return []

        scored: list[tuple[float, StoredChunk]] = []
        for item in indexed:
            score = 0.0
            for term in query_terms:
                frequency = item.term_frequencies.get(term, 0)
                if frequency == 0:
                    continue
                docs_with_term = document_frequency[term]
                idf = math.log(1 + (len(indexed) - docs_with_term + 0.5) / (docs_with_term + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * item.length / average_length
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, item.chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = scored[:top_k]
        return [
            Evidence(
                evidence_id=f"E{position}",
                chunk_id=chunk.id,
                report_id=chunk.report_id,
                page_number=chunk.page_number,
                text=chunk.text,
                score=round(score, 6),
            )
            for position, (score, chunk) in enumerate(selected, start=1)
        ]

