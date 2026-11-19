from __future__ import annotations

import re
from dataclasses import dataclass

from filinglens.models import PageText


NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"\s*(?:%|\uff05|\u4ebf\u5143|\u4e07\u5143|\u5343\u5143|\u5143|\u4ebf|\u4e07)?"
)

METRIC_TERMS: dict[str, tuple[str, ...]] = {
    "Revenue": (
        "revenue",
        "sales",
        "\u8425\u4e1a\u6536\u5165",
        "\u8425\u4e1a\u603b\u6536\u5165",
        "\u9500\u552e\u6536\u5165",
    ),
    "Net profit": (
        "net profit",
        "net income",
        "\u51c0\u5229\u6da6",
        "\u5f52\u6bcd\u51c0\u5229\u6da6",
    ),
    "Gross margin": (
        "gross margin",
        "gross profit margin",
        "\u6bdb\u5229\u7387",
    ),
    "R&D investment": (
        "research and development",
        "r&d",
        "\u7814\u53d1\u6295\u5165",
        "\u7814\u53d1\u8d39\u7528",
    ),
    "Operating cash flow": (
        "operating cash flow",
        "\u7ecf\u8425\u6d3b\u52a8\u4ea7\u751f\u7684\u73b0\u91d1\u6d41\u91cf\u51c0\u989d",
        "\u7ecf\u8425\u73b0\u91d1\u6d41",
    ),
    "Total assets": (
        "total assets",
        "\u8d44\u4ea7\u603b\u989d",
        "\u603b\u8d44\u4ea7",
    ),
    "Total liabilities": (
        "total liabilities",
        "\u8d1f\u503a\u603b\u989d",
        "\u603b\u8d1f\u503a",
    ),
}


@dataclass(frozen=True)
class FinancialFact:
    metric: str
    values: list[str]
    context: str
    page_number: int
    matched_term: str


def extract_financial_facts(
    pages: list[PageText],
    *,
    max_facts: int = 40,
) -> list[FinancialFact]:
    facts: list[FinancialFact] = []
    seen: set[tuple[str, int, tuple[str, ...]]] = set()

    for page in pages:
        for sentence in _sentences(page.text):
            for metric, terms in METRIC_TERMS.items():
                matched_term = next(
                    (
                        term
                        for term in terms
                        if term.lower() in sentence.lower()
                    ),
                    None,
                )
                if not matched_term:
                    continue
                values = list(
                    dict.fromkeys(
                        match.group(0).strip()
                        for match in NUMBER_RE.finditer(sentence)
                    )
                )
                if not values:
                    continue
                key = (metric, page.page_number, tuple(values))
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    FinancialFact(
                        metric=metric,
                        values=values[:4],
                        context=sentence.strip()[:500],
                        page_number=page.page_number,
                        matched_term=matched_term,
                    )
                )
                if len(facts) >= max_facts:
                    return facts
    return facts


def _sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[。！？!?；;])|\n+", text)
        if item.strip()
    ]

