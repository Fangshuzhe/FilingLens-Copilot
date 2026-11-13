from __future__ import annotations

import re
from dataclasses import dataclass

from filinglens.models import PageText


NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"\s*(?:%|％|亿元|万元|千元|元|亿|万)?"
)

METRIC_TERMS: dict[str, tuple[str, ...]] = {
    "营业收入": ("营业收入", "营业总收入", "销售收入", "revenue"),
    "净利润": ("净利润", "归母净利润", "net profit", "net income"),
    "毛利率": ("毛利率", "gross margin", "gross profit margin"),
    "研发投入": ("研发投入", "研发费用", "research and development"),
    "经营现金流": (
        "经营活动产生的现金流量净额",
        "经营现金流",
        "operating cash flow",
    ),
    "资产总额": ("资产总额", "总资产", "total assets"),
    "负债总额": ("负债总额", "总负债", "total liabilities"),
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
                        match.group(0)
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

