from filinglens.financial import extract_financial_facts
from filinglens.models import PageText


def test_extract_financial_facts_binds_metric_numbers_and_page():
    facts = extract_financial_facts(
        [
            PageText(
                page_number=12,
                text=(
                    "Revenue was 12.5 billion yuan, up 18.2% year over year. "
                    "Net profit was 1.2 billion yuan."
                ),
            )
        ]
    )

    revenue = next(item for item in facts if item.metric == "Revenue")
    profit = next(item for item in facts if item.metric == "Net profit")
    assert "12.5" in revenue.values
    assert "18.2%" in revenue.values
    assert revenue.page_number == 12
    assert "1.2" in profit.values


def test_extract_financial_facts_ignores_metric_without_number():
    facts = extract_financial_facts(
        [PageText(page_number=1, text="The company continues to increase R&D investment.")]
    )

    assert facts == []

