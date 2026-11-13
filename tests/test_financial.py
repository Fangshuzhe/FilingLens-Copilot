from filinglens.financial import extract_financial_facts
from filinglens.models import PageText


def test_extract_financial_facts_binds_metric_numbers_and_page():
    facts = extract_financial_facts(
        [
            PageText(
                page_number=12,
                text=(
                    "报告期内，公司实现营业收入 12.5 亿元，同比增长 18.2%。"
                    "归属于上市公司股东的净利润为 1.2 亿元。"
                ),
            )
        ]
    )

    revenue = next(item for item in facts if item.metric == "营业收入")
    profit = next(item for item in facts if item.metric == "净利润")
    assert "12.5 亿元" in revenue.values
    assert "18.2%" in revenue.values
    assert revenue.page_number == 12
    assert "1.2 亿元" in profit.values


def test_extract_financial_facts_ignores_metric_without_number():
    facts = extract_financial_facts(
        [PageText(page_number=1, text="公司持续提高研发投入并优化产品组合。")]
    )

    assert facts == []

