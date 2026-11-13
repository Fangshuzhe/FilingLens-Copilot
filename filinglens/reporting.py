from __future__ import annotations

import html

import markdown

from filinglens.models import (
    ChangeReport,
    Insight,
    Report,
    ResearchAnswer,
)


def insights_to_markdown(report: Report, insights: list[Insight]) -> str:
    lines = [
        f"# {report.company}（{report.year}）研究简报",
        "",
        f"- 年报文件：{report.filename}",
        f"- 页数：{report.page_count}",
        "",
    ]
    if not insights:
        lines.append("尚未收藏研究结论。")
        return "\n".join(lines).strip() + "\n"

    for index, insight in enumerate(insights, start=1):
        lines.extend(
            [
                f"## {index}. {insight.text}",
                "",
                f"- 类型：{_kind_label(insight.kind)}",
                "",
            ]
        )
        for evidence in insight.evidence:
            lines.extend(
                [
                    f"**证据 {evidence.evidence_id}｜第 {evidence.page_number} 页**",
                    "",
                    f"> {evidence.text.replace(chr(10), ' ')}",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def company_report_to_markdown(
    report: Report,
    *,
    answer: ResearchAnswer | None = None,
    insights: list[Insight] | None = None,
) -> str:
    lines = [
        f"# {report.company}（{report.year}）公司研究简报",
        "",
        f"- 年报文件：{report.filename}",
        f"- 页数：{report.page_count}",
        "",
    ]
    if answer and answer.is_answered:
        lines.extend(
            [
                "## 核心结论",
                "",
                answer.summary,
                "",
            ]
        )
        if answer.sections:
            for section in answer.sections:
                lines.extend([f"## {section.title}", ""])
                lines.extend(_claim_markdown(section.claims))
        else:
            lines.extend(_claim_markdown(answer.claims))

    if insights:
        lines.extend(["## 研究板结论", ""])
        for insight in insights:
            citations = " ".join(
                f"[{item.evidence_id}·第{item.page_number}页]"
                for item in insight.evidence
            )
            lines.append(f"- {insight.text} {citations}")
        lines.append("")

    if not answer and not insights:
        lines.append("尚未生成研究底稿或收藏研究结论。")

    return "\n".join(lines).strip() + "\n"


def change_report_to_markdown(
    current: Report,
    previous: Report,
    report: ChangeReport,
) -> str:
    lines = [
        f"# {current.company} 年报变化雷达",
        "",
        f"- 较新年报：{current.year}",
        f"- 较早年报：{previous.year}",
        "",
        "## 核心结论",
        "",
        report.summary,
        "",
    ]
    for item in report.changes:
        citations = " ".join(f"[{value}]" for value in item.evidence_ids)
        lines.extend(
            [
                f"## {item.dimension}｜{item.change_type}",
                "",
                f"- 上一年度：{item.previous}",
                f"- 本年度：{item.current}",
                f"- 证据：{citations}",
                "",
            ]
        )
    if report.evidence:
        lines.extend(["## 证据索引", ""])
        for evidence in report.evidence:
            lines.append(
                f"- [{evidence.evidence_id}] {evidence.company} "
                f"{evidence.year} 年第 {evidence.page_number} 页"
            )
        lines.append("")
    if not report.changes:
        lines.append("没有识别出可验证的年度变化。")
    return "\n".join(lines).strip() + "\n"


def company_comparison_to_markdown(
    report_a: Report,
    report_b: Report,
    answer: ResearchAnswer,
) -> str:
    lines = [
        f"# {report_a.company} vs {report_b.company} 公司对比",
        "",
        f"- A 公司：{report_a.company}（{report_a.year}）",
        f"- B 公司：{report_b.company}（{report_b.year}）",
        "",
        "## 核心结论",
        "",
        answer.summary,
        "",
    ]
    for section in answer.sections:
        lines.extend([f"## {section.title}", ""])
        lines.extend(_claim_markdown(section.claims))
    if not answer.sections:
        lines.extend(_claim_markdown(answer.claims))
    return "\n".join(lines).strip() + "\n"


def markdown_to_html(markdown_text: str, *, title: str) -> str:
    content = markdown.markdown(
        markdown_text,
        extensions=["tables", "sane_lists"],
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      max-width: 860px;
      margin: 40px auto;
      padding: 0 24px;
      color: #172033;
      font: 16px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1, h2 {{ line-height: 1.3; }}
    blockquote {{
      margin: 16px 0;
      padding: 8px 16px;
      border-left: 4px solid #0f766e;
      background: #f1f5f9;
    }}
    code {{
      padding: 2px 5px;
      border-radius: 4px;
      background: #e2e8f0;
    }}
  </style>
</head>
<body>
{content}
</body>
</html>
"""


def _claim_markdown(claims) -> list[str]:
    lines = []
    for claim in claims:
        citations = " ".join(
            f"[{evidence_id}]"
            for evidence_id in claim.evidence_ids
        )
        lines.append(f"- {claim.text} {citations}")
    lines.append("")
    return lines


def _kind_label(kind: str) -> str:
    return {
        "annual_report_fact": "年报事实",
        "model_judgment": "模型判断",
    }.get(kind, kind)

