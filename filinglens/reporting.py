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
        f"# {report.company} ({report.year}) Research Brief",
        "",
        f"- Annual report: {report.filename}",
        f"- Pages: {report.page_count}",
        "",
    ]
    if not insights:
        lines.append("No research conclusions have been saved yet.")
        return "\n".join(lines).strip() + "\n"

    for index, insight in enumerate(insights, start=1):
        lines.extend(
            [
                f"## {index}. {insight.text}",
                "",
                f"- Type: {_kind_label(insight.kind)}",
                "",
            ]
        )
        for evidence in insight.evidence:
            lines.extend(
                [
                    f"**Evidence {evidence.evidence_id} | Page {evidence.page_number}**",
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
        f"# {report.company} ({report.year}) Company Research Brief",
        "",
        f"- Annual report: {report.filename}",
        f"- Pages: {report.page_count}",
        "",
    ]
    if answer and answer.is_answered:
        lines.extend(
            [
                "## Key Conclusions",
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
        lines.extend(["## Saved Research Conclusions", ""])
        for insight in insights:
            citations = " ".join(
                f"[{item.evidence_id}, p. {item.page_number}]"
                for item in insight.evidence
            )
            lines.append(f"- {insight.text} {citations}")
        lines.append("")

    if not answer and not insights:
        lines.append("No research brief or saved conclusion is available.")

    return "\n".join(lines).strip() + "\n"


def change_report_to_markdown(
    current: Report,
    previous: Report,
    report: ChangeReport,
) -> str:
    lines = [
        f"# {current.company} Annual Report Change Radar",
        "",
        f"- Current report: {current.year}",
        f"- Previous report: {previous.year}",
        "",
        "## Key Findings",
        "",
        report.summary,
        "",
    ]
    for item in report.changes:
        citations = " ".join(f"[{value}]" for value in item.evidence_ids)
        lines.extend(
            [
                f"## {item.dimension} | {item.change_type}",
                "",
                f"- Previous report: {item.previous}",
                f"- Current report: {item.current}",
                f"- Evidence: {citations}",
                "",
            ]
        )
    if report.evidence:
        lines.extend(["## Evidence Index", ""])
        for evidence in report.evidence:
            lines.append(
                f"- [{evidence.evidence_id}] {evidence.company} "
                f"{evidence.year}, page {evidence.page_number}"
            )
        lines.append("")
    if not report.changes:
        lines.append("No verifiable year-over-year changes were identified.")
    return "\n".join(lines).strip() + "\n"


def company_comparison_to_markdown(
    report_a: Report,
    report_b: Report,
    answer: ResearchAnswer,
) -> str:
    lines = [
        f"# {report_a.company} vs {report_b.company}: Company Comparison",
        "",
        f"- Company A: {report_a.company} ({report_a.year})",
        f"- Company B: {report_b.company} ({report_b.year})",
        "",
        "## Key Conclusions",
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
<html lang="en">
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
        "annual_report_fact": "Annual report fact",
        "model_judgment": "Model judgment",
    }.get(kind, kind)

