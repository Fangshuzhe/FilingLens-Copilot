from __future__ import annotations

from filinglens.models import Insight, Report


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


def _kind_label(kind: str) -> str:
    return {
        "annual_report_fact": "年报事实",
        "model_judgment": "模型判断",
    }.get(kind, kind)

