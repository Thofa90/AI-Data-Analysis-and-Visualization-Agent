"""HTML and PDF summary report generation from real session data."""

from __future__ import annotations

import base64
import html
import io
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from services.chart_insight_service import generate_chart_insight
from services.chart_service import ChartSpec, create_chart, figure_to_png, prepare_figure_for_report
from services.metric_detector import MetricResult
from services.profile_models import DatasetProfile


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _simple_table_html(headers: list[str], rows: list[list[Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{header_html}</tr>{body}</table>"


def _pdf_table(headers: list[str], rows: list[list[Any]], *, col_widths: list[float] | None = None) -> Table:
    table = Table(
        [headers, *[[ _fmt(value) for value in row] for row in rows]],
        repeatRows=1,
        colWidths=col_widths,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173b70")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#ccd4e0")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    return table


def _dataset_summary(profile: DatasetProfile) -> str:
    return (
        f"The dataset contains {profile.row_count:,} rows and {profile.column_count:,} columns. "
        f"It has {profile.total_missing:,} missing cell(s) ({profile.missing_percentage:.2f}%), "
        f"{profile.duplicate_rows:,} duplicate row(s), and a data-quality score of "
        f"{profile.quality.score:.1f}/100 ({profile.quality.rating})."
    )


def _column_summary_rows(profile: DatasetProfile, limit: int = 30) -> list[list[Any]]:
    rows = []
    for column in profile.columns[:limit]:
        rows.append([
            column.name,
            column.kind,
            column.pandas_dtype,
            column.missing_count,
            f"{column.missing_percentage:.1f}%",
            column.unique_count,
            "Yes" if column.is_likely_id else "",
            column.potential_type_issue or "",
        ])
    return rows


def _statistics_rows(profile: DatasetProfile, limit: int = 20) -> list[list[Any]]:
    rows = []
    for column in profile.columns:
        if column.name not in profile.numeric_columns:
            continue
        rows.append([
            column.name,
            column.minimum,
            column.maximum,
            column.mean,
            column.median,
            column.standard_deviation,
            column.outlier_count,
        ])
        if len(rows) >= limit:
            break
    return rows


def _correlation_rows(dataframe: pd.DataFrame | None, profile: DatasetProfile, limit: int = 10) -> list[list[Any]]:
    if dataframe is None:
        return []
    numeric = [column for column in profile.numeric_columns if column in dataframe.columns]
    if len(numeric) < 2:
        return []
    corr = dataframe[numeric].corr(numeric_only=True).round(3)
    numeric = numeric[:limit]
    return [[column, *[corr.loc[column, other] for other in numeric]] for column in numeric]


def _saved_chart_sections(
    saved_insights: list[dict[str, Any]],
    dataframe: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for index, item in enumerate(saved_insights, start=1):
        chart_spec = item.get("chart_spec")
        if not chart_spec:
            continue
        try:
            spec = ChartSpec.model_validate(chart_spec)
            chart_rows = item.get("chart_data") or []
            chart_source = pd.DataFrame(chart_rows) if chart_rows else dataframe
            if chart_source is None or chart_source.empty:
                raise ValueError("No chart data was saved with this insight.")
            if chart_rows:
                spec = spec.model_copy(update={"filter_column": None, "filter_value": None, "limit": None})
            figure, chart_result = create_chart(chart_source, spec)
            insight = generate_chart_insight(chart_result)
            try:
                png = figure_to_png(prepare_figure_for_report(figure))
                image_base64 = base64.b64encode(png).decode("ascii")
            except ValueError as exc:
                png = None
                image_base64 = None
                item = {**item, "chart_warning": str(exc)}
            sections.append({
                "number": index,
                "question": item.get("question") or spec.title,
                "answer": item.get("answer") or "",
                "spec": spec,
                "insight": insight,
                "png": png,
                "image_base64": image_base64,
                "warning": item.get("chart_warning"),
            })
        except (TypeError, ValueError) as exc:
            sections.append({
                "number": index,
                "question": item.get("question", f"Saved insight {index}"),
                "answer": item.get("answer", ""),
                "warning": f"Chart could not be rebuilt: {exc}",
            })
    return sections


def _insight_detail_pairs(insight: Any) -> list[tuple[str, str]]:
    """Detailed chart explanation fields used by report outputs."""
    if not insight:
        return []
    pairs = [
        ("Key finding", insight.key_finding),
        ("Supporting evidence", insight.supporting_evidence),
        ("Interpretation", insight.interpretation),
        ("Caution", insight.caution),
        ("Recommended next step", insight.recommended_next_step),
        ("Evidence strength", str(insight.evidence_strength).title()),
    ]
    return [(label, value) for label, value in pairs if value]


def _chart_explanation_html(insight: Any) -> str:
    pairs = _insight_detail_pairs(insight)
    if not pairs:
        return ""
    items = "".join(
        f"<p><strong>{html.escape(label)}:</strong> {html.escape(str(value))}</p>"
        for label, value in pairs
    )
    return f"<div class='chart-explanation'><h4>Detailed chart explanation</h4>{items}</div>"


def generate_html_report(
    dataset_name: str,
    profile: DatasetProfile,
    metrics: list[MetricResult],
    observations: list[str],
    saved_insights: list[dict[str, Any]],
    analysis_history: list[dict[str, Any]],
    evaluation_history: list[dict[str, Any]] | None = None,
    dataframe: pd.DataFrame | None = None,
) -> bytes:
    """Generate a structured self-contained HTML report."""
    metric_rows = [
        [metric.label, metric.value, metric.aggregation]
        for metric in metrics
    ] or [["No detected metric", "", ""]]
    observation_items = "".join(f"<li>{html.escape(item)}</li>" for item in observations) or "<li>No EDA observations available.</li>"
    warning_items = "".join(
        f"<li>{html.escape(issue.detail)} Recommendation: {html.escape(issue.recommendation)}</li>"
        for issue in profile.quality.issues
    ) or "<li>No scored data-quality warnings.</li>"
    history_items = "".join(
        f"<li>{html.escape(str(item.get('question', '')))}</li>" for item in analysis_history[-20:]
    ) or "<li>No analysis history.</li>"
    evaluation_history = evaluation_history or []
    evaluation_summary = (
        f"{len(evaluation_history)} evaluated answer(s), average overall score "
        f"{sum(item.get('overall_score', 0) for item in evaluation_history) / len(evaluation_history):.1%}."
        if evaluation_history else "No answer evaluations have been run."
    )
    corr_rows = _correlation_rows(dataframe, profile)
    corr_headers = ["Column", *[row[0] for row in corr_rows]]
    chart_sections = _saved_chart_sections(saved_insights, dataframe)
    chart_html = ""
    for section in chart_sections:
        insight = section.get("insight")
        image = (
            f'<img class="chart" src="data:image/png;base64,{section["image_base64"]}" alt="Saved chart">'
            if section.get("image_base64")
            else ""
        )
        warning = f"<p class='warning'>{html.escape(section['warning'])}</p>" if section.get("warning") else ""
        explanation = _chart_explanation_html(insight)
        chart_html += (
            f"<section><h3>{section['number']}. {html.escape(str(section.get('question', 'Saved chart')))}</h3>"
            f"{image}{warning}<p>{html.escape(str(section.get('answer', '')))}</p>{explanation}</section>"
        )
    if not chart_html:
        chart_html = "<p>No saved chart insights were available. Save chart answers from Chat or Visualization first.</p>"
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Data Analysis Report</title>
<style>
body{{font-family:Arial;color:#172033;max-width:980px;margin:40px auto;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:10px 0 22px}}th,td{{border:1px solid #ccd4e0;padding:7px;text-align:left;font-size:13px}}
th{{background:#173b70;color:white}}h1,h2{{color:#173b70}}.chart{{width:100%;max-width:860px;border:1px solid #ccd4e0;margin:8px 0 12px}}
.chart-explanation{{background:#f7faff;border-left:4px solid #315eff;padding:10px 14px;margin:8px 0 18px}}
.chart-explanation h4{{margin:0 0 6px;color:#173b70}}.chart-explanation p{{margin:6px 0}}
.warning{{color:#8a4b00;background:#fff4dd;padding:8px}}</style></head><body>
<h1>AI Data Analysis Report</h1><p><strong>Dataset:</strong> {html.escape(dataset_name)}</p>
<h2>Executive Dataset Summary</h2><p>{html.escape(_dataset_summary(profile))}</p>
<h2>All Data Overview</h2>
{_simple_table_html(["Rows", "Columns", "Numeric", "Categorical", "Datetime", "ID", "Missing %", "Duplicates"], [[profile.row_count, profile.column_count, len(profile.numeric_columns), len(profile.categorical_columns), len(profile.datetime_columns), len(profile.id_columns), f"{profile.missing_percentage:.2f}%", profile.duplicate_rows]])}
<h2>Detected Business Metrics</h2>{_simple_table_html(["Metric", "Value", "Aggregation"], metric_rows)}
<h2>Column Summary</h2>{_simple_table_html(["Column", "Kind", "Pandas Type", "Missing", "Missing %", "Unique", "Likely ID", "Type Issue"], _column_summary_rows(profile))}
<h2>Statistical Analysis</h2>{_simple_table_html(["Column", "Min", "Max", "Mean", "Median", "Std Dev", "Outliers"], _statistics_rows(profile) or [["No numeric columns", "", "", "", "", "", ""]])}
<h2>Correlation Matrix</h2>{_simple_table_html(corr_headers, corr_rows) if corr_rows else "<p>At least two numeric columns are required for a correlation matrix.</p>"}
<h2>EDA Observations</h2><ul>{observation_items}</ul>
<h2>Saved Chart Insights</h2>{chart_html}
<h2>Analysis History</h2><ul>{history_items}</ul>
<h2>Evaluation Summary</h2><p>{html.escape(evaluation_summary)}</p>
<h2>Data Quality Warnings</h2><ul>{warning_items}</ul>
<h2>Limitations</h2><p>Potential outliers and inferred types require domain review. LLM explanations,
when enabled, are based only on verified tool outputs and limited dataset context.</p></body></html>"""
    return document.encode("utf-8")


def generate_pdf_report(
    dataset_name: str,
    profile: DatasetProfile,
    metrics: list[MetricResult],
    observations: list[str],
    saved_insights: list[dict[str, Any]],
    evaluation_history: list[dict[str, Any]] | None = None,
    dataframe: pd.DataFrame | None = None,
) -> bytes:
    """Generate a structured PDF report in memory."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=.55 * inch, leftMargin=.55 * inch)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AI Data Analysis Report", styles["Title"]),
        Paragraph(f"Dataset: {html.escape(dataset_name)}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Executive Dataset Summary", styles["Heading2"]),
        Paragraph(html.escape(_dataset_summary(profile)), styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("All Data Overview", styles["Heading2"]),
        _pdf_table(
            ["Rows", "Columns", "Numeric", "Categorical", "Datetime", "ID", "Missing %", "Duplicates"],
            [[profile.row_count, profile.column_count, len(profile.numeric_columns), len(profile.categorical_columns), len(profile.datetime_columns), len(profile.id_columns), f"{profile.missing_percentage:.2f}%", profile.duplicate_rows]],
        ),
        Spacer(1, 10),
        Paragraph("Detected Business Metrics", styles["Heading2"]),
        _pdf_table(
            ["Metric", "Value", "Aggregation"],
            [[metric.label, metric.value, metric.aggregation] for metric in metrics] or [["No detected metric", "", ""]],
            col_widths=[2.8 * inch, 1.4 * inch, 1.4 * inch],
        ),
        Spacer(1, 10),
        Paragraph("Column Summary", styles["Heading2"]),
        _pdf_table(
            ["Column", "Kind", "Type", "Missing", "Missing %", "Unique", "ID", "Issue"],
            _column_summary_rows(profile, limit=24),
            col_widths=[1.3 * inch, .75 * inch, .85 * inch, .55 * inch, .65 * inch, .55 * inch, .35 * inch, 1.4 * inch],
        ),
        Spacer(1, 10),
        Paragraph("Statistical Analysis", styles["Heading2"]),
        _pdf_table(
            ["Column", "Min", "Max", "Mean", "Median", "Std Dev", "Outliers"],
            _statistics_rows(profile, limit=16) or [["No numeric columns", "", "", "", "", "", ""]],
        ),
        Spacer(1, 10),
        Paragraph("Correlation Matrix", styles["Heading2"]),
    ]
    corr_rows = _correlation_rows(dataframe, profile, limit=7)
    if corr_rows:
        story.append(_pdf_table(["Column", *[row[0] for row in corr_rows]], corr_rows))
    else:
        story.append(Paragraph("At least two numeric columns are required for a correlation matrix.", styles["BodyText"]))
    story.extend([Spacer(1, 10), Paragraph("EDA Observations", styles["Heading2"])])
    if observations:
        story.extend(Paragraph(f"- {html.escape(item)}", styles["BodyText"]) for item in observations[:12])
    else:
        story.append(Paragraph("No EDA observations available.", styles["BodyText"]))
    story.extend([PageBreak(), Paragraph("Saved Chart Insights", styles["Heading2"])])
    chart_sections = _saved_chart_sections(saved_insights, dataframe)
    if chart_sections:
        for section in chart_sections[:20]:
            story.append(Paragraph(f"{section['number']}. {html.escape(str(section.get('question', 'Saved chart')))}", styles["Heading3"]))
            if section.get("png"):
                story.append(Image(io.BytesIO(section["png"]), width=6.6 * inch, height=3.7 * inch))
            if section.get("warning"):
                story.append(Paragraph(html.escape(section["warning"]), styles["BodyText"]))
            if section.get("answer"):
                story.append(Paragraph(html.escape(str(section["answer"])), styles["BodyText"]))
            insight = section.get("insight")
            if insight:
                story.append(Paragraph("Detailed chart explanation", styles["Heading4"]))
                story.extend(
                    Paragraph(f"<b>{html.escape(label)}:</b> {html.escape(str(value))}", styles["BodyText"])
                    for label, value in _insight_detail_pairs(insight)
                )
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No saved chart insights were available. Save chart answers from Chat or Visualization first.", styles["BodyText"]))
    evaluation_history = evaluation_history or []
    story.extend([Spacer(1, 10), Paragraph("Evaluation Summary", styles["Heading2"])])
    if evaluation_history:
        average = sum(item.get("overall_score", 0) for item in evaluation_history) / len(evaluation_history)
        story.append(Paragraph(
            f"{len(evaluation_history)} evaluated answer(s), average overall score {average:.1%}.",
            styles["BodyText"],
        ))
    else:
        story.append(Paragraph("No answer evaluations have been run.", styles["BodyText"]))
    story.extend([Spacer(1, 10), Paragraph("Data Quality Warnings", styles["Heading2"])])
    if profile.quality.issues:
        story.extend(
            Paragraph(f"- {html.escape(issue.detail)} Recommendation: {html.escape(issue.recommendation)}", styles["BodyText"])
            for issue in profile.quality.issues
        )
    else:
        story.append(Paragraph("No scored data-quality warnings.", styles["BodyText"]))
    document.build(story)
    return buffer.getvalue()
