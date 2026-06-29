"""Tests for in-memory exports and reports."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from services.dataset_profiler import profile_dataset
from services.export_service import dataframe_to_csv, dataframe_to_excel
from services.metric_detector import detect_key_metrics
from services.report_service import generate_html_report, generate_pdf_report
from services.chart_service import REPORT_CATEGORICAL_COLORS, REPORT_LINE_COLORS, prepare_figure_for_report


def test_data_and_report_exports_are_non_empty() -> None:
    dataframe = pd.DataFrame({"Region": ["W", "E"], "Sales": [10.0, 20.0]})
    profile = profile_dataset(dataframe)
    metrics = detect_key_metrics(dataframe, profile)
    assert b"Region,Sales" in dataframe_to_csv(dataframe)
    assert len(dataframe_to_excel(dataframe)) > 100
    html = generate_html_report("sales.csv", profile, metrics, ["Verified observation."], [], [])
    pdf = generate_pdf_report("sales.csv", profile, metrics, ["Verified observation."], [])
    assert b"sales.csv" in html
    assert pdf.startswith(b"%PDF")


def test_report_uses_detailed_chart_explanation() -> None:
    dataframe = pd.DataFrame({"Region": ["West", "East", "West"], "Sales": [10.0, 20.0, 30.0]})
    profile = profile_dataset(dataframe)
    metrics = detect_key_metrics(dataframe, profile)
    saved_insights = [{
        "question": "Chart sales by region",
        "answer": "Sales were grouped by region.",
        "chart_spec": {
            "chart_type": "bar",
            "x": "Region",
            "y": "Sales",
            "aggregation": "sum",
            "title": "Sales by Region",
        },
    }]

    html = generate_html_report(
        "sales.csv",
        profile,
        metrics,
        ["Verified observation."],
        saved_insights,
        [],
        dataframe=dataframe,
    )

    assert b"Detailed chart explanation" in html
    assert b"Supporting evidence" in html
    assert b"Caution" in html
    assert b"Evidence strength" in html


def test_report_chart_theme_is_light_and_readable() -> None:
    figure = go.Figure()
    figure.add_bar(x=["West", "East"], y=[10, 20])
    figure.update_layout(paper_bgcolor="#050b18", plot_bgcolor="#050b18", font={"color": "#ffffff"})

    report_figure = prepare_figure_for_report(figure)

    assert report_figure.layout.paper_bgcolor == "#ffffff"
    assert report_figure.layout.plot_bgcolor == "#ffffff"
    assert report_figure.layout.font.color == "#172033"
    assert report_figure.layout.xaxis.tickfont.color == "#334155"
    assert report_figure.data[0].marker.color == "#315eff"


def test_report_multi_line_chart_uses_soft_distinct_colors() -> None:
    figure = go.Figure()
    figure.add_scatter(x=[1, 2, 3], y=[1, 3, 2], mode="lines+markers", line={"color": "#000000"}, name="A")
    figure.add_scatter(x=[1, 2, 3], y=[2, 1, 4], mode="lines+markers", line={"color": "#000000"}, name="B")

    report_figure = prepare_figure_for_report(figure)

    assert report_figure.data[0].line.color == REPORT_LINE_COLORS[0]
    assert report_figure.data[1].line.color == REPORT_LINE_COLORS[1]
    assert report_figure.data[0].line.color != report_figure.data[1].line.color


def test_report_stacked_bar_uses_distinct_category_colors() -> None:
    figure = go.Figure()
    figure.add_bar(x=["West", "East"], y=[10, 12], marker_color="#000000", name="Furniture")
    figure.add_bar(x=["West", "East"], y=[7, 9], marker_color="#000000", name="Technology")
    figure.add_bar(x=["West", "East"], y=[4, 5], marker_color="#000000", name="Office Supplies")
    figure.update_layout(barmode="stack")

    report_figure = prepare_figure_for_report(figure)

    colors = [trace.marker.color for trace in report_figure.data]
    assert colors == list(REPORT_CATEGORICAL_COLORS[:3])
    assert len(set(colors)) == 3
