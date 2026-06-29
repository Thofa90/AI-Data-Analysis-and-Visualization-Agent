"""Render regressions for repeated chat-response controls."""

from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from services.dataset_profiler import profile_dataset
from services.metric_detector import detect_key_metrics


def test_multiple_tabular_answers_have_unique_download_buttons() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Europe"],
        "Country": ["France", "Germany"],
        "TotalRevenue": [20.0, 60.0],
    })
    profile = profile_dataset(dataframe)
    result = {
        "tool_name": "group_and_aggregate",
        "success": True,
        "summary": "Calculated sum of TotalRevenue by Country.",
        "data": [{"Country": "Germany", "TotalRevenue": 60.0}],
        "warnings": [],
        "execution_seconds": 0.01,
    }
    plan = {
        "tool_name": "group_and_aggregate",
        "arguments": {
            "group_by": "Country",
            "value_column": "TotalRevenue",
            "aggregation": "sum",
        },
        "chart_spec": None,
        "clarification": None,
        "safe_code": "df.groupby('Country')['TotalRevenue'].sum()",
    }
    history = [
        {
            "question": "First question",
            "answer": "First verified answer.",
            "plan": plan,
            "result": result,
            "chart_spec": None,
            "chart_data": [],
            "suggested_questions": [],
            "assumptions": [],
            "total_seconds": 0.1,
            "interpretation_seconds": 0.01,
            "tool_seconds": 0.01,
            "explanation_seconds": 0.01,
        },
        {
            "question": "Second question",
            "answer": "Second verified answer.",
            "plan": plan,
            "result": result,
            "chart_spec": None,
            "chart_data": [],
            "suggested_questions": [],
            "assumptions": [],
            "total_seconds": 0.1,
            "interpretation_seconds": 0.01,
            "tool_seconds": 0.01,
            "explanation_seconds": 0.01,
        },
    ]

    app = AppTest.from_file("app.py", default_timeout=15)
    app.run()
    app.session_state["active_dataframe"] = dataframe
    app.session_state["original_dataframe"] = dataframe.copy()
    app.session_state["dataset_profile"] = profile
    app.session_state["detected_metrics"] = detect_key_metrics(dataframe, profile)
    app.session_state["uploaded_file"] = {"name": "sales.csv", "size": 100}
    app.session_state["analysis_history"] = history
    app.sidebar.radio[0].set_value("Chat & Ask")
    app.run()

    assert not app.exception
    assert len(app.get("download_button")) == 2


def test_multiple_chat_charts_have_unique_insight_controls() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2021-01-01", "2021-02-01", "2021-03-01"],
        "TotalRevenue": [10.0, 20.0, 15.0],
    })
    profile = profile_dataset(dataframe)
    result = {
        "tool_name": "calculate_time_trend",
        "success": True,
        "summary": "Calculated the monthly sum trend for TotalRevenue.",
        "data": [
            {"Date": "2021-01", "TotalRevenue": 10.0},
            {"Date": "2021-02", "TotalRevenue": 20.0},
            {"Date": "2021-03", "TotalRevenue": 15.0},
        ],
        "warnings": [],
        "execution_seconds": 0.01,
    }
    chart_spec = {
        "chart_type": "line",
        "title": "Sum TotalRevenue by Month",
        "x": "Date",
        "y": "TotalRevenue",
        "aggregation": "sum",
        "time_grain": "month",
        "time_column": "Date",
    }
    plan = {
        "tool_name": "calculate_time_trend",
        "arguments": {
            "date_column": "Date",
            "value_column": "TotalRevenue",
            "aggregation": "sum",
            "frequency": "month",
        },
        "chart_spec": chart_spec,
        "clarification": None,
        "safe_code": "df.groupby('period')['TotalRevenue'].sum()",
    }
    history = [
        {
            "question": "First chart",
            "answer": "First chart answer.",
            "plan": plan,
            "result": result,
            "chart_spec": chart_spec,
            "chart_data": result["data"],
            "suggested_questions": [],
            "assumptions": [],
            "total_seconds": 0.1,
            "interpretation_seconds": 0.01,
            "tool_seconds": 0.01,
            "explanation_seconds": 0.01,
        },
        {
            "question": "Second chart",
            "answer": "Second chart answer.",
            "plan": plan,
            "result": result,
            "chart_spec": chart_spec,
            "chart_data": result["data"],
            "suggested_questions": [],
            "assumptions": [],
            "total_seconds": 0.1,
            "interpretation_seconds": 0.01,
            "tool_seconds": 0.01,
            "explanation_seconds": 0.01,
        },
    ]

    app = AppTest.from_file("app.py", default_timeout=15)
    app.run()
    app.session_state["active_dataframe"] = dataframe
    app.session_state["original_dataframe"] = dataframe.copy()
    app.session_state["dataset_profile"] = profile
    app.session_state["detected_metrics"] = detect_key_metrics(dataframe, profile)
    app.session_state["uploaded_file"] = {"name": "sales.csv", "size": 100}
    app.session_state["analysis_history"] = history
    app.sidebar.radio[0].set_value("Chat & Ask")
    app.run()

    assert not app.exception
    radio_keys = {radio.key for radio in app.radio}
    assert "insight_mode_chat_0" in radio_keys
    assert "insight_mode_chat_1" in radio_keys


def test_chat_page_renders_dataset_query_guidance() -> None:
    dataframe = pd.DataFrame({
        "Order Date": ["2017-01-01", "2017-02-01"],
        "Ship Date": ["2017-01-03", "2017-02-03"],
        "Ship Mode": ["Standard Class", "Second Class"],
        "Region": ["West", "East"],
        "Sales": [100.0, 200.0],
        "Profit": [10.0, 20.0],
    })
    profile = profile_dataset(dataframe)

    app = AppTest.from_file("app.py", default_timeout=15)
    app.run()
    app.session_state["active_dataframe"] = dataframe
    app.session_state["original_dataframe"] = dataframe.copy()
    app.session_state["dataset_profile"] = profile
    app.session_state["detected_metrics"] = detect_key_metrics(dataframe, profile)
    app.session_state["uploaded_file"] = {"name": "orders.csv", "size": 100}
    app.sidebar.radio[0].set_value("Chat & Ask")
    app.run()

    assert not app.exception
    guide = app.session_state["dataset_query_guide"]
    assert guide["date_columns"][0]["column_name"] == "Order Date"
    assert any(
        button.label == "Using Order Date, show monthly Profit for 2017."
        for button in app.button
    )
