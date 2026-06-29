"""Tests for deterministic intent interpretation and clarification."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.data_agent import deterministic_plan, run_agent, split_user_questions
from agent.schemas import AgentPlan, AgentResponse
from config.settings import get_settings
from services.dataset_profiler import profile_dataset


def _profile():
    dataframe = pd.DataFrame({
        "Order Date": ["2025-01-01", "2025-02-01"],
        "Region": ["West", "East"],
        "Sales": [10.0, 20.0],
        "Profit": [2.0, 4.0],
    })
    return profile_dataset(dataframe)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "What is total sales? What is average profit?",
            ["What is total sales", "What is average profit"],
        ),
        (
            "Show sales by region and then find the highest-profit region",
            ["Show sales by region", "find the highest-profit region"],
        ),
        (
            "1. Count regions\n2. Show total sales by region\n3. Find missing values",
            ["Count regions", "Show total sales by region", "Find missing values"],
        ),
        (
            "Plot sales and profit by region",
            ["Plot sales and profit by region"],
        ),
        (
            "How many quantity were sold through ship mode Standard Class.how many unique order id came from state new work",
            [
                "How many quantity were sold through ship mode Standard Class",
                "how many unique order id came from state new work",
            ],
        ),
    ],
)
def test_split_user_questions_preserves_independent_requests(
    message: str,
    expected: list[str],
) -> None:
    assert split_user_questions(message) == expected


def test_multiple_questions_execute_as_independent_analyses() -> None:
    dataframe = pd.DataFrame({
        "Region": ["East", "West", "East"],
        "Sales": [10.0, 20.0, 30.0],
        "Profit": [2.0, 8.0, 5.0],
    })
    profile = profile_dataset(dataframe)
    questions = split_user_questions(
        "What is total Sales? What is average Profit?"
    )

    responses = [
        run_agent(
            question,
            dataframe,
            profile,
            get_settings(),
            "qwen2.5:1.5b",
            "sales.csv",
            ollama_online=False,
        )
        for question in questions
    ]

    assert len(responses) == 2
    assert responses[0].plan.arguments == {
        "value_column": "Sales",
        "aggregation": "sum",
    }
    assert responses[1].plan.arguments == {
        "value_column": "Profit",
        "aggregation": "mean",
    }
    assert "**$60**" in responses[0].answer
    assert "**$5**" in responses[1].answer


def test_compound_asia_benchmark_request_never_includes_europe() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Europe"],
        "Country": ["Japan", "India", "China", "France", "Germany"],
        "TotalRevenue": [100.0, 50.0, 150.0, 1_000.0, 2_000.0],
    })
    message = (
        "find the mean total revenue of each country in region asia, "
        "and finaly show which countries are below from mean of total revenue of asia."
    )
    questions = split_user_questions(message)

    assert questions == [
        "find the mean total revenue of each country in region asia",
        "show which countries are below from mean of total revenue of asia",
    ]

    history = []
    responses = []
    for question in questions:
        response = run_agent(
            question,
            dataframe,
            profile_dataset(dataframe),
            get_settings(),
            "qwen2.5:1.5b",
            "sales.csv",
            history=history,
            ollama_online=False,
        )
        responses.append(response)
        history.append(response.model_dump(mode="json"))

    benchmark_response = responses[1]
    assert benchmark_response.plan.arguments["filter_column"] == "Region"
    assert benchmark_response.plan.arguments["filter_value"] == "Asia"
    assert benchmark_response.result
    assert benchmark_response.result.data == [{
        "Country": "India",
        "TotalRevenue": 50.0,
        "Benchmark": 100.0,
        "DifferenceFromBenchmark": -50.0,
    }]
    assert "Asia" in benchmark_response.answer
    assert "Europe" not in benchmark_response.answer


def test_benchmark_follow_up_inherits_previous_grouped_analysis_context() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe"],
        "Country": ["Japan", "India", "China", "France"],
        "TotalRevenue": [100.0, 50.0, 150.0, 5_000.0],
    })
    profile = profile_dataset(dataframe)
    history = []
    first_message = (
        "find mean revenue in region asia, and then "
        "find mean revenue for each country in asia"
    )
    for question in split_user_questions(first_message):
        response = run_agent(
            question,
            dataframe,
            profile,
            get_settings(),
            "qwen2.5:1.5b",
            "sales.csv",
            history=history,
            ollama_online=False,
        )
        history.append(response.model_dump(mode="json"))

    follow_up = run_agent(
        "from the above chat find the countries which have below mean of asia",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        history=history,
        ollama_online=False,
    )

    assert follow_up.plan.tool_name == "compare_grouped_to_benchmark"
    assert follow_up.plan.arguments == {
        "category_column": "Country",
        "value_column": "TotalRevenue",
        "aggregation": "mean",
        "benchmark": "mean",
        "comparison": "below",
        "benchmark_group_by": None,
        "filter_column": "Region",
        "filter_value": "Asia",
    }
    assert follow_up.result
    assert follow_up.result.data == [{
        "Country": "India",
        "TotalRevenue": 50.0,
        "Benchmark": 100.0,
        "DifferenceFromBenchmark": -50.0,
    }]
    assert "Europe" not in follow_up.answer


def test_this_region_inherits_previous_region_value() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "Country": ["Japan", "India", "France"],
        "TotalRevenue": [100.0, 50.0, 1_000.0],
    })
    profile = profile_dataset(dataframe)
    history = []

    for question in split_user_questions(
        "find mean revenue in asia and then "
        "find mean revenue of each country in this region"
    ):
        response = run_agent(
            question,
            dataframe,
            profile,
            get_settings(),
            "qwen2.5:1.5b",
            "sales.csv",
            history=history,
            ollama_online=False,
        )
        history.append(response.model_dump(mode="json"))

    second = AgentResponse.model_validate(history[-1])
    assert second.plan.tool_name == "group_and_aggregate"
    assert second.plan.arguments["group_by"] == "Country"
    assert second.plan.arguments["filter_column"] == "Region"
    assert second.plan.arguments["filter_value"] == "Asia"
    assert second.result
    assert {row["Country"] for row in second.result.data} == {"Japan", "India"}


def test_top_countries_in_region_filters_then_aggregates_before_ranking() -> None:
    dataframe = pd.DataFrame({
        "Country": [
            "Czech Republic", "Papua New Guinea", "Mongolia",
            "China", "China", "Japan", "India",
        ],
        "Region": [
            "Europe", "Australia and Oceania", "Asia",
            "Asia", "Asia", "Asia", "Asia",
        ],
        "TotalRevenue": [
            6_617_209.54, 6_580_454.69, 6_557_065.24,
            4_000_000.0, 4_000_000.0, 7_000_000.0, 5_000_000.0,
        ],
    })
    response = run_agent(
        "what are the top 3 country in region asia generating higher total revenue",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "Country",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "limit": 3,
        "filter_column": "Region",
        "filter_value": "Asia",
    }
    assert response.result
    assert response.result.data == [
        {"Country": "China", "TotalRevenue": 8_000_000.0},
        {"Country": "Japan", "TotalRevenue": 7_000_000.0},
        {"Country": "Mongolia", "TotalRevenue": 6_557_065.24},
    ]
    assert all(
        row["Country"] not in {"Czech Republic", "Papua New Guinea"}
        for row in response.result.data
    )


def test_top_five_countries_by_total_revenue_handles_plural_group_word() -> None:
    dataframe = pd.DataFrame({
        "Country": ["A", "B", "C", "D", "E", "F"],
        "TotalRevenue": [10.0, 60.0, 20.0, 50.0, 30.0, 40.0],
    })

    response = run_agent(
        "Show the top five countries by total revenue.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "Country",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "limit": 5,
        "filter_column": None,
        "filter_value": None,
    }
    assert response.chart_spec
    assert response.chart_spec.x == "Country"
    assert response.chart_spec.y == "TotalRevenue"
    assert response.chart_spec.limit == 5
    assert response.result
    assert [row["Country"] for row in response.result.data] == ["B", "D", "F", "E", "C"]


def test_country_breakdown_in_region_for_specific_year_uses_grouped_chart() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Asia"],
        "Country": ["Thailand", "India", "Thailand", "France", "India"],
        "Date": ["2021-01-01", "2021-02-01", "2022-01-01", "2021-01-01", "2021-03-01"],
        "TotalRevenue": [10.0, 20.0, 999.0, 40.0, 30.0],
    })

    response = run_agent(
        "total revenue for each country in region asia for year 2021",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "Country",
        "secondary_group_by": None,
        "value_column": "TotalRevenue",
        "value_columns": None,
        "aggregation": "sum",
        "limit": None,
        "filter_column": "Region",
        "filter_value": "Asia",
        "date_column": "Date",
        "start_date": pd.Timestamp("2021-01-01"),
        "end_date": pd.Timestamp("2021-12-31"),
    }
    assert response.chart_spec
    assert response.chart_spec.chart_type == "bar"
    assert response.chart_spec.x == "Country"
    assert response.chart_spec.filter_column == "Region"
    assert response.chart_spec.date_range_start == pd.Timestamp("2021-01-01")
    assert response.result
    assert response.result.data == [
        {"Country": "India", "TotalRevenue": 50.0},
        {"Country": "Thailand", "TotalRevenue": 10.0},
    ]
    assert response.chart_data == response.result.data


def test_monthly_country_revenue_in_region_year_uses_time_series_breakdown() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Asia", "Asia"],
        "Country": ["Thailand", "India", "Thailand", "France", "India", "Japan"],
        "Date": [
            "2021-01-01",
            "2021-01-15",
            "2021-02-01",
            "2021-01-01",
            "2022-01-01",
            "2021-02-15",
        ],
        "TotalRevenue": [10.0, 20.0, 30.0, 40.0, 999.0, 50.0],
    })

    response = run_agent(
        "For region Asia, find the monthly revenue of each country for the year 2021.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments == {
        "date_column": "Date",
        "value_column": "TotalRevenue",
        "breakdown_column": "Country",
        "aggregation": "sum",
        "frequency": "month",
        "start_date": pd.Timestamp("2021-01-01"),
        "end_date": pd.Timestamp("2021-12-31"),
        "filter_column": "Region",
        "filter_value": "Asia",
    }
    assert response.chart_spec
    assert response.chart_spec.chart_type == "line"
    assert response.chart_spec.x == "Date"
    assert response.chart_spec.y == "TotalRevenue"
    assert response.chart_spec.color == "Country"
    assert response.chart_spec.filter_column == "Region"
    assert response.chart_spec.filter_value == "Asia"
    assert response.chart_spec.time_grain == "month"
    assert response.result
    assert response.result.data == [
        {"Date": "2021-01", "Country": "India", "TotalRevenue": 20.0},
        {"Date": "2021-01", "Country": "Thailand", "TotalRevenue": 10.0},
        {"Date": "2021-02", "Country": "Japan", "TotalRevenue": 50.0},
        {"Date": "2021-02", "Country": "Thailand", "TotalRevenue": 30.0},
    ]
    assert all(row["Country"] != "France" for row in response.result.data)
    assert all(row["Date"].startswith("2021") for row in response.result.data)
    assert response.chart_data
    assert "where Region = Asia" in response.answer
    assert "month and country" in response.answer


def test_same_department_reference_is_dataset_dynamic() -> None:
    dataframe = pd.DataFrame({
        "Department": ["Support", "Support", "Engineering"],
        "Team": ["A", "B", "C"],
        "ResolutionTime": [4.0, 8.0, 20.0],
    })
    profile = profile_dataset(dataframe)
    first = run_agent(
        "find mean ResolutionTime in Support",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "operations.csv",
        ollama_online=False,
    )
    second = run_agent(
        "show mean ResolutionTime for each Team in the same Department",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "operations.csv",
        history=[first.model_dump(mode="json")],
        ollama_online=False,
    )

    assert second.plan.arguments["group_by"] == "Team"
    assert second.plan.arguments["filter_column"] == "Department"
    assert second.plan.arguments["filter_value"] == "Support"
    assert second.result
    assert {row["Team"] for row in second.result.data} == {"A", "B"}


def test_typo_tolerance_corrects_intent_columns_and_category_values() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "Country": ["Japan", "India", "France"],
        "TotalRevenue": [100.0, 50.0, 5_000.0],
    })
    response = run_agent(
        "find countires below the avarage total reveneu in Aisa",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "compare_grouped_to_benchmark"
    assert response.plan.arguments["category_column"] == "Country"
    assert response.plan.arguments["value_column"] == "TotalRevenue"
    assert response.plan.arguments["filter_column"] == "Region"
    assert response.plan.arguments["filter_value"] == "Asia"
    assert response.result
    assert response.result.data[0]["Country"] == "India"


def test_typo_tolerance_does_not_guess_ambiguous_metric() -> None:
    dataframe = pd.DataFrame({
        "Department": ["A", "B"],
        "SalesCost": [10.0, 20.0],
        "SalesCount": [1, 2],
    })
    plan = deterministic_plan(
        "show salescst by department",
        profile_dataset(dataframe),
        dataframe=dataframe,
    )

    assert not plan.tool_name
    assert plan.clarification


def test_plans_grouping_correlation_and_time_trend() -> None:
    profile = _profile()
    grouped = deterministic_plan("Show total Sales by Region", profile)
    assert grouped.tool_name == "group_and_aggregate"
    assert grouped.arguments["aggregation"] == "sum"
    correlation = deterministic_plan("Are Sales and Profit correlated?", profile)
    assert correlation.tool_name == "calculate_correlation"
    trend = deterministic_plan("Show monthly Sales trend using Order Date", profile)
    assert trend.tool_name == "calculate_time_trend"


@pytest.mark.parametrize(
    "question",
    [
        "Show me chart of Sales vs Sub-category",
        "show chart of total sales for all sub-category",
    ],
)
def test_chart_requests_with_hyphenated_category_group_sales(question: str) -> None:
    dataframe = pd.DataFrame({
        "Sub-category": ["Chairs", "Tables", "Chairs"],
        "Sales": [10.0, 20.0, 30.0],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "retail.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "Sub-category",
        "secondary_group_by": None,
        "value_column": "Sales",
        "value_columns": None,
        "aggregation": "sum",
        "limit": None,
        "filter_column": None,
        "filter_value": None,
    }
    assert response.chart_spec
    assert response.chart_spec.chart_type == "bar"
    assert response.chart_spec.x == "Sub-category"
    assert response.chart_spec.y == "Sales"
    assert response.result
    assert response.result.data == [
        {"Sub-category": "Chairs", "Sales": 40.0},
        {"Sub-category": "Tables", "Sales": 20.0},
    ]


def test_date_related_questions_route_to_time_trend() -> None:
    profile = _profile()

    yearly = deterministic_plan("Show Sales by year", profile)
    why = deterministic_plan("Why did Sales change over time?", profile)

    assert yearly.tool_name == "calculate_time_trend"
    assert yearly.arguments["frequency"] == "year"
    assert yearly.arguments["date_column"] == "Order Date"
    assert yearly.arguments["value_column"] == "Sales"
    assert why.tool_name == "calculate_time_trend"
    assert why.arguments["frequency"] == "month"


def test_date_related_chat_gets_deterministic_explanation() -> None:
    dataframe = pd.DataFrame({
        "Order Date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "Region": ["West", "East", "East"],
        "Sales": [10.0, 30.0, 20.0],
    })
    response = run_agent(
        "Why did Sales change by month?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert "Peak period" in response.answer
    assert "cannot prove why" in response.answer


def test_chat_answers_date_period_total_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.data_agent._current_date", lambda: pd.Timestamp("2026-06-19"))
    dataframe = pd.DataFrame({
        "Order Date": ["2026-06-01", "2026-06-18", "2026-07-01", "2025-06-01"],
        "TotalRevenue": [100.0, 200.0, 999.0, 50.0],
    })

    response = run_agent(
        "Give me total revenue for this month",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_date_aggregate"
    assert response.result
    assert response.result.data["value"] == pytest.approx(300.0)
    assert "Total revenue for" in response.answer
    assert "Order Date" in response.answer


@pytest.mark.parametrize(
    ("question", "expected_value", "expected_filter"),
    [
        ("How many unique order id were placed in order date 2017", 2, None),
        (
            "How many unique order id were placed in state California in order date 2017",
            1,
            ("State", "California"),
        ),
    ],
)
def test_chat_answers_date_filtered_distinct_count_questions(
    question: str,
    expected_value: int,
    expected_filter: tuple[str, str] | None,
) -> None:
    dataframe = pd.DataFrame({
        "Order ID": ["A", "A", "B", "C", "D"],
        "Order Date": ["2017-01-01", "2017-06-01", "2017-03-01", "2015-01-01", "2018-01-01"],
        "State": ["California", "California", "Nevada", "California", "California"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_date_aggregate"
    assert response.plan.arguments["date_column"] == "Order Date"
    assert response.plan.arguments["value_column"] == "Order ID"
    assert response.plan.arguments["aggregation"] == "nunique"
    assert response.plan.arguments["period_type"] == "year"
    assert response.plan.arguments["period_value"] == "2017"
    if expected_filter:
        assert response.plan.arguments["filter_column"] == expected_filter[0]
        assert response.plan.arguments["filter_value"] == expected_filter[1]
    else:
        assert response.plan.arguments["filter_column"] is None
        assert response.plan.arguments["filter_value"] is None
    assert response.result
    assert response.result.data["value"] == expected_value
    assert "unique order id count for 2017" in response.answer.lower()


@pytest.mark.parametrize(
    "question",
    [
        "Show the number of unique orders by month",
        "count the unique number of orders by month",
        "frequency of unique orders by month",
        "frequency of unique number by month",
    ],
)
def test_chat_answers_monthly_unique_order_count_questions(question: str) -> None:
    dataframe = pd.DataFrame({
        "Order ID": ["A", "A", "B", "C", "D"],
        "Order Date": ["2017-01-01", "2017-01-15", "2017-02-01", "2017-02-03", "2017-03-01"],
        "Profit": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments["date_column"] == "Order Date"
    assert response.plan.arguments["value_column"] == "Order ID"
    assert response.plan.arguments["aggregation"] == "nunique"
    assert response.plan.arguments["frequency"] == "month"
    assert response.chart_spec
    assert response.chart_spec.aggregation == "nunique"
    assert response.result
    assert response.result.data == [
        {"Order Date": "2017-01", "Order ID": 1},
        {"Order Date": "2017-02", "Order ID": 2},
        {"Order Date": "2017-03", "Order ID": 1},
    ]


def test_chat_answers_monthly_total_for_specific_year_with_chart() -> None:
    dataframe = pd.DataFrame({
        "Order Date": [
            "2021-01-01",
            "2021-01-20",
            "2021-02-01",
            "2021-03-01",
            "2022-01-01",
        ],
        "TotalRevenue": [10_000_000.0, 5_000_000.0, 20_000_000.0, 16_000_000.0, 999.0],
    })

    response = run_agent(
        "total revenue for each month in 2021",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments["frequency"] == "month"
    assert response.plan.arguments["aggregation"] == "sum"
    assert response.plan.arguments["start_date"] == pd.Timestamp("2021-01-01")
    assert response.plan.arguments["end_date"] == pd.Timestamp("2021-12-31")
    assert response.chart_spec is not None
    assert response.chart_spec.chart_type == "line"
    assert response.chart_spec.time_grain == "month"
    assert response.chart_spec.date_range_start == pd.Timestamp("2021-01-01")
    assert response.result
    assert response.result.data == [
        {"Order Date": "2021-01", "TotalRevenue": 15_000_000.0},
        {"Order Date": "2021-02", "TotalRevenue": 20_000_000.0},
        {"Order Date": "2021-03", "TotalRevenue": 16_000_000.0},
    ]
    assert response.chart_data

    alternate_response = run_agent(
        "show me total monthly revenue for year 2021",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert alternate_response.plan.tool_name == "calculate_time_trend"
    assert alternate_response.plan.arguments["frequency"] == "month"
    assert alternate_response.plan.arguments["start_date"] == pd.Timestamp("2021-01-01")
    assert alternate_response.plan.arguments["end_date"] == pd.Timestamp("2021-12-31")
    assert alternate_response.chart_spec
    assert alternate_response.chart_spec.time_grain == "month"
    assert alternate_response.result
    assert alternate_response.result.data == [
        {"Order Date": "2021-01", "TotalRevenue": 15_000_000.0},
        {"Order Date": "2021-02", "TotalRevenue": 20_000_000.0},
        {"Order Date": "2021-03", "TotalRevenue": 16_000_000.0},
    ]


def test_chat_last_month_across_year_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.data_agent._current_date", lambda: pd.Timestamp("2026-01-10"))
    dataframe = pd.DataFrame({
        "Order Date": ["2025-12-01", "2025-12-31", "2026-01-01"],
        "Revenue": [10.0, 20.0, 999.0],
    })

    response = run_agent(
        "How much revenue did we make last month?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.result
    assert response.result.data["value"] == pytest.approx(30.0)
    assert response.result.data["period_label"] == "December 2025"


def test_chat_explicit_year_and_average_month(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.data_agent._current_date", lambda: pd.Timestamp("2026-06-19"))
    dataframe = pd.DataFrame({
        "Order Date": ["2025-01-01", "2025-12-31", "2026-06-01"],
        "Revenue": [10.0, 20.0, 100.0],
    })

    year_response = run_agent(
        "Give me total revenue in 2025",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )
    average_response = run_agent(
        "What was the average revenue in June 2026?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert year_response.result
    assert year_response.result.data["value"] == pytest.approx(30.0)
    assert average_response.result
    assert average_response.result.data["aggregation"] == "mean"
    assert "average revenue" in average_response.answer.lower()
    assert "Total revenue" not in average_response.answer


def test_chat_answers_data_types_for_all_columns() -> None:
    dataframe = pd.DataFrame({
        "Order Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Region": ["Asia", "Europe"],
        "Revenue": [10.0, 20.0],
        "UnitsSold": [1, 2],
    })

    response = run_agent(
        "what are the data types of all column",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "get_column_information"
    assert response.plan.arguments == {}
    assert response.plan.response_mode == "text"
    assert response.result
    assert len(response.result.data) == 4
    assert "column data types" in response.answer.lower()
    assert "**Order Date**" in response.answer
    assert "**Revenue**" in response.answer


def test_chat_answers_year_metric_question_with_metric_typo() -> None:
    dataframe = pd.DataFrame({
        "Order Date": ["2021-01-01", "2021-12-31", "2022-01-01"],
        "Revenue": [10.0, 20.0, 100.0],
    })

    response = run_agent(
        "what is the reveue in 2021?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_date_aggregate"
    assert response.plan.arguments["value_column"] == "Revenue"
    assert response.plan.arguments["period_type"] == "year"
    assert response.plan.arguments["period_value"] == "2021"
    assert response.result
    assert response.result.data["value"] == pytest.approx(30.0)
    assert "2021" in response.answer
    assert "revenue" in response.answer.lower()


def test_chat_answers_filtered_time_series_for_country_and_region() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Asia"],
        "Country": ["Thailand", "Thailand", "India", "France", "Thailand"],
        "Date": ["2020-01-01", "2021-02-01", "2021-03-01", "2021-01-01", "2022-01-01"],
        "TotalRevenue": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    profile = profile_dataset(dataframe)

    thailand_year = run_agent(
        "Show total revenue for Thailand over all years.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )
    thailand_month = run_agent(
        "Show monthly total revenue for Thailand.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )
    asia_year = run_agent(
        "Show total revenue for Asia by year.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )
    scalar_thailand = run_agent(
        "What is the total revenue for Thailand?",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert thailand_year.plan.tool_name == "calculate_time_trend"
    assert thailand_year.plan.arguments["frequency"] == "year"
    assert thailand_year.plan.arguments["filter_column"] == "Country"
    assert thailand_year.plan.arguments["filter_value"] == "Thailand"
    assert thailand_year.result
    assert thailand_year.result.data == [
        {"Date": "2020", "TotalRevenue": 10.0},
        {"Date": "2021", "TotalRevenue": 20.0},
        {"Date": "2022", "TotalRevenue": 50.0},
    ]
    assert thailand_year.chart_spec
    assert thailand_year.chart_spec.filter_column == "Country"
    assert thailand_year.chart_spec.filter_value == "Thailand"
    assert thailand_year.chart_data
    assert "overall revenue where country = thailand" in thailand_year.answer.lower()
    assert "table and chart" in thailand_year.answer.lower()

    assert thailand_month.plan.tool_name == "calculate_time_trend"
    assert thailand_month.plan.arguments["frequency"] == "month"
    assert thailand_month.plan.arguments["filter_column"] == "Country"
    assert thailand_month.result
    assert thailand_month.result.data == [
        {"Date": "2020-01", "TotalRevenue": 10.0},
        {"Date": "2021-02", "TotalRevenue": 20.0},
        {"Date": "2022-01", "TotalRevenue": 50.0},
    ]
    assert thailand_month.chart_spec
    assert thailand_month.chart_spec.time_grain == "month"

    assert asia_year.plan.tool_name == "calculate_time_trend"
    assert asia_year.plan.arguments["frequency"] == "year"
    assert asia_year.plan.arguments["filter_column"] == "Region"
    assert asia_year.plan.arguments["filter_value"] == "Asia"
    assert asia_year.result
    assert asia_year.result.data == [
        {"Date": "2020", "TotalRevenue": 10.0},
        {"Date": "2021", "TotalRevenue": 50.0},
        {"Date": "2022", "TotalRevenue": 50.0},
    ]

    assert scalar_thailand.plan.tool_name == "calculate_filtered_aggregate"
    assert scalar_thailand.chart_spec is None
    assert scalar_thailand.result
    assert scalar_thailand.result.data["value"] == pytest.approx(80.0)


def test_chat_answers_multi_metric_time_series_with_dual_line_chart() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2020-01-01", "2021-01-01", "2021-06-01", "2022-01-01"],
        "SalesChannel": ["Online", "Offline", "Online", "Offline"],
        "TotalRevenue": [10.0, 20.0, 30.0, 40.0],
        "TotalProfit": [1.0, 2.0, 3.0, 4.0],
    })

    response = run_agent(
        "total revenue and total profit over the years in one chart",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments["date_column"] == "Date"
    assert response.plan.arguments["value_columns"] == ["TotalRevenue", "TotalProfit"]
    assert response.plan.arguments["frequency"] == "year"
    assert response.chart_spec
    assert response.chart_spec.chart_type == "dual_line"
    assert response.chart_spec.x == "Date"
    assert response.chart_spec.y == "TotalRevenue"
    assert response.chart_spec.secondary_y == "TotalProfit"
    assert response.chart_spec.time_grain == "year"
    assert response.result
    assert response.result.data == [
        {"Date": "2020", "TotalRevenue": 10.0, "TotalProfit": 1.0},
        {"Date": "2021", "TotalRevenue": 50.0, "TotalProfit": 5.0},
        {"Date": "2022", "TotalRevenue": 40.0, "TotalProfit": 4.0},
    ]
    assert response.chart_data
    assert "dual-line chart" in response.answer
    assert "**Revenue**" in response.answer
    assert "Total: $100" in response.answer
    assert "**Profit**" in response.answer
    assert response.answer.count("**") % 2 == 0


def test_multi_metric_highest_year_question_keeps_both_metrics_and_names_winning_years() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2020-01-01", "2021-01-01", "2021-06-01", "2022-01-01"],
        "TotalRevenue": [10.0, 50.0, 100.0, 40.0],
        "TotalProfit": [1.0, 5.0, 20.0, 6.0],
    })

    response = run_agent(
        "Which year has the highest total revenue and highest total profit?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments["value_columns"] == ["TotalRevenue", "TotalProfit"]
    assert response.plan.arguments["frequency"] == "year"
    assert response.result
    assert response.result.data == [
        {"Date": "2020", "TotalRevenue": 10.0, "TotalProfit": 1.0},
        {"Date": "2021", "TotalRevenue": 150.0, "TotalProfit": 25.0},
        {"Date": "2022", "TotalRevenue": 40.0, "TotalProfit": 6.0},
    ]
    assert "Highest revenue was 2021" in response.answer
    assert "Highest profit was 2021" in response.answer
    assert response.answer.count("**") % 2 == 0


def test_filtered_multi_metric_highest_year_question_applies_region_filter() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2020-01-01", "2021-01-01", "2021-06-01", "2022-01-01"],
        "Region": ["Asia", "Asia", "Europe", "Asia"],
        "TotalRevenue": [10.0, 50.0, 100.0, 40.0],
        "TotalProfit": [1.0, 5.0, 20.0, 6.0],
    })

    response = run_agent(
        "For Asia, which year has the highest revenue and profit?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments["value_columns"] == ["TotalRevenue", "TotalProfit"]
    assert response.plan.arguments["filter_column"] == "Region"
    assert response.plan.arguments["filter_value"] == "Asia"
    assert response.result
    assert response.result.data == [
        {"Date": "2020", "TotalRevenue": 10.0, "TotalProfit": 1.0},
        {"Date": "2021", "TotalRevenue": 50.0, "TotalProfit": 5.0},
        {"Date": "2022", "TotalRevenue": 40.0, "TotalProfit": 6.0},
    ]
    assert "Highest revenue was 2021" in response.answer
    assert "Highest profit was 2022" in response.answer
    assert "Region = Asia" in response.answer
    assert response.answer.count("**") % 2 == 0


def test_chat_answers_three_metric_yearly_time_series_with_multi_line_chart() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2021-01-01", "2021-02-01", "2022-01-01"],
        "TotalRevenue": [100.0, 200.0, 300.0],
        "TotalCost": [60.0, 80.0, 150.0],
        "TotalProfit": [40.0, 120.0, 150.0],
    })

    response = run_agent(
        "Show yearly revenue, cost, and profit.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments["date_column"] == "Date"
    assert response.plan.arguments["value_columns"] == [
        "TotalRevenue",
        "TotalCost",
        "TotalProfit",
    ]
    assert response.plan.arguments["frequency"] == "year"
    assert response.chart_spec
    assert response.chart_spec.chart_type == "line"
    assert response.chart_spec.x == "Date"
    assert response.chart_spec.y == "Value"
    assert response.chart_spec.color == "Metric"
    assert response.chart_spec.value_columns == [
        "TotalRevenue",
        "TotalCost",
        "TotalProfit",
    ]
    assert response.result
    assert response.result.data == [
        {"Date": "2021", "TotalRevenue": 300.0, "TotalCost": 140.0, "TotalProfit": 160.0},
        {"Date": "2022", "TotalRevenue": 300.0, "TotalCost": 150.0, "TotalProfit": 150.0},
    ]
    assert response.chart_data
    assert {row["Metric"] for row in response.chart_data} == {
        "TotalRevenue",
        "TotalCost",
        "TotalProfit",
    }
    assert "multi-line chart" in response.answer
    assert "**Profit**" in response.answer
    assert "Total: $310" in response.answer
    assert response.answer.count("**") % 2 == 0


def test_single_metric_time_chart_does_not_add_cost_from_total_token() -> None:
    dataframe = pd.DataFrame({
        "Item Type": ["Meat", "Meat", "Fruit", "Meat"],
        "Date": ["2020-01-01", "2021-01-01", "2021-06-01", "2022-01-01"],
        "TotalRevenue": [10.0, 20.0, 30.0, 40.0],
        "TotalCost": [1.0, 2.0, 3.0, 4.0],
    })

    response = run_agent(
        "chart for total revenue vs year for item type meat",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_time_trend"
    assert response.plan.arguments == {
        "date_column": "Date",
        "value_column": "TotalRevenue",
        "aggregation": "sum",
        "frequency": "year",
        "start_date": None,
        "end_date": None,
        "filter_column": "Item Type",
        "filter_value": "Meat",
    }
    assert "value_columns" not in response.plan.arguments
    assert response.chart_spec
    assert response.chart_spec.chart_type == "line"
    assert response.chart_spec.y == "TotalRevenue"
    assert response.chart_spec.secondary_y is None
    assert response.chart_spec.filter_column == "Item Type"
    assert response.result
    assert response.result.data == [
        {"Date": "2020", "TotalRevenue": 10.0},
        {"Date": "2021", "TotalRevenue": 20.0},
        {"Date": "2022", "TotalRevenue": 40.0},
    ]
    assert "TotalCost" not in response.result.data[0]


def test_grouped_extrema_query_returns_winner_per_region_chart() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Europe", "North America"],
        "ItemType": [
            "Household",
            "Cosmetics",
            "Household",
            "Cosmetics",
            "Office Supplies",
            "Office Supplies",
        ],
        "Date": ["2021-01-01"] * 6,
        "TotalRevenue": [100.0, 70.0, 50.0, 120.0, 80.0, 90.0],
        "TotalProfit": [10.0, 7.0, 5.0, 12.0, 8.0, 9.0],
    })

    response = run_agent(
        (
            "Create a chart with Region on the x-axis and the highest total revenue "
            "on the y-axis. For each region, also show the item type responsible "
            "for that region’s highest revenue."
        ),
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments == {
        "primary_group_column": "Region",
        "secondary_group_column": "ItemType",
        "metric_column": "TotalRevenue",
        "aggregation": "sum",
        "extremum": "max",
    }
    assert response.chart_spec
    assert response.chart_spec.chart_type == "grouped_extrema_bar"
    assert response.chart_spec.x == "Region"
    assert response.chart_spec.y == "TotalRevenue"
    assert response.chart_spec.color == "ItemType"
    assert response.result
    assert response.result.data == [
        {
            "Region": "Asia",
            "ItemType": "Household",
            "TotalRevenue": 150.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": "Cosmetics",
            "SecondPlaceValue": 70.0,
            "AbsoluteGap": 80.0,
            "PercentageGap": 114.28571428571428,
            "WinnerShareOfGroup": 68.18181818181817,
        },
        {
            "Region": "Europe",
            "ItemType": "Cosmetics",
            "TotalRevenue": 120.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": "Office Supplies",
            "SecondPlaceValue": 80.0,
            "AbsoluteGap": 40.0,
            "PercentageGap": 50.0,
            "WinnerShareOfGroup": 60.0,
        },
        {
            "Region": "North America",
            "ItemType": "Office Supplies",
            "TotalRevenue": 90.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": None,
            "SecondPlaceValue": None,
            "AbsoluteGap": None,
            "PercentageGap": None,
            "WinnerShareOfGroup": 100.0,
        },
    ]
    assert response.chart_data == [
        {"Region": "Asia", "ItemType": "Household", "TotalRevenue": 150.0, "Tie": False},
        {"Region": "Europe", "ItemType": "Cosmetics", "TotalRevenue": 120.0, "Tie": False},
        {
            "Region": "North America",
            "ItemType": "Office Supplies",
            "TotalRevenue": 90.0,
            "Tie": False,
        },
    ]
    assert "after first aggregating revenue by Region and Item Type" in response.answer


def test_grouped_extrema_generates_phrase_and_stacked_breakdown_stay_separate() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Europe", "North America"],
        "ItemType": [
            "Household",
            "Cosmetics",
            "Household",
            "Cosmetics",
            "Office Supplies",
            "Office Supplies",
        ],
        "TotalRevenue": [100.0, 70.0, 50.0, 120.0, 80.0, 90.0],
    })
    profile = profile_dataset(dataframe)

    extrema = run_agent(
        "Which item type generates the highest revenue in each region?",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )
    breakdown = run_agent(
        "Show total revenue by region and item type.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert extrema.plan.tool_name == "calculate_grouped_extrema"
    assert extrema.plan.arguments == {
        "primary_group_column": "Region",
        "secondary_group_column": "ItemType",
        "metric_column": "TotalRevenue",
        "aggregation": "sum",
        "extremum": "max",
    }
    assert extrema.chart_spec
    assert extrema.chart_spec.chart_type == "grouped_extrema_bar"
    assert extrema.result
    assert extrema.result.data == [
        {
            "Region": "Asia",
            "ItemType": "Household",
            "TotalRevenue": 150.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": "Cosmetics",
            "SecondPlaceValue": 70.0,
            "AbsoluteGap": 80.0,
            "PercentageGap": 114.28571428571428,
            "WinnerShareOfGroup": 68.18181818181817,
        },
        {
            "Region": "Europe",
            "ItemType": "Cosmetics",
            "TotalRevenue": 120.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": "Office Supplies",
            "SecondPlaceValue": 80.0,
            "AbsoluteGap": 40.0,
            "PercentageGap": 50.0,
            "WinnerShareOfGroup": 60.0,
        },
        {
            "Region": "North America",
            "ItemType": "Office Supplies",
            "TotalRevenue": 90.0,
            "Rank": 1,
            "Tie": False,
            "TieCount": 1,
            "SecondPlace": None,
            "SecondPlaceValue": None,
            "AbsoluteGap": None,
            "PercentageGap": None,
            "WinnerShareOfGroup": 100.0,
        },
    ]

    assert breakdown.plan.tool_name == "group_and_aggregate"
    assert breakdown.plan.arguments["group_by"] == "Region"
    assert breakdown.plan.arguments["secondary_group_by"] == "ItemType"
    assert breakdown.chart_spec
    assert breakdown.chart_spec.chart_type == "stacked_bar"
    assert breakdown.chart_spec.x == "Region"
    assert breakdown.chart_spec.y == "TotalRevenue"
    assert breakdown.chart_spec.color == "ItemType"
    assert breakdown.result
    assert len(breakdown.result.data) == 5


def test_grouped_extrema_supports_within_each_parent_phrase() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Europe"],
        "Country": ["Japan", "India", "France", "Germany"],
        "TotalProfit": [10.0, 30.0, 40.0, 20.0],
    })

    response = run_agent(
        "Which country generates the highest profit within each region",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments == {
        "primary_group_column": "Region",
        "secondary_group_column": "Country",
        "metric_column": "TotalProfit",
        "aggregation": "sum",
        "extremum": "max",
    }
    assert response.chart_spec
    assert response.chart_spec.chart_type == "grouped_extrema_bar"
    assert response.chart_spec.x == "Region"
    assert response.chart_spec.color == "Country"
    assert response.result
    assert [(row["Region"], row["Country"], row["TotalProfit"]) for row in response.result.data] == [
        ("Asia", "India", 30.0),
        ("Europe", "France", 40.0),
    ]


def test_grouped_extrema_within_each_phrase_is_dynamic_for_other_columns() -> None:
    dataframe = pd.DataFrame({
        "Department": ["Sales", "Sales", "Support", "Support"],
        "Employee": ["Ari", "Bo", "Cy", "Dee"],
        "Score": [70.0, 95.0, 88.0, 91.0],
    })

    response = run_agent(
        "Which employee generates the highest score within each department?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "people.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments == {
        "primary_group_column": "Department",
        "secondary_group_column": "Employee",
        "metric_column": "Score",
        "aggregation": "sum",
        "extremum": "max",
    }
    assert response.result
    assert [(row["Department"], row["Employee"], row["Score"]) for row in response.result.data] == [
        ("Sales", "Bo", 95.0),
        ("Support", "Dee", 91.0),
    ]


def test_grouped_extrema_handles_plural_item_types_brings_wording() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "Item Type": ["Meat", "Fruit", "Meat"],
        "TotalRevenue": [10.0, 20.0, 30.0],
    })

    response = run_agent(
        "which item types brings the total highest revenue for each region",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments == {
        "primary_group_column": "Region",
        "secondary_group_column": "Item Type",
        "metric_column": "TotalRevenue",
        "aggregation": "sum",
        "extremum": "max",
    }
    assert response.chart_spec
    assert response.chart_spec.chart_type == "grouped_extrema_bar"
    assert response.chart_spec.x == "Region"
    assert response.chart_spec.y == "TotalRevenue"
    assert response.chart_spec.color == "Item Type"
    assert response.result
    assert response.result.data[0]["Region"] == "Asia"
    assert response.result.data[0]["Item Type"] == "Fruit"
    assert response.result.data[0]["TotalRevenue"] == 20.0
    assert response.result.data[1]["Region"] == "Europe"
    assert response.result.data[1]["Item Type"] == "Meat"
    assert response.result.data[1]["TotalRevenue"] == 30.0


def test_grouped_extrema_reprofiles_when_chat_profile_is_stale() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "ItemType": ["Meat", "Fruit", "Meat"],
        "TotalRevenue": [10.0, 20.0, 30.0],
    })
    stale_profile = profile_dataset(pd.DataFrame({
        "Category": ["A", "B"],
        "Sales": [1.0, 2.0],
    }))

    response = run_agent(
        "which item types brings the total highest revenue for each region",
        dataframe,
        stale_profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments == {
        "primary_group_column": "Region",
        "secondary_group_column": "ItemType",
        "metric_column": "TotalRevenue",
        "aggregation": "sum",
        "extremum": "max",
    }
    assert response.result
    assert response.result.data[0]["Region"] == "Asia"
    assert response.result.data[0]["ItemType"] == "Fruit"


def test_grouped_extrema_handles_multiple_country_values() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Canada", "Canada", "USA", "USA", "Mexico"],
        "ItemType": ["Meat", "Fruit", "Meat", "Fruit", "Meat"],
        "TotalRevenue": [10.0, 30.0, 50.0, 20.0, 999.0],
    })

    response = run_agent(
        "which item type brings the highest revenue for country in canada and usa",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments == {
        "primary_group_column": "Country",
        "secondary_group_column": "ItemType",
        "metric_column": "TotalRevenue",
        "aggregation": "sum",
        "extremum": "max",
        "filter_column": "Country",
        "filter_values": ["Canada", "USA"],
    }
    assert response.result
    assert response.result.data[0]["Country"] == "Canada"
    assert response.result.data[0]["ItemType"] == "Fruit"
    assert response.result.data[1]["Country"] == "USA"
    assert response.result.data[1]["ItemType"] == "Meat"
    assert {row["Country"] for row in response.chart_data} == {"Canada", "USA"}
    assert "Mexico" not in response.answer


def test_grouped_extrema_handles_multiple_country_values_without_in_word() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Canada", "Canada", "USA", "USA", "Mexico"],
        "ItemType": ["Meat", "Fruit", "Meat", "Fruit", "Meat"],
        "TotalRevenue": [10.0, 30.0, 50.0, 20.0, 999.0],
    })

    response = run_agent(
        "which item type brings the highest revenue for country canada and usa",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments["filter_values"] == ["Canada", "USA"]
    assert response.result
    assert {row["Country"] for row in response.result.data} == {"Canada", "USA"}


def test_grouped_extrema_matches_country_value_acronym() -> None:
    dataframe = pd.DataFrame({
        "Country": [
            "Canada",
            "Canada",
            "United States of America",
            "United States of America",
            "Mexico",
        ],
        "ItemType": ["Meat", "Fruit", "Meat", "Fruit", "Meat"],
        "TotalRevenue": [10.0, 30.0, 50.0, 20.0, 999.0],
    })

    response = run_agent(
        "which item type brings the highest revenue for country canada and usa",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments["filter_values"] == [
        "Canada",
        "United States of America",
    ]
    assert response.result
    assert {row["Country"] for row in response.result.data} == {
        "Canada",
        "United States of America",
    }
    assert response.result.data[1]["ItemType"] == "Meat"


def test_grouped_extrema_does_not_match_hidden_acronym_inside_words() -> None:
    dataframe = pd.DataFrame({
        "Country": [
            "Canada",
            "Russia",
            "Spain",
            "Sao Tome and Principe",
            "Canada",
            "Russia",
            "Spain",
            "Sao Tome and Principe",
        ],
        "ItemType": [
            "Vegetables",
            "Cosmetics",
            "Household",
            "Cosmetics",
            "Beverages",
            "Baby Food",
            "Vegetables",
            "Cereal",
        ],
        "TotalProfit": [
            458_600.0,
            705_200.0,
            738_300.0,
            1_000_000.0,
            35_266.0,
            291_510.0,
            637_360.0,
            860_651.0,
        ],
    })

    response = run_agent(
        "which item type brings highest profit for country in canada, russia and spain",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_grouped_extrema"
    assert response.plan.arguments["filter_values"] == ["Canada", "Russia", "Spain"]
    assert response.result
    assert {row["Country"] for row in response.result.data} == {
        "Canada",
        "Russia",
        "Spain",
    }
    assert "Sao Tome and Principe" not in response.answer


def test_ranking_synonyms_route_to_grouped_aggregate_not_scalar_max() -> None:
    dataframe = pd.DataFrame({
        "ItemType": ["Meat", "Fruit", "Meat", "Fruit"],
        "TotalRevenue": [10.0, 20.0, 30.0, 5.0],
    })

    response = run_agent(
        "which item type brings the maximum total revenue",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )
    scalar_response = run_agent(
        "what is the maximum revenue",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "ItemType",
        "secondary_group_by": None,
        "value_column": "TotalRevenue",
        "value_columns": None,
        "aggregation": "sum",
        "limit": 1,
        "filter_column": None,
        "filter_value": None,
        "sort_descending": True,
    }
    assert response.result
    assert response.result.data == [{"ItemType": "Meat", "TotalRevenue": 40.0}]
    assert scalar_response.plan.tool_name == "calculate_scalar_aggregate"
    assert scalar_response.plan.arguments == {
        "value_column": "TotalRevenue",
        "aggregation": "max",
    }


@pytest.mark.parametrize("ship_mode_column", ["Ship Mode", "ShipMode"])
def test_generated_highest_sales_routes_to_grouped_sales_ranking(ship_mode_column: str) -> None:
    dataframe = pd.DataFrame({
        ship_mode_column: ["Standard Class", "Second Class", "Standard Class", "First Class"],
        "Ship Date": ["2017-01-01", "2017-02-01", "2017-03-01", "2017-04-01"],
        "Sales": [100.0, 200.0, 50.0, 75.0],
        "Profit": [10.0, 20.0, 5.0, 7.0],
    })

    response = run_agent(
        "Which ship mode generated the total highest sales",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["group_by"] == ship_mode_column
    assert response.plan.arguments["group_by"] != "Ship Date"
    assert response.plan.arguments["value_column"] == "Sales"
    assert response.plan.arguments["aggregation"] == "sum"
    assert response.plan.arguments["limit"] == 1
    assert response.plan.arguments["sort_descending"] is True
    assert response.result
    assert response.result.data == [{ship_mode_column: "Second Class", "Sales": 200.0}]
    assert "Second Class" in response.answer


def test_lowest_filtered_ranking_keeps_ascending_direction_and_answer_wording() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Europe", "Europe", "Asia"],
        "ItemType": ["Baby Food", "Office Supplies", "Meat", "Meat"],
        "TotalRevenue": [32_319_000.0, 89_300_000.0, 12_000_000.0, 120_000_000.0],
    })

    response = run_agent(
        "which item types brings the lowest total revenue in region europe",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "ItemType",
        "secondary_group_by": None,
        "value_column": "TotalRevenue",
        "value_columns": None,
        "aggregation": "sum",
        "limit": 1,
        "filter_column": "Region",
        "filter_value": "Europe",
        "sort_descending": False,
    }
    assert response.result
    assert response.result.data == [{"ItemType": "Meat", "TotalRevenue": 12_000_000.0}]
    assert response.chart_data == [{"ItemType": "Meat", "TotalRevenue": 12_000_000.0}]
    assert "lowest sum Total Revenue" in response.answer
    assert "highest sum TotalRevenue" not in response.answer


def test_lowest_profit_subcategory_uses_subcategory_not_parent_category() -> None:
    dataframe = pd.DataFrame({
        "Category": ["Furniture", "Furniture", "Furniture", "Technology"],
        "Sub-Category": ["Bookcases", "Tables", "Chairs", "Copiers"],
        "Profit": [-4_000.0, -18_000.0, 26_000.0, 55_000.0],
    })

    response = run_agent(
        "which sub-category has the lowest profit?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "superstore.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "Sub-Category",
        "secondary_group_by": None,
        "value_column": "Profit",
        "value_columns": None,
        "aggregation": "sum",
        "limit": 1,
        "filter_column": None,
        "filter_value": None,
        "sort_descending": False,
    }
    assert response.result
    assert response.result.data == [{"Sub-Category": "Tables", "Profit": -18_000.0}]
    assert response.chart_spec
    assert response.chart_spec.chart_type == "bar"
    assert response.chart_spec.color is None
    assert "Tables" in response.answer
    assert "Bookcases" not in response.answer


def test_lowest_word_does_not_match_west_region_from_previous_context() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "West", "West", "East"],
        "Sub-Category": ["Bookcases", "Tables", "Copiers", "Tables"],
        "Profit": [-1_600.0, 2_000.0, 100.0, -17_700.0],
    })
    profile = profile_dataset(dataframe)
    first = run_agent(
        "show profit by sub-category in west region",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "superstore.csv",
        ollama_online=False,
    )

    response = run_agent(
        "which sub-category has lowest profit",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "superstore.csv",
        history=[first.model_dump(mode="json")],
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["filter_column"] is None
    assert response.plan.arguments["filter_value"] is None
    assert response.result
    assert response.result.data == [{"Sub-Category": "Tables", "Profit": -15_700.0}]
    assert "Tables" in response.answer
    assert "Region = West" not in response.answer


def test_lowest_profit_ranking_ignores_llm_answer_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({
        "Category": ["Furniture", "Furniture", "Furniture", "Technology"],
        "Sub-Category": ["Bookcases", "Tables", "Chairs", "Copiers"],
        "Profit": [-4_000.0, -18_000.0, 26_000.0, 55_000.0],
    })

    def wrong_explanation(*_args, **_kwargs):
        return "**Bookcases** has the lowest profit.", 0.01

    monkeypatch.setattr("agent.data_agent.explain_with_ollama", wrong_explanation)

    response = run_agent(
        "which sub-category has the lowest profit?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "superstore.csv",
        ollama_online=True,
    )

    assert response.result
    assert response.result.data == [{"Sub-Category": "Tables", "Profit": -18_000.0}]
    assert "Tables" in response.answer
    assert "Bookcases" not in response.answer


def test_grouped_chart_answer_uses_verified_result_not_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({
        "Sub-Category": ["Bookcases", "Tables", "Copiers"],
        "Profit": [-3_500.0, -17_700.0, 55_000.0],
    })

    def wrong_explanation(*_args, **_kwargs):
        return "Bookcases has the lowest profit.", 0.01

    monkeypatch.setattr("agent.data_agent.explain_with_ollama", wrong_explanation)

    response = run_agent(
        "show chart of profit by sub-category",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "superstore.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.result
    assert response.result.data[-1] == {"Sub-Category": "Tables", "Profit": -17_700.0}
    assert "Bookcases has the lowest profit" not in response.answer


def test_singular_units_sold_wording_maps_to_dataset_column() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "UnitsSold": [10, 20, 5],
    })
    profile = profile_dataset(dataframe)

    response = run_agent(
        "UnitSold by region",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["group_by"] == "Region"
    assert response.plan.arguments["value_column"] == "UnitsSold"
    assert response.result
    assert response.result.data[0] == {"Region": "Europe", "UnitsSold": 20}
    assert response.chart_spec is not None


@pytest.mark.parametrize(
    ("question", "column", "semantic_type", "chart_type"),
    [
        ("Explain the Region column.", "Region", "categorical", "bar"),
        ("Tell me about TotalRevenue.", "TotalRevenue", "numerical", "histogram"),
        ("What type of column is OrderPriority?", "OrderPriority", "categorical", "bar"),
        ("Describe the Date column.", "Date", "datetime", "bar"),
        ("What does OrderID mean?", "OrderID", "identifier", None),
    ],
)
def test_column_profile_questions_route_to_type_aware_profile(
    question: str,
    column: str,
    semantic_type: str,
    chart_type: str | None,
) -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia", None],
        "OrderPriority": ["H", "L", "M", "H"],
        "OrderID": [1, 2, 2, 4],
        "TotalRevenue": [100.0, 200.0, 150.0, 0.0],
        "TotalCost": [60.0, 120.0, 90.0, 0.0],
        "TotalProfit": [40.0, 80.0, 60.0, 0.0],
        "UnitsSold": [10, 20, 15, 0],
        "UnitPrice": [10.0, 10.0, 10.0, 10.0],
        "Date": ["2021-01-01", "2021-02-01", "bad", "2022-01-01"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "profile_column"
    assert response.plan.arguments["column_name"] == column
    assert response.result
    assert response.result.data["profile"]["semantic_type"] == semantic_type
    assert response.result.data["table_rows"]
    if chart_type is None:
        assert response.chart_spec is None
    else:
        assert response.chart_spec
        assert response.chart_spec.chart_type == chart_type


def test_column_profile_humanized_and_case_insensitive_names_resolve() -> None:
    dataframe = pd.DataFrame({
        "TotalRevenue": [100.0, 200.0],
        "SalesChannel": ["Online", "Offline"],
    })

    humanized = run_agent(
        "profile total revenue",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )
    compact = run_agent(
        "What values does saleschannel contain?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert humanized.plan.tool_name == "profile_column"
    assert humanized.plan.arguments["column_name"] == "TotalRevenue"
    assert compact.plan.tool_name == "profile_column"
    assert compact.plan.arguments["column_name"] == "SalesChannel"


def test_unknown_column_profile_request_returns_suggestions_without_guessing() -> None:
    dataframe = pd.DataFrame({
        "TotalRevenue": [100.0, 200.0],
        "TotalProfit": [10.0, 20.0],
    })

    response = run_agent(
        "Explain the Warehouse Zone column.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == ""
    assert response.plan.clarification
    assert "could not find a column" in response.answer.lower()
    assert "TotalRevenue" in response.answer


@pytest.mark.parametrize(
    "question",
    [
        "How many units were sold in total",
        "total unitssold",
        "total units sold",
    ],
)
def test_total_units_sold_natural_phrasings_return_scalar_sum(question: str) -> None:
    dataframe = pd.DataFrame({
        "UnitsSold": [10, 20, 30],
        "TotalRevenue": [100.0, 200.0, 300.0],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_scalar_aggregate"
    assert response.plan.arguments == {
        "value_column": "UnitsSold",
        "aggregation": "sum",
    }
    assert response.result
    assert response.result.data["value"] == 60.0
    assert "total units sold" in response.answer.lower()


@pytest.mark.parametrize(
    "question",
    [
        "Give me a summary of UnitSold by region",
        "Find me unitsold by region from dataset",
    ],
)
def test_grouped_units_sold_phrasings_run_current_analysis(question: str) -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "UnitsSold": [10, 20, 5],
        "UnitPrice": [1.0, 2.0, 3.0],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["group_by"] == "Region"
    assert response.plan.arguments["value_column"] == "UnitsSold"
    assert response.result
    assert response.result.data == [
        {"Region": "Europe", "UnitsSold": 20},
        {"Region": "Asia", "UnitsSold": 15},
    ]


def test_categorical_count_ranking_with_filter_uses_record_count() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Turkey", "Turkey", "Turkey", "Canada"],
        "SalesChannel": ["Online", "Offline", "Online", "Online"],
    })

    response = run_agent(
        "which saleschannel has higher count for country turkey",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "SalesChannel",
        "secondary_group_by": None,
        "value_column": "Count",
        "value_columns": None,
        "aggregation": "count",
        "limit": 1,
        "filter_column": "Country",
        "filter_value": "Turkey",
        "sort_descending": True,
    }
    assert response.result
    assert response.result.data == [{"SalesChannel": "Online", "Count": 2}]
    assert response.chart_spec
    assert response.chart_spec.y == "Count"


def test_rank_order_priorities_by_total_revenue_does_not_add_sales_channel() -> None:
    dataframe = pd.DataFrame({
        "OrderPriority": ["H", "L", "H", "M"],
        "SalesChannel": ["Online", "Online", "Offline", "Offline"],
        "TotalRevenue": [100.0, 200.0, 300.0, 400.0],
    })

    response = run_agent(
        "Rank order priorities by total revenue.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "OrderPriority",
        "secondary_group_by": None,
        "value_column": "TotalRevenue",
        "value_columns": None,
        "aggregation": "sum",
        "limit": None,
        "filter_column": None,
        "filter_value": None,
    }
    assert response.chart_spec
    assert response.chart_spec.chart_type == "bar"
    assert response.chart_spec.color is None
    assert response.result
    assert response.result.data == [
        {"OrderPriority": "H", "TotalRevenue": 400.0},
        {"OrderPriority": "M", "TotalRevenue": 400.0},
        {"OrderPriority": "L", "TotalRevenue": 200.0},
    ]


def test_percentage_of_total_revenue_by_region_includes_share_column() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "TotalRevenue": [100.0, 300.0, 600.0],
    })

    response = run_agent(
        "What percentage of total revenue comes from each region?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["group_by"] == "Region"
    assert response.plan.arguments["value_column"] == "TotalRevenue"
    assert response.plan.arguments["include_percentage"] is True
    assert response.result
    assert response.result.data == [
        {"Region": "Europe", "TotalRevenue": 600.0, "PercentageOfTotal": 60.0},
        {"Region": "Asia", "TotalRevenue": 400.0, "PercentageOfTotal": 40.0},
    ]
    assert response.chart_spec
    assert response.chart_spec.y == "PercentageOfTotal"
    assert "60.0% of total" in response.answer


def test_parent_scope_share_uses_filtered_denominator_and_focus_value() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Asia"],
        "Country": ["Thailand", "India", "France", "Thailand"],
        "TotalRevenue": [100.0, 300.0, 600.0, 100.0],
    })

    response = run_agent(
        "What percentage of Asia's revenue comes from Thailand?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["filter_column"] == "Region"
    assert response.plan.arguments["filter_value"] == "Asia"
    assert response.plan.arguments["group_by"] == "Country"
    assert response.plan.arguments["focus_value"] == "Thailand"
    assert response.result
    thailand = next(row for row in response.result.data if row["Country"] == "Thailand")
    assert thailand["PercentageOfTotal"] == pytest.approx(40.0)
    assert "Thailand" in response.answer


def test_units_sold_percentage_distribution_by_item_type_uses_numeric_share() -> None:
    dataframe = pd.DataFrame({
        "ItemType": ["Meat", "Fruit", "Fruit"],
        "UnitsSold": [50, 25, 25],
    })

    response = run_agent(
        "Show the percentage distribution of units sold by item type.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["value_column"] == "UnitsSold"
    assert response.plan.arguments["group_by"] == "ItemType"
    assert response.plan.arguments["include_percentage"] is True
    assert response.result
    assert sum(row["PercentageOfTotal"] for row in response.result.data) == pytest.approx(100.0)


def test_monthly_revenue_change_routes_to_period_over_period() -> None:
    dataframe = pd.DataFrame({
        "Date": ["2020-12-01", "2021-01-01", "2021-02-01"],
        "Region": ["Asia", "Asia", "Europe"],
        "TotalRevenue": [50.0, 100.0, 200.0],
    })

    response = run_agent(
        "Show monthly revenue change compared with the previous month.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_period_over_period"
    assert response.plan.arguments["frequency"] == "month"
    assert response.chart_spec
    assert response.chart_spec.y == "PercentageChange"
    assert response.result
    assert response.result.data[1]["PreviousPeriodValue"] == 50.0


def test_unknown_filter_value_does_not_broaden_to_global_metric() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe"],
        "Country": ["Thailand", "France"],
        "TotalRevenue": [100.0, 200.0],
    })

    response = run_agent(
        "Show revenue for Atlantis.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == ""
    assert "No calculation was performed" in response.answer


def test_error_and_ambiguity_questions_do_not_silently_fallback() -> None:
    dataframe = pd.DataFrame({
        "Order ID": ["A", "B", "C", "D", "E", "F"],
        "Order Date": ["2017-01-01", "2017-02-01", "2017-03-01", "2017-04-01", "2017-05-01", "2017-06-01"],
        "Ship Date": ["2017-01-03", "2017-02-03", "2017-03-03", "2017-04-03", "2017-05-03", "2017-06-03"],
        "Ship Mode": ["Standard Class", "Second Class", "Standard Class", "First Class", "Same Day", "Standard Class"],
        "Region": ["South", "West", "East", "Central", "West", "West"],
        "State": ["Georgia", "Washington", "California", "New York", "Washington", "Illinois"],
        "City": ["Atlanta", "Seattle", "Springfield", "Springfield", "Olympia", "Springfield"],
        "Category": ["Furniture", "Office Supplies", "Technology", "Office Supplies", "Furniture", "Technology"],
        "Product Name": ["Chair", "Paper", "Phone", "Binder", "Table", "Laptop"],
        "Sales": [100.0, 200.0, 50.0, 75.0, 80.0, 300.0],
        "Profit": [10.0, 20.0, 5.0, 7.0, 8.0, 30.0],
    })
    profile = profile_dataset(dataframe)

    no_calc_questions = [
        ("Show sales for Atlantis.", "atlantis"),
        ("Show profit for a region called Northern Europe.", "northern europe"),
        ("Show value counts for Delivery Status.", "delivery status"),
        ("Show monthly revenue using Invoice Date.", "invoice date"),
        ("Show the average of Order ID.", "identifier column"),
        ("Show monthly sales without using a valid date column.", "valid date column"),
        ("Show the share of average Product Name.", "not a numeric measure"),
    ]
    for question, expected_text in no_calc_questions:
        response = run_agent(
            question,
            dataframe,
            profile,
            get_settings(),
            "qwen2.5:1.5b",
            "orders.csv",
            ollama_online=False,
        )
        assert response.plan.tool_name == ""
        assert expected_text in response.answer.lower()
        assert response.result is None

    georgia = run_agent(
        "Show sales for Georgia.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )
    washington = run_agent(
        "Show profit for Washington.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )
    springfield = run_agent(
        "Show sales for Springfield.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert georgia.plan.tool_name == "calculate_filtered_aggregate"
    assert georgia.plan.arguments["category_column"] == "State"
    assert georgia.plan.arguments["category_value"] == "Georgia"
    assert washington.plan.arguments["category_column"] == "State"
    assert washington.plan.arguments["category_value"] == "Washington"
    assert springfield.plan.arguments["category_column"] == "City"
    assert springfield.plan.arguments["category_value"] == "Springfield"


def test_date_ambiguity_requests_clarification_and_explicit_date_executes() -> None:
    dataframe = pd.DataFrame({
        "Order Date": ["2017-01-01", "2017-02-01", "2018-01-01"],
        "Ship Date": ["2017-01-05", "2017-02-05", "2018-01-05"],
        "Profit": [10.0, 20.0, 30.0],
    })
    profile = profile_dataset(dataframe)

    ambiguous = run_agent(
        "Show monthly profit for 2017.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )
    explicit = run_agent(
        "Using Order Date, show monthly Profit for 2017.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )
    selected = run_agent(
        f"Using Ship Date, {ambiguous.plan.arguments['original_query']}",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert ambiguous.plan.tool_name == ""
    assert ambiguous.plan.arguments["clarification_type"] == "ambiguous_date_column"
    assert ambiguous.plan.arguments["options"] == ["Order Date", "Ship Date"]
    assert ambiguous.result is None
    assert explicit.plan.tool_name == "calculate_time_trend"
    assert explicit.plan.arguments["date_column"] == "Order Date"
    assert selected.plan.tool_name == "calculate_time_trend"
    assert selected.plan.arguments["date_column"] == "Ship Date"


def test_ambiguous_categorical_value_requests_clarification() -> None:
    dataframe = pd.DataFrame({
        "State": ["New York", "California"],
        "City": ["New York", "Los Angeles"],
        "Sales": [100.0, 200.0],
    })

    response = run_agent(
        "Show sales for New York.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == ""
    assert response.plan.arguments["clarification_type"] == "ambiguous_filter_value"
    assert response.plan.arguments["options"] == ["State = New York", "City = New York"]


@pytest.mark.parametrize(
    "question",
    [
        "Show value counts of Ship Mode.",
        "Show the frequency of Ship Mode.",
        "Show the frequency count of Ship Mode.",
        "Show Ship Mode distribution.",
        "Show Ship Mode counts.",
    ],
)
def test_value_count_synonyms_resolve_to_categorical_counts(question: str) -> None:
    dataframe = pd.DataFrame({
        "Ship Mode": ["Standard Class", "Second Class", "Standard Class"],
        "Region": ["West", "West", "East"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "Ship Mode"
    assert response.plan.arguments["measure_type"] == "row_count"


def test_time_series_followups_preserve_and_update_previous_request() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "Country": ["Thailand", "India", "France"],
        "Date": ["2021-01-01", "2022-01-01", "2021-01-01"],
        "TotalRevenue": [1.0, 2.0, 3.0],
        "TotalProfit": [4.0, 5.0, 6.0],
    })
    profile = profile_dataset(dataframe)
    history = []

    for question in [
        "Show yearly revenue for Asia.",
        "Now show only Thailand.",
        "Now make it monthly.",
        "Show profit instead.",
        "Now use 2021 only.",
    ]:
        response = run_agent(
            question,
            dataframe,
            profile,
            get_settings(),
            "qwen2.5:1.5b",
            "sales.csv",
            history=history,
            ollama_online=False,
        )
        history.append(response.model_dump())

    final_response = run_agent(
        "Compare revenue and profit.",
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        history=history,
        ollama_online=False,
    )

    assert final_response.plan.tool_name == "calculate_time_trend"
    assert final_response.plan.arguments["filter_column"] == "Country"
    assert final_response.plan.arguments["filter_value"] == "Thailand"
    assert final_response.plan.arguments["frequency"] == "month"
    assert final_response.plan.arguments["start_date"] == pd.Timestamp("2021-01-01")
    assert final_response.plan.arguments["end_date"] == pd.Timestamp("2021-12-31")
    assert final_response.plan.arguments["value_columns"] == ["TotalRevenue", "TotalProfit"]


def test_value_count_for_each_region_routes_to_categorical_counts() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Europe", "Europe"],
        "OrderPriority": ["High", "Low", "High", "Critical", "High"],
    })

    response = run_agent(
        "Show me the value count of OrderPriority for each Region.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "OrderPriority"
    assert response.plan.arguments["primary_group_column"] == "Region"
    assert response.plan.arguments["normalization"] == "within_primary_group"
    assert response.chart_spec
    assert response.chart_spec.chart_type == "grouped_bar"
    assert response.chart_spec.x == "Region"
    assert response.chart_spec.y == "Count"
    assert response.chart_spec.color == "OrderPriority"
    assert response.result
    assert response.result.data["table_rows"] == [
        {"Region": "Asia", "OrderPriority": "High", "Count": 1, "Percentage": 50.0},
        {"Region": "Asia", "OrderPriority": "Low", "Count": 1, "Percentage": 50.0},
        {"Region": "Europe", "OrderPriority": "High", "Count": 2, "Percentage": pytest.approx(66.66666666666666)},
        {"Region": "Europe", "OrderPriority": "Critical", "Count": 1, "Percentage": pytest.approx(33.33333333333333)},
    ]
    assert response.chart_data == response.result.data["chart_rows"]


def test_order_priority_counts_by_region_bar_chart_request() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Europe"],
        "OrderPriority": ["High", "Low", "High", "Critical"],
    })

    response = run_agent(
        "Show a bar chart of OrderPriority counts by Region.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "OrderPriority"
    assert response.plan.arguments["primary_group_column"] == "Region"
    assert response.chart_spec
    assert response.chart_spec.chart_type == "grouped_bar"
    assert response.chart_spec.x == "Region"
    assert response.chart_spec.y == "Count"
    assert response.chart_spec.color == "OrderPriority"


def test_frequency_distribution_with_two_columns_uses_first_as_group_and_second_as_counted() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Europe"],
        "OrderPriority": ["High", "Low", "High", "Critical"],
    })

    response = run_agent(
        "Use Region and OrderPriority to show the frequency distribution.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "OrderPriority"
    assert response.plan.arguments["primary_group_column"] == "Region"
    assert response.chart_spec
    assert response.chart_spec.chart_type == "grouped_bar"
    assert response.chart_spec.x == "Region"
    assert response.chart_spec.color == "OrderPriority"


def test_listed_category_values_resolve_counted_column_for_each_group() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Asia", "Europe", "Europe", "Europe"],
        "OrderPriority": ["High", "Low", "Medium", "High", "Critical", "High"],
    })

    response = run_agent(
        "How many High, Medium, Low, and Critical priority orders are there in each Region?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "OrderPriority"
    assert response.plan.arguments["primary_group_column"] == "Region"
    assert response.plan.arguments["filters"] == []
    assert response.result
    assert response.result.data["table_rows"] == [
        {"Region": "Asia", "OrderPriority": "High", "Count": 1, "Percentage": pytest.approx(33.33333333333333)},
        {"Region": "Asia", "OrderPriority": "Low", "Count": 1, "Percentage": pytest.approx(33.33333333333333)},
        {"Region": "Asia", "OrderPriority": "Medium", "Count": 1, "Percentage": pytest.approx(33.33333333333333)},
        {"Region": "Europe", "OrderPriority": "High", "Count": 2, "Percentage": pytest.approx(66.66666666666666)},
        {"Region": "Europe", "OrderPriority": "Critical", "Count": 1, "Percentage": pytest.approx(33.33333333333333)},
    ]


def test_filtered_value_counts_by_compact_group_column() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Europe", "Europe", "Europe"],
        "OrderPriority": ["High", "High", "Critical", "Medium", "High"],
        "SalesChannel": ["Online", "Online", "Online", "Offline", "Offline"],
    })

    response = run_agent(
        "For Region Europe, show OrderPriority counts by SalesChannel.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "OrderPriority"
    assert response.plan.arguments["primary_group_column"] == "SalesChannel"
    assert response.plan.arguments["filters"] == [
        {"column": "Region", "operator": "equals", "value": "Europe"}
    ]
    assert response.chart_spec
    assert response.chart_spec.x == "SalesChannel"
    assert response.chart_spec.color == "OrderPriority"


def test_filtered_categorical_counts_do_not_ignore_region_filter() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Asia"],
        "SalesChannel": ["Online", "Offline", "Online", "Online"],
    })

    response = run_agent(
        "In region Asia, show SalesChannel counts.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "SalesChannel"
    assert response.plan.arguments["primary_group_column"] is None
    assert response.plan.arguments["filters"] == [
        {"column": "Region", "operator": "equals", "value": "Asia"}
    ]
    assert response.result
    assert response.result.data["table_rows"] == [
        {"SalesChannel": "Online", "Count": 2, "Percentage": pytest.approx(66.66666666666666)},
        {"SalesChannel": "Offline", "Count": 1, "Percentage": pytest.approx(33.33333333333333)},
    ]


def test_count_the_category_values_for_each_group_routes_to_categorical_counts() -> None:
    dataframe = pd.DataFrame({
        "ItemType": ["Meat", "Meat", "Fruit", "Fruit", "Meat"],
        "SalesChannel": ["Online", "Offline", "Online", "Online", "Offline"],
    })

    response = run_agent(
        "For each ItemType, count the SalesChannel values.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "SalesChannel"
    assert response.plan.arguments["primary_group_column"] == "ItemType"
    assert response.chart_spec
    assert response.chart_spec.chart_type == "grouped_bar"
    assert response.chart_spec.x == "ItemType"
    assert response.chart_spec.color == "SalesChannel"


def test_filtered_value_counts_for_country_spain_do_not_fall_back() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Spain", "Spain", "France", "Spain"],
        "OrderPriority": ["High", "Critical", "High", "High"],
    })

    response = run_agent(
        "For country Spain, what are the value counts of OrderPriority?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["counted_column"] == "OrderPriority"
    assert response.plan.arguments["primary_group_column"] is None
    assert response.plan.arguments["filters"] == [
        {"column": "Country", "operator": "equals", "value": "Spain"}
    ]
    assert response.result
    assert response.result.data["total_matching_rows"] == 3
    assert response.result.data["table_rows"] == [
        {"OrderPriority": "High", "Count": 2, "Percentage": pytest.approx(66.66666666666666)},
        {"OrderPriority": "Critical", "Count": 1, "Percentage": pytest.approx(33.33333333333333)},
    ]
    assert "France" not in str(response.result.data["table_rows"])


def test_unique_order_count_by_priority_for_spain() -> None:
    dataframe = pd.DataFrame({
        "Country": ["Spain", "Spain", "Spain", "France"],
        "OrderPriority": ["High", "High", "Low", "High"],
        "OrderID": [1, 1, 2, 3],
    })

    response = run_agent(
        "For country Spain, show unique OrderID count by OrderPriority.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["measure_type"] == "distinct_count"
    assert response.plan.arguments["distinct_column"] == "OrderID"
    assert response.plan.arguments["counted_column"] == "OrderPriority"
    assert response.result
    assert response.result.data["table_rows"] == [
        {"OrderPriority": "High", "Unique OrderID Count": 1, "Percentage": 50.0},
        {"OrderPriority": "Low", "Unique OrderID Count": 1, "Percentage": 50.0},
    ]


@pytest.mark.parametrize(
    ("question", "distinct_column", "group_column"),
    [
        ("Count distinct order id for each state", "Order ID", "State"),
        ("count unique Product ID by Region", "Product ID", "Region"),
    ],
)
def test_distinct_column_count_for_each_group_is_dynamic(
    question: str,
    distinct_column: str,
    group_column: str,
) -> None:
    dataframe = pd.DataFrame({
        distinct_column: ["A", "A", "B", "C"],
        group_column: ["West", "West", "West", "East"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["measure_type"] == "distinct_count"
    assert response.plan.arguments["distinct_column"] == distinct_column
    assert response.plan.arguments["counted_column"] == group_column
    assert response.result
    assert response.result.data["table_rows"] == [
        {group_column: "West", f"Unique {distinct_column} Count": 2, "Percentage": pytest.approx(66.66666666666666)},
        {group_column: "East", f"Unique {distinct_column} Count": 1, "Percentage": pytest.approx(33.33333333333333)},
    ]


def test_countries_below_average_regional_sales_uses_two_step_benchmark() -> None:
    dataframe = pd.DataFrame({
        "Region": ["East", "East", "East", "West", "West"],
        "Country": ["A", "A", "B", "C", "D"],
        "TotalSales": [40.0, 60.0, 300.0, 50.0, 150.0],
    })

    response = run_agent(
        "Find those countries which are below from average regional total sales",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "compare_grouped_to_benchmark"
    assert response.plan.arguments == {
        "category_column": "Country",
        "value_column": "TotalSales",
        "aggregation": "sum",
        "benchmark": "mean",
        "comparison": "below",
        "benchmark_group_by": "Region",
    }
    assert response.chart_spec is None
    assert response.result
    assert [row["Country"] for row in response.result.data] == ["A", "C"]
    assert "**A** in East" in response.answer
    assert "$100 below" in response.answer


def test_global_grouped_benchmark_works_for_unrelated_dataset() -> None:
    dataframe = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Units": [10, 30, 20],
    })

    response = run_agent(
        "Which products are above the average total Units?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "inventory.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "compare_grouped_to_benchmark"
    assert response.plan.arguments["category_column"] == "Product"
    assert response.plan.arguments["benchmark_group_by"] is None
    assert response.result
    assert response.result.data[0]["Product"] == "B"


def test_short_metric_reply_completes_previous_clarification() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "UnitsSold": [10, 20, 5],
        "UnitPrice": [1.0, 2.0, 3.0],
    })
    history = [{
        "question": "Find a metric by region",
        "answer": "Which metric should I use?",
        "plan": {
            "tool_name": "",
            "arguments": {},
            "chart_spec": None,
            "clarification": "Which metric should I use?",
        },
    }]

    plan = deterministic_plan(
        "UnitsSold",
        profile_dataset(dataframe),
        history,
        dataframe,
    )

    assert plan.tool_name == "group_and_aggregate"
    assert plan.arguments["group_by"] == "Region"
    assert plan.arguments["value_column"] == "UnitsSold"


@pytest.mark.parametrize(
    "question",
    [
        "For which Regions Offline channel is available?",
        "For which Regions Offline SalesChannel is available?",
    ],
)
def test_offline_channel_regions_return_text_only(question: str) -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Africa", "Asia"],
        "SalesChannel": ["Offline", "Online", "Offline", "Online"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "list_distinct_values"
    assert response.plan.arguments["target_column"] == "Region"
    assert response.plan.arguments["filter_column"] == "SalesChannel"
    assert response.plan.arguments["filter_value"] == "Offline"
    assert response.plan.response_mode == "text"
    assert response.chart_spec is None
    assert "Africa, Asia" in response.answer


def test_how_many_regions_returns_distinct_count_text_only() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Africa", "Asia"],
        "SalesChannel": ["Offline", "Online", "Offline", "Online"],
    })

    response = run_agent(
        "How many Regions are there?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "count_distinct_values"
    assert response.plan.response_mode == "text"
    assert response.result
    assert response.result.data["count"] == 3
    assert response.answer == "There are **3** distinct Region values."


@pytest.mark.parametrize(
    "question",
    [
        "How many time country name Germany appear?",
        "How many times does Germany appear in the Country column?",
    ],
)
def test_how_many_times_category_value_appears_counts_matching_rows(question: str) -> None:
    dataframe = pd.DataFrame({
        "Country": ["Germany", "Spain", "Germany", "France"],
        "Region": ["Europe", "Europe", "Europe", "Europe"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.response_mode == "text"
    assert response.plan.arguments["counted_column"] == "Country"
    assert response.plan.arguments["filters"] == [
        {"column": "Country", "operator": "equals", "value": "Germany"}
    ]
    assert response.result
    assert response.result.data["total_matching_rows"] == 2
    assert response.result.data["table_rows"] == [
        {"Country": "Germany", "Count": 2, "Percentage": 100.0}
    ]


def test_region_names_question_lists_distinct_region_values() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Africa", "Asia"],
        "SalesChannel": ["Offline", "Online", "Offline", "Online"],
    })

    response = run_agent(
        "what are the region names in region column",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "list_distinct_values"
    assert response.plan.arguments["target_column"] == "Region"
    assert response.plan.response_mode == "text"
    assert response.chart_spec is None
    assert response.result
    assert response.result.data["values"] == ["Africa", "Asia", "Europe"]
    assert response.answer == "Region values are **Africa, Asia, Europe** (3 distinct value(s))."


@pytest.mark.parametrize(
    "question",
    [
        "show me all countries in region Asia",
        "which countries are in region asia",
        "name all country for region asia",
        "name all countries under region asia",
        "give me all countries within region asia",
    ],
)
def test_filtered_distinct_values_question_lists_target_values(question: str) -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe", "Asia", "Europe"],
        "Country": ["Thailand", "India", "France", "Japan", "Spain"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "list_distinct_values"
    assert response.plan.arguments == {
        "target_column": "Country",
        "filter_column": "Region",
        "filter_value": "Asia",
    }
    assert response.plan.response_mode == "text"
    assert response.result
    assert response.result.data["values"] == ["India", "Japan", "Thailand"]
    assert "France" not in response.answer


def test_high_units_low_profit_country_question_uses_dual_axis_analysis() -> None:
    dataframe = pd.DataFrame({
        "Country": ["A", "B", "C", "D"],
        "UnitsSold": [100, 80, 40, 20],
        "TotalProfit": [10.0, 50.0, 5.0, 30.0],
    })

    response = run_agent(
        "Which countries sell many units but generate low profit?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "analyze_high_volume_low_outcome"
    assert response.plan.arguments["category_column"] == "Country"
    assert response.plan.arguments["volume_column"] == "UnitsSold"
    assert response.plan.arguments["outcome_column"] == "TotalProfit"
    assert response.chart_spec
    assert response.chart_spec.chart_type == "dual_axis"
    assert response.chart_spec.secondary_y == "TotalProfit"
    assert response.result
    assert response.result.data["candidates"][0]["Country"] == "A"
    assert "**A**" in response.answer
    assert "median" in response.answer


def test_dual_axis_request_works_with_arbitrary_dataset_columns() -> None:
    dataframe = pd.DataFrame({
        "ProductLine": ["Core", "Core", "Plus", "Plus"],
        "Visitors": [100, 120, 80, 90],
        "SupportCost": [20.0, 25.0, 50.0, 55.0],
    })

    response = run_agent(
        "Plot Visitors and SupportCost by ProductLine on a dual axis",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "products.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["group_by"] == "ProductLine"
    assert response.plan.arguments["value_columns"] == ["Visitors", "SupportCost"]
    assert response.chart_spec
    assert response.chart_spec.chart_type == "dual_axis"
    assert response.chart_spec.y == "Visitors"
    assert response.chart_spec.secondary_y == "SupportCost"
    assert response.chart_data
    assert response.chart_data[0] == {
        "ProductLine": "Core",
        "Visitors": 220,
        "SupportCost": 45.0,
    }


def test_generic_high_metric_low_metric_request_uses_dataset_schema() -> None:
    dataframe = pd.DataFrame({
        "ProductLine": ["Core", "Plus", "Basic", "Pro"],
        "Visitors": [100, 80, 40, 20],
        "ConversionRate": [0.1, 0.5, 0.05, 0.3],
    })

    response = run_agent(
        "Which ProductLine has high Visitors but low ConversionRate?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "products.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "analyze_high_volume_low_outcome"
    assert response.plan.arguments["category_column"] == "ProductLine"
    assert response.plan.arguments["volume_column"] == "Visitors"
    assert response.plan.arguments["outcome_column"] == "ConversionRate"
    assert response.result
    assert response.result.data["candidates"][0]["ProductLine"] == "Core"
    assert "Visitors:" in response.answer
    assert "Conversion Rate:" in response.answer


def test_generic_high_low_request_respects_reversed_wording() -> None:
    dataframe = pd.DataFrame({
        "Service": ["A", "B", "C", "D"],
        "SupportCost": [10.0, 40.0, 20.0, 50.0],
        "QualityScore": [90.0, 80.0, 60.0, 50.0],
    })

    response = run_agent(
        "Which Service has low SupportCost but high QualityScore?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "services.csv",
        ollama_online=True,
    )

    assert response.plan.arguments["volume_column"] == "QualityScore"
    assert response.plan.arguments["outcome_column"] == "SupportCost"
    assert response.result
    assert response.result.data["candidates"][0]["Service"] == "A"


def test_incomplete_dual_axis_request_explains_required_columns() -> None:
    dataframe = pd.DataFrame({
        "Department": ["A", "B"],
        "Headcount": [10, 20],
        "Budget": [100.0, 200.0],
    })

    response = run_agent(
        "Create a dual-axis chart",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "departments.csv",
        ollama_online=True,
    )

    assert not response.plan.tool_name
    assert "one categorical column and two numeric columns" in response.answer


def test_malformed_ollama_tool_arguments_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe"],
        "UnitsSold": [10, 20],
    })

    def malformed_plan(*_args, **_kwargs):
        return (
            AgentPlan(
                tool_name="inspect_dataset",
                arguments={"dataset_path": "/invented/path.csv"},
            ),
            0.01,
        )

    monkeypatch.setattr("agent.data_agent.plan_with_ollama", malformed_plan)
    response = run_agent(
        "perform an unsupported custom operation",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=True,
    )

    assert response.result is None
    assert response.plan.tool_name == ""
    assert "could not map" in response.answer.lower()


def test_ambiguous_question_requests_clarification() -> None:
    plan = deterministic_plan("Show performance over time", _profile())
    assert plan.clarification
    assert "Which date and metric" in plan.clarification


def test_follow_up_reuses_previous_grouping_context() -> None:
    profile = _profile()
    history = [{
        "question": "Show total Sales by Region",
        "plan": {
            "tool_name": "group_and_aggregate",
            "arguments": {"group_by": "Region", "value_column": "Sales", "aggregation": "sum"},
        },
    }]
    plan = deterministic_plan("Now show only the top three", profile, history)
    assert plan.tool_name == "group_and_aggregate"
    assert plan.arguments["group_by"] == "Region"
    assert plan.arguments["limit"] == 3


def test_semantic_revenue_ranking_with_region_filter() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Europe", "Asia"],
        "Country": ["France", "Germany", "Japan"],
        "UnitsSold": [2, 3, 4],
        "UnitPrice": [10.0, 20.0, 30.0],
        "TotalRevenue": [20.0, 60.0, 120.0],
    })
    profile = profile_dataset(dataframe)
    for question in (
        "which country has the highest sales revenue in europe region?",
        "highest total sales in country in europe",
    ):
        plan = deterministic_plan(question, profile, dataframe=dataframe)
        assert plan.tool_name == "group_and_aggregate"
        assert plan.arguments["group_by"] == "Country"
        assert plan.arguments["value_column"] == "TotalRevenue"
        assert plan.arguments["filter_column"] == "Region"
        assert plan.arguments["filter_value"] == "Europe"
        assert plan.arguments["limit"] == 1


def test_sales_performance_across_region_and_channel() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Europe", "Asia", "Asia"],
        "SalesChannel": ["Online", "Offline", "Online", "Offline"],
        "UnitsSold": [2, 3, 4, 5],
        "TotalRevenue": [20.0, 60.0, 120.0, 80.0],
    })
    profile = profile_dataset(dataframe)

    plan = deterministic_plan(
        "show me sales performance across all region for each sales channel",
        profile,
        dataframe=dataframe,
    )

    assert plan.tool_name == "group_and_aggregate"
    assert plan.arguments["group_by"] == "Region"
    assert plan.arguments["secondary_group_by"] == "SalesChannel"
    assert plan.arguments["value_column"] == "TotalRevenue"
    assert plan.chart_spec
    assert plan.chart_spec.color == "SalesChannel"
    assert plan.chart_spec.limit is None


def test_explicit_multi_metric_request_does_not_reuse_stale_context() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe"],
        "SalesChannel": ["Offline", "Online"],
        "OrderPriority": ["L", "H"],
        "TotalRevenue": [100.0, 200.0],
        "TotalProfit": [10.0, 30.0],
    })
    history = [{
        "question": "Show profit by region and channel where priority is L",
        "plan": {
            "tool_name": "group_and_aggregate",
            "arguments": {
                "group_by": "Region",
                "secondary_group_by": "SalesChannel",
                "value_column": "TotalProfit",
                "aggregation": "sum",
                "filter_column": "OrderPriority",
                "filter_value": "L",
            },
        },
    }]

    plan = deterministic_plan(
        "total sales revenue and total profit by region",
        profile_dataset(dataframe),
        history,
        dataframe,
    )

    assert plan.arguments["group_by"] == "Region"
    assert plan.arguments["secondary_group_by"] is None
    assert plan.arguments["value_columns"] == ["TotalRevenue", "TotalProfit"]
    assert plan.arguments["filter_column"] is None
    assert plan.arguments["filter_value"] is None
    assert plan.chart_spec
    assert plan.chart_spec.value_columns == ["TotalRevenue", "TotalProfit"]
    assert "TotalRevenue and TotalProfit" in plan.chart_spec.title


def test_difference_between_two_regions_is_calculated_and_explained() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Europe", "Asia", "Europe", "Asia"],
        "TotalProfit": [30.0, 10.0, 20.0, 15.0],
        "TotalRevenue": [100.0, 50.0, 80.0, 60.0],
    })
    profile = profile_dataset(dataframe)
    question = "give me the difference in profit between europe and asia"

    plan = deterministic_plan(question, profile, dataframe=dataframe)
    response = run_agent(
        question,
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=True,
    )

    assert plan.tool_name == "compare_category_values"
    assert plan.arguments["category_column"] == "Region"
    assert plan.arguments["value_column"] == "TotalProfit"
    assert plan.arguments["first_value"] == "Europe"
    assert plan.arguments["second_value"] == "Asia"
    assert plan.chart_spec
    assert plan.chart_spec.include_values == ["Europe", "Asia"]
    assert response.result
    assert response.result.data["absolute_difference"] == 25.0
    assert "Europe exceeds Asia by $25" in response.answer


def test_single_filtered_total_returns_text_only() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "TotalProfit": [10.0, 30.0, 4.0],
    })
    profile = profile_dataset(dataframe)
    question = "what is the total profit in region asia?"

    response = run_agent(
        question,
        dataframe,
        profile,
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "calculate_filtered_aggregate"
    assert response.plan.response_mode == "text"
    assert response.chart_spec is None
    assert response.chart_data == []
    assert response.result
    assert response.result.data["value"] == 14.0
    assert response.answer == "Applied filter: Region = **Asia**. The total profit is **$14**."


@pytest.mark.parametrize(
    ("question", "metric_column", "expected_value"),
    [
        ("what is the total profit for the category office supplies in region central", "Profit", 8_838.11),
        ("What is the score for Hardware in the Enterprise segment", "Score", 7.0),
    ],
)
def test_scalar_filtered_aggregate_applies_multiple_category_filters(
    question: str,
    metric_column: str,
    expected_value: float,
) -> None:
    if metric_column == "Profit":
        dataframe = pd.DataFrame({
            "Category": ["Office Supplies", "Office Supplies", "Office Supplies", "Technology"],
            "Sub-Category": ["Supplies", "Binders", "Paper", "Supplies"],
            "Region": ["Central", "Central", "Central", "Central"],
            "Profit": [-661.89, 5_000.0, 4_500.0, 100.0],
        })
        expected_filters = [
            {"column": "Category", "operator": "equals", "value": "Office Supplies"},
            {"column": "Region", "operator": "equals", "value": "Central"},
        ]
    else:
        dataframe = pd.DataFrame({
            "ProductLine": ["Hardware", "Hardware", "Software", "Hardware"],
            "Segment": ["Enterprise", "SMB", "Enterprise", "Enterprise"],
            "Score": [3.0, 100.0, 1_000.0, 4.0],
        })
        expected_filters = [
            {"column": "ProductLine", "operator": "equals", "value": "Hardware"},
            {"column": "Segment", "operator": "equals", "value": "Enterprise"},
        ]

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "data.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_filtered_aggregate"
    assert response.plan.arguments["filters"] == expected_filters
    assert response.result
    assert response.result.data["value"] == expected_value
    assert response.result.data["filters"] == expected_filters


def test_multi_metric_filtered_total_answers_all_requested_metrics() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "TotalProfit": [10.0, 30.0, 4.0],
        "TotalRevenue": [100.0, 300.0, 40.0],
    })

    response = run_agent(
        "what is the total profit and total revenue in region asia",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "calculate_filtered_aggregate"
    assert response.plan.arguments == {
        "category_column": "Region",
        "category_value": "Asia",
        "value_column": "TotalProfit",
        "value_columns": ["TotalProfit", "TotalRevenue"],
        "aggregation": "sum",
    }
    assert response.plan.response_mode == "text"
    assert response.chart_spec is None
    assert response.result
    assert response.result.data["values"] == [
        {"value_column": "TotalProfit", "aggregation": "sum", "value": 14.0},
        {"value_column": "TotalRevenue", "aggregation": "sum", "value": 140.0},
    ]


def test_how_many_quantity_sold_through_category_filters_and_sums() -> None:
    dataframe = pd.DataFrame({
        "Quantity": [2, 3, 5, 7],
        "Ship Mode": ["Standard Class", "Standard Class", "Second Class", "Standard Class"],
    })

    response = run_agent(
        "How many quantity were sold through ship mode Standard Class",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "calculate_filtered_aggregate"
    assert response.plan.arguments == {
        "category_column": "Ship Mode",
        "category_value": "Standard Class",
        "value_column": "Quantity",
        "aggregation": "sum",
    }
    assert response.result
    assert response.result.data["value"] == 12.0


def test_how_many_unique_id_from_filtered_category_counts_distinct_values() -> None:
    dataframe = pd.DataFrame({
        "Order ID": ["A", "A", "B", "C"],
        "State": ["New York", "New York", "California", "New York"],
    })

    response = run_agent(
        "how many unique order id came from state new work",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "orders.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "analyze_categorical_value_counts"
    assert response.plan.arguments["measure_type"] == "distinct_count"
    assert response.plan.arguments["distinct_column"] == "Order ID"
    assert response.plan.arguments["filters"] == [
        {"column": "State", "operator": "equals", "value": "New York"}
    ]
    assert response.result
    assert response.result.data["table_rows"] == [
        {"State": "New York", "Unique Order ID Count": 2, "Percentage": 100.0}
    ]
    assert "unique Order ID" in response.answer


@pytest.mark.parametrize(
    "question",
    [
        "what is the mean of total sales revenue in the uploaded excel file",
        "what is the mean of total revenue in the uploaded excel file",
    ],
)
def test_dataset_wide_mean_returns_text_only(question: str) -> None:
    dataframe = pd.DataFrame({
        "TotalRevenue": [10_000_000.0, 20_000_000.0, 30_000_000.0],
        "Region": ["Asia", "Europe", "Africa"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.xlsx",
        ollama_online=True,
    )

    assert response.plan.tool_name == "calculate_scalar_aggregate"
    assert response.plan.arguments == {
        "value_column": "TotalRevenue",
        "aggregation": "mean",
    }
    assert response.plan.response_mode == "text"
    assert response.chart_spec is None
    assert response.result
    assert response.result.data["value"] == 20_000_000.0
    assert response.answer == "The mean revenue is **$20M**."


@pytest.mark.parametrize(
    ("question", "aggregation"),
    [
        ("what is the total revenue", "sum"),
        ("what is the median revenue", "median"),
        ("what is the minimum revenue", "min"),
        ("what is the maximum revenue", "max"),
    ],
)
def test_dataset_wide_scalar_aggregation_variants(
    question: str,
    aggregation: str,
) -> None:
    dataframe = pd.DataFrame({"TotalRevenue": [10.0, 20.0, 30.0]})
    plan = deterministic_plan(
        question,
        profile_dataset(dataframe),
        dataframe=dataframe,
    )

    assert plan.tool_name == "calculate_scalar_aggregate"
    assert plan.arguments["aggregation"] == aggregation


@pytest.mark.parametrize(
    ("question", "column", "aggregation", "expected"),
    [
        ("What is the highest single sales value?", "Sales", "max", 50.0),
        ("What is the lowest recorded profit?", "Profit", "min", -7.0),
        ("What is the maximum single ResponseTime value?", "ResponseTime", "max", 90.0),
    ],
)
def test_global_highest_lowest_numeric_value_uses_scalar_extrema(
    question: str,
    column: str,
    aggregation: str,
    expected: float,
) -> None:
    dataframe = pd.DataFrame({
        column: [10.0, expected, 30.0],
        "Category": ["A", "B", "A"],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "metrics.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "calculate_scalar_aggregate"
    assert response.plan.arguments == {
        "value_column": column,
        "aggregation": aggregation,
    }
    assert response.result
    assert response.result.data == {
        "value_column": column,
        "aggregation": aggregation,
        "value": expected,
    }


@pytest.mark.parametrize(
    ("question", "column"),
    [
        ("What are the minimum, maximum, mean, and median values of Sales", "Sales"),
        ("Give me min, max, average and median for ResponseTime", "ResponseTime"),
    ],
)
def test_multi_stat_numeric_column_question_uses_summary_statistics(
    question: str,
    column: str,
) -> None:
    dataframe = pd.DataFrame({
        column: [10.0, 20.0, 30.0, 40.0],
        "OtherMetric": [1.0, 2.0, 3.0, 4.0],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "metrics.csv",
        ollama_online=True,
    )

    assert response.plan.tool_name == "calculate_summary_statistics"
    assert response.plan.arguments == {"columns": [column]}
    assert response.plan.response_mode == "text"
    assert response.result
    assert len(response.result.data) == 1
    row = response.result.data[0]
    assert row["column"] == column
    assert row["min"] == 10.0
    assert row["max"] == 40.0
    assert row["mean"] == 25.0
    assert row["50%"] == 25.0
    answer = response.answer.lower()
    assert "minimum" in answer
    assert "maximum" in answer
    assert "mean" in answer
    assert "median" in answer


def test_multi_metric_average_question_answers_all_requested_metrics() -> None:
    dataframe = pd.DataFrame({
        "TotalRevenue": [1_000_000.0, 2_000_000.0],
        "TotalProfit": [100_000.0, 300_000.0],
        "UnitsSold": [10.0, 30.0],
    })

    response = run_agent(
        "find me the average of total revenue, total profit and unitsolds?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.xlsx",
        ollama_online=True,
    )

    assert response.plan.tool_name == "calculate_multi_scalar_aggregate"
    assert response.plan.arguments == {
        "value_columns": ["TotalRevenue", "TotalProfit", "UnitsSold"],
        "aggregation": "mean",
    }
    assert response.plan.response_mode == "text"
    assert response.chart_spec is None
    assert response.result
    assert response.result.data == [
        {"value_column": "TotalRevenue", "aggregation": "mean", "value": 1_500_000.0},
        {"value_column": "TotalProfit", "aggregation": "mean", "value": 200_000.0},
        {"value_column": "UnitsSold", "aggregation": "mean", "value": 20.0},
    ]
    answer = response.answer.lower()
    assert "mean revenue" in answer
    assert "mean profit" in answer
    assert "mean units sold" in answer


def test_grouped_multi_metric_question_keeps_all_requested_metrics() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Asia", "Europe"],
        "TotalProfit": [10.0, 20.0, 5.0],
        "UnitsSold": [1, 2, 3],
    })

    response = run_agent(
        "what is the total profit and total unitssold in each region?",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "Region",
        "secondary_group_by": None,
        "value_column": "TotalProfit",
        "value_columns": ["TotalProfit", "UnitsSold"],
        "aggregation": "sum",
        "limit": None,
        "filter_column": None,
        "filter_value": None,
    }
    assert response.result
    assert response.result.data == [
        {"Region": "Asia", "TotalProfit": 30.0, "UnitsSold": 3},
        {"Region": "Europe", "TotalProfit": 5.0, "UnitsSold": 3},
    ]
    assert response.chart_spec
    assert response.chart_spec.value_columns == ["TotalProfit", "UnitsSold"]
    assert {row["Metric"] for row in response.chart_data} == {"TotalProfit", "UnitsSold"}


@pytest.mark.parametrize(
    "question",
    [
        "What is the average sales value by category",
        "What is the average sales value per category",
    ],
)
def test_average_metric_per_category_routes_to_grouped_mean(question: str) -> None:
    dataframe = pd.DataFrame({
        "Category": ["A", "A", "B"],
        "Sales": [10.0, 30.0, 50.0],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments == {
        "group_by": "Category",
        "secondary_group_by": None,
        "value_column": "Sales",
        "value_columns": None,
        "aggregation": "mean",
        "limit": None,
        "filter_column": None,
        "filter_value": None,
    }
    assert response.result
    assert response.result.data == [
        {"Category": "B", "Sales": 50.0},
        {"Category": "A", "Sales": 20.0},
    ]


def test_average_unit_price_and_unit_cost_by_item_type_excludes_total_cost() -> None:
    dataframe = pd.DataFrame({
        "ItemType": ["A", "A", "B", "B"],
        "UnitPrice": [2.0, 6.0, 4.0, 8.0],
        "UnitCost": [1.0, 3.0, 2.0, 4.0],
        "TotalCost": [5.0, 20.0, 10.0, 30.0],
    })

    response = run_agent(
        "Show average unit price and average unit cost by item type.",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "items.csv",
        ollama_online=False,
    )

    assert response.plan.tool_name == "group_and_aggregate"
    assert response.plan.arguments["group_by"] == "ItemType"
    assert response.plan.arguments["value_columns"] == ["UnitPrice", "UnitCost"]
    assert response.chart_spec
    assert response.chart_spec.value_columns == ["UnitPrice", "UnitCost"]
    assert response.result
    assert response.result.data == [
        {"ItemType": "B", "UnitPrice": 6.0, "UnitCost": 3.0},
        {"ItemType": "A", "UnitPrice": 4.0, "UnitCost": 2.0},
    ]
    assert "TotalCost" not in str(response.result.data)
    assert response.chart_data
    assert {row["Metric"] for row in response.chart_data} == {"UnitPrice", "UnitCost"}


def test_explicit_plot_request_keeps_visual_output() -> None:
    dataframe = pd.DataFrame({
        "Region": ["Asia", "Europe", "Asia"],
        "TotalProfit": [10.0, 30.0, 4.0],
    })
    plan = deterministic_plan(
        "plot total profit by region",
        profile_dataset(dataframe),
        dataframe=dataframe,
    )

    assert plan.tool_name == "group_and_aggregate"
    assert plan.response_mode == "full"
    assert plan.chart_spec is not None
    assert plan.chart_spec.limit is None


def test_chat_chart_does_not_truncate_categories_by_default() -> None:
    dataframe = pd.DataFrame({
        "Region": [f"Region {index}" for index in range(25)],
        "TotalProfit": [float(index) for index in range(25)],
    })

    response = run_agent(
        "plot total profit by region",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen2.5:1.5b",
        "sales.csv",
        ollama_online=False,
    )

    assert response.chart_spec
    assert response.chart_spec.limit is None
    assert len(response.chart_data) == 25


def test_previous_chart_summary_returns_text_without_new_chart() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "East", "South"],
        "Sales": [30.0, 20.0, 10.0],
    })
    profile = profile_dataset(dataframe)
    history = [{
        "question": "Show total Sales by Region",
        "answer": "Calculated sales by region.",
        "plan": {
            "tool_name": "group_and_aggregate",
            "arguments": {"group_by": "Region", "value_column": "Sales", "aggregation": "sum"},
        },
        "result": {
            "tool_name": "group_and_aggregate",
            "success": True,
            "summary": "Calculated sales by region.",
            "data": [
                {"Region": "West", "Sales": 30.0},
                {"Region": "East", "Sales": 20.0},
                {"Region": "South", "Sales": 10.0},
            ],
            "warnings": [],
            "execution_seconds": 0.01,
        },
        "chart_spec": {
            "chart_type": "bar",
            "title": "Sales by Region",
            "x": "Region",
            "y": "Sales",
            "aggregation": "sum",
        },
        "chart_data": [
            {"Region": "West", "Sales": 30.0},
            {"Region": "East", "Sales": 20.0},
            {"Region": "South", "Sales": 10.0},
        ],
    }]

    plan = deterministic_plan("make a short summary of 2 line from above chart", profile, history)
    assert plan.tool_name == "summarize_previous"
    assert plan.arguments["line_count"] == 2

    response = run_agent(
        "make a short summary of 2 line from above chart",
        dataframe,
        profile,
        get_settings(),
        "qwen3:4b",
        "sales.csv",
        history=history,
        ollama_online=False,
    )
    assert response.chart_spec is None
    assert response.chart_data == []
    assert response.result is None
    assert len(response.answer.splitlines()) == 2
    assert "West" in response.answer
    assert "South" in response.answer


@pytest.mark.parametrize(
    ("question", "expected_lines"),
    [
        ("Summarize the previous chart in one line", 1),
        ("Explain the last chart in 3 lines", 3),
        ("Give me a summary of this chart", 2),
    ],
)
def test_previous_chart_summary_phrasing_variants(question: str, expected_lines: int) -> None:
    dataframe = pd.DataFrame({"Region": ["West", "East"], "Sales": [30.0, 20.0]})
    profile = profile_dataset(dataframe)
    history = [{
        "question": "Show Sales by Region",
        "answer": "Sales by region.",
        "plan": {"tool_name": "group_and_aggregate", "arguments": {}},
        "chart_spec": {
            "chart_type": "bar",
            "title": "Sales by Region",
            "x": "Region",
            "y": "Sales",
            "aggregation": "sum",
        },
        "chart_data": [
            {"Region": "West", "Sales": 30.0},
            {"Region": "East", "Sales": 20.0},
        ],
    }]

    response = run_agent(
        question,
        dataframe,
        profile,
        get_settings(),
        "qwen3:4b",
        "sales.csv",
        history=history,
        ollama_online=False,
    )

    assert len(response.answer.splitlines()) == expected_lines
    assert response.chart_spec is None


def test_dataset_summary_returns_verified_text_without_chart() -> None:
    dataframe = pd.DataFrame({
        "Region": ["West", "East", "West", None],
        "Sales": [30.0, 20.0, 10.0, 40.0],
        "Profit": [5.0, 4.0, 1.0, 8.0],
    })
    profile = profile_dataset(dataframe)

    response = run_agent(
        "Summarize the dataset and give me key insights in 4 lines",
        dataframe,
        profile,
        get_settings(),
        "qwen3:4b",
        "sales.csv",
        history=[],
        ollama_online=False,
    )

    assert response.plan.tool_name == "summarize_dataset"
    assert len(response.answer.splitlines()) == 4
    assert "4 rows" in response.answer
    assert response.chart_spec is None
    assert response.result is None

    insight_response = run_agent(
        "Give me the main insights and takeaways",
        dataframe,
        profile,
        get_settings(),
        "qwen3:4b",
        "sales.csv",
        history=[],
        ollama_online=False,
    )
    assert insight_response.plan.tool_name == "summarize_dataset"
    assert insight_response.chart_spec is None


@pytest.mark.parametrize(
    "question",
    [
        "tell me about this dataset",
        "what is this dataset about",
        "describe the dataset",
    ],
)
def test_dataset_overview_phrasings_route_to_summary(question: str) -> None:
    dataframe = pd.DataFrame({
        "Category": ["A", "B"],
        "Sales": [10.0, 20.0],
        "Profit": [1.0, 2.0],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen3:4b",
        "sales.csv",
        history=[],
        ollama_online=False,
    )

    assert response.plan.tool_name == "summarize_dataset"
    assert "2 rows" in response.answer
    assert "3 columns" in response.answer
    assert response.chart_spec is None


def test_row_and_column_count_question_routes_to_inspect_dataset() -> None:
    dataframe = pd.DataFrame({
        "Category": ["A", "B"],
        "Sales": [10.0, 20.0],
        "Profit": [1.0, 2.0],
    })

    response = run_agent(
        "how many column and row",
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen3:4b",
        "sales.csv",
        history=[],
        ollama_online=False,
    )

    assert response.plan.tool_name == "inspect_dataset"
    assert response.plan.response_mode == "text"
    assert response.result
    assert response.result.data["rows"] == 2
    assert response.result.data["columns"] == 3
    assert "2 rows" in response.answer
    assert "3 columns" in response.answer


def test_explain_previous_result_without_chart() -> None:
    dataframe = pd.DataFrame({"A": [1, 2], "B": [2, 4]})
    profile = profile_dataset(dataframe)
    history = [{
        "question": "Are A and B correlated?",
        "answer": "The correlation is 1.0.",
        "plan": {"tool_name": "calculate_correlation", "arguments": {}},
        "result": {
            "tool_name": "calculate_correlation",
            "success": True,
            "summary": "Correlation calculated.",
            "data": {
                "first_column": "A",
                "second_column": "B",
                "correlation": 1.0,
            },
            "warnings": [],
            "execution_seconds": 0.01,
        },
        "chart_spec": None,
        "chart_data": [],
    }]

    response = run_agent(
        "What does this mean? Explain the previous result in 2 lines",
        dataframe,
        profile,
        get_settings(),
        "qwen3:4b",
        "data.csv",
        history=history,
        ollama_online=False,
    )

    assert response.plan.tool_name == "summarize_previous"
    assert len(response.answer.splitlines()) == 2
    assert "strong positive correlation" in response.answer
    assert response.chart_spec is None


@pytest.mark.parametrize(
    ("question", "tool_name", "chart_type", "x", "y", "value_columns"),
    [
        (
            "Create a scatter plot of Sales versus Profit.",
            "create_scatter_plot",
            "scatter",
            "Sales",
            "Profit",
            [],
        ),
        (
            "Create a histogram of Sales.",
            "create_histogram",
            "histogram",
            "Sales",
            None,
            [],
        ),
        (
            "Create a box plot of Profit by Category.",
            "create_box_plot",
            "box",
            "Category",
            "Profit",
            [],
        ),
        (
            "Create a pie chart of sales share by Region.",
            "create_pie_chart",
            "pie",
            "Region",
            "Sales",
            [],
        ),
        (
            "Create a correlation heatmap for Sales, Quantity, Discount, and Profit.",
            "create_heatmap",
            "heatmap",
            None,
            None,
            ["Sales", "Quantity", "Discount", "Profit"],
        ),
    ],
)
def test_explicit_chart_type_requests_are_dynamic(
    question: str,
    tool_name: str,
    chart_type: str,
    x: str | None,
    y: str | None,
    value_columns: list[str],
) -> None:
    dataframe = pd.DataFrame({
        "Category": ["Furniture", "Technology", "Office Supplies", "Furniture"],
        "Region": ["West", "East", "West", "Central"],
        "Sales": [100.0, 200.0, 50.0, 75.0],
        "Profit": [10.0, 20.0, -5.0, 7.0],
        "Quantity": [2, 3, 1, 4],
        "Discount": [0.1, 0.0, 0.2, 0.15],
    })

    response = run_agent(
        question,
        dataframe,
        profile_dataset(dataframe),
        get_settings(),
        "qwen3:4b",
        "orders.csv",
        history=[],
        ollama_online=False,
    )

    assert response.plan.tool_name == tool_name
    assert response.chart_spec
    assert response.chart_spec.chart_type == chart_type
    assert response.chart_spec.x == x
    assert response.chart_spec.y == y
    assert response.chart_spec.value_columns == value_columns
    assert response.chart_data
