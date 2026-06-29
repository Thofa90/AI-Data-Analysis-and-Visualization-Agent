"""Conversational analysis UI backed by approved tools."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agent.data_agent import run_agent, split_user_questions
from agent.memory import store_response
from config.settings import Settings
from agent.schemas import AgentResponse
from services.evaluation_service import evaluate_answer
from datetime import datetime, timezone
from services.chart_service import create_chart
from services.chart_insight_service import generate_chart_insight
from components.chart_panel import render_chart_insight
from components.chat_narrative import render_chat_narrative
from services.export_service import dataframe_to_csv
from utils.formatting import format_currency, is_currency_column
from services.query_guide import (
    DatasetQueryGuide,
    build_dataset_query_guide,
    build_hint_chips,
    build_query_examples,
    dataset_fingerprint,
    suggest_columns,
)


def _get_query_guide(dataframe: pd.DataFrame, profile) -> DatasetQueryGuide:
    fingerprint = dataset_fingerprint(dataframe)
    cached = st.session_state.get("dataset_query_guide")
    if cached and cached.get("fingerprint") == fingerprint:
        return DatasetQueryGuide.model_validate(cached)
    guide = build_dataset_query_guide(dataframe, profile)
    st.session_state.dataset_query_guide = guide.model_dump(mode="json")
    return guide


def _append_draft(text: str) -> None:
    current = st.session_state.get("query_builder_draft", "")
    separator = " " if current and not current.endswith(" ") else ""
    st.session_state.query_builder_draft = f"{current}{separator}{text}".strip()


def _plan_preview(plan: dict) -> dict[str, str]:
    args = plan.get("arguments", {}) or {}
    preview: dict[str, str] = {}
    tool = plan.get("tool_name")
    if tool:
        preview["Intent"] = str(tool).replace("_", " ").title()
    if args.get("value_column"):
        preview["Metric"] = str(args["value_column"])
    if args.get("aggregation"):
        preview["Aggregation"] = str(args["aggregation"]).title()
    if args.get("date_column"):
        preview["Date column"] = str(args["date_column"])
    if args.get("frequency"):
        preview["Time grouping"] = str(args["frequency"]).title()
    if args.get("filter_column"):
        preview["Filter"] = f"{args.get('filter_column')} = {args.get('filter_value')}"
    if args.get("category_column"):
        preview["Filter"] = f"{args.get('category_column')} = {args.get('category_value')}"
    if args.get("group_by"):
        preview["Group by"] = str(args["group_by"])
    if args.get("secondary_group_by"):
        preview["Breakdown"] = str(args["secondary_group_by"])
    if args.get("counted_column"):
        preview["Counted column"] = str(args["counted_column"])
    if args.get("measure_type"):
        preview["Measure"] = str(args["measure_type"]).replace("_", " ").title()
    return preview


def _render_result_table(data, response_index: int) -> None:
    if isinstance(data, list) and data:
        table = pd.DataFrame(data)
        display_table = table.copy()
        for column in display_table.columns:
            if is_currency_column(str(column)) and pd.api.types.is_numeric_dtype(
                display_table[column]
            ):
                display_table[column] = display_table[column].map(
                    lambda value: format_currency(
                        value,
                        symbol=st.session_state.currency_symbol,
                    )
                )
        st.dataframe(display_table, width="stretch", hide_index=True)
        st.download_button(
            "Download result CSV",
            dataframe_to_csv(table),
            file_name=f"analysis_result_{response_index + 1}.csv",
            mime="text/csv",
            key=f"download_chat_result_{response_index}",
        )
    elif isinstance(data, dict):
        st.json(data)


def _render_response(item: dict, index: int, settings: Settings) -> None:
    with st.chat_message("assistant"):
        render_chat_narrative(item.get("narrative"), item.get("answer", ""))
        result = item.get("result")
        plan = item.get("plan", {})
        arguments = plan.get("arguments", {}) or {}
        if plan.get("clarification") and arguments.get("options"):
            st.session_state.pending_clarification = arguments
            columns = st.columns(min(len(arguments["options"]), 4))
            for option_index, option in enumerate(arguments["options"]):
                if columns[option_index % len(columns)].button(
                    f"Use {option}",
                    key=f"clarify_{index}_{option_index}",
                ):
                    original = arguments.get("original_query", item.get("question", ""))
                    if arguments.get("clarification_type") == "ambiguous_date_column":
                        st.session_state.pending_question = f"Using {option}, {original}"
                    elif arguments.get("clarification_type") == "ambiguous_filter_value":
                        st.session_state.pending_question = f"{original} where {option}"
                    else:
                        st.session_state.pending_question = f"{original} {option}"
                    st.rerun()
            suggested = arguments.get("suggested_queries") or []
            if suggested:
                with st.expander("Suggested precise queries"):
                    for suggestion in suggested[:4]:
                        if st.button(suggestion, key=f"suggested_rewrite_{index}_{suggestion}"):
                            st.session_state.pending_question = suggestion
                            st.rerun()
        response_mode = plan.get("response_mode", "full")
        if response_mode != "text" and result and result.get("data") is not None:
            data = result["data"]
            if (
                plan.get("tool_name") in {
                    "analyze_categorical_value_counts",
                    "profile_column",
                }
                and isinstance(data, dict)
                and data.get("table_rows")
            ):
                _render_result_table(data["table_rows"], index)
            else:
                _render_result_table(data, index)
        chart_spec = item.get("chart_spec")
        if (
            response_mode != "text"
            and chart_spec
            and st.session_state.active_dataframe is not None
        ):
            try:
                from services.chart_service import ChartSpec
                chart_definition = ChartSpec.model_validate(chart_spec).model_copy(
                    update={
                        "currency_symbol": st.session_state.currency_symbol,
                    }
                )
                chart_rows = item.get("chart_data") or []
                use_verified_chart_rows = (
                    bool(chart_rows)
                    and plan.get("tool_name") in {
                        "group_and_aggregate",
                        "calculate_grouped_extrema",
                        "analyze_categorical_value_counts",
                        "profile_column",
                        "calculate_period_over_period",
                        "analyze_advanced_request",
                    }
                )
                chart_source = (
                    pd.DataFrame(chart_rows)
                    if use_verified_chart_rows
                    else st.session_state.active_dataframe
                )
                chart_definition = (
                    chart_definition.model_copy(
                        update={
                            "filter_column": None,
                            "filter_value": None,
                            "limit": None,
                        }
                    )
                    if use_verified_chart_rows
                    else chart_definition
                )
                figure, chart_result = create_chart(chart_source, chart_definition)
                st.plotly_chart(figure, width="stretch", key=f"chat_chart_{index}")
                render_chart_insight(
                    generate_chart_insight(chart_result),
                    key_prefix=f"chat_{index}",
                )
            except ValueError as exc:
                st.warning(str(exc))
        if response_mode != "text" and plan.get("safe_code"):
            with st.expander("Safe Pandas operation (Beta)"):
                st.code(plan["safe_code"], language="python")
        preview = _plan_preview(plan)
        if preview:
            with st.expander("How this question was understood"):
                st.table(pd.DataFrame(
                    [{"Role": key, "Resolved as": value} for key, value in preview.items()]
                ))
        controls = st.columns(2)
        if controls[0].button("Save insight", key=f"save_{index}"):
            metadata = st.session_state.uploaded_file or {}
            saved = {
                "dataset_name": metadata.get("name", "dataset"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **item,
            }
            st.session_state.saved_insights = [*st.session_state.saved_insights, saved]
            st.toast("Insight saved.")
        if controls[1].button("Evaluate This Answer", key=f"evaluate_{index}"):
            status = st.session_state.ollama_status
            scores = evaluate_answer(
                AgentResponse.model_validate(item),
                settings,
                st.session_state.selected_model,
                use_llm_judge=bool(
                    status and status.online and st.session_state.selected_model in status.models
                ),
            )
            record = {
                "question": item["question"],
                **scores.model_dump(mode="json"),
            }
            st.session_state.evaluation_history = [*st.session_state.evaluation_history, record]
            st.session_state.analysis_history[index]["evaluation"] = scores.model_dump(mode="json")
            st.rerun()
        if item.get("evaluation"):
            scores = item["evaluation"]
            with st.expander("Evaluation result", expanded=True):
                st.write(f"Correctness: **{scores['correctness']:.0%}**")
                st.write(f"Faithfulness: **{scores['faithfulness']:.0%}**")
                st.write(f"Relevancy: **{scores['relevancy']:.0%}**")
                st.write(f"Completeness: **{scores['completeness']:.0%}**")
                st.write(f"Tool Accuracy: **{scores['tool_accuracy']:.0%}**")
                st.write(f"Chart Accuracy: **{scores['chart_accuracy']:.0%}**")
                st.write(f"Overall Evaluation Score: **{scores['overall_score']:.0%}**")


def _render_query_guidance(guide: DatasetQueryGuide) -> None:
    examples = build_query_examples(guide)
    with st.expander("How to ask questions about this dataset", expanded=False):
        st.caption(
            "For more accurate results, mention the metric, date field, time level, "
            "filters as Column = Value, and grouping column."
        )
        if examples:
            for example in examples[:8]:
                cols = st.columns([0.85, 0.15])
                cols[0].markdown(f"- {example}")
                if cols[1].button("Use", key=f"use_example_{example}"):
                    st.session_state.pending_question = example
                    st.rerun()
        chips = build_hint_chips(guide)
        st.divider()
        st.caption("Hint chips add text to the query builder draft.")
        for group, values in chips.items():
            if not values:
                continue
            st.markdown(f"**{group}**")
            columns = st.columns(min(len(values), 6))
            for index, value in enumerate(values):
                if columns[index % len(columns)].button(value, key=f"chip_{group}_{value}"):
                    _append_draft(value)
                    st.rerun()

    with st.expander("Build a precise question", expanded=False):
        analysis_type = st.selectbox(
            "Analysis type",
            [
                "Trend over time",
                "Filtered metric",
                "Value counts",
                "Grouped comparison",
                "Percentage share",
                "Period-over-period change",
            ],
            key="query_builder_analysis_type",
        )
        date_names = [column.display_name for column in guide.date_columns]
        metric_names = [column.display_name for column in guide.numeric_columns]
        category_names = [column.display_name for column in guide.categorical_columns]
        metric = st.selectbox("Metric", metric_names or [""], key="query_builder_metric")
        date_column = st.selectbox("Date column", [""] + date_names, key="query_builder_date")
        time_grouping = st.selectbox("Time grouping", ["monthly", "yearly", "weekly", "daily"], key="query_builder_time")
        year = st.text_input("Year", placeholder="2017", key="query_builder_year")
        filter_column = st.selectbox("Filter column", [""] + category_names, key="query_builder_filter_column")
        filter_value_options = [""]
        selected_filter = next((column for column in guide.categorical_columns if column.display_name == filter_column), None)
        if selected_filter:
            filter_value_options.extend(selected_filter.example_values[:20])
        filter_value = st.selectbox("Filter value", filter_value_options, key="query_builder_filter_value")
        group_by = st.selectbox("Group-by column", [""] + category_names, key="query_builder_group_by")
        counted_column = st.selectbox("Counted column", [""] + category_names, key="query_builder_counted")

        generated = ""
        if analysis_type == "Trend over time" and metric and date_column:
            generated = f"Using {date_column}, show {time_grouping} {metric}"
            if year:
                generated += f" for {year}"
            if filter_column and filter_value:
                generated += f" where {filter_column} = {filter_value}"
        elif analysis_type == "Filtered metric" and metric and filter_column and filter_value:
            generated = f"Show total {metric} where {filter_column} = {filter_value}"
        elif analysis_type == "Value counts" and counted_column:
            generated = f"Show value counts of {counted_column}"
            if filter_column and filter_value:
                generated += f" where {filter_column} = {filter_value}"
        elif analysis_type == "Grouped comparison" and metric and group_by:
            generated = f"Show total {metric} by {group_by}"
        elif analysis_type == "Percentage share" and metric and group_by:
            generated = f"Show each {group_by}'s share of total {metric}"
        elif analysis_type == "Period-over-period change" and metric and date_column:
            generated = f"Using {date_column}, show {time_grouping} {metric} change compared with the previous period"

        draft_value = generated or st.session_state.get("query_builder_draft", "")
        draft = st.text_input("Generated question", value=draft_value, key="query_builder_generated")
        action_cols = st.columns(2)
        if action_cols[0].button("Ask generated question", disabled=not draft):
            st.session_state.pending_question = draft
            st.session_state.query_builder_draft = ""
            st.rerun()
        if action_cols[1].button("Clear draft"):
            st.session_state.query_builder_draft = ""
            st.rerun()


def _render_column_browser(guide: DatasetQueryGuide) -> None:
    with st.expander("Available columns and values", expanded=False):
        rows = []
        for column in [
            *guide.date_columns,
            *guide.numeric_columns,
            *guide.categorical_columns,
            *guide.identifier_columns,
        ]:
            rows.append({
                "Column": column.display_name,
                "Type": column.semantic_type.title(),
                "Unique": column.unique_count,
                "Example values": ", ".join(column.example_values[:3]),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        for column in guide.categorical_columns:
            if column.unique_count <= 20:
                with st.expander(f"Values in {column.display_name}", expanded=False):
                    st.write(", ".join(column.example_values[:20]) or "No example values available.")
            elif column.example_values:
                st.caption(
                    f"{column.display_name}: showing {len(column.example_values)} example values "
                    f"of {column.unique_count:,} unique values."
                )


def render_chat_page(settings: Settings) -> None:
    """Render messages, verified results, charts, and safe code."""
    st.title("Chat & Ask")
    dataframe = st.session_state.active_dataframe
    profile = st.session_state.dataset_profile
    if dataframe is None or profile is None:
        st.info("Upload a dataset before asking analytical questions.")
        return
    st.caption("All calculations use approved pandas tools. Ollama may plan or explain, but does not calculate values.")
    guide = _get_query_guide(dataframe, profile)
    examples = build_query_examples(guide)
    _render_query_guidance(guide)
    _render_column_browser(guide)
    typed_fragment = st.text_input(
        "Column suggestions",
        placeholder="Type part of a column, e.g. order, ship, sales",
        key="column_suggestion_fragment",
    )
    if typed_fragment:
        matches = suggest_columns(typed_fragment, guide)
        if matches:
            st.caption("Suggestions: " + ", ".join(matches))
    suggestions = examples[:4] or [
        "What is the average of each numeric column?",
        "Which columns contain missing values?",
        "Find potential outliers.",
        "Show a correlation matrix.",
    ]
    suggestion_columns = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        if suggestion_columns[index % 2].button(suggestion, key=f"suggestion_{index}", width="stretch"):
            st.session_state.pending_question = suggestion
    for index, item in enumerate(st.session_state.analysis_history):
        with st.chat_message("user"):
            st.markdown(item["question"])
        _render_response(item, index, settings)

    placeholder = (
        f"Example: {examples[0]}"
        if examples
        else "Ask anything about your data..."
    )
    question = st.chat_input(placeholder)
    question = question or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.chat_messages = [
            *st.session_state.chat_messages,
            {"role": "user", "content": question},
        ]
        metadata = st.session_state.uploaded_file or {}
        status = st.session_state.ollama_status
        questions = split_user_questions(question)
        responses = []
        working_history = st.session_state.analysis_history
        with st.spinner("Running verified analysis..."):
            for subquestion in questions:
                response = run_agent(
                    subquestion,
                    dataframe,
                    profile,
                    settings,
                    st.session_state.selected_model,
                    metadata.get("name", "dataset"),
                    history=working_history,
                    ollama_online=bool(
                        status
                        and status.online
                        and st.session_state.selected_model in status.models
                    ),
                )
                responses.append(response)
                working_history = store_response(working_history, response)
        st.session_state.analysis_history = working_history
        st.session_state.chat_messages = [
            *st.session_state.chat_messages,
            {
                "role": "assistant",
                "content": "\n\n".join(
                    f"{index + 1}. {response.answer}"
                    for index, response in enumerate(responses)
                ),
            },
        ]
        st.rerun()
