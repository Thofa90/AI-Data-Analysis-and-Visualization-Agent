"""Saved analytical insights."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.chart_panel import render_chart_insight
from services.chart_insight_service import generate_chart_insight
from services.chart_service import ChartSpec, create_chart
from services.export_service import history_to_json


def render_saved_insights_page() -> None:
    st.title("Saved Insights")
    insights = st.session_state.saved_insights
    if not insights:
        st.info("No insights have been saved yet.")
        return
    for index, insight in enumerate(insights):
        with st.expander(insight.get("question", f"Insight {index + 1}"), expanded=index == 0):
            st.caption(f"Dataset: {insight.get('dataset_name', 'dataset')}")
            st.write(insight.get("answer", ""))
            result = insight.get("result")
            if result and isinstance(result.get("data"), list):
                st.dataframe(pd.DataFrame(result["data"]), width="stretch", hide_index=True)
            chart_spec = insight.get("chart_spec")
            if chart_spec and st.session_state.active_dataframe is not None:
                try:
                    chart_definition = ChartSpec.model_validate(chart_spec).model_copy(
                        update={"currency_symbol": st.session_state.currency_symbol}
                    )
                    chart_rows = insight.get("chart_data") or []
                    chart_source = (
                        pd.DataFrame(chart_rows)
                        if chart_rows
                        else st.session_state.active_dataframe
                    )
                    if chart_rows:
                        chart_definition = chart_definition.model_copy(
                            update={
                                "filter_column": None,
                                "filter_value": None,
                                "limit": None,
                            }
                        )
                    figure, chart_result = create_chart(chart_source, chart_definition)
                    st.plotly_chart(figure, width="stretch", key=f"saved_chart_{index}")
                    render_chart_insight(
                        generate_chart_insight(chart_result),
                        key_prefix=f"saved_{index}",
                    )
                except ValueError as exc:
                    st.warning(str(exc))
            if st.button("Remove insight", key=f"remove_insight_{index}"):
                st.session_state.saved_insights = [
                    item for position, item in enumerate(insights) if position != index
                ]
                st.rerun()
    st.download_button(
        "Download saved insights",
        history_to_json(insights),
        file_name="saved_insights.json",
        mime="application/json",
    )
