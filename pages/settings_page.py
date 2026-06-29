"""Application, safety, and multi-dataset settings."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from services.file_loader import FileLoadError, load_dataset


def render_settings_page(settings: Settings) -> None:
    """Render effective local configuration without exposing secrets."""
    st.title("Settings")
    st.markdown("### Local AI")
    st.text_input("Ollama base URL", settings.ollama_base_url, disabled=True)
    st.text_input("Selected model", st.session_state.selected_model, disabled=True)
    st.caption("Runtime overrides are loaded from `.env`; restart Streamlit after changing them.")

    st.markdown("### Data Safety")
    st.write(f"Maximum upload size: **{settings.max_upload_size_mb} MB**")
    st.write(f"Maximum sample rows for future LLM context: **{settings.max_sample_rows_for_llm}**")
    st.write("Conversation and analysis memory are stored only in this Streamlit session.")
    st.write("No complete dataset is sent to Ollama.")

    st.markdown("### Experimental Multi-Dataset Comparison")
    files = st.file_uploader(
        "Upload two or more CSV/Excel datasets",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="multi_dataset_upload",
    )
    if files:
        loaded = {}
        errors = []
        for file in files:
            try:
                dataset = load_dataset(
                    file.name,
                    file.getvalue(),
                    max_size_mb=settings.max_upload_size_mb,
                )
                loaded[file.name] = dataset.dataframe
            except FileLoadError as exc:
                errors.append(f"{file.name}: {exc}")
        for error in errors:
            st.error(error)
        if len(loaded) >= 2:
            names = list(loaded)
            selection = st.columns(2)
            left_name = selection[0].selectbox("First dataset", names, key="multi_left")
            right_options = [name for name in names if name != left_name]
            right_name = selection[1].selectbox("Second dataset", right_options, key="multi_right")
            left, right = loaded[left_name], loaded[right_name]
            common = sorted(set(left.columns) & set(right.columns))
            compatibility = [
                column for column in common if str(left[column].dtype) == str(right[column].dtype)
            ]
            st.write(f"Common columns: **{', '.join(common) or 'none'}**")
            st.write(f"Type-compatible columns: **{', '.join(compatibility) or 'none'}**")
            preview_columns = st.columns(2)
            preview_columns[0].dataframe(left.head(10), width="stretch", hide_index=True)
            preview_columns[1].dataframe(right.head(10), width="stretch", hide_index=True)
            if common:
                join_column = st.selectbox("Optional join column", common)
                confirmed = st.checkbox("I confirm this join may duplicate rows when keys are not unique")
                if st.button("Preview confirmed join", disabled=not confirmed):
                    joined = left.merge(
                        right,
                        on=join_column,
                        how="inner",
                        suffixes=(f"_{left_name}", f"_{right_name}"),
                    )
                    st.write(f"Joined preview: **{len(joined):,} rows**")
                    st.dataframe(joined.head(30), width="stretch", hide_index=True)
