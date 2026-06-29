"""Streamlit entry point for the AI Data Analysis & Visualization Agent."""

from __future__ import annotations

import streamlit as st

from components.sidebar import render_sidebar
from config.settings import get_settings
from pages.home_page import render_home_page
from pages.eda_page import render_eda_page
from pages.visualization_page import render_visualization_page
from pages.correlation_page import render_correlation_page
from pages.overview_page import render_overview_page
from pages.settings_page import render_settings_page
from pages.chat_page import render_chat_page
from pages.cleaning_page import render_cleaning_page
from pages.reports_page import render_reports_page
from pages.saved_insights_page import render_saved_insights_page
from pages.evaluation_page import render_evaluation_page
from utils.logging_config import configure_logging
from utils.session_state import initialize_session_state


def main() -> None:
    """Configure and render the application."""
    settings = get_settings()
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="DI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    configure_logging(settings.log_level)
    initialize_session_state(settings.default_model)

    with open("assets/styles.css", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

    selected_page = render_sidebar(settings)
    if selected_page == "Home":
        render_home_page(settings)
    elif selected_page == "Chat & Ask":
        render_chat_page(settings)
    elif selected_page == "Data Overview":
        render_overview_page()
    elif selected_page == "EDA Dashboard":
        render_eda_page()
    elif selected_page == "Visualizations":
        render_visualization_page()
    elif selected_page == "Correlations":
        render_correlation_page()
    elif selected_page == "Data Cleaning":
        render_cleaning_page()
    elif selected_page == "Reports":
        render_reports_page()
    elif selected_page == "Saved Insights":
        render_saved_insights_page()
    elif selected_page == "Evaluation":
        render_evaluation_page()
    elif selected_page == "Settings":
        render_settings_page(settings)
    else:
        st.error("Unknown page selection.")


if __name__ == "__main__":
    main()
