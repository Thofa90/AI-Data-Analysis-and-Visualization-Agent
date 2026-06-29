"""Regression tests for one-click radio and callback navigation."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_sidebar_radio_changes_page_on_first_click() -> None:
    app = AppTest.from_file("app.py", default_timeout=15)
    app.run()

    assert app.session_state["current_page"] == "Home"
    app.sidebar.radio[0].set_value("Settings")
    app.run()

    assert not app.exception
    assert app.session_state["current_page"] == "Settings"
    assert app.title[0].value == "Settings"


def test_home_feature_button_updates_radio_backed_page() -> None:
    app = AppTest.from_file("app.py", default_timeout=15)
    app.run()

    feature_button = next(button for button in app.button if button.label == "Evaluation")
    feature_button.click()
    app.run()

    assert not app.exception
    assert app.session_state["current_page"] == "Evaluation"
    assert app.sidebar.radio[0].value == "Evaluation"
    assert app.title[0].value == "Evaluation Dashboard"
