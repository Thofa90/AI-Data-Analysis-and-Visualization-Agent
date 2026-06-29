"""Configuration regressions for the custom sidebar navigation."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_builtin_sidebar_navigation_is_disabled() -> None:
    config = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf-8"))
    assert config["client"]["showSidebarNavigation"] is False
    assert config["client"]["toolbarMode"] == "minimal"


def test_sidebar_css_hides_builtin_navigation_and_starts_with_brand() -> None:
    css = Path("assets/styles.css").read_text(encoding="utf-8")
    assert '[data-testid="stSidebarNav"]' in css
    assert "display: none !important" in css
    assert '[data-testid="stSidebarContent"]' in css
    assert "color: #ffffff !important" in css
    assert "-webkit-text-fill-color: #ffffff !important" in css
    assert '[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]' in css
    assert '[data-testid="stSidebar"] [data-testid="stCaptionContainer"]' in css
    assert "background: rgb(49, 94, 255)" in css
    assert "background: rgb(79, 119, 255)" in css
    assert '[data-testid="stFileUploaderDropzone"] button' in css


def test_chat_markdown_text_is_white() -> None:
    css = Path("assets/styles.css").read_text(encoding="utf-8")

    assert (
        '[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]'
        in css
    )
    assert (
        '[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *'
        in css
    )
    assert "color: #dce7f8 !important" in css
    assert "-webkit-text-fill-color: #dce7f8 !important" in css
    assert '[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong' in css
    assert '[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em' in css
    assert "font-size: 1rem !important" in css
    assert "line-height: 1.6 !important" in css


def test_chat_bottom_and_input_use_dark_theme() -> None:
    css = Path("assets/styles.css").read_text(encoding="utf-8")

    assert '[data-testid="stBottom"]' in css
    assert '[data-testid="stBottom"] > div' in css
    assert "background: var(--bg) !important" in css
    assert '[data-testid="stChatInput"]' in css
    assert "background: #081326 !important" in css
    assert "border-radius: .5rem !important" in css
    assert '[data-baseweb="base-input"]' in css
    assert '[data-baseweb="textarea"] > div' in css
    assert "background-color: #081326 !important" in css
    assert '[data-testid="stChatInput"]:focus-within' in css
    assert "border-color: #477dff !important" in css
    assert '[data-baseweb="base-input"]:focus-within' in css
    assert '[data-testid="stChatInput"] textarea:focus-visible' in css
    assert "border-color: transparent !important" in css
    assert "outline: none !important" in css
    assert '[data-testid="stChatInput"] textarea::placeholder' in css
    assert "color: #8296b5 !important" in css
    assert '[data-testid="stChatInput"] button:hover' in css
