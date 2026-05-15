"""Shared theme management for light and dark themes.

Provides theme-aware stylesheet definitions, a global apply function,
and system theme detection (Qt 6.5+ QStyleHints with palette fallback).
"""

from typing import Literal, Optional

ThemeName = Literal["dark", "light"]

# ── Dark Theme Palette (current defaults) ──────────────────────────────────
DARK = {
    "bg_main": "#1e1e1e",
    "bg_panel": "#252526",
    "bg_graphics": "#2b2b2b",
    "border": "#3c3c3c",
    "title": "#007acc",
    "text": "#ffffff",
    "text_muted": "#cccccc",
    "button_bg": "#0e639c",
    "button_hover": "#1177bb",
    "button_pressed": "#0d538f",
    "button_disabled": "#4a4a4a",
    "button_disabled_text": "#888888",
    "input_bg": "#252526",
    "input_border": "#3c3c3c",
    "input_border_focus": "#007acc",
    "slider_bg": "#3c3c3c",
    "slider_handle": "#007acc",
    "slider_handle_hover": "#1a8beb",
    "slider_handle_border": "#005a9e",
    "combo_view_bg": "#252526",
    "combo_view_sel": "#0e639c",
    "hint_bg": "#1e1e1e",
    "status_text": "#cccccc",
    "status_border": "#3c3c3c",
}

# ── Light Theme Palette ────────────────────────────────────────────────────
LIGHT = {
    "bg_main": "#f3f3f3",
    "bg_panel": "#ffffff",
    "bg_graphics": "#e8e8e8",
    "border": "#c0c0c0",
    "title": "#0066cc",
    "text": "#333333",
    "text_muted": "#555555",
    "button_bg": "#0078d4",
    "button_hover": "#1a86d9",
    "button_pressed": "#0063b1",
    "button_disabled": "#d0d0d0",
    "button_disabled_text": "#999999",
    "input_bg": "#ffffff",
    "input_border": "#c0c0c0",
    "input_border_focus": "#0078d4",
    "slider_bg": "#c0c0c0",
    "slider_handle": "#0078d4",
    "slider_handle_hover": "#1a86d9",
    "slider_handle_border": "#005a9e",
    "combo_view_bg": "#ffffff",
    "combo_view_sel": "#0078d4",
    "hint_bg": "#f3f3f3",
    "status_text": "#555555",
    "status_border": "#c0c0c0",
}

PALETTES = {
    "dark": DARK,
    "light": LIGHT,
}


def detect_system_theme(app=None) -> ThemeName:
    """Detect the operating system's preferred theme color scheme.

    Uses a three-tier fallback approach:
    1. **QStyleHints.colorScheme** (Qt 6.5+) — native OS preference on
       Wayland, Windows 11, and macOS.  Returns ``None`` on older Qt
       versions or X11 systems without a detected scheme.
    2. **Palette lightness** — checks whether the application's current
       window palette is visually dark (lightness < 0.5).
    3. **Hard-coded default** — returns ``"dark"`` as the safe fallback.

    Parameters
    ----------
    app : QApplication, optional
        If provided the palette lightness check is used.  When ``None``
        only the QStyleHints path is attempted (returns ``"dark"`` if
        unavailable).

    Returns
    -------
    str
        ``"dark"`` or ``"light"``.
    """
    # --- Tier 1: QStyleHints (Qt 6.5+) ---
    try:
        from PySide6.QtCore import QStyleHints
        from PySide6.QtGui import QPalette

        style_hints = QStyleHints()
        scheme = style_hints.colorScheme  # type: ignore[attr-defined]

        if scheme is not None:
            if scheme == QPalette.ColorScheme.Dark:  # type: ignore[attr-defined]
                return "dark"
            elif scheme == QPalette.ColorScheme.Light:  # type: ignore[attr-defined]
                return "light"
    except Exception:
        pass

    # --- Tier 2: Palette lightness check ---
    if app is not None:
        try:
            palette = app.palette()
            window_color = palette.window()
            lightness = window_color.lightnessF()
            if lightness < 0.5:
                return "dark"
            else:
                return "light"
        except Exception:
            pass

    # --- Tier 3: Safe default ---
    return "dark"


def get_initial_theme(settings, app=None) -> ThemeName:
    """Determine the initial theme to apply on startup.

    Checks the saved user preference first; if no preference exists
    (first launch), falls back to system theme detection.

    Parameters
    ----------
    settings : QSettings
        The application QSettings instance.
    app : QApplication, optional
        Passed through to :func:`detect_system_theme`.

    Returns
    -------
    str
        ``"dark"`` or ``"light"``.
    """
    saved = settings.value(
        "ui/theme",
        defaultValue=None,
        type=str,
    )
    if saved in ("dark", "light"):
        return saved

    return detect_system_theme(app)


def _build_stylesheet(theme_name: str) -> str:
    """Build a complete Qt stylesheet string for the given theme name."""
    p = PALETTES[theme_name]

    return f"""
        QMainWindow {{
            background-color: {p['bg_main']};
        }}
        QGroupBox {{
            background-color: {p['bg_main']};
            color: {p['text']};
            border: 1px solid {p['border']};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 15px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {p['title']};
        }}
        QLabel {{
            color: {p['text']};
            background-color: {p['bg_main']};
        }}
        QPushButton {{
            background-color: {p['button_bg']};
            color: white;
            border: none;
            padding: 5px 12px;
            border-radius: 3px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {p['button_hover']};
        }}
        QPushButton:pressed {{
            background-color: {p['button_pressed']};
        }}
        QPushButton:disabled {{
            background-color: {p['button_disabled']};
            color: {p['button_disabled_text']};
        }}
        QComboBox {{
            background-color: {p['input_bg']};
            color: {p['text']};
            border: 1px solid {p['input_border']};
            border-radius: 3px;
            padding: 3px 6px;
        }}
        QComboBox:disabled {{
            background-color: {p['bg_main']};
            color: {p['button_disabled_text']};
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {p['combo_view_bg']};
            color: {p['text']};
            selection-background-color: {p['combo_view_sel']};
        }}
        QCheckBox {{
            color: {p['text']};
            background-color: {p['bg_main']};
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
        }}
        QSlider::groove:horizontal {{
            border: 1px solid {p['border']};
            height: 6px;
            background: {p['slider_bg']};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {p['slider_handle']};
            border: 1px solid {p['slider_handle_border']};
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {p['slider_handle_hover']};
        }}
        QStatusBar {{
            background-color: {p['bg_main']};
            color: {p['status_text']};
            border-top: 1px solid {p['status_border']};
        }}
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        #scrollContent {{
            background-color: {p['bg_main']};
        }}
        QGraphicsView {{
            background-color: {p['bg_graphics']};
            border: 1px solid {p['border']};
        }}
        QLineEdit {{
            background-color: {p['input_bg']};
            color: {p['text']};
            border: 1px solid {p['input_border']};
            border-radius: 3px;
            padding: 3px 6px;
        }}
        QLineEdit:focus {{
            border: 1px solid {p['input_border_focus']};
        }}
    """.strip()


# ── Global state ────────────────────────────────────────────────────────────
_current_theme: ThemeName = "dark"


def get_current_theme() -> ThemeName:
    """Return the currently active theme name."""
    return _current_theme


def set_app_theme(theme_name: ThemeName, app=None) -> str:
    """Apply a theme globally to the application.

    Parameters
    ----------
    theme_name : "dark" | "light"
        Name of the theme to apply.
    app : QApplication, optional
        If provided, the stylesheet is applied to the entire application
        via ``app.setStyleSheet(...)``.  When ``None`` the caller is
        expected to apply the returned string manually.

    Returns
    -------
    str
        The computed Qt stylesheet string (useful for debugging / testing).
    """
    global _current_theme
    if theme_name not in PALETTES:
        raise ValueError(f"Unknown theme '{theme_name}'. Available: {list(PALETTES.keys())}")
    _current_theme = theme_name
    style = _build_stylesheet(theme_name)
    if app is not None:
        app.setStyleSheet(style)
    return style