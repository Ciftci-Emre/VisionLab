"""Koyu tema — tüm uygulama genelinde kullanılan QSS."""

DARK_STYLE = """
/* ── Genel ── */
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #1e1e2e;
}

/* ── Üst bar ── */
#topbar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
}

/* ── Sol panel ── */
#left_panel {
    background-color: #181825;
    border-right: 1px solid #313244;
}

#category_label {
    color: #6c7086;
    font-size: 10px;
    font-weight: bold;
    padding: 12px 14px 4px 14px;
    letter-spacing: 1px;
}

QPushButton#tool_btn {
    background-color: transparent;
    color: #cdd6f4;
    border: none;
    border-left: 3px solid transparent;
    text-align: left;
    padding: 8px 14px;
    font-size: 13px;
    border-radius: 0px;
}

QPushButton#tool_btn:hover {
    background-color: #313244;
}

QPushButton#tool_btn[active="true"] {
    background-color: #2a2a3d;
    border-left: 3px solid #89b4fa;
    color: #89b4fa;
}

/* ── Sağ panel ── */
#right_panel {
    background-color: #181825;
    border-left: 1px solid #313244;
}

#param_title {
    font-size: 13px;
    font-weight: bold;
    color: #cdd6f4;
    padding: 12px 12px 6px 12px;
}

#param_label {
    color: #a6adc8;
    font-size: 11px;
    padding: 6px 12px 2px 12px;
}

#info_label {
    color: #6c7086;
    font-size: 11px;
    padding: 6px 12px;
    line-height: 1.6;
}

/* ── Slider ── */
QSlider::groove:horizontal {
    height: 4px;
    background: #313244;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: #89b4fa;
    border-radius: 7px;
}

QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 2px;
}

/* ── ComboBox ── */
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 10px;
    color: #cdd6f4;
    margin: 2px 10px;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #89b4fa;
    width: 0;
    height: 0;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
    color: #cdd6f4;
    outline: none;
}

/* ── CheckBox ── */
QCheckBox {
    padding: 4px 12px;
    color: #cdd6f4;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #45475a;
    border-radius: 4px;
    background: #313244;
}

QCheckBox::indicator:checked {
    background: #89b4fa;
    border-color: #89b4fa;
}

/* ── Butonlar ── */
QPushButton#apply_btn {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 7px;
    padding: 9px;
    font-weight: bold;
    font-size: 13px;
    margin: 6px 10px 3px 10px;
}

QPushButton#apply_btn:hover {
    background-color: #b4d0fb;
}

QPushButton#apply_btn:pressed {
    background-color: #74a8f9;
}

QPushButton#secondary_btn {
    background-color: transparent;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-radius: 7px;
    padding: 7px;
    margin: 0 10px 6px 10px;
}

QPushButton#secondary_btn:hover {
    background-color: #313244;
    color: #cdd6f4;
}

/* ── Üst bar butonları ── */
QPushButton#topbar_btn {
    background-color: #313244;
    color: #cdd6f4;
    border: none;
    border-radius: 7px;
    padding: 7px 16px;
    font-size: 13px;
}

QPushButton#topbar_btn:hover {
    background-color: #45475a;
}

QPushButton#topbar_btn:pressed {
    background-color: #585b70;
}

QPushButton#topbar_btn_accent {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 7px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton#topbar_btn_accent:hover {
    background-color: #b4d0fb;
}

/* ── Canvas ── */
#canvas_frame {
    background-color: #1e1e2e;
}

#image_slot {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 10px;
}

#slot_title {
    color: #6c7086;
    font-size: 11px;
    padding: 6px 10px 0 10px;
}

/* ── Status bar ── */
#statusbar {
    background-color: #11111b;
    border-top: 1px solid #313244;
    color: #6c7086;
    font-size: 11px;
    padding: 4px 14px;
}

#status_item {
    color: #6c7086;
    font-size: 11px;
    padding: 0 12px;
}

/* ── Scroll bar ── */
QScrollBar:vertical {
    background: #1e1e2e;
    width: 6px;
    border-radius: 3px;
}

QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 3px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #585b70;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── Splitter ── */
QSplitter::handle {
    background: #313244;
    width: 1px;
}

/* ── Separator ── */
#separator {
    background-color: #313244;
    max-height: 1px;
    margin: 4px 8px;
}
"""
