"""Sol panel — araç listesi."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt

TOOLS = [
    ("🎨", "Gri Dönüşüm"),
    ("⬛", "Binary Dönüşüm"),
    ("🌈", "Renk Uzayı Dönüşümleri"),
    ("↻", "Görüntü Döndürme"),
    ("✂", "Görüntü Kırpma"),
    ("🔍", "Yaklaştırma / Uzaklaştırma"),
    ("☀", "Parlaklık Artırma"),
    ("📊", "Histogram & Germe"),
    ("〰", "Konvolüsyon İşlemi (Gauss)"),
    ("◎", "Blurring"),
    ("△", "Kenar Bulma (Sobel)"),
    ("⊥", "Eşikleme (Adaptif)"),
    ("※", "Görüntüye Gürültü Ekleme"),
    ("±", "Aritmetik İşlemler"),
    ("⬡", "Morfolojik İşlemler"),
]


class LeftPanel(QScrollArea):
    tool_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("left_panel")
        self.setFixedWidth(215)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        self._buttons: dict[str, QPushButton] = {}
        self._active: str = ""

        container = QWidget()
        container.setObjectName("left_panel")
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(0, 6, 0, 12)
        self._layout.setSpacing(0)

        self._build()
        self._layout.addStretch()
        self.setWidget(container)

    def _build(self):
        for icon, name in TOOLS:
            btn = QPushButton(f"  {icon}  {name}")
            btn.setObjectName("tool_btn")
            btn.setProperty("active", "false")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, n=name: self._select(n))
            self._layout.addWidget(btn)
            self._buttons[name] = btn

    def _select(self, name: str):
        if self._active and self._active in self._buttons:
            self._buttons[self._active].setProperty("active", "false")
            self._buttons[self._active].style().unpolish(self._buttons[self._active])
            self._buttons[self._active].style().polish(self._buttons[self._active])

        self._buttons[name].setProperty("active", "true")
        self._buttons[name].style().unpolish(self._buttons[name])
        self._buttons[name].style().polish(self._buttons[name])
        self._active = name
        self.tool_selected.emit(name)
