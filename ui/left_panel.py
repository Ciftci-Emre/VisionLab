"""Sol panel — kategorilere ayrılmış araç listesi."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt

TOOLS = [
    ("DÖNÜŞÜMLER", [
        ("🎨", "Gri Dönüşüm"),
        ("⬛", "Binary Dönüşüm"),
        ("🌈", "Renk Uzayı Dönüşümleri"),
    ]),
    ("GEOMETRİ", [
        ("↻",  "Görüntü Döndürme"),
        ("✂",  "Görüntü Kırpma"),
        ("🔍", "Yaklaştırma / Uzaklaştırma"),
    ]),
    ("PARLAKLIK & KONTRAST", [
        ("☀",  "Parlaklık Artırma"),
        ("📊", "Histogram & Germe"),
    ]),
    ("FİLTRELER", [
        ("〰", "Konvolüsyon İşlemi (Gauss)"),
        ("◎",  "Blurring"),
        ("△",  "Kenar Bulma (Sobel)"),
        ("⊥",  "Eşikleme (Adaptif)"),
    ]),
    ("GÜRÜLTÜ & ARİTMETİK", [
        ("※",  "Görüntüye Gürültü Ekleme"),
        ("±",  "Aritmetik İşlemler"),
    ]),
    ("MORFOLOJİ", [
        ("⬡",  "Morfolojik İşlemler"),
    ]),
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
        for category, tools in TOOLS:
            cat_lbl = QLabel(category)
            cat_lbl.setObjectName("category_label")
            self._layout.addWidget(cat_lbl)

            for icon, name in tools:
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
