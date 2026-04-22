"""Alt durum çubuğu."""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
import numpy as np


class StatusBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusbar")
        self.setFixedHeight(28)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 14, 0)
        layout.setSpacing(0)

        self._size   = self._item("Boyut: —")
        self._ch     = self._item("Kanal: —")
        self._dtype  = self._item("Tip: —")

        layout.addWidget(self._size)
        layout.addWidget(self._sep())
        layout.addWidget(self._ch)
        layout.addWidget(self._sep())
        layout.addWidget(self._dtype)
        layout.addStretch()



    def _item(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("status_item")
        return lbl

    def _sep(self) -> QLabel:
        s = QLabel("|")
        s.setObjectName("status_item")
        s.setStyleSheet("color: #313244;")
        return s

    def update_info(self, img: np.ndarray | None):
        if img is None:
            self._size.setText("Boyut: —")
            self._ch.setText("Kanal: —")
            self._dtype.setText("Tip: —")
        else:
            h, w = img.shape[:2]
            ch = 1 if img.ndim == 2 else img.shape[2]
            self._size.setText(f"Boyut: {w}×{h}")
            self._ch.setText(f"Kanal: {ch}")
            self._dtype.setText(f"Tip: {img.dtype}")
