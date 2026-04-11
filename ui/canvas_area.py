"""Orta tuval — giriş/çıkış resimleri yan yana, mouse seçimi destekli."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import cv2
import numpy as np


def ndarray_to_pixmap(img: np.ndarray) -> QPixmap:
    """OpenCV BGR → QPixmap"""
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ImageCanvas(QLabel):
    """Tek resim alanı — seçim desteği opsiyonel."""
    selection_changed = pyqtSignal(QRect)

    def __init__(self, allow_selection=False, parent=None):
        super().__init__(parent)
        self.setObjectName("image_slot")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 160)
        self.setText("Resim yüklenmedi")
        self.setStyleSheet("color: #45475a; font-size: 13px;")

        self._allow_selection = allow_selection
        self._orig_img: np.ndarray | None = None
        self._pixmap_base: QPixmap | None = None
        self._sel_start: QPoint | None = None
        self._sel_rect: QRect | None = None
        self._selecting = False

        if allow_selection:
            self.setCursor(Qt.CrossCursor)

    # ── Public ──────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray | None):
        self._orig_img = img
        self._sel_rect = None
        if img is None:
            self._pixmap_base = None
            self.setText("Resim yüklenmedi")
            self.setPixmap(QPixmap())
        else:
            self.setText("")
            self._pixmap_base = ndarray_to_pixmap(img)
            self._refresh_display()

    def get_selection_pixels(self) -> tuple | None:
        """Canvas koordinatlarını orijinal resim koordinatlarına çevirir."""
        if self._sel_rect is None or self._orig_img is None:
            return None
        pm = self._pixmap_base
        if pm is None:
            return None

        # Resim, label içinde nasıl konumlandı?
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        scale = min(lw / pw, lh / ph)
        disp_w, disp_h = int(pw * scale), int(ph * scale)
        off_x = (lw - disp_w) // 2
        off_y = (lh - disp_h) // 2

        r = self._sel_rect
        x1 = max(0, int((r.left()   - off_x) / scale))
        y1 = max(0, int((r.top()    - off_y) / scale))
        x2 = min(self._orig_img.shape[1], int((r.right()  - off_x) / scale))
        y2 = min(self._orig_img.shape[0], int((r.bottom() - off_y) / scale))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def clear_selection(self):
        self._sel_rect = None
        self._refresh_display()

    def enable_selection(self, enable: bool):
        self._allow_selection = enable
        self.setCursor(Qt.CrossCursor if enable else Qt.ArrowCursor)
        if not enable:
            self.clear_selection()

    # ── Mouse ────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._allow_selection and event.button() == Qt.LeftButton:
            self._sel_start = event.pos()
            self._sel_rect = QRect(self._sel_start, self._sel_start)
            self._selecting = True

    def mouseMoveEvent(self, event):
        if self._selecting and self._sel_start:
            self._sel_rect = QRect(self._sel_start, event.pos()).normalized()
            self._refresh_display()

    def mouseReleaseEvent(self, event):
        if self._selecting:
            self._selecting = False
            if self._sel_rect:
                self.selection_changed.emit(self._sel_rect)

    # ── Paint ────────────────────────────────────────────────────────

    def _refresh_display(self):
        if self._pixmap_base is None:
            return
        pm = self._pixmap_base.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if self._sel_rect:
            pm = pm.copy()
            painter = QPainter(pm)
            pen = QPen(QColor("#89b4fa"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(137, 180, 250, 30))

            # Seçimi görüntülenen pixmap boyutuna taşı
            lw, lh = self.width(), self.height()
            pw, ph = pm.width(), pm.height()
            off_x = (lw - pw) // 2
            off_y = (lh - ph) // 2
            shifted = self._sel_rect.translated(-off_x, -off_y)
            painter.drawRect(shifted)
            painter.end()
        self.setPixmap(pm)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_display()


class CanvasArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvas_frame")
        self._active_tool = ""
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Giriş
        in_box = QWidget(); in_box.setObjectName("image_slot")
        in_layout = QVBoxLayout(in_box)
        in_layout.setContentsMargins(0, 0, 0, 0)
        in_layout.setSpacing(0)
        in_title = QLabel("📥  Giriş"); in_title.setObjectName("slot_title")
        self.canvas_in = ImageCanvas(allow_selection=False)
        in_layout.addWidget(in_title)
        in_layout.addWidget(self.canvas_in)
        root.addWidget(in_box)

        # Ok
        arrow = QLabel("→")
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setFixedWidth(28)
        arrow.setStyleSheet("color: #45475a; font-size: 20px;")
        root.addWidget(arrow)

        # Çıkış
        out_box = QWidget(); out_box.setObjectName("image_slot")
        out_layout = QVBoxLayout(out_box)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.setSpacing(0)
        out_title = QLabel("📤  Çıkış"); out_title.setObjectName("slot_title")
        self.canvas_out = ImageCanvas(allow_selection=False)
        out_layout.addWidget(out_title)
        out_layout.addWidget(self.canvas_out)
        root.addWidget(out_box)

    # ── Public ──────────────────────────────────────────────────────

    def show_images(self, src, dst):
        self.canvas_in.set_image(src)
        self.canvas_out.set_image(dst)

    def set_tool(self, tool_name: str):
        self._active_tool = tool_name
        select_tools = {"Görüntü Kırpma", "Görüntüye Gürültü Ekleme"}
        self.canvas_in.enable_selection(tool_name in select_tools)

    def get_selection(self):
        return self.canvas_in.get_selection_pixels()
