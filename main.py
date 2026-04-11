"""
Görüntü İşleme Stüdyosu — PyQt5
=================================
Kurulum:
    pip install PyQt5 opencv-python pillow numpy

Çalıştırma:
    python main.py
"""

import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.theme import DARK_STYLE
from ui.left_panel import LeftPanel
from ui.right_panel import RightPanel
from ui.canvas_area import CanvasArea
from ui.statusbar import StatusBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionLab")
        self.resize(1280, 780)
        self.setMinimumSize(960, 620)

        # ── Uygulama durumu ─────────────────────────────────────────
        self.original_image: np.ndarray | None = None
        self.current_image:  np.ndarray | None = None
        self.output_image:   np.ndarray | None = None
        self.second_image:   np.ndarray | None = None
        self.history:        list[np.ndarray]  = []
        self.active_tool:    str = ""

        self._build_ui()
        self._connect_signals()

    # ── UI Kurulumu ──────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Üst bar
        root.addWidget(self._build_topbar())

        # Ana alan
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.left_panel  = LeftPanel()
        self.canvas_area = CanvasArea()
        self.right_panel = RightPanel()

        body_layout.addWidget(self.left_panel)
        body_layout.addWidget(self.canvas_area, stretch=1)

        body_layout.addWidget(self.right_panel)
        root.addWidget(body, stretch=1)

        # Alt durum çubuğu
        self.statusbar_widget = StatusBar()
        root.addWidget(self.statusbar_widget)

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topbar")
        bar.setFixedHeight(52)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(8)

        title = QLabel("VisionLab")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(title)

        tag = QLabel("PyQt5")
        tag.setStyleSheet(
            "color: #89b4fa; background: #2a2a3d; border-radius: 5px;"
            "padding: 2px 8px; font-size: 11px;"
        )
        layout.addWidget(tag)
        layout.addStretch()

        for text, name, slot in [
            ("📂  Aç",       "topbar_btn_accent", self.load_image),
            ("💾  Kaydet",   "topbar_btn",        self.save_image),
            ("✓  Resmi Al", "topbar_btn",    self.accept_output),
            ("↩  Geri Al",  "topbar_btn",        self.undo),
            ("🔄  Sıfırla", "topbar_btn",        self.reset_image),
        ]:
            btn = QPushButton(text)
            btn.setObjectName(name)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        return bar

    # ── Bağlantılar ─────────────────────────────────────────────────

    def _connect_signals(self):
        self.left_panel.tool_selected.connect(self.on_tool_select)
        self.right_panel.apply_requested.connect(self.apply_tool)
        self.right_panel.undo_requested.connect(self.undo)
        self.right_panel.load_second_image_requested.connect(self.load_second_image)

    # ── Resim İşlemleri ─────────────────────────────────────────────

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Resim Seç", "",
            "Resim Dosyaları (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;Tümü (*)"
        )
        if not path:
            return
        # Türkçe karakterler ve boşluk içeren yollarla çalışmak için
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Hata", "Resim yüklenemedi.")
            return

        self.original_image = img.copy()
        self.current_image  = img.copy()
        self.output_image   = None
        self.history.clear()
        self._refresh_view()

    def save_image(self):
        img = self.output_image if self.output_image is not None else self.current_image
        if img is None:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek resim yok.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Kaydet", "", "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            # Türkçe karakterler ve boşluk içeren yollarla çalışmak için
            _, ext = path.rsplit('.', 1)
            ext = '.' + ext
            success, buffer = cv2.imencode(ext, img)
            if success:
                buffer.tofile(path)
                QMessageBox.information(self, "Kaydedildi", f"Resim kaydedildi:\n{path}")
            else:
                QMessageBox.critical(self, "Hata", "Resim kaydedilemedi.")

    def undo(self):
        if not self.history:
            self.output_image = None
            self._refresh_view()
            return
        self.output_image = self.history.pop()
        self._refresh_view()

    def reset_image(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.output_image  = None
        self.second_image  = None
        self.history.clear()
        self._refresh_view()

    def load_second_image(self):
        """Aritmetik işlemler için ikinci resmi yükle."""
        path, _ = QFileDialog.getOpenFileName(
            self, "İkinci Resim Seç", "",
            "Resim Dosyaları (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;Tümü (*)"
        )
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Hata", "Resim yüklenemedi.")
            return
        self.second_image = img.copy()
        QMessageBox.information(self, "Bilgi", "İkinci resim yüklendi. Şimdi 'Uygula' tuşuna basın.")

    def accept_output(self):
        """Çıkış resmini girdi resmi olarak kabul et."""
        if self.output_image is None:
            QMessageBox.warning(self, "Uyarı", "Çıkış resmi yok.")
            return
        
        self.current_image = self.output_image.copy()
        self.output_image = None
        self.history.clear()
        self._refresh_view()
        QMessageBox.information(self, "Bilgi", "Çıkış resmi girdi resmi olarak ayarlandı.")

    # ── Araç Seçimi & Uygulama ──────────────────────────────────────

    def on_tool_select(self, tool_name: str):
        self.active_tool = tool_name
        self.right_panel.show_params(tool_name)
        self.canvas_area.set_tool(tool_name)
        self.statusbar_widget.update_info(self.current_image, tool_name)

    def apply_tool(self, tool_name: str, params: dict):
        if self.current_image is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir resim yükleyin.")
            return

        # Aritmetik işlemler için ikinci resmi kontrol et
        if tool_name == "Aritmetik İşlemler":
            if self.second_image is None:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce ikinci resmi yükleyin.")
                return
            
            from tools.methods import resim_ekleme, resim_carpma
            operation = params.get("operation", "Ekleme (Add)")
            
            try:
                if operation == "Ekleme (Add)":
                    result = resim_ekleme(self.current_image, self.second_image)
                elif operation == "Çarpma (Multiply)":
                    result = resim_carpma(self.current_image, self.second_image)
                else:
                    QMessageBox.warning(self, "Hata", f"Bilinmeyen işlem: {operation}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "İşlem Hatası", str(e))
                return
            
            # Önceki çıkışı history'ye ekle
            if self.output_image is not None:
                self.history.append(self.output_image.copy())
            
            self.output_image = result
            self._refresh_view()
            return

        # Kırpma için seçim koordinatlarını params'a ekle
        if tool_name in ("Görüntü Kırpma", "Görüntüye Gürültü Ekleme"):
            sel = self.canvas_area.get_selection()
            if sel:
                params["x1"], params["y1"], params["x2"], params["y2"] = sel

        from tools.methods import registry
        
        if tool_name not in registry:
            QMessageBox.information(self, "Bilgi", "Bu özellik henüz aktive edilmemiştir.")
            return

        fn = registry.get(tool_name)
        if fn is None:
            QMessageBox.warning(self, "Bilinmeyen Araç", f"'{tool_name}' henüz eklenmedi.")
            return

        try:
            result = fn(self.current_image.copy(), params)
        except Exception as e:
            QMessageBox.critical(self, "İşlem Hatası", str(e))
            return

        # Önceki çıkışı history'ye ekle
        if self.output_image is not None:
            self.history.append(self.output_image.copy())
        
        # Yalnızca çıkış resmi değişir, giriş resmi aynı kalır
        self.output_image = result
        self._refresh_view()

    # ── Yenile ──────────────────────────────────────────────────────

    def _refresh_view(self):
        self.canvas_area.show_images(self.current_image, self.output_image)
        self.statusbar_widget.update_info(self.current_image, self.active_tool)


# ── Giriş noktası ────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
