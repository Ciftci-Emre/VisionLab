import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage

# Projenin diğer bileşenleri
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
        self.history:        list = []
        self.active_tool:    str = ""

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Üst Bar
        root.addWidget(self._build_topbar())

        # Ana Alan
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
        layout.addStretch()

        btns = [
            ("📂  Aç",       "topbar_btn_accent", self.load_image),
            ("💾  Kaydet",   "topbar_btn",        self.save_image),
            ("✓  Resmi Al",  "topbar_btn",        self.accept_output),
            ("↩  Geri Al",   "topbar_btn",        self.undo),
            ("🔄  Sıfırla",  "topbar_btn",        self.reset_image),
        ]

        for text, name, slot in btns:
            btn = QPushButton(text)
            btn.setObjectName(name)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        return bar

    def _connect_signals(self):
        self.left_panel.tool_selected.connect(self.on_tool_select)
        self.right_panel.apply_requested.connect(self.apply_tool)
        self.right_panel.load_second_image_requested.connect(self.load_second_image)

    # ── Yardımcı Dönüştürücü ────────────────────────────────────────

    def qimage_to_numpy(self, qimg: QImage) -> np.ndarray:
        """QImage nesnesini güvenli bir şekilde NumPy dizisine çevirir."""
        qimg = qimg.convertToFormat(QImage.Format_RGB888)
    
        width = qimg.width()
        height = qimg.height()
        bytes_per_line = qimg.bytesPerLine()
    
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
    
        arr = np.array(ptr, copy=True)
        arr = arr.reshape(height, bytes_per_line)
        arr = arr[:, :width * 3]
        arr = arr.reshape(height, width, 3)

        # RENK DÜZELTME: Eğer renkler tersse (RGB -> BGR takası)
        # Dizinin son boyutunu (kanalları) ters çeviriyoruz [:, :, ::-1]
        return arr[:, :, ::-1].copy()

    def numpy_to_qimage(self, arr: np.ndarray) -> QImage:
        """NumPy dizisini PyQt'de göstermek veya kaydetmek için QImage'e çevirir."""
        # Gri resimse (tek kanal) boyut hatası almamak için 3 kanala genişlet
        if len(arr.shape) == 2:
            rgb = np.stack((arr, arr, arr), axis=-1)
        else:
            # Algoritmalarımız BGR çalışıyor, kaydederken tekrar RGB'ye döndürmeliyiz!
            rgb = arr[:, :, ::-1].copy()
            
        height, width, channel = rgb.shape
        bytes_per_line = 3 * width
        return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

    # ── Resim İşlemleri ─────────────────────────────────────────────

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resimler (*.png *.jpg *.bmp)")
        if not path: return
        
        qimg = QImage(path)
        if qimg.isNull():
            QMessageBox.critical(self, "Hata", "Resim dosyası okunamadı.")
            return
        
        self.original_image = self.qimage_to_numpy(qimg)
        self.current_image = self.original_image.copy()
        self.output_image = None
        self.history.clear()
        self._refresh_view()

    def save_image(self):
        img_arr = self.output_image if self.output_image is not None else self.current_image
        if img_arr is None: return

        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "", "PNG (*.png);;JPG (*.jpg)")
        if path:
            qimg = self.numpy_to_qimage(img_arr)
            qimg.save(path)
            QMessageBox.information(self, "Bilgi", "Başarıyla kaydedildi.")

    def load_second_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "İkinci Resim", "", "Resimler (*.png *.jpg)")
        if path:
            qimg = QImage(path)
            self.second_image = self.qimage_to_numpy(qimg)
            QMessageBox.information(self, "Bilgi", "İkinci resim hafızaya alındı.")

    def undo(self):
        if self.history:
            prev_in, _ = self.history.pop()
            self.current_image = prev_in
            self.output_image = None
            self._refresh_view()

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.output_image = None
            self.history.clear()
            self._refresh_view()

    def accept_output(self):
        if self.output_image is not None:
            self.history.append((self.current_image.copy(), self.output_image.copy()))
            self.current_image = self.output_image.copy()
            self.output_image = None
            self._refresh_view()

    # ── Araçlar ─────────────────────────────────────────────────────

    def on_tool_select(self, tool_name: str):
        self.active_tool = tool_name
        self.right_panel.show_params(tool_name)
        self.canvas_area.set_tool(tool_name)

    def apply_tool(self, tool_name: str, params: dict):
        if self.current_image is None: return

        # Kendi yazdığın algoritmaların olduğu dosya
        from tools.methods import registry, resim_ekleme, resim_carpma
        
        try:
            if tool_name == "Aritmetik İşlemler":
                if self.second_image is None:
                    QMessageBox.warning(self, "Uyarı", "Lütfen önce ikinci resmi yükleyin.")
                    return
                op = params.get("operation", "Ekleme (Add)")
                # Kendi toplama/çarpma fonksiyonlarını çağır
                if op == "Ekleme (Add)":
                    self.output_image = resim_ekleme(self.current_image, self.second_image)
                else:
                    self.output_image = resim_carpma(self.current_image, self.second_image)
            
            elif tool_name in registry:
                # Kırpma alanı seçilmişse parametrelere ekle
                if tool_name in ("Görüntü Kırpma", "Görüntüye Gürültü Ekleme"):
                    sel = self.canvas_area.get_selection()
                    if sel: params["x1"], params["y1"], params["x2"], params["y2"] = sel
                
                # İlgili fonksiyonu registry'den bul ve çalıştır
                fn = registry.get(tool_name)
                self.output_image = fn(self.current_image.copy(), params)
            
            self._refresh_view()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Algoritma hatası: {str(e)}")

    def _refresh_view(self):
        self.canvas_area.show_images(self.current_image, self.output_image)
        self.statusbar_widget.update_info(self.current_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())