"""Sağ panel — aktif araca göre dinamik parametre arayüzü."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QCheckBox, QScrollArea,
    QHBoxLayout, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal

TOOL_PARAMS: dict[str, list[dict]] = {
    "Gri Dönüşüm": [],

    "Binary Dönüşüm": [
        {"label": "Yöntem", "key": "method", "type": "combo",
         "options": ["Eşik (Threshold)", "Otsu"], "default": "Otsu"},
        {"label": "Eşik değeri (Eşik için)", "key": "threshold", "type": "slider", "min": 0, "max": 255, "default": 127, "display": "raw"},
    ],

    "Renk Uzayı Dönüşümleri": [
        {"label": "Dönüşüm", "key": "colorspace", "type": "combo",
         "options": ["RGB'den NTSC'ye", "NTSC'den RGB'ye"], "default": "RGB'den NTSC'ye"},
    ],

    "Görüntü Döndürme": [
        {"label": "Açı (°)", "key": "angle", "type": "slider", "min": 0, "max": 360, "default": 90, "display": "raw"},
    ],

    "Görüntü Kırpma": [
        {"label": None, "key": "_info", "type": "info",
         "text": "Giriş tuvali üzerinde\nmouse ile alan seçin,\nsonra Uygula'ya basın."},
    ],

    "Yaklaştırma / Uzaklaştırma": [
        {"label": "Oran (x)", "key": "scale", "type": "slider", "min": 1, "max": 30, "default": 10},
    ],

    "Kontrast/Parlaklık": [
        {"label": "Parlaklık",  "key": "beta",  "type": "slider", "min": -255, "max": 255, "default": 0, "display": "raw"},
        {"label": "Kontrast",   "key": "alpha", "type": "slider", "min": 1,    "max": 30,  "default": 10},
    ],

    "Histogram & Germe": [
        {"label": "Alt kesim (0-255)",  "key": "p_low",  "type": "slider", "min": 0, "max": 255, "default": 0,   "display": "raw"},
        {"label": "Üst kesim (0-255)", "key": "p_high", "type": "slider", "min": 0, "max": 255, "default": 255, "display": "raw"},
        {"label": "Histogram göster", "key": "show_histogram", "type": "check", "default": True},
    ],

    "Konvolüsyon İşlemi (Gauss)": [
        {"label": "Kernel boyutu (tek)", "key": "ksize", "type": "combo",
         "options": ["3","5","7","9","11","13","15","17","19","21"], "default": "5"},
        {"label": "Sigma",              "key": "sigma", "type": "slider", "min": 1, "max": 200, "default": 10},
    ],

    "Blurring": [
        {"label": "Disk Kernel Yarıçapı", "key": "radius", "type": "slider", "min": 3, "max": 50, "default": 5, "display": "raw"},
    ],

    "Kenar Bulma (Sobel)": [
        {"label": "Yöntem", "key": "method", "type": "combo",
         "options": ["Sobel XY","Sobel X","Sobel Y"],                         "default": "Sobel XY"},
    ],

    "Filtreler": [
        {"label": "Filtre tipi",    "key": "filter_type", "type": "combo",
         "options": ["Mean Filtre", "Median Filtre"], "default": "Mean Filtre"},
        {"label": "Kernel boyutu", "key": "ksize",       "type": "combo",
         "options": ["3","5","7","9","11","13","15","17","19","21"], "default": "3"},
    ],

    "Görüntüye Gürültü Ekleme": [
        {"label": "Yoğunluk (%)", "key": "amount", "type": "slider", "min": 1, "max": 50, "default": 5},
        {"label": None, "key": "_info2", "type": "info",
         "text": "Mouse ile bölge seçerek\nsadece o alana gürültü\neklenebilir."},
    ],

    "Aritmetik İşlemler": [
        {"label": "İşlem tipi", "key": "operation", "type": "combo",
         "options": ["Ekleme (Add)","Çarpma (Multiply)"],
         "default": "Ekleme (Add)"},
    ],

    "Morfolojik İşlemler": [
        {"label": "İşlem", "key": "operation", "type": "combo",
         "options": ["Genişleme (Dilate)","Aşınma (Erode)","Açma (Opening)","Kapama (Closing)"],
         "default": "Genişleme (Dilate)"},
        {"label": "Kernel tipi",   "key": "kernel_shape",  "type": "combo",
         "options": ["Dikdörtgen","Elips"], "default": "Dikdörtgen"},
        {"label": "Kernel boyutu", "key": "ksize",         "type": "slider", "min": 3, "max": 20, "default": 3, "display": "raw"},
        {"label": "Elips Yarıçapı","key": "elips_radius",  "type": "slider", "min": 3, "max": 50, "default": 5, "display": "raw"},
    ],
}


class RightPanel(QScrollArea):
    apply_requested = pyqtSignal(str, dict)
    undo_requested  = pyqtSignal()
    load_second_image_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("right_panel")
        self.setFixedWidth(225)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        self._widgets: dict[str, QWidget] = {}
        self._current_tool = ""

        self._container = QWidget()
        self._container.setObjectName("right_panel")
        self._vbox = QVBoxLayout(self._container)
        self._vbox.setContentsMargins(0, 0, 0, 12)
        self._vbox.setSpacing(0)
        self.setWidget(self._container)

        self._show_placeholder()

    # ── Build ────────────────────────────────────────────────────────

    def _clear(self):
        while self._vbox.count():
            item = self._vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()

    def _show_placeholder(self):
        self._clear()
        lbl = QLabel("Bir araç seçin")
        lbl.setObjectName("info_label")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #45475a; padding: 40px;")
        self._vbox.addWidget(lbl)
        self._vbox.addStretch()

    def show_params(self, tool_name: str):
        self._clear()
        self._current_tool = tool_name
        params = TOOL_PARAMS.get(tool_name, [])

        # Başlık
        title = QLabel(tool_name)
        title.setObjectName("param_title")
        title.setWordWrap(True)
        self._vbox.addWidget(title)

        # Ayraç
        sep = QFrame(); sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine); sep.setFixedHeight(1)
        self._vbox.addWidget(sep)

        # Parametreler
        for p in params:
            self._add_param(p)

        # Aritmetik İşlemler için ikinci resim yükleme butonu
        if tool_name == "Aritmetik İşlemler":
            sep = QFrame(); sep.setObjectName("separator")
            sep.setFrameShape(QFrame.HLine); sep.setFixedHeight(1)
            self._vbox.addWidget(sep)
            
            load_second_btn = QPushButton("📂  İkinci Resim Seç")
            load_second_btn.setObjectName("secondary_btn")
            load_second_btn.setCursor(Qt.PointingHandCursor)
            load_second_btn.clicked.connect(self.load_second_image_requested.emit)
            self._vbox.addWidget(load_second_btn)

        self._vbox.addStretch()

        # Uygula
        apply_btn = QPushButton("✅  Uygula")
        apply_btn.setObjectName("apply_btn")
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.clicked.connect(self._emit_apply)
        self._vbox.addWidget(apply_btn)

    def _add_param(self, p: dict):
        ptype = p["type"]

        if ptype == "info":
            lbl = QLabel(p["text"])
            lbl.setObjectName("info_label")
            lbl.setWordWrap(True)
            self._vbox.addWidget(lbl)
            return

        if p.get("label"):
            lbl = QLabel(p["label"])
            lbl.setObjectName("param_label")
            self._vbox.addWidget(lbl)

        if ptype == "slider":
            row = QWidget()
            row.setObjectName("right_panel")
            hl = QHBoxLayout(row)
            hl.setContentsMargins(10, 0, 10, 0)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(p["min"])
            slider.setMaximum(p["max"])
            slider.setValue(p["default"])

            if p.get("display") == "raw":
                val_lbl = QLabel(str(p["default"]))
                slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            else:
                val_lbl = QLabel(f"{p['default']/10:.1f}")
                slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(f"{v/10:.1f}"))
            val_lbl.setFixedWidth(34)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("color: #89b4fa; font-size: 12px;")
            hl.addWidget(slider)
            hl.addWidget(val_lbl)
            self._vbox.addWidget(row)
            self._widgets[p["key"]] = slider

        elif ptype == "combo":
            cb = QComboBox()
            cb.addItems(p["options"])
            cb.setCurrentText(p["default"])
            self._vbox.addWidget(cb)
            self._widgets[p["key"]] = cb

        elif ptype == "check":
            chk = QCheckBox("Açık")
            chk.setChecked(p["default"])
            chk.setContentsMargins(12, 0, 0, 0)
            self._vbox.addWidget(chk)
            self._widgets[p["key"]] = chk

    # ── Collect & Emit ───────────────────────────────────────────────

    def _emit_apply(self):
        params = {}
        for key, w in self._widgets.items():
            if isinstance(w, QSlider):
                if key in ("scale", "alpha", "sigma"):
                    params[key] = w.value() / 10.0
                else:
                    params[key] = w.value()
            elif isinstance(w, QComboBox):
                params[key] = w.currentText()
            elif isinstance(w, QCheckBox):
                params[key] = w.isChecked()
        self.apply_requested.emit(self._current_tool, params)
