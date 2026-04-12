"""Görüntü işleme metodları."""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def gri_donusum(img: np.ndarray, params: dict) -> np.ndarray:
    """Renk görüntüsünü gri (grayscale) görüntüye dönüştür.
    
    Formül: Gray = 0.229*R + 0.587*G + 0.114*B
    (OpenCV BGR sırasında: 0.114*B + 0.587*G + 0.229*R)
    """
    # BGR kanallarını ayır
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    
    # Manuel formülü uygula - tek kanal olarak döndür
    gray = (0.114 * b + 0.587 * g + 0.229 * r).astype(np.uint8)
    
    return gray


def resim_ekleme(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """İki resmin pixellerini toplayarak ekleme işlemi yap.
    
    - 3 kanallı + 3 kanallı: Kanal başına toplama (R+R, G+G, B+B)
    - 3 kanallı + 1 kanallı: Her kanala 1 kanallı değer ekleme (R+değer, G+değer, B+değer)
    - 1 kanallı + 1 kanallı: Normal toplama
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Boyutları eşitleme - daha küçük boyutu kullan
    h = min(h1, h2)
    w = min(w1, w2)
    
    img1_crop = img1[:h, :w]
    img2_crop = img2[:h, :w]
    
    # Kanal sayısını kontrol et
    is_img1_color = len(img1_crop.shape) == 3
    is_img2_color = len(img2_crop.shape) == 3
    
    img1_uint16 = img1_crop.astype(np.uint16)
    img2_uint16 = img2_crop.astype(np.uint16)
    
    # 3 kanallı + 3 kanallı: Kanal başına toplama
    if is_img1_color and is_img2_color:
        result = np.minimum(img1_uint16 + img2_uint16, 255).astype(np.uint8)
    # 3 kanallı + 1 kanallı: Her kanala 1 kanallı değer ekle
    elif is_img1_color and not is_img2_color:
        img2_broadcast = np.stack([img2_uint16, img2_uint16, img2_uint16], axis=2)
        result = np.minimum(img1_uint16 + img2_broadcast, 255).astype(np.uint8)
    # 1 kanallı + 3 kanallı: Her kanala 1 kanallı değer ekle
    elif not is_img1_color and is_img2_color:
        img1_broadcast = np.stack([img1_uint16, img1_uint16, img1_uint16], axis=2)
        result = np.minimum(img1_broadcast + img2_uint16, 255).astype(np.uint8)
    # 1 kanallı + 1 kanallı: Normal toplama
    else:
        result = np.minimum(img1_uint16 + img2_uint16, 255).astype(np.uint8)
    
    return result


def resim_carpma(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """İki resmin pixellerini çarparak çarpma işlemi yap.
    
    - 3 kanallı + 3 kanallı: Kanal başına çarpma (R*R, G*G, B*B)
    - 3 kanallı + 1 kanallı: Her kanala 1 kanallı değer çarpma (R*değer, G*değer, B*değer)
    - 1 kanallı + 1 kanallı: Normal çarpma
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Boyutları eşitleme - daha küçük boyutu kullan
    h = min(h1, h2)
    w = min(w1, w2)
    
    img1_crop = img1[:h, :w]
    img2_crop = img2[:h, :w]
    
    # Kanal sayısını kontrol et
    is_img1_color = len(img1_crop.shape) == 3
    is_img2_color = len(img2_crop.shape) == 3
    
    img1_float = img1_crop.astype(np.float32) / 255.0
    img2_float = img2_crop.astype(np.float32) / 255.0
    
    # 3 kanallı + 3 kanallı: Kanal başına çarpma
    if is_img1_color and is_img2_color:
        result = (img1_float * img2_float * 255).astype(np.uint8)
    # 3 kanallı + 1 kanallı: Her kanala 1 kanallı değer çarp
    elif is_img1_color and not is_img2_color:
        img2_broadcast = np.stack([img2_float, img2_float, img2_float], axis=2)
        result = (img1_float * img2_broadcast * 255).astype(np.uint8)
    # 1 kanallı + 3 kanallı: Her kanala 1 kanallı değer çarp
    elif not is_img1_color and is_img2_color:
        img1_broadcast = np.stack([img1_float, img1_float, img1_float], axis=2)
        result = (img1_broadcast * img2_float * 255).astype(np.uint8)
    # 1 kanallı + 1 kanallı: Normal çarpma
    else:
        result = (img1_float * img2_float * 255).astype(np.uint8)
    
    return result


def binary_donusum(img: np.ndarray, params: dict) -> np.ndarray:
    """Binary dönüşüm yap (Eşik veya Otsu yöntemi).
    
    - Eşik (Threshold): Verilen eşik değerine göre bölüm
    - Otsu: Otomatik eşik değeri hesaplayarak bölüm
    """
    # 3 kanallı ise griye dönüştür
    if len(img.shape) == 3:
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        gray = (0.114 * b + 0.587 * g + 0.229 * r).astype(np.uint8)
    else:
        gray = img.astype(np.uint8)
    
    method = params.get("method", "Otsu")
    
    if method == "Eşik (Threshold)":
        # Manuel eşik yöntemi
        threshold = params.get("threshold", 127)
        binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    else:  # Otsu
        # Histogram hesapla
        histogram = np.bincount(gray.flatten(), minlength=256)
        
        # Tüm pixel toplamı
        total_pixels = gray.size
        total_sum = sum([i * histogram[i] for i in range(256)])
        
        # Otsu yöntemi - eşik değerini bul
        max_variance = 0
        threshold = 0
        
        w0_count = 0  # Arka plan piksel sayısı
        w0_sum = 0    # Arka plan piksel değerleri toplamı
        
        for t in range(255):
            w0_count += histogram[t]
            w0_sum += t * histogram[t]
            
            # Arka plan ve ön plan oranları
            w0 = w0_count / total_pixels
            w1 = 1 - w0
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Ortalama pixel değerleri
            mu0 = w0_sum / w0_count if w0_count > 0 else 0
            mu1 = (total_sum - w0_sum) / (total_pixels - w0_count) if (total_pixels - w0_count) > 0 else 0
            
            # Sınıflar arası varyans
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                threshold = t
        
        # Binary görüntü oluştur
        binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    
    return binary


def gauss_konvolüsyon(img: np.ndarray, params: dict) -> np.ndarray:
    """Gauss çekirdeği ile konvolüsyon işlemi yap.
    
    Gauss filtresi kullanarak görüntüyü bulandırır (blur).
    Her kanal ayrı ayrı işleme tabi tutulur.
    """
    # Kernel boyutu (tek olması gerekli)
    ksize = int(params.get("ksize", 5))
    if ksize % 2 == 0:
        ksize += 1
    
    # Sigma X (0 ise otomatik hesapla)
    sigma_x = float(params.get("sigma_x", 1.0))
    if sigma_x == 0:
        sigma_x = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    # Gauss çekirdeğini oluştur
    kernel = _gauss_kernel(ksize, sigma_x)
    
    # 3 kanallı mı yoksa 1 kanallı mı kontrol et
    if len(img.shape) == 3:
        # Her kanal için ayrı konvolüsyon
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            result[:, :, c] = _apply_convolution(img[:, :, c].astype(np.float32), kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        # Tek kanal
        result = _apply_convolution(img.astype(np.float32), kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def _gauss_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Gauss çekirdeği oluştur."""
    # Çekirdek merkezi
    center = ksize // 2
    
    # 1D Gauss profili
    kernel_1d = np.zeros(ksize)
    for i in range(ksize):
        x = i - center
        kernel_1d[i] = np.exp(-(x ** 2) / (2 * sigma ** 2))
    
    # Normalize et
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    # 2D Gauss çekirdeği (separable convolution)
    kernel = np.outer(kernel_1d, kernel_1d)
    
    return kernel


def _apply_convolution(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Konvolüsyon işlemini uygula (padding ile)."""
    ksize = kernel.shape[0]
    pad = ksize // 2
    
    # Resimi pad et (REFLECT padding)
    padded = np.pad(img, pad, mode='reflect')
    
    # Konvolüsyon
    output = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + ksize, j:j + ksize]
            output[i, j] = np.sum(region * kernel)
    
    return output


def goruntu_dongme(img: np.ndarray, params: dict) -> np.ndarray:
    """Görüntüyü belirtilen açıda döndür (bilineer interpolasyon).

    Parametreler:
        angle (float): Döndürme açısı (derece, saat yönünün tersine pozitif).
        expand (bool): True ise çıktı boyutu tüm döndürülmüş görüntüyü kapsayacak
                       şekilde genişletilir; False ise orijinal boyut korunur.
    """
    angle_deg = float(params.get("angle", 45.0))
    expand = bool(params.get("expand", True))

    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0          # orijinal merkez

    if expand:
        # Yeni boyutları hesapla
        new_w = int(np.ceil(abs(w * cos_a) + abs(h * sin_a)))
        new_h = int(np.ceil(abs(w * sin_a) + abs(h * cos_a)))
    else:
        new_w, new_h = w, h

    ncx, ncy = new_w / 2.0, new_h / 2.0  # yeni merkez

    # Çıktı piksel koordinatları
    out_y, out_x = np.mgrid[0:new_h, 0:new_w].astype(np.float32)

    # Yeni koordinatları orijinal görüntüye geri dönüştür
    # (ters rotasyon: negatif açı)
    src_x = cos_a * (out_x - ncx) + sin_a * (out_y - ncy) + cx
    src_y = -sin_a * (out_x - ncx) + cos_a * (out_y - ncy) + cy

    # Bilineer interpolasyon
    result = _bilineer_interpolasyon(img, src_x, src_y, new_h, new_w)
    return result


def goruntu_olcekleme(img: np.ndarray, params: dict) -> np.ndarray:
    """Görüntüyü yaklaştır veya uzaklaştır (bilineer interpolasyon).

    Parametreler:
        scale_x (float): Yatay ölçek katsayısı (örn. 2.0 → 2 kat büyüt).
        scale_y (float): Dikey ölçek katsayısı (örn. 0.5 → 2 kat küçült).
                         Belirtilmezse scale_x değerini alır.
    """
    scale_x = float(params.get("scale_x", 150)) / 100.0
    scale_y = float(params.get("scale_y", scale_x * 100)) / 100.0

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale_x)))
    new_h = max(1, int(round(h * scale_y)))

    # Çıktı koordinatlarını kaynak görüntüye eşle
    out_y, out_x = np.mgrid[0:new_h, 0:new_w].astype(np.float32)
    src_x = out_x * (w / new_w)
    src_y = out_y * (h / new_h)

    result = _bilineer_interpolasyon(img, src_x, src_y, new_h, new_w)
    return result


def _bilineer_interpolasyon(
    img: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Kaynak koordinatları (src_x, src_y) için bilineer interpolasyon uygula."""
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3

    # Sınır dışı koordinatları kırp
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    # Kesirli kısımlar (ağırlıklar)
    wx = (src_x - np.floor(src_x)).astype(np.float32)
    wy = (src_y - np.floor(src_y)).astype(np.float32)

    # Sınır dışı pikselleri siyah yap
    mask = (
        (src_x >= 0) & (src_x <= w - 1) &
        (src_y >= 0) & (src_y <= h - 1)
    )

    if is_color:
        channels = img.shape[2]
        result = np.zeros((out_h, out_w, channels), dtype=np.float32)
        for c in range(channels):
            tl = img[y0, x0, c].astype(np.float32)
            tr = img[y0, x1, c].astype(np.float32)
            bl = img[y1, x0, c].astype(np.float32)
            br = img[y1, x1, c].astype(np.float32)
            interp = (
                tl * (1 - wx) * (1 - wy) +
                tr * wx * (1 - wy) +
                bl * (1 - wx) * wy +
                br * wx * wy
            )
            result[:, :, c] = np.where(mask, interp, 0)
    else:
        tl = img[y0, x0].astype(np.float32)
        tr = img[y0, x1].astype(np.float32)
        bl = img[y1, x0].astype(np.float32)
        br = img[y1, x1].astype(np.float32)
        interp = (
            tl * (1 - wx) * (1 - wy) +
            tr * wx * (1 - wy) +
            bl * (1 - wx) * wy +
            br * wx * wy
        )
        result = np.where(mask, interp, 0)

    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_germe(img: np.ndarray, params: dict) -> np.ndarray:
    """Histogram germe / kontrast genişletme (contrast stretching).

    Her kanal bağımsız olarak [p_low, p_high] yüzdelik dilimleri arasındaki
    yoğunluk aralığını tam [0, 255] aralığına uzatır.

    Parametreler:
        p_low  (float): Alt kesim yüzdeliği (örn. 1.0 → %1 alt kesim). Varsayılan: 1.0
        p_high (float): Üst kesim yüzdeliği (örn. 99.0 → %99 üst kesim). Varsayılan: 99.0
        show_histogram (bool): True ise giriş ve çıkış histogramlarını matplotlib ile göster.
    """
    p_low = float(params.get("p_low", 1.0))
    p_high = float(params.get("p_high", 99.0))
    show_histogram = bool(params.get("show_histogram", True))

    img_f = img.astype(np.float32)
    is_color = len(img.shape) == 3
    channels = img.shape[2] if is_color else 1

    result = np.zeros_like(img_f)

    # Kanal başına germe
    for c in range(channels):
        channel = img_f[:, :, c] if is_color else img_f
        v_min = np.percentile(channel, p_low)
        v_max = np.percentile(channel, p_high)
        if v_max <= v_min:
            stretched = channel.copy()
        else:
            stretched = (channel - v_min) / (v_max - v_min) * 255.0
        stretched = np.clip(stretched, 0, 255)
        if is_color:
            result[:, :, c] = stretched
        else:
            result = stretched

    result = result.astype(np.uint8)

    # Histogram gösterimi
    if show_histogram:
        _histogram_goster(img, result, is_color, channels)

    return result


def _histogram_goster(
    img_orig: np.ndarray,
    img_result: np.ndarray,
    is_color: bool,
    channels: int,
) -> None:
    """Orijinal ve gerdirilmiş görüntünün histogramını matplotlib ile göster."""
    kanal_renkleri = ["blue", "green", "red"]
    kanal_adlari = ["B", "G", "R"]

    fig, axes = plt.subplots(2, channels, figsize=(5 * channels, 6))
    fig.suptitle("Histogram Germe — Orijinal vs. Gerdirilen", fontsize=13)

    # Tek kanallı durumda axes boyutunu düzenle
    if channels == 1:
        axes = np.array([[axes[0]], [axes[1]]]) if channels == 1 and axes.ndim == 1 \
            else axes.reshape(2, 1)

    for c in range(channels):
        if is_color:
            orig_ch = img_orig[:, :, c].flatten()
            res_ch = img_result[:, :, c].flatten()
            renk = kanal_renkleri[c]
            ad = kanal_adlari[c]
        else:
            orig_ch = img_orig.flatten()
            res_ch = img_result.flatten()
            renk = "gray"
            ad = "Gray"

        # Orijinal histogram
        hist_orig, bins = np.histogram(orig_ch, bins=256, range=(0, 255))
        axes[0][c].bar(bins[:-1], hist_orig, width=1, color=renk, alpha=0.75)
        axes[0][c].set_title(f"Orijinal — {ad} Kanalı")
        axes[0][c].set_xlim(0, 255)
        axes[0][c].set_xlabel("Piksel Değeri")
        axes[0][c].set_ylabel("Frekans")

        # Gerdirilmiş histogram
        hist_res, bins = np.histogram(res_ch, bins=256, range=(0, 255))
        axes[1][c].bar(bins[:-1], hist_res, width=1, color=renk, alpha=0.75)
        axes[1][c].set_title(f"Gerdirilen — {ad} Kanalı")
        axes[1][c].set_xlim(0, 255)
        axes[1][c].set_xlabel("Piksel Değeri")
        axes[1][c].set_ylabel("Frekans")

    plt.tight_layout()
    plt.show()


# Mevcut araçlar (diğerleri henüz devre dışı)
registry = {
    "Gri Dönüşüm": gri_donusum,
    "Binary Dönüşüm": binary_donusum,
    "Konvolüsyon İşlemi (Gauss)": gauss_konvolüsyon,
    "Görüntü Döndürme": goruntu_dongme,
    "Yaklaştırma / Uzaklaştırma": goruntu_olcekleme,
    "Histogram & Germe": histogram_germe,
}
