"""Görüntü işleme metodları."""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def gri_donusum(img: np.ndarray, params: dict) -> np.ndarray:
    """Renk görüntüsünü gri (grayscale) görüntüye dönüştür.
    
    Formül: Gray = 0.229*R + 0.587*G + 0.114*B
    """
    # BGR kanallarını ayır
    if len(img.shape)==3:
        r = img[:, :, 2].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 0].astype(np.float32)
    
    # Manuel formülü uygula - tek kanal olarak döndür
    gray = (0.114 * b + 0.587 * g + 0.229 * r).astype(np.uint8)
    
    return gray


def resim_ekleme(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """İki resmin pixellerini toplayarak ekleme işlemi yap.
    
    3 kanallı + 3 kanallı: Kanal başına toplama (R+R, G+G, B+B)
    3 kanallı + 1 kanallı: Her kanala 1 kanallı değer ekleme (R+değer, G+değer, B+değer)
    1 kanallı + 1 kanallı: Normal toplama
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Boyutları eşitleme - daha küçük boyutu kullan
    h = min(h1, h2)
    w = min(w1, w2)
    
    img1_kirpilmis = img1[:h, :w]
    img2_kirpilmis = img2[:h, :w]
    
    img1_uint16 = img1_kirpilmis.astype(np.uint16)
    img2_uint16 = img2_kirpilmis.astype(np.uint16)
    
    # 3 kanallı + 3 kanallı: Kanal başına toplama
    if len(img1_kirpilmis.shape)==3 and len(img2_kirpilmis.shape)==3:
        result = np.minimum(img1_uint16 + img2_uint16, 255).astype(np.uint8)
    # 3 kanallı + 1 kanallı: Her kanala 1 kanallı değer ekle
    elif len(img1_kirpilmis.shape)==3 and len(img2_kirpilmis.shape)!=3:
        for i in range(h):
            for j in range(w):
                for k in range(3):
                    img1_uint16[i, j, k] = min(img1_uint16[i, j, k] + img2_uint16[i, j], 255)
        result=img1_uint16.astype(np.uint8)
    # 1 kanallı + 3 kanallı: Her kanala 1 kanallı değer ekle
    elif len(img1_kirpilmis.shape)!=3 and len(img2_kirpilmis.shape)==3:
        for i in range(h):
            for j in range(w):
                for k in range(3):
                    img2_uint16[i, j, k] = min(img2_uint16[i, j, k] + img1_uint16[i, j], 255)
        result=img2_uint16.astype(np.uint8)
    # 1 kanallı + 1 kanallı: Normal toplama
    else:
        result = np.minimum(img1_uint16 + img2_uint16, 255).astype(np.uint8)
    
    return result


def resim_carpma(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """İki resmin pixellerini çarparak çarpma işlemi yap.
    
    3 kanallı + 3 kanallı: Kanal başına çarpma (R*R, G*G, B*B)
    3 kanallı + 1 kanallı: Her kanala 1 kanallı değer çarpma (R*değer, G*değer, B*değer)
    1 kanallı + 1 kanallı: Normal çarpma
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Boyutları eşitleme - daha küçük boyutu kullan
    h = min(h1, h2)
    w = min(w1, w2)
    
    img1_kirpilmis = img1[:h, :w]
    img2_kirpilmis = img2[:h, :w]
    
    img1_float = img1_kirpilmis.astype(np.float32) / 255.0
    img2_float = img2_kirpilmis.astype(np.float32) / 255.0
    
    # 3 kanallı + 3 kanallı: Kanal başına çarpma
    if len(img1_kirpilmis.shape)==3 and len(img2_kirpilmis.shape)==3:
        result = (img1_float * img2_float * 255).astype(np.uint8)
    # 3 kanallı + 1 kanallı: Her kanala 1 kanallı değer çarp
    elif len(img1_kirpilmis.shape)==3 and len(img2_kirpilmis.shape)!=3:
        for i in range(h):
            for j in range(w):
                for k in range(3):
                    img1_float[i,j,k]*=img2_float[i,j]
        img1_float=img1_float*255
        result=img1_float.astype(np.uint8)
    # 1 kanallı + 3 kanallı: Her kanala 1 kanallı değer çarp
    elif len(img1_kirpilmis.shape)!=3 and len(img2_kirpilmis.shape)==3:
        for i in range(h):
            for j in range(w):
                for k in range(3):
                    img2_float[i,j,k]*=img1_float[i,j]
        img2_float=img2_float*255
        result=img2_float.astype(np.uint8)
    # 1 kanallı + 1 kanallı: Normal çarpma
    else:
        result = (img1_float * img2_float * 255).astype(np.uint8)
    
    return result


def binary_donusum(img: np.ndarray, params: dict) -> np.ndarray:
    """Binary dönüşüm yap (Eşik veya Otsu yöntemi).
    
    Eşik (Threshold): Verilen eşik değerine göre bölüm
    Otsu: Otomatik eşik değeri hesaplayarak bölüm
    """
    # 3 kanallı ise griye dönüştür
    if(len(img.shape) == 3):
        gray = gri_donusum(img, params)
    else:
        gray = img.astype(np.uint8)

    method = params.get("method", "Otsu")
    
    if method == "Eşik (Threshold)":
        threshold = params.get("threshold", 127)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i, j] >= threshold:
                    gray[i, j] = 255
                else:
                    gray[i, j] = 0
    else:  # Otsu
        # Histogram hesapla
        hist = histogram(gray, params)
        toplam_pixel=sum(hist)
        olasilik_dagilimi=np.zeros(256)
        max_varyans=0
        threshold=0
        for i in range(256):
            olasilik_dagilimi[i]=hist[i]/toplam_pixel

        for t in range(0, 256):
            w0=sum(olasilik_dagilimi[:t])
            w1=sum(olasilik_dagilimi[t:])
            if w0 == 0 or w1 == 0:
                continue
            ortalama_0=sum([i * olasilik_dagilimi[i] for i in range(t)]) / w0
            ortalama_1=sum([i * olasilik_dagilimi[i] for i in range(t, 256)]) / w1
            varyans=w0 * w1 * (ortalama_0 - ortalama_1) ** 2
            if varyans > max_varyans:
                max_varyans = varyans
                threshold = t

        # Binary görüntü oluştur
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i, j] >= threshold:
                    gray[i, j] = 255
                else:
                    gray[i, j] = 0
    return gray


def gauss_konvolüsyon(img: np.ndarray, params: dict) -> np.ndarray:
    """Gauss çekirdeği ile konvolüsyon işlemi yap.
    
    Gauss filtresi kullanarak görüntüyü bulanıklaştırır.
    Her kanal ayrı ayrı işleme tabi tutulur.
    """
    # Kernel boyutu (tek olması gerekli)
    ksize = int(params.get("ksize", 5))
    if ksize % 2 == 0:
        ksize += 1
    
    sigma = float(params.get("sigma", 1.0))

    # Gauss çekirdeğini oluştur
    kernel = gauss_kernel(ksize, sigma)
    
    # 3 kanallı mı yoksa 1 kanallı mı kontrol et
    if len(img.shape) == 3:
        # Her kanal için ayrı konvolüsyon
        result = np.zeros_like(img, dtype=np.float32)
        for kanal in range(img.shape[2]):
            result[:, :, kanal] = konvolusyon(img[:, :, kanal].astype(np.float32), kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        # Tek kanal
        result = konvolusyon(img.astype(np.float32), kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def gauss_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Gauss çekirdeği oluştur."""
    kernel=np.zeros((ksize, ksize))
    center=ksize//2
    for i in range(ksize):
        for j in range(ksize):
            x=i-center
            y=j-center
            kernel[i,j]=(1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel=kernel/kernel.sum() #normalize etmezsek resim kararıyor
    return kernel


def konvolusyon(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Konvolüsyon işlemini uygula (padding ile)."""
    kernel=np.flip(kernel)
    ksize = kernel.shape[0]
    padding=ksize//2 #zero padding yapacağız

    padding_img=np.zeros((img.shape[0]+2*padding, img.shape[1]+2*padding))
    padding_img[padding:padding+img.shape[0], padding:padding+img.shape[1]]=img
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            toplam=0
            for k in range(ksize):
                for l in range(ksize):
                    toplam += padding_img[i+k, j+l] * kernel[k, l]
            output[i, j] = toplam

    output = np.clip(output, 0, 255)
    return output #unutma output float32 dönüyor


def goruntu_dongme(img: np.ndarray, params: dict) -> np.ndarray:
    """Görüntüyü belirtilen açıda döndür (bilineer interpolasyon).

    Parametreler:
        angle (float): Döndürme açısı (derece, saat yönünün tersine pozitif).
    """
    angle_deg = float(params.get("angle", 45.0))

    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0          # orijinal merkez

    # Yeni boyutları hesapla ve taşan pikselleri koru
    new_w = int(np.ceil(abs(w * cos_a) + abs(h * sin_a)))
    new_h = int(np.ceil(abs(w * sin_a) + abs(h * cos_a)))
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
        scale (float): Hem yatay hem dikey için eşit ölçek oranı.
    """
    scale = float(params.get("scale", 150)) / 100.0

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

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

    Formül: P_yeni = (P_eski - a) × (d - c) / (b - a) + c
    
    Parametreler:
        c (int): Alt kesim (0-255). Varsayılan: 0
        d (int): Üst kesim (0-255). Varsayılan: 255
        show_histogram (bool): True ise histogramları göster.
    
    Değişkenler:
        a: Resimdeki en düşük piksel değeri
        b: Resimdeki en yüksek piksel değeri
    """
    c = float(params.get("p_low", 0))      # Alt kesim
    d = float(params.get("p_high", 255))   # Üst kesim
    show_histogram = bool(params.get("show_histogram", True))

    if d <= c:
        raise ValueError("Üst kesim değeri alt kesim değerinden büyük olmalıdır.")

    img_f = img.astype(np.float32)
    is_color = len(img.shape) == 3
    channels = img.shape[2] if is_color else 1

    result = np.zeros_like(img_f)

    # Kanal başına germe
    for ch in range(channels):
        channel = img_f[:, :, ch] if is_color else img_f
        a = channel.min()  # Resimdeki en düşük değer
        b = channel.max()  # Resimdeki en yüksek değer
        
        if b <= a:
            # Tüm pikseller aynı değerse
            stretched = channel.copy()
        else:
            # P_yeni = (P_eski - a) × (d - c) / (b - a) + c
            stretched = (channel - a) / (b - a) * (d - c) + c
        
        stretched = np.clip(stretched, 0, 255)
        if is_color:
            result[:, :, ch] = stretched
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

def histogram(img: np.ndarray, params: dict) -> np.ndarray:
    """Görüntünün histogramını verir"""
    if len(img.shape) == 3:
        gray=gri_donusum(img, params)
    else:
        gray=img.astype(np.uint8)
    
    histogram=np.zeros(256)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            histogram[gray[i,j]]+=1

    return histogram


def goruntu_kirpma(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Görüntüyü döngüler kullanarak manuel olarak kırpıyoruz.
    Formül: B(i,j) = A(y1 + i, x1 + j)
    """
    # UI'dan (canvas_area.py) gelen seçim koordinatlarını al mouse ile
    x1 = params.get("x1", 0)
    y1 = params.get("y1", 0)
    x2 = params.get("x2", img.shape[1])
    y2 = params.get("y2", img.shape[0])

    # Sınır denetimi (Boundary check)
    H, W = img.shape[:2]
    y1, y2 = max(0, y1), min(H, y2)
    x1, x2 = max(0, x1), min(W, x2)

    # Hedef boyut hesaplama ve boş matris tahsisi
    h_new = y2 - y1
    w_new = x2 - x1
    channels = img.shape[2] if len(img.shape) == 3 else 1

    if channels > 1:
        cropped_img = np.zeros((h_new, w_new, channels), dtype=np.uint8)
    else:
        cropped_img = np.zeros((h_new, w_new), dtype=np.uint8)

    # Piksel Transferi: Döngüler ile manuel kırpma
    for i in range(h_new):
        for j in range(w_new):
            if channels > 1:
                for c in range(channels):
                    cropped_img[i, j, c] = img[y1 + i, x1 + j, c]
            else:
                cropped_img[i, j] = img[y1 + i, x1 + j]

    return cropped_img


def parlaklik_artirma(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Görüntü parlaklığını ve kontrastını doğrusal formül ile ayarlıyoruz.
    Formül: g(x) = alpha * f(x) + beta
    """
    # alpha: Kontrast, beta: Parlaklık değerlerini slider'dan alıyor
    alpha = float(params.get("alpha", 10)) / 10.0  # Slider 10-30 -> 1.0-3.0 kat
    beta = int(params.get("beta", 30))

    # Taşmaları (overflow) önlemek için clip kullandık
    # İşlem: f(x) * alpha + beta
    result = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return result


def renk_uzayi_donusumleri(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Görüntüyü seçilen hedef renk uzayına dönüştürür.
    """
    if len(img.shape) != 3:  # Görüntü zaten gri ise işlem yapmaya gerek yok
        return img

    target = params.get("colorspace", "HSV")

    if target == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif target == "YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif target == "Grayscale":

        weights = np.array([0.114, 0.587, 0.299])
        gray = np.dot(img[..., :3], weights).astype(np.uint8)
        return gray

    return img

# Mevcut araçlar (diğerleri henüz devre dışı)
registry = {
    "Gri Dönüşüm": gri_donusum,
    "Binary Dönüşüm": binary_donusum,
    "Konvolüsyon İşlemi (Gauss)": gauss_konvolüsyon,
    "Görüntü Döndürme": goruntu_dongme,
    "Yaklaştırma / Uzaklaştırma": goruntu_olcekleme,
    "Histogram & Germe": histogram_germe,
    "Görüntü Kırpma": goruntu_kirpma, 
    "Parlaklık Artırma": parlaklik_artirma,
    "Renk Uzayı Dönüşümleri": renk_uzayi_donusumleri,
}
