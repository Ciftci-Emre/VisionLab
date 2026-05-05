"""Görüntü işleme metodları."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


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
def konvolusyon_uygula(resim_kanali: np.ndarray, maske_matrisi: np.ndarray) -> np.ndarray:
    """Verilen 2 boyutlu görüntü kanalı üzerinde konvolüsyon (matris çarpımı) işlemini gerçekleştirir."""
    maske_boyutu = maske_matrisi.shape[0]
    kenar = maske_boyutu // 2
    yukseklik, genislik = resim_kanali.shape
    
    # Görüntünün kenarlarına yansıtma (reflect) yöntemiyle dolgu (pad) ekliyoruz
    cerceveli_resim = np.pad(resim_kanali, kenar, mode='reflect')
    sonuc_matrisi = np.zeros_like(resim_kanali, dtype=np.float32)

    for y in range(yukseklik):
        for x in range(genislik):
            # O anki pikselin etrafındaki komşuluk penceresi
            pencere = cerceveli_resim[y : y + maske_boyutu, x : x + maske_boyutu]
            # Pencere ile filtre maskesini karşılıklı çarpıp toplayarak konvolüsyonu tamamlıyoruz
            sonuc_matrisi[y, x] = np.sum(pencere * maske_matrisi)
            
    return sonuc_matrisi

# SOBEL KENAR BULMA
def sobel_kenar_bulma(resim: np.ndarray, params: dict) -> np.ndarray:
    """Sobel operatörü ile x, y veya çift yönlü kenar tespiti yapan ana fonksiyon."""
    
    # Sobel filtre matrisleri (X ve Y yönleri için)
    Gx = np.array([[-1,  0,  1],
                   [-2,  0,  2],
                   [-1,  0,  1]], dtype=np.float32)
                   
    Gy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    tum_icerik = str(params.values()).lower()
    if "sobel x" in tum_icerik and "y" not in tum_icerik: yontem = "X"
    elif "sobel y" in tum_icerik and "x" not in tum_icerik: yontem = "Y"
    else: yontem = "XY"

    renkli_mi = len(resim.shape) == 3
    
    if renkli_mi:
        sonuc_resmi = np.zeros_like(resim, dtype=np.float32)
        for c in range(3):
            kanal = resim[:, :, c].astype(np.float32)
            
            # Ortak konvolüsyon motorunu çağırıyoruz
            gx_sonuc = konvolusyon_uygula(kanal, Gx)
            gy_sonuc = konvolusyon_uygula(kanal, Gy)
            
            if yontem == "X": sonuc_resmi[:, :, c] = np.abs(gx_sonuc)
            elif yontem == "Y": sonuc_resmi[:, :, c] = np.abs(gy_sonuc)
            else: sonuc_resmi[:, :, c] = np.sqrt(gx_sonuc**2 + gy_sonuc**2)
                
        return np.clip(sonuc_resmi, 0, 255).astype(np.uint8)
        
    else:
        resim_float = resim.astype(np.float32)
        
        # Ortak konvolüsyon motorunu çağırıyoruz
        gx_sonuc = konvolusyon_uygula(resim_float, Gx)
        gy_sonuc = konvolusyon_uygula(resim_float, Gy)
        
        if yontem == "X": buyukluk = np.abs(gx_sonuc)
        elif yontem == "Y": buyukluk = np.abs(gy_sonuc)
        else: buyukluk = np.sqrt(gx_sonuc**2 + gy_sonuc**2)
            
        return np.clip(buyukluk, 0, 255).astype(np.uint8)

# BULANIKLAŞTIRMA (BLUR) İŞLEMLERİ
def ortalama_bulaniklastirma(resim: np.ndarray, params: dict) -> np.ndarray:
    """Çeşitli bulanıklaştırma (blur) filtrelerini uygulayan ana fonksiyon."""
    tum_icerik = str(params).lower()
    if "median" in tum_icerik or "medyan" in tum_icerik: secilen_filtre = "Median"
    elif "bilateral" in tum_icerik or "çift" in tum_icerik: secilen_filtre = "Bilateral"
    elif "gaussian" in tum_icerik or "gauss" in tum_icerik: secilen_filtre = "Gaussian"
    else: secilen_filtre = "Average"

    maske_boyutu = int(params.get("ksize", 3))
    if maske_boyutu % 2 == 0: maske_boyutu += 1
    kenar = maske_boyutu // 2

    renkli_mi = len(resim.shape) == 3
    yukseklik, genislik = resim.shape[:2]
    sonuc_resmi = np.zeros_like(resim, dtype=np.float32)

    # 1. ORTALAMA (AVERAGE) BLUR
    if secilen_filtre == "Average":
        maske = np.ones((maske_boyutu, maske_boyutu), dtype=np.float32) / (maske_boyutu**2)
        if renkli_mi:
            for c in range(3): 
                sonuc_resmi[:, :, c] = konvolusyon_uygula(resim[:, :, c].astype(np.float32), maske)
        else: 
            sonuc_resmi = konvolusyon_uygula(resim.astype(np.float32), maske)

    # 2. GAUSSIAN BLUR
    elif secilen_filtre == "Gaussian":
        sigma = float(params.get("sigma", maske_boyutu / 3.0))
        maske = np.zeros((maske_boyutu, maske_boyutu), dtype=np.float32)
        
        for ky in range(maske_boyutu):
            for kx in range(maske_boyutu):
                maske[ky, kx] = math.exp(-((kx - kenar)**2 + (ky - kenar)**2) / (2 * sigma**2))
        maske /= np.sum(maske) 
        
        if renkli_mi:
            for c in range(3): 
                sonuc_resmi[:, :, c] = konvolusyon_uygula(resim[:, :, c].astype(np.float32), maske)
        else: 
            sonuc_resmi = konvolusyon_uygula(resim.astype(np.float32), maske)

    # 3. MEDYAN (MEDIAN) BLUR
    elif secilen_filtre == "Median":
        kanallar = range(3) if renkli_mi else [None]
        for c in kanallar:
            kanal_verisi = resim[:, :, c] if renkli_mi else resim
            cerceveli = np.pad(kanal_verisi, kenar, mode='reflect')
            for y in range(yukseklik):
                for x in range(genislik):
                    pencere = cerceveli[y : y + maske_boyutu, x : x + maske_boyutu]
                    if renkli_mi: sonuc_resmi[y, x, c] = np.median(pencere)
                    else: sonuc_resmi[y, x] = np.median(pencere)

    # 4. ÇİFT YÖNLÜ (BILATERAL) BLUR
    elif secilen_filtre == "Bilateral":
        sigma_mesafe = 75.0
        sigma_renk = 75.0
        
        mesafe_maskesi = np.zeros((maske_boyutu, maske_boyutu), dtype=np.float32)
        for ky in range(maske_boyutu):
            for kx in range(maske_boyutu):
                mesafe_maskesi[ky, kx] = math.exp(-((kx - kenar)**2 + (ky - kenar)**2) / (2 * sigma_mesafe**2))

        kanallar = range(3) if renkli_mi else [None]
        for c in kanallar:
            kanal_verisi = resim[:, :, c].astype(np.float32) if renkli_mi else resim.astype(np.float32)
            cerceveli = np.pad(kanal_verisi, kenar, mode='reflect')
            
            for y in range(yukseklik):
                for x in range(genislik):
                    pencere = cerceveli[y : y + maske_boyutu, x : x + maske_boyutu]
                    merkez_piksel = pencere[kenar, kenar]
                    
                    renk_maskesi = np.exp(-((pencere - merkez_piksel)**2) / (2 * sigma_renk**2))
                    toplam_maske = mesafe_maskesi * renk_maskesi
                    
                    yeni_deger = np.sum(pencere * toplam_maske) / np.sum(toplam_maske)
                    
                    if renkli_mi: sonuc_resmi[y, x, c] = yeni_deger
                    else: sonuc_resmi[y, x] = yeni_deger

    return np.clip(sonuc_resmi, 0, 255).astype(np.uint8)

def morfolojik_genisletme(resim: np.ndarray, params: dict) -> np.ndarray:
    maske_boyutu = int(params.get("ksize", 3))
    if maske_boyutu % 2 == 0: maske_boyutu += 1
    kenar = maske_boyutu // 2
    tekrar_sayisi = int(params.get("iterations", params.get("iterasyon", 1)))
    kernel_tipi = str(params.get("kernel_shape", params.get("kernel_type", "Dikdörtgen"))).lower()

    maske = np.zeros((maske_boyutu, maske_boyutu), dtype=np.uint8)
    merkez = maske_boyutu // 2
    if "haç" in kernel_tipi or "artı" in kernel_tipi:
        for i in range(maske_boyutu): maske[merkez, i] = 1; maske[i, merkez] = 1
    elif "daire" in kernel_tipi or "elips" in kernel_tipi:
        for y in range(maske_boyutu):
            for x in range(maske_boyutu):
                if (x - merkez)**2 + (y - merkez)**2 <= merkez**2: maske[y, x] = 1
    else:
        maske = np.ones((maske_boyutu, maske_boyutu), dtype=np.uint8)

    gecici_resim = resim.copy()
    renkli_mi = len(gecici_resim.shape) == 3

    for _ in range(tekrar_sayisi):
        yukseklik, genislik = gecici_resim.shape[:2]
        sonuc_resmi = np.zeros_like(gecici_resim)
        
        if renkli_mi:
            cerceveli = np.pad(gecici_resim, ((kenar, kenar), (kenar, kenar), (0, 0)), mode='reflect')
            for y in range(yukseklik):
                for x in range(genislik):
                    pencere = cerceveli[y : y + maske_boyutu, x : x + maske_boyutu, :]
                    for c in range(3):
                        sonuc_resmi[y, x, c] = np.max(pencere[:, :, c][maske == 1])
        else:
            cerceveli = np.pad(gecici_resim, kenar, mode='reflect')
            for y in range(yukseklik):
                for x in range(genislik):
                    pencere = cerceveli[y : y + maske_boyutu, x : x + maske_boyutu]
                    sonuc_resmi[y, x] = np.max(pencere[maske == 1])
        gecici_resim = sonuc_resmi.copy()
    return gecici_resim

def morfolojik_asindirma(resim: np.ndarray, params: dict) -> np.ndarray:
    maske_boyutu = int(params.get("ksize", 3))
    if maske_boyutu % 2 == 0: maske_boyutu += 1
    kenar = maske_boyutu // 2
    tekrar_sayisi = int(params.get("iterations", params.get("iterasyon", 1)))
    kernel_tipi = str(params.get("kernel_shape", params.get("kernel_type", "Dikdörtgen"))).lower()

    maske = np.zeros((maske_boyutu, maske_boyutu), dtype=np.uint8)
    merkez = maske_boyutu // 2
    if "haç" in kernel_tipi or "artı" in kernel_tipi:
        for i in range(maske_boyutu): maske[merkez, i] = 1; maske[i, merkez] = 1
    elif "daire" in kernel_tipi or "elips" in kernel_tipi:
        for y in range(maske_boyutu):
            for x in range(maske_boyutu):
                if (x - merkez)**2 + (y - merkez)**2 <= merkez**2: maske[y, x] = 1
    else:
        maske = np.ones((maske_boyutu, maske_boyutu), dtype=np.uint8)

    gecici_resim = resim.copy()
    renkli_mi = len(gecici_resim.shape) == 3

    for _ in range(tekrar_sayisi):
        yukseklik, genislik = gecici_resim.shape[:2]
        sonuc_resmi = np.zeros_like(gecici_resim)
        
        if renkli_mi:
            cerceveli = np.pad(gecici_resim, ((kenar, kenar), (kenar, kenar), (0, 0)), mode='reflect')
            for y in range(yukseklik):
                for x in range(genislik):
                    pencere = cerceveli[y : y + maske_boyutu, x : x + maske_boyutu, :]
                    for c in range(3):
                        sonuc_resmi[y, x, c] = np.min(pencere[:, :, c][maske == 1])
        else:
            cerceveli = np.pad(gecici_resim, kenar, mode='reflect')
            for y in range(yukseklik):
                for x in range(genislik):
                    pencere = cerceveli[y : y + maske_boyutu, x : x + maske_boyutu]
                    sonuc_resmi[y, x] = np.min(pencere[maske == 1])
        gecici_resim = sonuc_resmi.copy()
    return gecici_resim

def morfolojik_acma(resim: np.ndarray, params: dict) -> np.ndarray:
    return morfolojik_genisletme(morfolojik_asindirma(resim, params), params)

def morfolojik_kapama(resim: np.ndarray, params: dict) -> np.ndarray:
    return morfolojik_asindirma(morfolojik_genisletme(resim, params), params)

def morfolojik_islem_yonlendirici(resim: np.ndarray, params: dict) -> np.ndarray:
    """Arayüzden gelen komuta göre ilgili fonksiyona yönlendirir."""
    tum_icerik = str(params).lower()
    if "aşınma" in tum_icerik or "asinma" in tum_icerik or "aşındırma" in tum_icerik or "erode" in tum_icerik: 
        return morfolojik_asindirma(resim, params)
    elif "açma" in tum_icerik or "acma" in tum_icerik or "opening" in tum_icerik: 
        return morfolojik_acma(resim, params)
    elif "kapama" in tum_icerik or "closing" in tum_icerik: 
        return morfolojik_kapama(resim, params)
    else: 
        return morfolojik_genisletme(resim, params)
registry = {
    "Gri Dönüşüm": gri_donusum,
    "Binary Dönüşüm": binary_donusum,
    "Konvolüsyon İşlemi (Gauss)": gauss_konvolüsyon,
    "Görüntü Döndürme": goruntu_dongme,
    "Yaklaştırma / Uzaklaştırma": goruntu_olcekleme,
    "Histogram & Germe": histogram_germe,
    "Kenar Bulma (Sobel)": sobel_kenar_bulma,
    "Blurring": ortalama_bulaniklastirma,
    "Morfolojik İşlemler": morfolojik_islem_yonlendirici
}