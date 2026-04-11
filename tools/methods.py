"""Görüntü işleme metodları."""

import cv2
import numpy as np


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


# Mevcut araçlar (diğerleri henüz devre dışı)
registry = {
    "Gri Dönüşüm": gri_donusum,
    "Binary Dönüşüm": binary_donusum,
    "Konvolüsyon İşlemi (Gauss)": gauss_konvolüsyon,
}
