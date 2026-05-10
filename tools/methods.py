"""Görüntü işleme metodları."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def gri_donusum(img: np.ndarray, params: dict) -> np.ndarray:
    """Renk görüntüsünü gri (grayscale) görüntüye dönüştür.
    Formül: Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(img.shape) == 2:
        return img
    # RGB kanallarını ayır
    if len(img.shape)==3:
        r = img[:, :, 2].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 0].astype(np.float32)
    
    gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    
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
    # Kernel boyutu
    ksize = int(params.get("ksize", 5))
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

def konvolusyon(img: np.ndarray, kernel: np.ndarray, dondurme=True) -> np.ndarray:
    """Konvolüsyon işlemini hazır kütüphane fonksiyonu kullanmadan, 
    matris kaydırma (shift and add) yöntemiyle çok hızlı şekilde uygula."""
    if dondurme:
        kernel = np.flip(kernel)
    
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape
    
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    #Zero Padding Uygula
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output = np.zeros_like(img, dtype=np.float32)
    for y in range(img_h):
        for x in range(img_w):
            # Kernel'ın altına denk gelen bölgeyi (pencereyi) kesip alıyoruz
            # Bu pencere kernel ile aynı boyutta
            pencere = padded_img[y : y + k_h, x : x + k_w]
            
            # Penceredeki her pikseli kernel'daki karşılığıyla çarp ve hepsini topla
            deger = np.sum(pencere * kernel)
            output[y, x] = deger
            
    return output #sonuç float32 dönüyor unutma

def goruntu_dondurme(img: np.ndarray, params: dict) -> np.ndarray:
    """Görüntüyü belirtilen açıda döndür (nearest neighbor interpolasyonu).
    Parametreler:
        angle (float): Döndürme açısı (derece, saat yönünün tersine pozitif).
    """
    angle_deg = float(params.get("angle", 45.0))
    angle_rad = np.deg2rad(angle_deg)

    h, w = img.shape[:2]
    
    # Yeni boyutlar(Resmin sığması için)
    yeni_w = int(abs(w * np.cos(angle_rad)) + abs(h * np.sin(angle_rad)))
    yeni_h = int(abs(w * np.sin(angle_rad)) + abs(h * np.cos(angle_rad)))

    if len(img.shape) == 3:
        yeni_resim = np.zeros((yeni_h, yeni_w, img.shape[2]), dtype=img.dtype)
    else:
        yeni_resim = np.zeros((yeni_h, yeni_w), dtype=img.dtype)

    merkez_x, merkez_y = w / 2.0, h / 2.0
    yeni_merkez_x, yeni_merkez_y = yeni_w / 2.0, yeni_h / 2.0

    for y_yeni in range(yeni_h):
        for x_yeni in range(yeni_w):

            x_eski = np.cos(angle_rad) * (x_yeni - yeni_merkez_x) + np.sin(angle_rad) * (y_yeni - yeni_merkez_y) + merkez_x
            y_eski = -np.sin(angle_rad) * (x_yeni - yeni_merkez_x) + np.cos(angle_rad) * (y_yeni - yeni_merkez_y) + merkez_y

            #EN YAKIN KOMŞU(Nearest Neighbor)
            x_eski_int = int(round(x_eski))
            y_eski_int = int(round(y_eski))

            #Eğer hesaplanan eski koordinat orijinal resmin içindeyse rengi kopyala
            if 0 <= x_eski_int < w and 0 <= y_eski_int < h:
                yeni_resim[y_yeni, x_yeni] = img[y_eski_int, x_eski_int]
    return yeni_resim


def goruntu_olcekleme(img: np.ndarray, params: dict) -> np.ndarray:
    """Görüntüyü yaklaştır veya uzaklaştır (nearest neighbor interpolasyonu).
    Parametreler:
        scale (float): Hem yatay hem dikey için eşit ölçek oranı.
    """
    scale = float(params.get("scale", 1.0))

    h,w=img.shape[:2]
    yeni_w=int(w*scale)
    yeni_h=int(h*scale)

    if len(img.shape) == 3:
        yeni_resim = np.zeros((yeni_h, yeni_w, img.shape[2]), dtype=img.dtype)
    else:
        yeni_resim = np.zeros((yeni_h, yeni_w), dtype=img.dtype)

    for y in range(yeni_h):
        for x in range(yeni_w):
            eski_x = int(x / scale)
            eski_y = int(y / scale)

            eski_x = min(eski_x, w - 1) #bazen eski_x veya eski_y orijinal resmin sınırlarını aşabiliyor, bu yüzden kırpıyoruz
            eski_y = min(eski_y, h - 1)
            
            yeni_resim[y,x]=img[eski_y, eski_x]
    return yeni_resim


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
    """Orijinal ve işlem görmüş görüntünün histogramını 'histogram' fonksiyonu ile gösterir."""
    kanal_renkleri = ["blue", "green", "red"]
    kanal_adlari = ["B", "G", "R"]

    fig, axes = plt.subplots(2, channels, figsize=(5 * channels, 6))
    fig.suptitle("Histogram - Orijinal vs. İşlenmiş", fontsize=13)

    # Subplot dizinleme hatasını önlemek için (tek kanal durumu)
    if channels == 1:
        axes = axes.reshape(2, 1)

    x_ekseni = np.arange(256) # 0'dan 255'e kadar değerler

    for c in range(channels):
        # Kanal seçimi
        if is_color:
            orig_ch = img_orig[:, :, c]
            res_ch = img_result[:, :, c]
            renk = kanal_renkleri[c]
            ad = kanal_adlari[c]
        else:
            orig_ch = img_orig
            res_ch = img_result
            renk = "gray"
            ad = "Gray"

        #'histogram' fonksiyonunu çağırıyoruz
        # Boş bir params sözlüğü gönderiyoruz
        hist_orig = histogram(orig_ch, {})
        hist_res = histogram(res_ch, {})

        # Orijinal histogramı çizdir
        axes[0][c].bar(x_ekseni, hist_orig, width=1, color=renk, alpha=0.75)
        axes[0][c].set_title(f"Orijinal — {ad}")
        axes[0][c].set_xlim(0, 255)

        # İşlenmiş (Gerdirilmiş/Ölçeklenmiş) histogramı çizdir
        axes[1][c].bar(x_ekseni, hist_res, width=1, color=renk, alpha=0.75)
        axes[1][c].set_title(f"İşlenmiş — {ad}")
        axes[1][c].set_xlim(0, 255)

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

def resim_kirpma(img: np.ndarray, params: dict) -> np.ndarray:
    """Görüntüyü seçilen alana göre kırp.

    Parametreler (main.py tarafından otomatik eklenir):
        x1, y1 (int): Sol üst köşe koordinatları.
        x2, y2 (int): Sağ alt köşe koordinatları.
    """
    x1 = params.get("x1")
    y1 = params.get("y1")
    x2 = params.get("x2")
    y2 = params.get("y2")

    if None in (x1, y1, x2, y2):
        raise ValueError("Kırpma için lütfen giriş resmi üzerinde bir alan seçin.")

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    #sınırları resim boyutuna göre kırp
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Geçersiz seçim alanı. Lütfen daha büyük bir alan seçin.")

    return img[y1:y2, x1:x2]


def parlaklik_artirma(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Görüntü parlaklığını ve kontrastını doğrusal formül ile ayarlıyoruz.
    Formül: g(x) = alpha * f(x) + beta
    """
    # alpha: Kontrast, beta: Parlaklık değerlerini slider'dan alıyor
    alpha = float(params.get("alpha", 1.0))
    beta = int(params.get("beta", 0))

    # Taşmaları (overflow) önlemek için clip kullandık
    # İşlem: f(x) * alpha + beta
    result = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return result

def renk_uzayi_donusumu(img: np.ndarray, params: dict) -> np.ndarray:
    """RGB, NTSC (YIQ) renk uzayı dönüşümü.

    RGB'den NTSC'ye:
        Y =  0.299R + 0.587G + 0.114B       (0–255)
        I =  0.596R - 0.274G - 0.322B       (-152 – 152) -> +128 ile 0–255'e taşınır
        Q =  0.212R - 0.523G + 0.311B       (-134 – 134) -> +128 ile 0–255'e taşınır
    Çıkış BGR sırasıyla saklanır: B=Q+128, G=I+128, R=Y

    NTSC'den RGB'ye:
        Ters matris uygulanır, I ve Q'dan önce 128 çıkarılır.
    """
    colorspace = params.get("colorspace", "RGB'den NTSC'ye")

    if len(img.shape) == 2:
        raise ValueError("Bu dönüşüm için renkli (3 kanallı) bir görüntü gereklidir.")

    img_f = img.astype(np.float32)
    R = img_f[:, :, 2]
    G = img_f[:, :, 1]
    B = img_f[:, :, 0]

    if colorspace == "RGB'den NTSC'ye":
        Y =  0.299 * R + 0.587 * G + 0.114 * B
        I =  0.596 * R - 0.275 * G - 0.321 * B
        Q =  0.212 * R - 0.523 * G + 0.311 * B

        # I ve Q'yu 0-255 aralığına taşı (+128 offset)
        Y_out = np.clip(Y,       0, 255)
        I_out = np.clip(I + 128, 0, 255)
        Q_out = np.clip(Q + 128, 0, 255)

        # BGR olarak döndür: B=Q_out, G=I_out, R=Y_out
        result = np.stack([Q_out, I_out, Y_out], axis=2).astype(np.uint8)

    else:  # NTSC'den RGB'ye
        # Girdi: B=Q+128, G=I+128, R=Y
        Y =  R.copy()
        I =  G - 128.0
        Q =  B - 128.0

        R_out =  Y + 0.956 * I + 0.621 * Q
        G_out =  Y - 0.272 * I - 0.647 * Q
        B_out =  Y - 1.105 * I + 1.702 * Q

        result = np.stack([
            np.clip(B_out, 0, 255),
            np.clip(G_out, 0, 255),
            np.clip(R_out, 0, 255)
        ], axis=2).astype(np.uint8)

    return result

def disk_kernel(r, normalizasyon):
    size=2*r+1
    kernel=np.zeros((size,size))
    
    merkez_x=int((size-1)/2)
    merkez_y=int((size-1)/2)

    for y in range(size):
        for x in range(size):
            if((x-merkez_x)**2+(y-merkez_y)**2)**(1/2) <= r:
                kernel[y,x]=1

    if normalizasyon==True:
        kernel=kernel/np.sum(kernel)
    
    return kernel

def blurring(img: np.ndarray, params: dict) -> np.ndarray:
    radius = int(params.get("radius", 3))
    kernel = disk_kernel(radius, True)

    result = np.zeros_like(img, dtype=np.uint8)
    
    if len(img.shape) == 3:
        # Renkli görüntü: Her kanalı tek tek konvolüsyona sok
        for kanal in range(img.shape[2]):
            # Dikkat: konvolusyon fonksiyonun tüm kanalı döndürmeli!
            result[:, :, kanal] = konvolusyon(img[:, :, kanal], kernel, False)
    else:
        #Gri görüntü
        result = konvolusyon(img, kernel, False)
        
    return result.astype(np.uint8)

def sobel(img: np.ndarray, params: dict) -> np.ndarray:
    """Sobel kenar bulma operatörü.

    Parametreler:
    Parametreler:
        method (str): "Tüm Kenarlar (Sobel XY)" (her iki yön), "Dikey Kenarlar (Sobel X)",
                      "Yatay Kenarlar (Sobel Y)".
    """
    method = params.get("method", "Tüm Kenarlar (Sobel XY)")

    if len(img.shape) == 3:
        gray = gri_donusum(img, params).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    if method == "Dikey Kenarlar (Sobel X)":
        Gx = konvolusyon(gray, Kx, dondurme=False)
        result = np.abs(Gx)
    elif method == "Yatay Kenarlar (Sobel Y)":
        Gy = konvolusyon(gray, Ky, dondurme=False)
        result = np.abs(Gy)
    else:  # Tüm Kenarlar (Sobel XY)
        Gx = konvolusyon(gray, Kx, dondurme=False)
        Gy = konvolusyon(gray, Ky, dondurme=False)
        result = np.sqrt(Gx ** 2 + Gy ** 2)

    return np.clip(result, 0, 255).astype(np.uint8)

def filtreler(img: np.ndarray, params: dict) -> np.ndarray:
    """Mean veya Median filtre uygular.

    Parametreler:
        filter_type (str): "Mean Filtre" veya "Median Filtre".
        ksize (int): Pencere boyutu (tek sayi, 3-21).
    """
    filter_type = params.get("filter_type", "Mean Filtre")
    ksize = int(params.get("ksize", 3))
    
    result=np.zeros_like(img, dtype=np.uint8)
    
    if filter_type=="Mean Filtre":
        kernel=np.full((ksize, ksize), 1.0/(ksize*ksize), dtype=np.float32)
        if(len(img.shape)==3):
            for kanal in range(3):
                result[:,:,kanal]=konvolusyon(img[:,:,kanal], kernel, dondurme=False)
        else:
            result = konvolusyon(img, kernel, dondurme=False);
    
    else:  # Median Filtre
        padding=ksize//2
        h,w=img.shape[:2]
        orta=(ksize*ksize)//2
        if len(img.shape)==3:
            for kanal in range(3):
                kanal_img=img[:,:,kanal].astype(np.float32)
                padded=np.pad(kanal_img,padding,mode="constant", constant_values=0)
                out=np.zeros_like(kanal_img, dtype=np.float32)
                for y in range(h):
                    for x in range(w):
                        kernel_degerleri=padded[y:y+ksize, x:x+ksize].flatten()
                        kernel_degerleri.sort()
                        out[y,x]=kernel_degerleri[orta]
                result[:,:,kanal]=np.clip(out,0,255).astype(np.uint8)
        else:
            padded=np.pad(img.astype(np.float32),padding,mode="constant", constant_values=0)
            out=np.zeros(img.shape,dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    kernel_degerleri=padded[y:y+ksize, x:x+ksize].flatten()
                    kernel_degerleri.sort()
                    out[y,x]=kernel_degerleri[orta]
            result=np.clip(out,0,255).astype(np.uint8)

    return result.astype(np.uint8)

def gurultu_ekleme(img: np.ndarray, params: dict) -> np.ndarray:
    """Salt & Pepper gürültüsü ekler.

    Parametreler:
        amount (int): Gürültü yoğunluğu (yüzde olarak).
                      Tüm piksellerin amount% kadarı siyah (pepper)
                      veya beyaz (salt) yapılır.
    """
    amount = int(params.get("amount", 5))
    x1 = params.get("x1")
    y1 = params.get("y1")
    x2 = params.get("x2")
    y2 = params.get("y2")

    if None in (x1, y1, x2, y2):
        x1=0
        y1=0
        x2=img.shape[1]
        y2=img.shape[0]

    result = img.copy()
    toplam_piksel = (x2-x1)*(y2-y1)
    gurultu_sayisi = int(toplam_piksel * amount / 100)

    #hangi pikselleri seçeceğimiz
    rng = np.random.default_rng()
    y_koord = rng.integers(y1, y2, gurultu_sayisi)
    x_koord = rng.integers(x1, x2, gurultu_sayisi)

    yari = gurultu_sayisi // 2

    result[y_koord[:yari], x_koord[:yari]] = 255
    result[y_koord[yari:], x_koord[yari:]] = 0

    return result

def morfolojik_islemler(img: np.ndarray, params: dict) -> np.ndarray:
    """Morfolojik işlemler: Genişleme, Aşınma, Açma, Kapama.

    Yalnızca binary (0/255) görüntüler üzerinde çalışır.
    Renkli veya gri görüntü verilirse Otsu yöntemiyle önce binary'e dönüştürülür.

    Parametreler:
        operation    (str): "Genişleme (Dilate)", "Aşınma (Erode)",
                            "Açma (Opening)", "Kapama (Closing)".
        kernel_shape (str): "Dikdörtgen" veya "Disk".
        ksize        (int): Dikdörtgen kernel boyutu (3-21 arası tek sayılar).
        elips_radius (int): Disk yarıçapı (Disk seçilince kullanılır).
    """
    operation    = params.get("operation",    "Genişleme (Dilate)")
    kernel_shape = params.get("kernel_shape", "Dikdörtgen")
    ksize        = int(params.get("ksize",        3))
    elips_radius = int(params.get("elips_radius", 5))

    #  Girdiyi binary'e dönüştür
    binary = binary_donusum(img, {"method": "Otsu"})  # her zaman tek kanal

    #  Kernel oluştur
    if kernel_shape == "Dikdörtgen":
        kernel = np.ones((ksize, ksize), dtype=np.uint8)
    else:  # Disk — disk_kernel normalizasyonsuz kullan
        kernel = disk_kernel(elips_radius, normalizasyon=False).astype(np.uint8)

    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    #  Dilate / Erode (binary)
    def dilate(resim: np.ndarray) -> np.ndarray:
        """Genişleme Algoritması"""
        h, w = resim.shape
        padded = np.pad(resim, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
        out = np.zeros_like(resim)
        for y in range(h):
            for x in range(w):
                giris_pikseli = resim[y, x]
                bolge = padded[y:y + kernel_h, x:x + kernel_w]
                komsular = bolge[kernel == 1]
                
                tum_komsular_1 = np.all(komsular == 255)
                tum_komsular_0 = np.all(komsular == 0)
                bazi_1_bazi_0 = not (tum_komsular_1 or tum_komsular_0)

                if giris_pikseli == 255 and tum_komsular_1:
                    out[y, x] = 255
                elif giris_pikseli == 255 and bazi_1_bazi_0:
                    out[y, x] = 255
                elif giris_pikseli == 0 and bazi_1_bazi_0:
                    out[y, x] = 255
                elif giris_pikseli == 0 and tum_komsular_0:
                    out[y, x] = 0
        return out

    def erode(resim: np.ndarray) -> np.ndarray:
        """Aşınma Algoritması"""
        h, w = resim.shape
        padded = np.pad(resim, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
        out = np.zeros_like(resim)
        for y in range(h):
            for x in range(w):
                giris_pikseli = resim[y, x]
                bolge = padded[y:y + kernel_h, x:x + kernel_w]
                komsular = bolge[kernel == 1]
                
                tum_komsular_1 = np.all(komsular == 255)
                tum_komsular_0 = np.all(komsular == 0)
                bazi_1_bazi_0 = not (tum_komsular_1 or tum_komsular_0)

                if giris_pikseli == 255 and tum_komsular_1:
                    out[y, x] = 255
                elif giris_pikseli == 255 and bazi_1_bazi_0:
                    out[y, x] = 0
                elif giris_pikseli == 0 and bazi_1_bazi_0:
                    out[y, x] = 0
                elif giris_pikseli == 0 and tum_komsular_0:
                    out[y, x] = 0
        return out

    #  İşlemi uygula
    if operation == "Genişleme (Dilate)":
        result = dilate(binary)
    elif operation == "Aşınma (Erode)":
        result = erode(binary)
    elif operation == "Açma (Opening)":
        result = dilate(erode(binary))
    else:  # Kapama (Closing)
        result = erode(dilate(binary))

    return result

# Mevcut araçlar
registry = {
    "Renk Uzayı Dönüşümleri": renk_uzayi_donusumu,
    "Gri Dönüşüm": gri_donusum,
    "Binary Dönüşüm": binary_donusum,
    "Konvolüsyon İşlemi (Gauss)": gauss_konvolüsyon,
    "Görüntü Döndürme": goruntu_dondurme,
    "Yaklaştırma / Uzaklaştırma": goruntu_olcekleme,
    "Görüntü Kırpma": resim_kirpma,
    "Histogram & Germe": histogram_germe,
    "Kontrast/Parlaklık": parlaklik_artirma,
    "Blurring": blurring,
    "Kenar Bulma (Sobel)": sobel,
    "Filtreler": filtreler,
    "Görüntüye Gürültü Ekleme": gurultu_ekleme,
    "Morfolojik İşlemler": morfolojik_islemler,
}
