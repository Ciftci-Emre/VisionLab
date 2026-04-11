import cv2
import numpy as np

def _odd(v):
    v = max(1, int(v))
    return v if v % 2 == 1 else v + 1

def gri_donusum(img, p):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def binary_donusum(img, p):
    thresh = int(p.get("threshold", 127)); maxval = int(p.get("maxval", 255))
    type_map = {"Binary": cv2.THRESH_BINARY, "Binary Inv": cv2.THRESH_BINARY_INV,
                "Trunc": cv2.THRESH_TRUNC, "ToZero": cv2.THRESH_TOZERO, "ToZero Inv": cv2.THRESH_TOZERO_INV}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, r = cv2.threshold(gray, thresh, maxval, type_map.get(p.get("method","Binary"), cv2.THRESH_BINARY))
    return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

def renk_uzayi(img, p):
    code_map = {"HSV": cv2.COLOR_BGR2HSV,
                "YCrCb": cv2.COLOR_BGR2YCrCb, "Grayscale": cv2.COLOR_BGR2GRAY}
    c = cv2.cvtColor(img, code_map.get(p.get("colorspace","HSV"), cv2.COLOR_BGR2HSV))
    return cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) if c.ndim == 2 else c

def goruntu_dondurme(img, p):
    angle = float(p.get("angle", 90)); scale = float(p.get("scale", 100)) / 100.0
    expand = bool(p.get("expand", True)); h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), -angle, scale)
    if expand:
        ca, sa = abs(M[0,0]), abs(M[0,1])
        nw, nh = int(h*sa+w*ca), int(h*ca+w*sa)
        M[0,2] += (nw-w)/2; M[1,2] += (nh-h)/2
        return cv2.warpAffine(img, M, (nw, nh))
    return cv2.warpAffine(img, M, (w, h))

def goruntu_kirpma(img, p):
    x1,y1,x2,y2 = int(p.get("x1",0)),int(p.get("y1",0)),int(p.get("x2",img.shape[1])),int(p.get("y2",img.shape[0]))
    r = img[y1:y2, x1:x2]
    if r.size == 0: raise ValueError("Geçersiz seçim. Tuval üzerinde bir alan seçin.")
    return r

def yaklastirma(img, p):
    pct = float(p.get("percent", 150)) / 100.0
    im = {"Linear": cv2.INTER_LINEAR, "Cubic": cv2.INTER_CUBIC,
          "Nearest": cv2.INTER_NEAREST, "Lanczos": cv2.INTER_LANCZOS4}
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*pct), int(h*pct)), interpolation=im.get(p.get("interpolation","Linear"), cv2.INTER_LINEAR))

def parlaklik(img, p):
    return cv2.convertScaleAbs(img, alpha=float(p.get("alpha",10))/10.0, beta=float(p.get("beta",30)))

def histogram_germe(img, p):
    method = p.get("method","Histogram Çıkarma")
    if method == "Histogram Çıkarma":
        return img
    if method == "Histogram Germe":
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV); yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    if method == "Histogram Genişletme":
        return cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
    return img

def konvolusyon_gauss(img, p):
    k = _odd(int(p.get("ksize",5))); s = float(p.get("sigma_x",0))
    return cv2.GaussianBlur(img, (k,k), s)

def blurring(img, p):
    bt = p.get("blur_type","Gaussian"); k = _odd(int(p.get("ksize",5)))
    if bt == "Gaussian":       return cv2.GaussianBlur(img, (k,k), 0)
    if bt == "Average (Box)":  return cv2.blur(img, (k,k))
    if bt == "Median":         return cv2.medianBlur(img, k)
    if bt == "Bilateral":      return cv2.bilateralFilter(img, k, 75, 75)
    return img

def kenar_bulma(img, p):
    m = p.get("method","Sobel XY"); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if m == "Sobel XY":
        sx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); sy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        r = cv2.convertScaleAbs(cv2.magnitude(sx,sy))
    elif m == "Sobel X": r = cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3))
    elif m == "Sobel Y": r = cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3))
    else: r = gray
    return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

def esikleme(img, p):
    m = p.get("method","Adaptive Gaussian"); bs = _odd(int(p.get("block_size",11))); C = int(p.get("C",2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if m == "Otsu":             _, r = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif m == "Global":         _, r = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    elif m == "Adaptive Mean":  r = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,bs,C)
    else:                       r = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bs,C)
    return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

def gurultu_ekle(img, p):
    amount = float(p.get("amount",5))/100.0; r = img.copy()
    n = int(img.size * amount / 2)
    c = [np.random.randint(0,i-1,n) for i in img.shape[:2]]
    r[c[0],c[1]] = 255; c = [np.random.randint(0,i-1,n) for i in img.shape[:2]]; r[c[0],c[1]] = 0
    d = p.get("denoise","Yok")
    if d == "Mean":    r = cv2.blur(r,(3,3))
    elif d == "Median": r = cv2.medianBlur(r,3)
    return r

def aritmetik(img, p):
    op = p.get("operation","Ekleme (Add)")
    img2 = cv2.flip(img, 1)
    if op == "Ekleme (Add)":      return cv2.add(img, img2)
    if op == "Çarpma (Multiply)": return cv2.multiply(img, img2)
    return img

def morfoloji(img, p):
    op = p.get("operation","Genişleme (Dilate)"); ks = int(p.get("ksize",3)); it = int(p.get("iterations",1))
    sm = {"Dikdörtgen": cv2.MORPH_RECT, "Elips": cv2.MORPH_ELLIPSE, "Çapraz": cv2.MORPH_CROSS}
    om = {"Genişleme (Dilate)": cv2.MORPH_DILATE, "Aşınma (Erode)": cv2.MORPH_ERODE,
          "Açma (Opening)": cv2.MORPH_OPEN, "Kapama (Closing)": cv2.MORPH_CLOSE}
    kernel = cv2.getStructuringElement(sm.get(p.get("kernel_shape","Dikdörtgen"), cv2.MORPH_RECT), (ks,ks))
    return cv2.morphologyEx(img, om.get(op, cv2.MORPH_DILATE), kernel, iterations=it)

registry = {
    "Gri Dönüşüm": gri_donusum,
    "Binary Dönüşüm": binary_donusum,
    "Renk Uzayı Dönüşümleri": renk_uzayi,
    "Görüntü Döndürme": goruntu_dondurme,
    "Görüntü Kırpma": goruntu_kirpma,
    "Yaklaştırma / Uzaklaştırma": yaklastirma,
    "Parlaklık Artırma": parlaklik,
    "Histogram & Germe": histogram_germe,
    "Konvolüsyon İşlemi (Gauss)": konvolusyon_gauss,
    "Blurring": blurring,
    "Kenar Bulma (Sobel)": kenar_bulma,
    "Eşikleme (Adaptif)": esikleme,
    "Görüntüye Gürültü Ekleme": gurultu_ekle,
    "Aritmetik İşlemler": aritmetik,
    "Morfolojik İşlemler": morfoloji,
}
