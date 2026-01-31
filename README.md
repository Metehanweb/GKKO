# GKKO
Yapay zeka destekli ürün kontrol otomasyonu - Colab

-------------------------------------------------------------
#HÜCRE 1
-------------------------------------------------------------
!pip -q install ultralytics gradio opencv-python-headless
-------------------------------------------------------------
#HÜCRE 2
-------------------------------------------------------------

import os
import cv2
import numpy as np
import torch

from ultralytics import YOLO
import gradio as gr
-------------------------------------------------------------
#HÜCRE 3 !!!!!!!MODEL_PATH DİKKAT!!!!!!
-------------------------------------------------------------
# =========================
# KULLANICI AYARLARI
# =========================
MODEL_PATH = "/content/banana_yolov8n.pt"   # <-- kendi best.pt yolunu buraya yaz
IMG_SIZE   = 640                 # inference boyutu (performans için)
CONF_THRES = 0.25                # düşükse daha çok tespit gelir
IOU_THRES  = 0.45

# Model sınıflarını Türkçe isimlere eşle
CLASS_NAMES = ["tam olgun", "olgun", "olgun degil"]

# Eğer modelin class sırası farklıysa burayı değiştir:
# Örn: modelde 0="tam olgun", 1="olgun", 2="olgun değil" ise buna göre düzelt
CLASS_MAP = {
    0: "tam olgun",
    1: "olgun",
    2: "olgun degil"
}

# GPU otomatik kullan
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", DEVICE)

# =========================
# MODEL YÜKLEME
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model dosyası bulunamadı!\n"
        f"MODEL_PATH yanlış olabilir: {MODEL_PATH}\n"
        f"Çözüm: best.pt dosyanı /content altına yükleyip MODEL_PATH'i güncelle."
    )

try:
    model = YOLO(MODEL_PATH)
    # Bazı ultralytics sürümlerinde model.to çalışır; çalışmazsa predict'te device veriyoruz.
    try:
        model.to(DEVICE)
    except Exception:
        pass
    print("Model yüklendi:", MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        "Model yüklenirken hata oluştu. Lütfen best.pt dosyanın sağlam olduğundan ve yolun doğru olduğundan emin ol.\n"
        f"Hata detayı: {e}"
    )
-------------------------------------------------------------
#HÜCRE 4
-------------------------------------------------------------   
def _format_pct(conf: float) -> str:
    """0-1 arası confidence değerini yüzde metnine çevirir."""
    return f"%{conf*100:.1f}"

def predict_frame(frame: np.ndarray):
    """
    Gradio webcam'den gelen kareyi alır (RGB numpy),
    YOLOv8 ile tahmin eder, bbox+etiket çizer ve metin özet döndürür.
    """
    if frame is None:
        return None, "Kamera görüntüsü alınamadı. Tarayıcı kamera iznini kontrol et."

    # Gradio genelde RGB verir; cv2 çizim için BGR'a çeviriyoruz
    img_rgb = frame.copy()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Performans için istersen görüntüyü küçült (orantılı)
    h, w = img_bgr.shape[:2]
    scale = 1.0
    target = IMG_SIZE
    # En büyük kenarı target'a yaklaştır
    max_side = max(h, w)
    if max_side > target:
        scale = target / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # YOLO inference
    try:
        results = model.predict(
            source=img_bgr,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=0 if DEVICE == "cuda" else "cpu",
            verbose=False
        )
    except Exception as e:
        # Kullanıcıya net hata
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), f"Inference hatası: {e}"

    r = results[0]
    boxes = r.boxes

    # Hiç tespit yoksa
    if boxes is None or len(boxes) == 0:
        out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        summary = (
            "Genel değerlendirme: Muz tespit edilmedi\n"
            "Güven skoru: -\n"
            "olgun değil: 0 | olgun: 0 | tam olgun: 0"
        )
        return out_rgb, summary

    # Sınıf sayacı
    counts = {name: 0 for name in CLASS_NAMES}

    # En yüksek güvenli tespit (genel değerlendirme için)
    best_conf = -1.0
    best_label = None

    # Çizim ayarları
    thickness = max(2, int(0.003 * (img_bgr.shape[0] + img_bgr.shape[1]) / 2))
    font_scale = max(0.5, 0.6 * (img_bgr.shape[1] / 800))

    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)   # (x1,y1,x2,y2)
        cls_id = int(b.cls[0].cpu().numpy())
        conf = float(b.conf[0].cpu().numpy())

        label = CLASS_MAP.get(cls_id, f"class_{cls_id}")
        if label in counts:
            counts[label] += 1

        if conf > best_conf:
            best_conf = conf
            best_label = label

        x1, y1, x2, y2 = xyxy.tolist()

        # bbox çiz
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # etiket metni
        text = f"{label} ({_format_pct(conf)})"

        # yazı arka planı
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness-1))
        y_text = max(th + 10, y1)
        cv2.rectangle(img_bgr, (x1, y_text - th - 10), (x1 + tw + 10, y_text + baseline), (0, 255, 0), -1)
        cv2.putText(img_bgr, text, (x1 + 5, y_text - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, thickness-1), cv2.LINE_AA)

    # Metin özet
    summary = (
        f"Genel değerlendirme: {best_label}\n"
        f"Güven skoru: {_format_pct(best_conf)}\n"
        f"olgun değil: {counts.get('olgun değil', 0)} | "
        f"olgun: {counts.get('olgun', 0)} | "
        f"tam olgun: {counts.get('tam olgun', 0)}"
    )

    out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, summary
-------------------------------------------------------------
#HÜCRE 5
-------------------------------------------------------------
    with gr.Blocks() as demo:
    gr.Markdown(
        "## 🍌 YOLOv8 Muz Olgunluk Canlı Kamera\n"
        "- Kameradan gelen görüntüde muzları **olgun değil / olgun / tam olgun** olarak sınıflandırır.\n"
        "- Her tespit için bbox üstünde **etiket + % güven** gösterir.\n"
        "- Metin alanında **genel değerlendirme** (en yüksek güvenli tespit) ve sınıf adetleri yazılır.\n\n"
        "**Not:** Webcam için tarayıcı kamera izni vermelisin. Colab'da genelde `share=True` linki (HTTPS) ile daha sorunsuz çalışır."
    )

    with gr.Row():
        inp = gr.Image(
            label="Webcam (Canlı)",
            sources=["webcam"],      # bazı sürümlerde source="webcam" olabilir
            streaming=True
        )
        out_img = gr.Image(label="İşlenmiş Görüntü (bbox + etiket + % güven)")

    out_txt = gr.Textbox(label="Genel Sonuç", lines=4)

    # live=True → webcam akışında kare geldikçe fonksiyon çağrılır
    gr.Interface(
        fn=predict_frame,
        inputs=inp,
        outputs=[out_img, out_txt],
        live=True,
        allow_flagging="never"
    )

demo
-------------------------------------------------------------
#HÜCRE 6
-------------------------------------------------------------
    demo.launch(share=True, debug=True)
-------------------------------------------------------------
