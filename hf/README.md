---
title: FaceSwap
emoji: 🎭
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: High-quality video face swap with Gradio UI
---

# 🎭 FaceSwap — Ahmet Gedik

**High-quality video face swap**, tarayıcıdan: video + kaynak yüz fotoğrafı yükle → kimlikleri analiz et → önizle → tam takas.

> Made by **Ahmet Gedik** · [instagram.com/ahmetgedik67](https://www.instagram.com/ahmetgedik67)

## Nasıl kullanılır

1. **Kurulum** sekmesi: *Sistem Kontrolü*'ne bas. Modeller ilk açılışta otomatik
   indirilir (~950 MB, birkaç dakika — Space yeniden başlatıldığında tekrar indirilmez).
2. **Yüz Değiştir** sekmesi:
   - Videoyu yükle
   - Kaynak yüz fotoğrafını yükle (**dosya adı kişinin adıdır**, ör. `ahmet.jpeg`)
   - Eşleme kuralı: `top` (ana konuşmacı) · `left` (soldaki) · `first` (ilk çıkan) · `age` (en yaşlı)
   - **🔍 Analiz Et** → **🖼 Önizle** → **▶️ Tam Swap**

## Performans notu

- Bu Space **CPU** ile çalışır (ücretsiz katman). Yaklaşık **1–3 saniye/kare**.
  30 saniyelik bir klip için ~1 dakika sürebilir.
- **GPU (T4) hardware'ı** seçersen çok daha hızlıdır (Space ayarlarından
  "Hardware" → GPU seçebilirsin; ücretsiz krediyle birkaç saat).
- Yüklenen videolar Space'e kaydedilmez; iş bittikten sonra dosyalar
  `workspace/` altında tutulur ve Space yeniden başlatıldığında temizlenir.

## Nasıl çalışır

```
inswapper_128 kimlik takası (128px latent)
  → pixel-boost (256/512px, çok geçişli hizalama)
  → GFPGAN v1.4 restorasyon (ağırlıklı harman)
  → SegFormer face-parsing anlamsal maske (saç/gözlük korunur)
  → LAB ten rengi + sahne ışığı uyumu
  → yumuşak maske ile geri yapıştırma
  → FFmpeg tek geçiş: H.264 + orijinal ses (AAC)
```

## ⚠️ Etik kullanım

Yalnızca yasal ve etik amaçlar için: kendi yüzünüzü veya **yazılı izni aldığınız**
kişilerin yüzünü değiştirin. Başkalarını taklit etmek, aldatmak veya mahremiyet
ihlali için kullanmayın.

## Kaynak kod

Engine, CLI ve yerel kurulum: **github.com/Volvox6767/FaceSwap**

Teşekkürler: [InsightFace](https://github.com/deepinsight/insightface) ·
[GFPGAN](https://github.com/TencentARC/GFPGAN) ·
[jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) ·
[FaceFusion](https://github.com/facefusion/facefusion-assets) · [Gradio](https://gradio.app)

Made with ❤️ by **Ahmet Gedik** · [instagram.com/ahmetgedik67](https://www.instagram.com/ahmetgedik67)