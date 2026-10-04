---
title: FaceSwap
emoji: 🎭
colorFrom: indigo
colorTo: gray
sdk: static
license: mit
pinned: false
---

# 🎭 FaceSwap — Ahmet Gedik

**Tarayıcıda çalışan video yüz değiştirme.** Sunucu yok, ücret yok, kurulum yok —
tüm işlemler kendi bilgisayarınızda ONNX Runtime ile yapılır.

> Made by **Ahmet Gedik** · [instagram.com/ahmetgedik67](https://www.instagram.com/ahmetgedik67)

## Nasıl kullanılır

1. **⬇️ Modelleri indir ve başlat** — toplam ~745 MB, tarayıcı önbelleğinde tutulur
2. **Kaynak yüz fotoğrafı** yükle (net, tek yüz içermesi iyi olur)
3. **Video** yükle ve **▶️ Videoyu işle**

Chrome / Edge önerilir (WebGPU). İlk açılışta model indirmesi birkaç dakika sürer.

## Nasıl çalışır

| Aşama | Model | Boyut | Nerede çalışır |
|---|---|---|---|
| Yüz tespiti + 5 nokta | SCRFD `det_10g` | 17 MB | WASM |
| Kimlik vektörü (512-D) | ArcFace `w600k_r50` | 174 MB | WASM |
| Yüz geçiş matrisi | `emap` | 1 MB | — |
| Takas üretimi | `inswapper_128` | 554 MB | WebGPU (yoksa WASM) |

Modeller [volvox67/FaceSwap-models](https://huggingface.co/volvox67/FaceSwap-models)
deposundan, CORS açık olarak sunulur.

**Tünelleme:** SCRFD ve ArcFace her karede saniyeler sürdüğü için tespit ve kimlik
*eşleştirmesi* her N karede bir yapılır, ara karelerde landmark'lar yeniden
kullanılır. Takas (inswapper) her karede çalışır.

## ⚠️ Bu sürümün sınırları

- **Ses korunmaz** — tarayıcıda ses mux desteklenmiyor (FFmpeg ile sonradan eklenebilir)
- **GFPGAN restorasyonu, anlamsal yüz maskesi ve LAB ten rengi uyumu yok** — bunlar
  tarayıcıda çok ağır. Sonuç sunucu sürümünden belirgin şekilde daha yumuşak olur.
- Uzun videolar yavaş (CPU'da ~2 sn/kare)

**Tam kalite (sunucu sürümü):** [github.com/Volvox6767/FaceSwap](https://github.com/Volvox6767/FaceSwap)

## ⚠️ Etik kullanım

Yalnızca yasal ve etik amaçlar için: kendi yüzünüzü veya **yazılı izni aldığınız**
kişilerin yüzünü değiştirin. Başkalarını taklit etmek veya mahremiyet ihlali için kullanmayın.

## Teşekkürler

[InsightFace](https://github.com/deepinsight/insightface) ·
[GFPGAN](https://github.com/TencentARC/GFPGAN) ·
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

Made with ❤️ by **Ahmet Gedik** · [instagram.com/ahmetgedik67](https://www.instagram.com/ahmetgedik67)
