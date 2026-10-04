# 🎭 FaceSwap — High-Quality Video Face Swap

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Web UI](https://img.shields.io/badge/UI-Gradio-orange)](#-web-aray%C3%BCz%C3%BC-ile-kullan%C4%B1m)

**Yüksek kaliteli video yüz değiştirme aracı** — inswapper_128 kimlik takası,
pixel-boost çok geçişli çözünürlük artışı, GFPGAN yüz restorasyonu, anlamsal
yüz maskesi (face parsing) ve LAB ten rengi aktarımını tek bir hattta birleştirir.

> **Made by Ahmet Gedik** — [instagram.com/ahmetgedik67](https://www.instagram.com/ahmetgedik67)

---

## ✨ Özellikler

- 🎬 **Video yüz takası** — kare kare işleme, orijinal ses korunarak tek geçişte H.264 kodlama
- 🧠 **Otomatik kimlik analizi** — videoyu örnekler, kişileri kümeler, istatistik verir
- 🖼 **Önizleme modu** — tam videoyu işlemeden birkaç karede eşlemeyi doğrula
- 🎯 **Esnek eşleme** — en sık görünen (ana konuşmacı), en soldaki, ilk görünen, en yaşlı
  ya da `id=kişi` elle eşleme (`--map "0=ahmet,3=erkan"`)
- 💎 **Kalite profilleri** — `natural` / `ultra` / `fast`
- 🖥 **Web arayüzü** — tarayıcıdan sürükle-bırak, günlük akışı, galeri ve sonuç videosu
- 🎧 **Ses koruma** — orijinal ses AAC 192k olarak geri eklenir
- ⚡ **GPU desteği** — NVIDIA (CUDA) ve AMD/Intel (DirectML) otomatik algılanır

## ⚠️ Etik Kullanım

Bu araç **yalnızca yasal ve etik amaçlar** içindir:

- Kendi yüzünüzü veya **yazılı iznini aldığınız** kişilerin yüzünü değiştirin
- Başkalarının kimliğine bürünmek, aldatmak, taciz etmek veya yanlış bilgi
  yaymak için kullanmayın
- Oluşturduğu içeriklerin yapay olduğunu gizlemeyin
- Kişilerin rızası olmadan yüzlerini işleyerek mahremiyetlerini ihlal etmeyin

Kullanımın hukuki sorumluluğu kullanıcıya aittir.

## 📦 Kurulum

### 1) Depoyu indir

```bash
git clone https://github.com/<kullanıcı-adı>/FaceSwap.git
cd FaceSwap
```

### 2) Python sanal ortamı (Python 3.10–3.11 önerilir)

**Windows:**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `insightface` Windows'ta önceden derlenmiş tekerlek bulamazsa Visual C++
> Build Tools gerekebilir: `winget install Microsoft.VisualStudio.2022.BuildTools`

### 3) FFmpeg

Kodlama için FFmpeg `PATH` üzerinde olmalı:

- **Windows:** `winget install Gyan.FFmpeg` (veya [gyan.dev](https://www.gyan.dev/ffmpeg/builds/))
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### 4) Modeller

`models/` klasörüne 3 dosya gerekir. En kolayı **Web UI → Kurulum →
“Modelleri İndir”** düğmesidir. Elle indirmek isterseniz:

| Dosya | Kaynak |
|---|---|
| `inswapper_128.onnx` | [facefusion-assets → models](https://github.com/facefusion/facefusion-assets/releases/tag/models) |
| `gfpgan_1.4.onnx` | [facefusion-assets → models](https://github.com/facefusion/facefusion-assets/releases/tag/models) |
| `face_parsing.onnx` | [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) (`onnx/model.onnx`) |

```bash
# örnek (Linux/macOS)
mkdir -p models
curl -L -o models/inswapper_128.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
curl -L -o models/gfpgan_1.4.onnx    "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx"
curl -L -o models/face_parsing.onnx  "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model.onnx"
```

> Model lisanslarına dikkat: `inswapper_128` yalnızca **kişisel/araştırma**
> kullanımı için lisanslanmıştır — ticari kullanmayın.

### 🚀 Tek tıkla kurulum (Windows)

`install.bat` çift tıklayın: sanal ortam + bağımlılıklar + modelleri indirir,
ardından Web UI'yi açar.

## 🖥 Web Arayüzü ile Kullanım

```bash
python app.py
```

Tarayıcıda `http://127.0.0.1:7860` açılır.

1. **⚙️ Kurulum** sekmesinden *Sistem Kontrolü* ve *Modelleri İndir*
2. **🎬 Yüz Değiştir** sekmesi:
   - Hedef videoyu yükleyin (`.mp4`)
   - Kaynak yüz fotoğraflarını yükleyin — **dosya adı kişinin adıdır**
     (ör. `ahmet.jpeg` → ahmet yüzü)
   - *Eşleme kuralı* seçin:
     | Kural | Ne yapar |
     |---|---|
     | `top` | En sık görünen kişi = ana konuşmacı → ilk kaynak |
     | `left` | En soldaki kişi → ilk kaynak (ikinci kaynak sağdakine) |
     | `first` | İlk görünen kişi → ilk kaynak, sonra gelenler sıradaki kaynaklara |
     | `age` | En yaşlı görünen kişi → ilk kaynak |
   - **🔍 Analiz Et** → kimlik kümelerini ve yüz kırpıntılarını görün
   - **🖼 Önizle** → 4 örnek karede takası kontrol edin
   - **▶️ Tam Swap** → tüm videoyu işleyin (orijinal ses korunur)
   - Sonuç: `workspace/uploads/<proje>/video_swap_hq.mp4`

## ⌨️ Komut Satırı ile Kullanım

Kaynak fotoğrafları videonun yanına koyun (ör. `workspace/klip/ahmet.jpeg`),
ardından:

```bash
# 1) Kimlikleri analiz et
python swap_engine.py --stage analyze --clip-dir workspace/klip

# 2) Birkaç karede önizle
python swap_engine.py --stage preview --clip-dir workspace/klip --sources ahmet

# 3) Tam videoyu işle
python swap_engine.py --stage run --clip-dir workspace/klip --sources ahmet,erkan
```

### Sık kullanılan seçenekler

| Seçenek | Açıklama |
|---|---|
| `--map-by top\|left\|first\|age` | Kimlik → kaynak eşleme kuralı |
| `--map "0=ahmet,3=erkan"` | Kimlikleri elle ata (analiz çıktısındaki id numaraları) |
| `--min-count 10` | Bir kimliğin sayılması için en az örnek sayısı |
| `--match-thresh 0.30` | Yüzün bir kimliğe atanma eşiği (düşürmek zor açılar için) |
| `--only-top N` | Yalnız en sık N kimliği değiştir, arka planı koru |
| `--quality natural\|ultra\|fast` | Kalite profili (`ultra` = 512px pixel-boost) |
| `--resolution 1080p\|720p\|source` | Çıktı çözünürlüğü (kaynakla sınırlı, büyütmez) |
| `--max-frames 100` | Hızlı deneme için ilk 100 kare |
| `--out sonuc.mp4` | Çıktı dosyası |

Örnek — “soldaki kişi Ümit, sağdaki uzun olan Ahmet olsun”:

```bash
python swap_engine.py --stage run --clip-dir workspace/klip --sources umit,ahmet --map-by left
```

Örnek — arka plandaki yüzler korunarak yalnız ana konuşmacı değiştirilir:

```bash
python swap_engine.py --stage run --clip-dir workspace/klip --sources ahmet --map-by top --only-top 1
```

## 🔧 Nasıl Çalışır

```
video ──► FaceAnalysis (buffalo_l) ──► kimlik kümeleri (embedding eşleştirme)
              │
              ▼
   her kare için:
     1. inswapper_128 kimlik takası (128px latent)
     2. pixel-boost: aynı hizalı yüz 256/512px'e bölünerek çok geçişli takas
     3. GFPGAN v1.4 restorasyon (ağırlıklı harman)
     4. face-parsing (SegFormer) anlamsal maske — saç/gözlük/arka plan korunur
     5. LAB ten rengi aktarımı + sahne ışığı uyumu + doku katkısı
     6. yumuşak maske ile kareye geri yapıştırma
              │
              ▼
   FFmpeg tek geçiş: H.264 (CRF 16) + orijinal ses (AAC 192k)
```

Kalite profilleri `face_quality.json` içinde ayarlanabilir.

## 🤗 Hugging Face Space (gelecek)

Bu depo Hugging Face Space olarak da yayınlanmaya uygundur: `app.py` Gradio
tabanlıdır. models/ klasörü repoda bulunmadığı için Space üzerinde
`download_models.py` ya da Kurulum sekmesindeki "Modelleri İndir" düğmesi
kullanılmalıdır (lfs veya runtime indirme). Yayınlama planına göre
güncellenecektir.

## 🐞 Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `model missing or incomplete` | `models/` klasörüne 3 modeli indirin (Kurulum sekmesi) |
| `FFmpeg bulunamadı` | FFmpeg'i kurun ve PATH'e ekleyin |
| `Could not pick one video` | `--video` ile dosyayı belirtin |
| Değişken kare hızı hatası | `ffmpeg -i in.mp4 -vsync cfr -r 25 -c:v libx264 -crf 16 out.mp4` |
| Çok yavaş / GPU kullanılmıyor | `onnxruntime-gpu` (NVIDIA) veya `onnxruntime-directml` kurun; diğer GPU uygulamalarını kapatın |
| Kimlik yanlış kişiye gitti | Analiz kırpıntılarını kontrol edip `--map "id=kişi"` ile elle atayın |
| Zor açılarda yüz atlanıyor | `--match-thresh 0.20` gibi daha düşük eşik deneyin |

## 📁 Proje Yapısı

```
FaceSwap/
├── app.py               # Gradio web arayüzü
├── swap_engine.py       # Ana motor: analyze / preview / run
├── realism.py           # Pixel-boost, anlamsal maske, renk/ışık uyumu
├── video_io.py          # FFmpeg kodlayıcı (ses korumalı, atomik çıktı)
├── face_parser.py       # SegFormer yüz ayrıştırma (19 sınıf)
├── runtime_utils.py     # ONNX Runtime oturum ayarları
├── gpu_devices.py       # DirectML aygıt seçimi (Windows)
├── face_quality.json    # Kalite profilleri
├── download_models.py   # Model indirici (3 ONNX modeli)
├── install.bat / install.sh  # Tek tıkla kurulum
├── models/              # (indirilir) 3 ONNX modeli
└── workspace/           # videolarınız, kaynak fotoğraflar, çıktılar
```

## 🙏 Teşekkürler

- [InsightFace](https://github.com/deepinsight/insightface) — buffalo_l & inswapper_128
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — yüz restorasyonu
- [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) — anlamsal maske
- [FaceFusion](https://github.com/facefusion/facefusion-assets) — model aynası
- [Gradio](https://gradio.app) — web arayüzü

---

Made by **Ahmet Gedik** · [instagram.com/ahmetgedik67](https://www.instagram.com/ahmetgedik67)

*Yasal ve etik kullanım için yapılmıştır. Sorumlu olun.* 🎭
