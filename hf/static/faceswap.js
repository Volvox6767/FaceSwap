// FaceSwap — tarayıcı içi yüz değiştirme motoru.
// ONNX Runtime Web + InsightFace SCRFD tespit + ArcFace kimlik + inswapper_128 takas.
// Made by Ahmet Gedik — instagram.com/ahmetgedik67
'use strict';

const MODELS = {
  det:  'https://huggingface.co/volvox67/FaceSwap-models/resolve/main/det_10g.onnx',
  rec:  'https://huggingface.co/volvox67/FaceSwap-models/resolve/main/w600k_r50.onnx',
  swap: 'https://huggingface.co/volvox67/FaceSwap-models/resolve/main/inswapper_128.onnx',
  emap: 'https://huggingface.co/volvox67/FaceSwap-models/resolve/main/inswapper_emap.bin',
};

const ORT_VERSION = '1.22.0';
const ORT_JS  = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`;

// arcface 5-nokta referans şablonu (insightface utils/face_align.py)
const ARCFACE_DST = [
  [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
  [41.5493, 92.3655], [70.7299, 92.2041],
];

const DET_SIZE = 640;   // SCRFD giriş kenarı (insightface varsayılanı)

// ---------------------------------------------------------------- yardımcılar

function onProgress(el, loaded, total, label) {
  if (!el) return;
  const mb = (b) => (b / 1048576).toFixed(1);
  const pct = total ? ((loaded / total) * 100).toFixed(1) : '?';
  el.textContent = total
    ? `${label}: ${mb(loaded)} / ${mb(total)} MB (%${pct})`
    : `${label}: ${mb(loaded)} MB`;
}

async function fetchWithProgress(url, onChunk) {
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`Model indirilemedi (${res.status}): ${url}`);
  const total = +(res.headers.get('content-length') || 0);
  if (!res.body || !onChunk) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onChunk(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

/** Benzerlik dönüşümü: 5 nokta -> M (2x3), skimage SimilarityTransform ile aynı. */
function estimateNorm(lmk, imageSize) {
  const ratio = imageSize % 112 === 0 ? imageSize / 112 : imageSize / 128;
  const diffX = imageSize % 112 === 0 ? 0 : 8 * ratio;
  const dst = ARCFACE_DST.map(([x, y]) => [x * ratio + diffX, y * ratio]);
  return umeyama(lmk, dst);
}

/** Benzerlik donusumu (skimage SimilarityTransform.estimate ile ayni).
 *  M = [[a, -b], [b, a]];  dondurulen dizi canvas setTransform sirasi:
 *  [xx, yx, xy, yy, e, f] -> x' = xx*x + xy*y + e, y' = yx*x + yy*y + f */
function umeyama(src, dst) {
  const n = src.length;
  let mx = 0, my = 0, mdx = 0, mdy = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i][0]; my += src[i][1];
    mdx += dst[i][0]; mdy += dst[i][1];
  }
  mx /= n; my /= n; mdx /= n; mdy /= n;

  let den = 0, n1 = 0, n2 = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - mx, sy = src[i][1] - my;
    const dx = dst[i][0] - mdx, dy = dst[i][1] - mdy;
    den += sx * sx + sy * sy;
    n1 += sx * dx + sy * dy;
    n2 += sx * dy - sy * dx;
  }
  if (Math.abs(den) < 1e-12) return [1, 0, 0, 1, 0, 0];
  const a = n1 / den, b = n2 / den;
  const tx = mdx - (a * mx - b * my);
  const ty = mdy - (b * mx + a * my);
  return [a, b, -b, a, tx, ty];
}

/** Ters affine (M^-1) — 2x3. */
function invertAffine(m) {
  const [a, b, c, d, e, f] = m;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) return null;
  const ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
  return [ia, ib, ic, id, -(ia * e + ic * f), -(ib * e + id * f)];
}

/** M ile hizalanmış kırpm. `src` üzerine affine dönüşüm uygular. */
function warpCrop(src, m, outW, outH) {
  const out = makeScratch(outW, outH);
  const ctx = out.getContext('2d', { willReadFrequently: true });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, outW, outH);
  ctx.imageSmoothingQuality = 'high';
  ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
  ctx.drawImage(src, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return ctx.getImageData(0, 0, outW, outH);
}

function makeScratch(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c._src = null;
  return c;
}

// ------------------------------------------------------------------- SCRFD

class SCRFD {
  constructor(session, detSize = DET_SIZE) {
    this.sess = session;
    this.detSize = detSize;      // 320/480/640 — kucuk boyut belirgin hizlandirir
  }

  /**
   * SCRFD'de AveragePool + ceil() shape hesabi WebGPU'da desteklenmiyor:
   * "using ceil() in shape computation is not yet supported for AveragePool".
   * Bu yuzden tespit her zaman WASM ile calisir (hizli ve guvenilir).
   */
  static EP = ['wasm'];
  static STRIDES = [8, 16, 32];
  static NUM_ANCHORS = 2;     // det_10g: her hucrede 2 anchor (12800 = 80*80*2)
  static NMS_THRESH = 0.4;
  static centerCache = new Map();

  /** kare (ImageData/canvas) -> [{bbox:[x1,y1,x2,y2], kps:[[x,y]x5], score}] */
  async detect(canvas, threshold = 0.5) {
    const S = this.detSize;
    // insightface FaceAnalysis.detect: en-boy korunur, sonla copyMakeBorder ile S'e tamamlanir
    let scale = S / Math.max(canvas.width, canvas.height);
    scale = Math.min(scale, 1.0);
    const rw = Math.floor(canvas.width * scale + 0.5);
    const rh = Math.floor(canvas.height * scale + 0.5);

    // siyah dolgulu SxS (sol-ust hizali)
    const sq = makeScratch(S, S);
    const qctx = sq.getContext('2d', { willReadFrequently: true });
    qctx.fillStyle = '#000';
    qctx.fillRect(0, 0, S, S);
    qctx.imageSmoothingQuality = 'high';
    qctx.drawImage(canvas, 0, 0, rw, rh);

    const { data } = qctx.getImageData(0, 0, S, S);
    // blobFromImage(img, 1/128, None, (127.5,127.5,127.5), swapRB=True)
    const blob = new ort.Tensor('float32', rgbaToNCHW(data, S, S, 1 / 128, 127.5),
                                 [1, 3, S, S]);

    // cikti sirasi: once 3 seviye score, sonra 3 seviye bbox, sonra 3 seviye kps
    const outNames = this.sess.outputNames;      // ['448','471','494','451','474','497','454','477','500']
    const out = await this.sess.run({ [this.sess.inputNames[0]]: blob }, outNames);

    // insightface: her seviye stride ile carpar, ızgara merkezlerine distance eklenir
    const strides = SCRFD.STRIDES;
    const centers = SCRFD.centerCache;
    const faces = [];

    for (let idx = 0; idx < 3; idx++) {
      const stride = strides[idx];
      const scores = out[outNames[idx]].data;
      const bboxP = out[outNames[idx + 3]].data;
      const kpsP = out[outNames[idx + 6]].data;

      const height = S / stride, width = S / stride;
      const key = `${height}x${width}x${stride}`;
      let ac = centers.get(key);
      if (!ac) {
        // her hucre icin NUM_ANCHORS kez ayni merkez (insightface stack+reshape)
        const cells = height * width;
        ac = new Float32Array(cells * SCRFD.NUM_ANCHORS * 2);
        let w2 = 0;
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            for (let a = 0; a < SCRFD.NUM_ANCHORS; a++) {
              ac[w2] = x * stride; ac[w2 + 1] = y * stride;
              w2 += 2;
            }
          }
        }
        centers.set(key, ac);
      }

      const n = scores.length;
      for (let j = 0; j < n; j++) {
        if (scores[j] < threshold) continue;
        const px = ac[j * 2], py = ac[j * 2 + 1];
        const o1 = j * 4, o2 = j * 10;
        const face = {
          bbox: [
            px - bboxP[o1] * stride,
            py - bboxP[o1 + 1] * stride,
            px + bboxP[o1 + 2] * stride,
            py + bboxP[o1 + 3] * stride,
          ],
          kps: [],
          score: scores[j],
        };
        for (let k = 0; k < 5; k++) {
          // distance2kps: points[:, i%2] — i%2 daima 0, yani hep ayni merkez
          face.kps.push([px + kpsP[o2 + k * 2] * stride, py + kpsP[o2 + k * 2 + 1] * stride]);
        }
        faces.push(face);
      }
    }

    // NMS (insightface nms_thresh = 0.4)
    faces.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const f of faces) {
      let overlap = false;
      for (const k of keep) {
        const ix = Math.min(f.bbox[2], k.bbox[2]) - Math.max(f.bbox[0], k.bbox[0]) + 1;
        const iy = Math.min(f.bbox[3], k.bbox[3]) - Math.max(f.bbox[1], k.bbox[1]) + 1;
        if (ix <= 0 || iy <= 0) continue;
        const inter = ix * iy;
        const areaF = (f.bbox[2] - f.bbox[0] + 1) * (f.bbox[3] - f.bbox[1] + 1);
        const areaK = (k.bbox[2] - k.bbox[0] + 1) * (k.bbox[3] - k.bbox[1] + 1);
        if (inter / (areaF + areaK - inter) > SCRFD.NMS_THRESH) { overlap = true; break; }
      }
      if (!overlap) keep.push(f);
    }
    const out2 = keep;
    // kareye geri ölçekle: 640 uzayindaki c -> c / scale
    const det = scale;
    for (const f of out2) {
      f.bbox = f.bbox.map((v) => v / det);
      f.kps = f.kps.map(([x, y]) => [x / det, y / det]);
    }
    out2.sort((a, b) => {
      const aa = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
      const ba = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
      return ba - aa;   // en buyuk yuz once
    });
    return out2;
  }
}

// ---------------------------------------------------------------- ArcFace

class ArcFace {
  constructor(session, emap) { this.sess = session; this.emap = emap; }

  /** 112x112 hizalanmış kırpm -> {emb: Float32Array(512), latent: Float32Array(512)} */
  async embed(canvas, kps) {
    const m = estimateNorm(kps, 112);
    const data = warpCrop(canvas, m, 112, 112).data;
    // blobFromImage(img, 1/127.5, (127.5,)*3, swapRB=True) — insightface norm_crop
    const blob = new ort.Tensor('float32', rgbaToNCHW(data, 112, 112, 1 / 127.5, 127.5),
                                 [1, 3, 112, 112]);
    const res = await this.sess.run({ [this.sess.inputNames[0]]: blob });
    const emb = res[this.sess.outputNames[0]].data;
    return { emb, latent: applyEmap(emb, this.emap) };
  }
}

/** latent = emb @ emap, sonra L2 normalizasyon. */
function applyEmap(emb, emap) {
  const out = new Float32Array(512);
  for (let i = 0; i < 512; i++) {
    let s = 0;
    for (let j = 0; j < 512; j++) s += emb[j] * emap[j * 512 + i];
    out[i] = s;
  }
  let norm = 0;
  for (let i = 0; i < 512; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < 512; i++) out[i] /= norm;
  return out;
}

function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s; // ikisi de L2-normalize
}

// ---------------------------------------------------------------- inswapper

class Inswapper {
  constructor(session) { this.sess = session; }

  /** 128x128 hizalanmış hedef + latent -> 128x128 RGBA kırpm (ham float) */
  async swap(canvas, kps, latent) {
    const m = estimateNorm(kps, 128);
    const aimg = warpCrop(canvas, m, 128, 128).data;
    // inswapper input_std = 255, input_mean = 0
    const blob = new ort.Tensor('float32', rgbaToNCHW(aimg, 128, 128, 1 / 255, 0),
                                 [1, 3, 128, 128]);
    const res = await this.sess.run({
      [this.sess.inputNames[0]]: blob,
      [this.sess.inputNames[1]]: new ort.Tensor('float32', latent, [1, 512]),
    });
    const o = res[this.sess.outputNames[0]].data;   // [1,3,128,128] RGB, 0..1
    const out = new Uint8ClampedArray(128 * 128 * 4);
    const d = o;
    const plane = 128 * 128;
    for (let i = 0; i < plane; i++) {
      out[i * 4 + 0] = d[i] * 255;             // R
      out[i * 4 + 1] = d[i + plane] * 255;     // G
      out[i * 4 + 2] = d[i + 2 * plane] * 255; // B
      out[i * 4 + 3] = 255;
    }
    return { pixels: out, m };
  }
}

/** RGBA -> NCHW float32; istege bagli (v - mean) / std normalizasyonu. */
function rgbaToNCHW(rgba, w, h, scale = 1, mean = 0) {
  const out = new Float32Array(3 * w * h);
  const plane = w * h;
  for (let i = 0; i < plane; i++) {
    out[i] = (rgba[i * 4] - mean) * scale;
    out[plane + i] = (rgba[i * 4 + 1] - mean) * scale;
    out[2 * plane + i] = (rgba[i * 4 + 2] - mean) * scale;
  }
  return out;
}

// ------------------------------------------------------------------ Engine

class FaceSwapEngine {
  constructor() {
    this.det = null; this.rec = null; this.swapper = null;
    this.providers = [];
    this.detSize = 640;
  }

  async init(onStatus, opts = {}) {
    const ort = window.ort;
    if (!ort) throw new Error('ONNX Runtime yüklenemedi (internet bağlantısı?).');

    // WebGPU varsa tercih et, yoksa WASM
    let ep = ['wasm'];
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) { ep = ['webgpu', 'wasm']; }
      } catch (_) { /* WASM'a duser */ }
    }
    this.providers = ep;
    ort.env.wasm.numThreads = 1;   // crossOriginIsolated yoksa tek thread zorunlu
    ort.env.wasm.simd = true;
    onStatus(`Hesaplayici tercihi: ${ep.join(' + ')}`);

    // Her model icin WebGPU denenir, duserse WASM'a dusulur.
    const createSession = async (buf, label) => {
      try {
        return await ort.InferenceSession.create(buf, {
          executionProviders: ep, graphOptimizationLevel: 'all',
        });
      } catch (e) {
        if (!ep.includes('wasm')) throw e;
        onStatus(`${label}: WebGPU calismadi, WASM'a dusuluyor (${e.message || e}).`);
        return ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      }
    };

    // 1) emap (kucuk, once)
    onStatus('Yuz gecis matrisi (1 MB) indiriliyor...');
    const emapBuf = await fetchWithProgress(MODELS.emap,
      (l, t) => onStatus(`Yuz gecis matrisi: ${(l / 1048576).toFixed(1)} / ${(t / 1048576).toFixed(1)} MB`));
    const emap = new Float32Array(emapBuf.buffer, emapBuf.byteOffset, 512 * 512);

    // 2) SCRFD tespit (17 MB)
    onStatus('Yuz tespit modeli indiriliyor (17 MB)...');
    const detBuf = await fetchWithProgress(MODELS.det,
      (l, t) => onProgress(null, l, t, 'Yuz tespit'));
    onStatus('Yuz tespit modeli yukleniyor...');
    const detSess = await ort.InferenceSession.create(detBuf, {
      executionProviders: SCRFD.EP, graphOptimizationLevel: 'all',
    });
    this.detSize = opts.detSize || 640;
    this.det = new SCRFD(detSess, this.detSize);

    // 3) ArcFace kimlik (174 MB)
    onStatus('Kimlik modeli indiriliyor (174 MB)...');
    const recBuf = await fetchWithProgress(MODELS.rec,
      (l, t) => onStatus(`Kimlik modeli: ${(l / 1048576).toFixed(0)} / ${(t / 1048576).toFixed(0)} MB`));
    onStatus('Kimlik modeli yukleniyor...');
    const recSess = await createSession(recBuf, 'Kimlik');
    this.rec = new ArcFace(recSess, emap);

    // 4) inswapper (554 MB)
    onStatus('Takas modeli indiriliyor (554 MB) — birkac dakika surebilir...');
    const swapBuf = await fetchWithProgress(MODELS.swap,
      (l, t) => onStatus(`Takas modeli: ${(l / 1048576).toFixed(0)} / ${(t / 1048576).toFixed(0)} MB`));
    onStatus('Takas modeli yukleniyor...');
    const swapSess = await createSession(swapBuf, 'Takas');
    this.swapper = new Inswapper(swapSess);

    onStatus('Hazir.');
    return true;
  }

  /** Kaynak fotoğraftan kimlik latent'i çıkarır. */
  async identityFromPhoto(img) {
    const c = makeScratch(img.width, img.height);
    c.getContext('2d').drawImage(img, 0, 0);
    const faces = await this.det.detect(c, 0.4);
    if (!faces.length) throw new Error('Kaynak fotografta yuz bulunamadi.');
    const face = faces[0];
    const { emb, latent } = await this.rec.embed(c, face.kps);
    return { emb, latent, face };
  }

  /**
   * Tek kareyi isler. `canvas` uzerine yazar.
   *
   * SCRFD ve ArcFace tarayicida WASM ile calisir ve kare basina saniyeler
   * surer; bu yuzden `state` ile zamansal yeniden kullanim yapilir:
   *   - tespit ve embedding yalnizca `state.every` karede bir yapilir,
   *   - ara karelerde onceki landmark'lar ve latent yeniden kullanilir,
   *   - inswapper (WebGPU) her karede calisir.
   */
  async processFrame(canvas, identities, threshold, state = null) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const st = state || { frame: 0, tracks: [], every: 1 };

    const refresh = !state || (st.frame % st.every) === 0;

    if (refresh) {
      const faces = await this.det.detect(canvas, threshold);
      st.tracks = [];
      for (const face of faces) {
        const { emb } = await this.rec.embed(canvas, face.kps);
        let best = null, bestSim = -1;
        for (const id of identities) {
          const sim = cosine(emb, id.emb);
          if (sim > bestSim) { bestSim = sim; best = id; }
        }
        st.tracks.push({ kps: face.kps, latent: best ? best.latent : null, sim: bestSim });
      }
    }

    let swapped = 0;
    for (const t of st.tracks) {
      if (!t.latent) continue;                 // eslesme yoksa dokunma
      const { pixels, m } = await this.swapper.swap(canvas, t.kps, t.latent);
      pasteBack(ctx, canvas, pixels, m, t.sim);
      swapped++;
    }
    st.frame++;
    return { detected: st.tracks.length, swapped };
  }
}

/** Takilan 128x128 kırpmi affine ile geri yapistirir (yumuşak maske ile). */
function pasteBack(ctx, frameCanvas, pixels, m, sim) {
  const im = invertAffine(m);
  if (!im) return;
  const W = frameCanvas.width, H = frameCanvas.height;

  const tmp = makeScratch(128, 128);
  tmp.getContext('2d').putImageData(new ImageData(pixels, 128, 128), 0, 0);

  // hedef kirpm (maske ve harman icin)
  const target = makeScratch(128, 128);
  const tctx = target.getContext('2d', { willReadFrequently: true });
  tctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
  tctx.drawImage(frameCanvas, 0, 0);
  tctx.setTransform(1, 0, 0, 1, 0, 0);
  const aimg = tctx.getImageData(0, 0, 128, 128);

  // fark maskesi: yalnizca gercekten degisen pikseller
  const fake = new ImageData(pixels, 128, 128);
  const fdata = fake.data;
  const adata = aimg.data;
  const mask = new Uint8ClampedArray(128 * 128 * 4);
  for (let i = 0; i < 128 * 128; i++) {
    const dr = Math.abs(fdata[i * 4] - adata[i * 4]);
    const dg = Math.abs(fdata[i * 4 + 1] - adata[i * 4 + 1]);
    const db = Math.abs(fdata[i * 4 + 2] - adata[i * 4 + 2]);
    const diff = (dr + dg + db) / 3;
    const v = diff < 10 ? 0 : 255;
    mask[i * 4] = v; mask[i * 4 + 1] = v; mask[i * 4 + 2] = v; mask[i * 4 + 3] = 255;
  }
  // kenarlari kapat (insightface ayni seyi yapar)
  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) {
      if (x < 2 || y < 2 || x > 125 || y > 125) {
        const i = (y * 128 + x) * 4;
        mask[i] = mask[i + 1] = mask[i + 2] = 0;
      }
    }
  }

  // keskinlik: benzerlik yuksekse daha yumusak gecis
  const blur = sim > 0.45 ? 9 : 5;
  const mCanvas = makeScratch(128, 128);
  const mctx = mCanvas.getContext('2d');
  mctx.putImageData(new ImageData(mask, 128, 128), 0, 0);
  mctx.filter = `blur(${blur}px)`;
  mctx.drawImage(mCanvas, 0, 0);
  mctx.filter = 'none';
  const alpha = mctx.getImageData(0, 0, 128, 128);

  // harman: alpha/255 * fake + (1 - alpha/255) * orig
  const blended = new Uint8ClampedArray(128 * 128 * 4);
  for (let i = 0; i < 128 * 128; i++) {
    const a = alpha.data[i * 4 + 3] / 255;
    blended[i * 4]     = fdata[i * 4] * a + adata[i * 4] * (1 - a);
    blended[i * 4 + 1] = fdata[i * 4 + 1] * a + adata[i * 4 + 1] * (1 - a);
    blended[i * 4 + 2] = fdata[i * 4 + 2] * a + adata[i * 4 + 2] * (1 - a);
    blended[i * 4 + 3] = 255;
  }

  const paste = makeScratch(128, 128);
  paste.getContext('2d').putImageData(new ImageData(blended, 128, 128), 0, 0);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.imageSmoothingQuality = 'high';
  ctx.setTransform(im[0], im[1], im[2], im[3], im[4], im[5]);
  ctx.drawImage(paste, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

window.FaceSwap = { MODELS, ORT_JS, FaceSwapEngine, ARCFACE_DST };
