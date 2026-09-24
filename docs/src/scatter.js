/**
 * scatter.js -- the whole test population, one dot per pupil.
 *
 * Data is the PUBLISHED evaluation of the noiseless U-Net over the test split:
 * every dot is one pupil's metric averaged over 1000 held-out wavefronts, read from
 * test_loss_vs_asym_metrics.npz beside the checkpoint. Nothing here is recomputed in
 * the browser, so the dots are the paper's own numbers.
 *
 * Assets (fetched lazily, on first draw):
 *   assets/scatter.json  metadata + the 30-bin published means
 *   assets/scatter.bin   1500 x 5 float32, column-major: alpha, loss, psfdiff,
 *                        throughput, strehl
 *   assets/shapes_all.bin  uint16 vertex counts x1500, then uint8 xy pairs (exact
 *                          integer pixel coords, already truncated as cv2 does)
 */

import { forAlpha, forAlphaA, outline } from './plasma.js';

const N = 128;

/* Content stamp on every asset fetch. These files change whenever the evaluation is
   re-run, and a browser happily serves a stale scatter.json against a fresh
   scatter.bin -- which silently draws the guide curve at the wrong x positions. */
const V = '1f9e781ac4';

let data = null;           // {n, alpha, loss, psfdiff, thru, meta}
let shapes = null;         // Array<Float32Array> of xy pairs in 128-grid pixels
let loading = null;

export const FIELDS = Object.freeze([
  { id: 'loss',    label: 'Estimation error',  unit: 'MSE (rad²)', better: 'lower'  },
  { id: 'psfdiff', label: 'PSF difference',    unit: '',           better: 'higher' },
  { id: 'strehl',  label: 'Corrected Strehl',  unit: '',           better: 'higher' },
]);

export async function load(base = './assets/') {
  if (data) return data;
  if (loading) return loading;
  loading = (async () => {
    const [meta, bin, shp] = await Promise.all([
      fetch(base + 'scatter.json?v=' + V).then(r => r.json()),
      fetch(base + 'scatter.bin?v=' + V).then(r => r.arrayBuffer()),
      fetch(base + 'shapes_all.bin?v=' + V).then(r => r.arrayBuffer()),
    ]);
    const n = meta.n;
    const f = new Float32Array(bin);
    const want = n * meta.fields.length;
    if (f.length !== want) {
      throw new Error(`scatter.bin has ${f.length} floats, metadata expects ${want} `
        + `(${meta.fields.length} fields x ${n}) -- mismatched assets, likely a stale cache`);
    }
    data = {
      n, meta,
      alpha:   f.subarray(0 * n, 1 * n),
      loss:    f.subarray(1 * n, 2 * n),
      psfdiff: f.subarray(2 * n, 3 * n),
      thru:    f.subarray(3 * n, 4 * n),
      strehl:  f.subarray(4 * n, 5 * n),
    };
    const counts = new Uint16Array(shp, 0, n);
    const xy = new Uint8Array(shp, n * 2);
    shapes = []; let o = 0;
    for (let i = 0; i < n; i++) {
      const c = counts[i], p = new Float32Array(c * 2);
      for (let k = 0; k < c * 2; k++) p[k] = xy[o + k];
      shapes.push(p); o += c * 2;
    }
    return data;
  })();
  return loading;
}

export function get() { return data; }
export function shapeOf(i) { return shapes ? shapes[i] : null; }

/* The exact published masks, fetched only if someone actually selects a pupil.
   3 MB, same packing as ladder.bin: 2048 B per frame, 1 bit/px, p = y*128+x, LSB-first.
   Rasterising the outlines in JS instead gets within ~1% of area but NOT bit-exact --
   cv2.fillPoly's fixed-point edge rule differs -- and an inexact mask would report an
   alpha that disagrees with the paper, so we ship the real thing. */
let masks = null, masksLoading = null;
export async function loadMasks(base = './assets/') {
  if (masks) return masks;
  if (masksLoading) return masksLoading;
  masksLoading = fetch(base + 'masks_all.bin?v=' + V)
    .then(r => r.arrayBuffer())
    .then(b => (masks = new Uint8Array(b)));
  return masksLoading;
}
export function maskOf(i) {
  if (!masks) return null;
  const out = new Float64Array(N * N), off = i * 2048;
  for (let p = 0; p < N * N; p++) out[p] = (masks[off + (p >> 3)] >> (p & 7)) & 1;
  return out;
}
export function masksReady() { return !!masks; }

/* ---------- drawing ---------- */

function niceTicks(lo, hi, want = 5) {
  const span = hi - lo || 1;
  const raw = span / want;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

/**
 * @param {HTMLCanvasElement} cv
 * @param {object} o  {field, tokens:{ink,muted,faint,line,surface,accent,sunk},
 *                     marked:number[]|null, current:number|null, hover:number|null}
 * @returns {object|null} the plot geometry, for hit-testing
 */
export function draw(cv, o = {}) {
  if (!data) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect();
  if (!r.width) return null;
  const W = r.width, H = r.height;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  const t = o.tokens || {};
  const fieldId = o.field || 'loss';
  const ys = data[fieldId];
  const L = 58, R = 14, T = 12, B = 38;
  const iw = W - L - R, ih = H - T - B;
  if (iw < 40 || ih < 40) return null;

  let ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < data.n; i++) { const v = ys[i]; if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  const pad = (ymax - ymin) * 0.06;
  ymin = Math.max(0, ymin - pad); ymax += pad;
  const xmin = 0, xmax = 0.36;
  const X = a => L + ((a - xmin) / (xmax - xmin)) * iw;
  const Y = v => T + ih - ((v - ymin) / (ymax - ymin)) * ih;

  const mono = '11px ' + (t.mono || 'monospace');
  g.font = mono; g.lineWidth = 1;

  // grid + axes, labelled at values the data actually reaches
  g.strokeStyle = t.line || '#ddd'; g.fillStyle = t.faint || '#888';
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const v of niceTicks(ymin, ymax, 5)) {
    const y = Y(v);
    if (y < T - 1 || y > T + ih + 1) continue;
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(2), L - 8, y);
  }
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const a of niceTicks(xmin, xmax, 5)) {
    if (a < xmin - 1e-9 || a > xmax + 1e-9) continue;
    g.fillText(a.toFixed(2), X(a), T + ih + 8);
  }
  g.fillStyle = t.muted || '#666';
  g.fillText('asymmetry α', L + iw / 2, T + ih + 22);
  g.save(); g.translate(14, T + ih / 2); g.rotate(-Math.PI / 2);
  g.textAlign = 'center'; g.textBaseline = 'middle';
  const fdef = FIELDS.find(f => f.id === fieldId);
  g.fillText(fdef ? fdef.label : fieldId, 0, 0); g.restore();

  // the population
  // dots carry the paper's plasma scale, so a pupil has the same colour here as on the
  // slider -- you can match a chip to its dot by eye
  const acc = t.accent || '#1B96DC';
  for (let i = 0; i < data.n; i++) {
    g.fillStyle = forAlphaA(data.alpha[i], 0.5);
    g.beginPath(); g.arc(X(data.alpha[i]), Y(ys[i]), 2.1, 0, Math.PI * 2); g.fill();
  }

  /* Guide curve: the MEDIAN, with an interquartile band.
     The mean is also shipped, but within a bin these distributions are skewed -- for
     Strehl the mean runs up to 0.31 IQR below the median -- so a mean line plots below
     the dense band the eye reads as "the cluster" and looks misplaced even though it is
     correct. The median tracks that band, and the shaded quartiles show the asymmetry
     instead of hiding it. */
  const bc = data.meta.bin_centers;
  const med = data.meta['bin_median_' + fieldId];
  const q1 = data.meta['bin_q1_' + fieldId], q3 = data.meta['bin_q3_' + fieldId];
  if (bc && med) {
    if (q1 && q3) {
      g.beginPath();
      for (let i = 0; i < bc.length; i++) g[i ? 'lineTo' : 'moveTo'](X(bc[i]), Y(q3[i]));
      for (let i = bc.length - 1; i >= 0; i--) g.lineTo(X(bc[i]), Y(q1[i]));
      g.closePath();
      g.fillStyle = withAlpha(t.ink || '#111', 0.10);
      g.fill();
    }
    g.strokeStyle = t.ink || '#111'; g.lineWidth = 2; g.beginPath();
    for (let i = 0; i < bc.length; i++) g[i ? 'lineTo' : 'moveTo'](X(bc[i]), Y(med[i]));
    g.stroke(); g.lineWidth = 1;
  }

  // the pupils that appear on the slider
  if (o.marked) {
    for (const i of o.marked) {
      if (i == null) continue;
      g.beginPath(); g.arc(X(data.alpha[i]), Y(ys[i]), 3.4, 0, Math.PI * 2);
      g.fillStyle = forAlpha(data.alpha[i]); g.fill();
      g.lineWidth = 1; g.strokeStyle = outline(!!o.dark); g.stroke();
    }
  }
  // where the slider is right now
  if (o.current != null) {
    const x = X(data.alpha[o.current]), y = Y(ys[o.current]);
    g.beginPath(); g.arc(x, y, 7, 0, Math.PI * 2);
    g.fillStyle = forAlpha(data.alpha[o.current]); g.fill();
    g.lineWidth = 3; g.strokeStyle = t.surface || '#fff'; g.stroke();
    g.lineWidth = 1.2; g.strokeStyle = outline(!!o.dark); g.stroke();
  }
  if (o.hover != null && o.hover !== o.current) {
    const x = X(data.alpha[o.hover]), y = Y(ys[o.hover]);
    g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2);
    g.fillStyle = t.ink || '#111'; g.fill();
    g.lineWidth = 2; g.strokeStyle = t.surface || '#fff'; g.stroke(); g.lineWidth = 1;
  }
  return { L, R, T, B, iw, ih, X, Y, field: fieldId, W, H };
}

function withAlpha(hex, a) {
  const h = hex.replace('#', '');
  const s = h.length === 3 ? h.split('').map(c => c + c).join('') : h;
  return `rgba(${parseInt(s.slice(0, 2), 16)},${parseInt(s.slice(2, 4), 16)},${parseInt(s.slice(4, 6), 16)},${a})`;
}

/** Nearest dot to a point in CSS pixels, or null past `maxPx`. */
export function pick(geom, px, py, maxPx = 14) {
  if (!geom || !data) return null;
  const ys = data[geom.field];
  let best = -1, bd = maxPx * maxPx;
  for (let i = 0; i < data.n; i++) {
    const dx = geom.X(data.alpha[i]) - px, dy = geom.Y(ys[i]) - py;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best < 0 ? null : best;
}

/** Draw pupil `i`'s outline into a small canvas. */
export function drawShape(cv, i, fill, bg, stroke) {
  const p = shapeOf(i); if (!p) return;
  const g = cv.getContext('2d'), W = cv.width;
  g.clearRect(0, 0, W, W);
  if (bg) { g.fillStyle = bg; g.fillRect(0, 0, W, W); }
  const VX = 40, VS = 48, k = W / VS;
  g.fillStyle = fill; g.beginPath();
  for (let v = 0; v < p.length; v += 2) {
    const x = (p[v] - VX) * k, y = (p[v + 1] - VX) * k;
    v ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.closePath(); g.fill();
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.5; g.stroke(); }
}
