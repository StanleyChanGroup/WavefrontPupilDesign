/**
 * net.js -- browser-side inference for the published wavefront U-Net.
 *
 * Runs `assets/unet_fp16.onnx` with onnxruntime-web (WASM execution provider,
 * single-threaded) and wires the result into the verified forward model in
 * `optics.js`. No UI, no DOM, no globals: everything here is a pure function of
 * its arguments except the module-level session cache.
 *
 * ---------------------------------------------------------------------------
 * Runtime choice (see vendor/VENDOR.md for the full rationale)
 * ---------------------------------------------------------------------------
 *   backend      : 'wasm' (CPU). No WebGPU/WebGL -- the fp16 graph's
 *                  Conv/ConvTranspose set is fully covered by the WASM EP, the
 *                  numbers below are proven against PyTorch, and WebGPU would
 *                  need a second verification pass plus a fallback path.
 *   threads      : ort.env.wasm.numThreads = 1.
 *                  The shipped wasm binary is the `-threaded` build, but its
 *                  worker pool needs `SharedArrayBuffer` sharing, which needs
 *                  COOP/COEP response headers, which GitHub Pages cannot set.
 *                  Pinning numThreads = 1 means no worker is ever spawned, so
 *                  the page works identically with and without cross-origin
 *                  isolation. (ORT would clamp to 1 by itself off an isolated
 *                  origin; we set it explicitly so behaviour is not
 *                  environment-dependent.) The module itself imports a *shared*
 *                  WebAssembly.Memory, which browsers allow to be constructed
 *                  without cross-origin isolation -- that is the emscripten
 *                  path ORT relies on, and it is why no COOP/COEP is needed.
 *   proxy        : false (a proxy worker is pointless with one thread).
 *   SIMD         : required. onnxruntime-web >= 1.18 ships no non-SIMD binary.
 *                  Baseline: Chrome 91+, Firefox 89+, Safari 16.4+.
 *   wasm path    : ort.env.wasm.wasmPaths is set to the OBJECT form
 *                  `{ wasm: <url> }` (a directory string would make ORT fetch a
 *                  second glue file we deliberately do not vendor -- see the
 *                  comment in getOrt()), resolved RELATIVE to this module's own
 *                  URL (`../vendor/`), so the site can be served from any
 *                  sub-path of any static host with no configuration.
 *
 * ---------------------------------------------------------------------------
 * Piston
 * ---------------------------------------------------------------------------
 * The network output carries an arbitrary constant offset. That is not a defect
 * -- piston is physically unobservable here:
 *
 *   PSF(phi + c) = |FFT{ P e^{i(phi+c)} }|^2 = |e^{ic}|^2 |FFT{ P e^{i phi} }|^2
 *                = |FFT{ P e^{i phi} }|^2
 *
 * because a constant phase factors out of the (linear) transform as a
 * unit-modulus scalar and the modulus-squared then discards it. So the
 * measurement the network sees is blind to piston, and every intensity-domain
 * quantity built from it -- the corrected PSF, and therefore the Strehl ratio
 * -- is exactly piston-invariant. `strehlOf()` needs no piston handling at all.
 *
 * The MSE is a phase-domain quantity and is *not* invariant: adding c to the
 * prediction changes sum((pred - gt)^2) by 2c*sum(pred-gt) + c^2*n. Penalising
 * an unobservable degree of freedom would train the network to guess a constant
 * it cannot see, so the published loss subtracts the in-pupil mean from both
 * fields first. `optics.maskedPistonCorrectedMse` (re-exported here as `mseOf`)
 * does exactly that, so it also needs no piston handling from the caller.
 *
 * Only *display* needs it: a raw residual map is offset by an arbitrary
 * constant and looks wrong. Use `removePiston(field, mask)` for that, or
 * `residual(gt, pred, mask, { removePiston: true })`.
 */

import {
  N_DEFAULT,
  RADIUS_DEFAULT,
  psfFromPupil,
  psfPeak,
  strehlRatio,
  maskedPistonCorrectedMse,
  maskedPhase,
  simulateMeasurement,
  throughput,
  asymmetryCorrected,
  synthPhase,
  PAPER_MODES,
} from './optics.js';

/* ------------------------------------------------------------------ *
 * Constants -- the published network contract
 * ------------------------------------------------------------------ */

/** Pinned onnxruntime-web version vendored under vendor/. */
export const ORT_VERSION = '1.30.0';
/** ORT entry point, relative to this module. WASM EP only, glue JS inlined. */
export const ORT_MODULE_URL = '../vendor/ort.wasm.bundle.min.mjs';
/** Directory holding `ort-wasm-simd-threaded.wasm`, relative to this module. */
export const ORT_WASM_DIR = '../vendor/';
/** The one wasm binary the bundle build needs. */
export const ORT_WASM_FILE = 'ort-wasm-simd-threaded.wasm';
/** Default model path, relative to this module. */
export const MODEL_URL = '../assets/unet_fp16.onnx';

/** Graph input name: float32 [1,2,128,128]. */
export const INPUT_NAME = 'input';
/** Graph output name: float32 [1,1,128,128], dense phase in radians. */
export const OUTPUT_NAME = 'phase';
/** Spatial size. */
export const N = N_DEFAULT;
/** Pupil radius the weights were trained at. */
export const RADIUS = RADIUS_DEFAULT;
const N2 = N * N;

/* ------------------------------------------------------------------ *
 * ORT bootstrap
 * ------------------------------------------------------------------ */

let ortPromise = null;
/** Overrides applied by `configure()` before ORT is first imported. */
let ortConfig = { wasmPaths: null, wasmBinary: null, numThreads: 1, logLevel: 'error' };

/**
 * Override the ORT environment. Must be called before the first `loadModel()`.
 * The browser never needs this; the Node verification harness uses it to hand
 * ORT the wasm bytes directly (Node has no relative-URL fetch).
 *
 * @param {object} opts
 * @param {string|{wasm:string}} [opts.wasmPaths]  directory (trailing '/') or {wasm: url}
 * @param {ArrayBuffer|Uint8Array} [opts.wasmBinary]  pre-read wasm, bypasses wasmPaths
 * @param {number} [opts.numThreads=1]
 * @param {string} [opts.logLevel='error']
 * @param {string} [opts.ortModuleUrl]  override the ORT bundle location
 */
export function configure(opts = {}) {
  if (ortPromise) throw new Error('net.js: configure() must be called before loadModel()');
  ortConfig = { ...ortConfig, ...opts };
}

/**
 * Import and configure onnxruntime-web exactly once.
 * @returns {Promise<object>} the `ort` namespace, already configured.
 */
export async function getOrt() {
  if (ortPromise) return ortPromise;
  ortPromise = (async () => {
    const url = ortConfig.ortModuleUrl ?? new URL(ORT_MODULE_URL, import.meta.url).href;
    const ort = await import(/* @vite-ignore */ url);

    // Single-threaded, no proxy worker: works without COOP/COEP (see header).
    ort.env.wasm.numThreads = ortConfig.numThreads ?? 1;
    ort.env.wasm.proxy = false;
    ort.env.logLevel = ortConfig.logLevel ?? 'error';

    if (ortConfig.wasmBinary) {
      // Node / bundler path: hand ORT the bytes, no locateFile lookup at all.
      const b = ortConfig.wasmBinary;
      ort.env.wasm.wasmBinary = b instanceof Uint8Array ? b.buffer : b;
    } else {
      // Browser path.
      //
      // Two traps, both load-bearing:
      //
      // 1. `wasmPaths` MUST be the object form `{ wasm: <url to the .wasm> }`.
      //    Setting it to a *directory string* makes ORT decide the caller is
      //    hosting the split build and go fetch `ort-wasm-simd-threaded.mjs`
      //    from that directory -- discarding the emscripten glue that is
      //    already inlined in ort.wasm.bundle.min.mjs. vendor/ deliberately does
      //    not contain that second file, so a string would 404 (in Node it fails
      //    outright: "Only URLs with a scheme in: file and data are supported").
      //    The object form leaves the inlined glue in place and only overrides
      //    the binary's location, which is exactly what we want.
      //
      // 2. The URL is resolved against THIS MODULE's url, not the document's.
      //    A bare relative string would be resolved by ORT against the page URL
      //    and break the moment the site is served from a sub-path such as
      //    https://user.github.io/pupil-page/.
      const wp = ortConfig.wasmPaths;
      ort.env.wasm.wasmPaths =
        wp == null
          ? { wasm: new URL(ORT_WASM_DIR + ORT_WASM_FILE, import.meta.url).href }
          : typeof wp === 'string'
            ? { wasm: new URL(ORT_WASM_FILE, wp.endsWith('/') ? wp : wp + '/').href }
            : wp;
    }
    return ort;
  })();
  return ortPromise;
}

/* ------------------------------------------------------------------ *
 * Model loading
 * ------------------------------------------------------------------ */

/** url -> Promise<ModelHandle>. Makes loadModel() idempotent. */
const sessionCache = new Map();

/**
 * @typedef {object} ModelHandle
 * @property {object} session        the live ort.InferenceSession
 * @property {object} ort            the configured ort namespace
 * @property {string} url            resolved model URL
 * @property {number} bytes          model size actually transferred
 * @property {string} inputName
 * @property {string} outputName
 * @property {{downloadMs:number, initMs:number, totalMs:number,
 *             firstInferenceMs:number|null}} timing
 * @property {number} inferences     completed predictPhase() calls
 */

/**
 * Load the ONNX model and create an inference session. Idempotent: calling it
 * again with the same URL returns the same promise (and therefore the same
 * session) without re-downloading. Concurrent callers share one download.
 *
 * @param {string} [url=MODEL_URL]  model URL, resolved against this module
 * @param {object} [opts]
 * @param {ArrayBuffer|Uint8Array} [opts.buffer]  pre-fetched model bytes (skips download)
 * @param {(p:{loaded:number,total:number})=>void} [opts.onProgress]  download progress
 * @param {boolean} [opts.reload=false]  drop any cached session first
 * @param {string[]} [opts.executionProviders=['wasm']]
 * @returns {Promise<ModelHandle>}
 */
export function loadModel(url = MODEL_URL, opts = {}) {
  const resolved = opts.buffer ? String(url) : new URL(url, import.meta.url).href;
  if (opts.reload) sessionCache.delete(resolved);
  const hit = sessionCache.get(resolved);
  if (hit) return hit;

  const p = (async () => {
    const t0 = now();
    const ort = await getOrt();

    let bytes;
    if (opts.buffer) {
      bytes = opts.buffer instanceof Uint8Array ? opts.buffer : new Uint8Array(opts.buffer);
    } else {
      bytes = await fetchWithProgress(resolved, opts.onProgress);
    }
    const tDl = now();

    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: opts.executionProviders ?? ['wasm'],
      graphOptimizationLevel: opts.graphOptimizationLevel ?? 'all',
    });
    const tInit = now();

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    if (inputName !== INPUT_NAME || outputName !== OUTPUT_NAME) {
      // Not fatal -- we use the reported names -- but the contract changed.
      console.warn(
        `net.js: expected IO ${INPUT_NAME}->${OUTPUT_NAME}, got ${inputName}->${outputName}`,
      );
    }

    return {
      session,
      ort,
      url: resolved,
      bytes: bytes.byteLength,
      inputName,
      outputName,
      timing: {
        downloadMs: tDl - t0,
        initMs: tInit - tDl,
        totalMs: tInit - t0,
        firstInferenceMs: null,
      },
      inferences: 0,
      _queue: Promise.resolve(),
    };
  })();

  sessionCache.set(resolved, p);
  // A failed load must not poison the cache.
  p.catch(() => sessionCache.delete(resolved));
  return p;
}

/** Release a session and forget it. Safe to call on an unloaded URL. */
export async function disposeModel(handleOrUrl) {
  const key =
    typeof handleOrUrl === 'string'
      ? new URL(handleOrUrl, import.meta.url).href
      : handleOrUrl?.url;
  const p = sessionCache.get(key);
  sessionCache.delete(key);
  if (!p) return;
  try {
    const h = await p;
    await h.session.release?.();
  } catch { /* already broken; nothing to release */ }
}

/** True if this URL is loaded (or loading). */
export function isLoaded(url = MODEL_URL) {
  return sessionCache.has(new URL(url, import.meta.url).href);
}

/* ------------------------------------------------------------------ *
 * Inference
 * ------------------------------------------------------------------ */

/**
 * Pack the two-channel network input, NCHW float32 [1,2,128,128].
 * Channel 0 = measurement (output of `optics.simulateMeasurement`),
 * channel 1 = the binary pupil mask {0,1}. Row-major, p = y*128 + x.
 *
 * @param {ArrayLike<number>} measurement  length 16384
 * @param {ArrayLike<number>} mask         length 16384
 * @param {Float32Array} [out]             optional destination, length 32768
 * @returns {Float32Array} length 2*16384
 */
export function buildInput(measurement, mask, out = null) {
  if (measurement.length !== N2) throw new Error(`measurement length ${measurement.length} != ${N2}`);
  if (mask.length !== N2) throw new Error(`mask length ${mask.length} != ${N2}`);
  const x = out ?? new Float32Array(2 * N2);
  for (let p = 0; p < N2; p++) x[p] = measurement[p];
  for (let p = 0; p < N2; p++) x[N2 + p] = mask[p];
  return x;
}

/** Accept either a ModelHandle or a bare ort.InferenceSession. */
function asHandle(h) {
  if (h && h.session) return h;
  if (h && h.run && h.inputNames) {
    return { session: h, inputName: h.inputNames[0], outputName: h.outputNames[0], timing: {}, _queue: null };
  }
  throw new Error('net.js: first argument must be a ModelHandle from loadModel()');
}

/**
 * Run the U-Net.
 *
 * @param {ModelHandle|object} handle  from `loadModel()`
 * @param {ArrayLike<number>} measurement  16384, from `optics.simulateMeasurement`
 * @param {ArrayLike<number>} mask         16384, binary pupil {0,1}
 * @returns {Promise<Float32Array>} 16384 dense phase values in RADIANS --
 *   unmasked (values outside the pupil are unconstrained and can be large),
 *   unactivated, and carrying an arbitrary piston. Mask it with
 *   `optics.maskedPhase` before displaying, and let `mseOf`/`strehlOf` deal
 *   with the piston (see the header note).
 */
export function predictPhase(handle, measurement, mask) {
  const h = asHandle(handle);
  const run = async () => {
    const ort = h.ort ?? (await getOrt());
    const x = buildInput(measurement, mask);
    const t0 = now();
    const feeds = { [h.inputName]: new ort.Tensor('float32', x, [1, 2, N, N]) };
    const res = await h.session.run(feeds);
    const dt = now() - t0;
    if (h.timing) {
      if (h.timing.firstInferenceMs == null) h.timing.firstInferenceMs = dt;
      h.timing.lastInferenceMs = dt;
    }
    if (typeof h.inferences === 'number') h.inferences++;
    const out = res[h.outputName];
    // Copy out of ORT's arena; the tensor's backing store is reused.
    return new Float32Array(out.data);
  };
  // ORT wasm sessions are not re-entrant; serialise calls on the same session.
  if (!h._queue) return run();
  const next = h._queue.then(run, run);
  h._queue = next.then(() => {}, () => {});
  return next;
}

/**
 * Convenience: pupil + Zernike coefficients -> everything the page shows.
 * One call does the forward model (optics.js), the network, and the metrics.
 *
 * @param {ModelHandle} handle
 * @param {ArrayLike<number>} mask     binary pupil {0,1}, 16384
 * @param {ArrayLike<number>} coeffs   33 Zernike coefficients, radians, PAPER_MODES order
 * @returns {Promise<{phase, gt, measurement, pred, residual, strehl, mse,
 *                    throughput, alpha}>}
 */
export async function predictFromCoeffs(handle, mask, coeffs, opts = {}) {
  const radius = opts.radius ?? RADIUS;
  const phase = synthPhase(coeffs, N, radius, opts.modes ?? PAPER_MODES);
  const gt = maskedPhase(mask, phase, N);
  const measurement = simulateMeasurement(mask, gt, { N, radius, ...opts });
  const pred = await predictPhase(handle, measurement, mask);
  return {
    phase,
    gt,
    measurement,
    pred,
    residual: residual(gt, pred, mask),
    strehl: strehlOf(mask, gt, pred),
    mse: mseOf(mask, gt, pred),
    throughput: throughput(mask, N, radius),
    alpha: opts.alpha ?? asymmetryCorrected(mask, N),
  };
}

/* ------------------------------------------------------------------ *
 * Helpers the page needs -- all optics via optics.js, nothing reimplemented
 * ------------------------------------------------------------------ */

/**
 * Residual wavefront the corrected PSF is formed from: `(gt - pred) * mask`.
 *
 * This is bit-for-bit the field `optics.strehlRatio` builds internally, so
 * `correctedPsf(mask, residual(gt, pred, mask))` and `strehlOf(mask, gt, pred)`
 * are guaranteed consistent.
 *
 * The raw residual carries the network's arbitrary piston. That is invisible to
 * `correctedPsf`/`strehlOf` (see the header) but not to the eye, so pass
 * `{ removePiston: true }` when the array is going onto a colour map.
 *
 * @param {ArrayLike<number>} gt    masked ground-truth phase (mask * phase), radians
 * @param {ArrayLike<number>} pred  raw network output, radians
 * @param {ArrayLike<number>} mask  binary pupil
 * @param {{removePiston?: boolean}} [opts]
 * @returns {Float64Array} length 16384, zero outside the pupil
 */
export function residual(gt, pred, mask, opts = {}) {
  const r = new Float64Array(N2);
  for (let p = 0; p < N2; p++) {
    const m = mask[p];
    if (m !== 0) r[p] = (gt[p] - pred[p]) * m;
  }
  return opts.removePiston ? removePiston(r, mask) : r;
}

/**
 * Subtract the in-pupil mean, then re-mask. Exactly the piston correction the
 * published loss applies (GOLDEN.md §7). Returns a new array; the input is
 * untouched. Display-only -- no metric here needs it.
 */
export function removePiston(field, mask) {
  let s = 0, n = 0;
  for (let p = 0; p < N2; p++) {
    const m = mask[p];
    if (m === 0) continue;
    n += m;
    s += field[p] * m;
  }
  const mean = s / Math.max(n, 1e-9);
  const out = new Float64Array(N2);
  for (let p = 0; p < N2; p++) {
    const m = mask[p];
    if (m !== 0) out[p] = (field[p] - mean) * m;
  }
  return out;
}

/**
 * PSF after correction: |FFT{ mask * exp(i * residual) }|^2, unnormalised,
 * fftshift-centred -- `optics.psfFromPupil` with the residual as the phase.
 *
 * Piston-invariant: a constant added to `residual` multiplies the transform by
 * a unit-modulus scalar, which |.|^2 removes. Feeding it a piston-corrected or
 * a raw residual gives the same array.
 *
 * @param {ArrayLike<number>} mask
 * @param {ArrayLike<number>} res  from `residual()`
 * @returns {Float64Array} length 16384
 */
export function correctedPsf(mask, res) {
  return psfFromPupil(mask, res, N, false);
}

/**
 * The diffraction-limited reference the Strehl divides by:
 * |FFT{ mask }|^2 with zero phase. Peak only.
 */
export function referencePsfPeak(mask) {
  return psfPeak(mask, null, N);
}

/** Uncorrected PSF of this pupil and ground truth -- the "before" panel. */
export function aberratedPsf(mask, gt) {
  return psfFromPupil(mask, gt, N, false);
}

/**
 * Strehl ratio of the correction -- `optics.strehlRatio`, raw and unclamped.
 * Exactly piston-invariant, so the network's arbitrary offset costs nothing.
 * Aggregate with `Math.min(1, s)` the way evaluate.py does.
 *
 * @param {ArrayLike<number>} mask
 * @param {ArrayLike<number>} gt    masked ground truth, radians
 * @param {ArrayLike<number>} pred  raw network output, radians
 * @returns {number}
 */
export function strehlOf(mask, gt, pred) {
  return strehlRatio(mask, gt, pred, N);
}

/**
 * Masked, piston-corrected MSE in rad^2 -- `optics.maskedPistonCorrectedMse`,
 * the published loss. Unlike the Strehl this is a phase-domain quantity and is
 * NOT piston-invariant, which is why the function subtracts the in-pupil mean
 * of both fields before differencing. Callers pass the raw network output.
 */
export function mseOf(mask, gt, pred) {
  return maskedPistonCorrectedMse(mask, pred, gt);
}

/** RMS of the in-pupil residual in radians, piston removed. */
export function rmsResidual(mask, gt, pred) {
  return Math.sqrt(mseOf(mask, gt, pred));
}

/* re-export the few optics entry points the page would otherwise import twice */
export { maskedPhase, simulateMeasurement, throughput, asymmetryCorrected, synthPhase, PAPER_MODES };

/* ------------------------------------------------------------------ *
 * Plumbing
 * ------------------------------------------------------------------ */

const now =
  typeof performance !== 'undefined' && performance.now
    ? () => performance.now()
    : () => Date.now();

/**
 * Fetch with optional progress. Falls back to a plain arrayBuffer() when the
 * response is not streamable or has no Content-Length.
 */
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`net.js: ${res.status} ${res.statusText} fetching ${url}`);
  const total = Number(res.headers.get('content-length') || 0);
  if (!onProgress || !res.body || !total) {
    const buf = await res.arrayBuffer();
    onProgress?.({ loaded: buf.byteLength, total: total || buf.byteLength });
    return new Uint8Array(buf);
  }
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress({ loaded, total });
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out;
}
