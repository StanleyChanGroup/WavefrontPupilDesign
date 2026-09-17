/**
 * net_verify.mjs -- numerical proof that src/net.js (onnxruntime-web, WASM EP,
 * single-threaded, unet_fp16.onnx) reproduces the published PyTorch model.
 *
 * Run:
 *   node test/net_verify.mjs [cases.json] [out.json]
 *
 * The reference side is produced by scratchpad/net_ref.py, which writes
 *   cases.bin   -- per case: mask u8[16384], phase f64[16384],
 *                            meas_f32[16384], pred_f32[16384]
 *   cases.json  -- indices, corrected asymmetry, PyTorch-computed metrics
 *
 * Two JS inference runs are compared against the PyTorch prediction:
 *   A "same input"  -- fed the float32 measurement PyTorch itself built.
 *                      Isolates the ONNX/fp16/WASM inference error.
 *   B "full JS"     -- fed the measurement optics.js builds from (mask, phase).
 *                      The end-to-end path a visitor actually gets.
 * If B is materially worse than A, the JS forward model is the culprit; if both
 * are worse than the known fp16 budget, the JS inference path has a bug.
 *
 * The harness serves the project over real HTTP on 127.0.0.1 so that net.js
 * exercises exactly the browser code path: fetch() for the model, and ORT's own
 * fetch for the wasm through env.wasm.wasmPaths.
 */
import { readFileSync, writeFileSync, createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { performance } from 'node:perf_hooks';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const ROOT = normalize(join(HERE, '..'));
const SCRATCH = '/tmp/claude-979489/-depot-chan129-users-aalmuall/63a2bda4-40f1-435b-960c-8ea418acfc7a/scratchpad';
const CASES_JSON = process.argv[2] ?? join(SCRATCH, 'cases.json');
const CASES_BIN = CASES_JSON.replace(/\.json$/, '.bin');
const OUT_JSON = process.argv[3] ?? join(SCRATCH, 'net_verify.json');

const N = 128, N2 = N * N;

/* ---------- a minimal static server, so fetch() and ORT behave as in a browser ---------- */
const MIME = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream', '.json': 'application/json', '.html': 'text/html' };
const server = createServer((req, res) => {
  const p = join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  if (!p.startsWith(ROOT)) { res.writeHead(403).end(); return; }
  try {
    const st = statSync(p);
    res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream',
      'content-length': st.size });
    createReadStream(p).pipe(res);
  } catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const BASE = `http://127.0.0.1:${server.address().port}`;
console.log(`serving ${ROOT} at ${BASE}`);

/* ---------- load net.js ----------
 * Node refuses `import('http://...')` (ERR_UNSUPPORTED_ESM_URL_SCHEME), so the
 * module itself comes off disk; everything it *fetches* -- the model and ORT's
 * wasm -- comes over the HTTP server above, which is the part that matters.
 * The module-relative default for wasmPaths is asserted separately below. */
const net = await import(new URL('../src/net.js', import.meta.url).href);
const optics = await import(new URL('../src/optics.js', import.meta.url).href);

// The browser default: wasmPaths resolved against net.js's own URL, not the
// document's. Assert it lands on the vendored binary before overriding it.
{
  const netUrl = new URL('../src/net.js', import.meta.url).href;
  const def = new URL(net.ORT_WASM_DIR + net.ORT_WASM_FILE, netUrl).href;
  const want = new URL('../vendor/ort-wasm-simd-threaded.wasm', import.meta.url).href;
  if (def !== want) throw new Error(`default wasmPaths ${def} != ${want}`);
  statSync(fileURLToPath(def));
  statSync(fileURLToPath(new URL(net.ORT_MODULE_URL, netUrl)));
  console.log(`default wasmPaths resolves to ${def} (present)`);
}
net.configure({ wasmPaths: `${BASE}/vendor/`, numThreads: 1, logLevel: 'warning' });

/* ---------- cold-load timings ---------- */
const tOrt0 = performance.now();
await net.getOrt();                              // imports bundle, no wasm yet
const ortImportMs = performance.now() - tOrt0;

let progressTicks = 0;
const handle = await net.loadModel(`${BASE}/assets/unet_fp16.onnx`, {
  onProgress: () => { progressTicks++; },
});
console.log(`ORT import ${ortImportMs.toFixed(1)} ms | model ${handle.bytes} B ` +
  `| download ${handle.timing.downloadMs.toFixed(1)} ms | session init ${handle.timing.initMs.toFixed(1)} ms ` +
  `| progress callbacks ${progressTicks}`);
console.log(`IO: ${handle.inputName} -> ${handle.outputName}`);

/* ---------- read the reference cases ---------- */
const meta = JSON.parse(readFileSync(CASES_JSON, 'utf8'));
const bin = readFileSync(CASES_BIN);
const REC = N2 * (1 + 8 + 4 + 4);
if (bin.length !== meta.n * REC) throw new Error(`cases.bin ${bin.length} != ${meta.n * REC}`);
console.log(`${meta.n} cases, alpha ${Math.min(...meta.alpha).toFixed(4)}..${Math.max(...meta.alpha).toFixed(4)}`);

function readCase(i) {
  let o = i * REC;
  const mask = new Float64Array(N2);
  for (let p = 0; p < N2; p++) mask[p] = bin[o + p];
  o += N2;
  const phase = new Float64Array(bin.buffer.slice(bin.byteOffset + o, bin.byteOffset + o + N2 * 8));
  o += N2 * 8;
  const measPy = new Float32Array(bin.buffer.slice(bin.byteOffset + o, bin.byteOffset + o + N2 * 4));
  o += N2 * 4;
  const predPy = new Float32Array(bin.buffer.slice(bin.byteOffset + o, bin.byteOffset + o + N2 * 4));
  return { mask, phase, measPy, predPy };
}

/* ---------- deviation helpers ---------- */
function dev(a, b, mask) {
  let rawMax = 0, rawSq = 0, inMax = 0, inSq = 0, npix = 0;
  for (let p = 0; p < N2; p++) {
    const d = a[p] - b[p], ad = Math.abs(d);
    if (ad > rawMax) rawMax = ad;
    rawSq += d * d;
    if (mask[p] !== 0) { npix++; inSq += d * d; if (ad > inMax) inMax = ad; }
  }
  return { rawMax, rawRms: Math.sqrt(rawSq / N2), inMax, inRms: Math.sqrt(inSq / Math.max(npix, 1)), npix };
}

/* ---------- main loop ---------- */
const rows = [];
let warm = 0, warmN = 0, firstInf = null;
const t0 = performance.now();
for (let i = 0; i < meta.n; i++) {
  const { mask, phase, measPy, predPy } = readCase(i);
  const gt = optics.maskedPhase(mask, phase, N);
  const measJs = optics.simulateMeasurement(mask, gt, { N, radius: 19 });

  const tA = performance.now();
  const predA = await net.predictPhase(handle, measPy, mask);
  const msA = performance.now() - tA;
  const tB = performance.now();
  const predB = await net.predictPhase(handle, measJs, mask);
  const msB = performance.now() - tB;
  if (firstInf === null) firstInf = msA;
  if (i >= 2) { warm += msA + msB; warmN += 2; }

  // measurement agreement (float64 JS vs float32 torch)
  let mMax = 0, mMaxRel = 0, mPeak = 0;
  for (let p = 0; p < N2; p++) {
    const d = Math.abs(measJs[p] - measPy[p]);
    if (d > mMax) mMax = d;
    if (measPy[p] > mPeak) mPeak = measPy[p];
  }
  mMaxRel = mMax / Math.max(mPeak, 1e-30);

  const r = {
    i, alpha: meta.alpha[i], band: meta.band[i],
    npix: 0,
    measMax: mMax, measMaxRel: mMaxRel,
    // metrics computed IN JS for all three predictions
    mseJsPy: optics.maskedPistonCorrectedMse(mask, predPy, gt),
    mseA: net.mseOf(mask, gt, predA),
    mseB: net.mseOf(mask, gt, predB),
    stJsPy: optics.strehlRatio(mask, gt, predPy, N),
    stA: net.strehlOf(mask, gt, predA),
    stB: net.strehlOf(mask, gt, predB),
    // metrics PyTorch computed itself, for the metric-implementation cross-check
    msePy: meta.py_mse[i], stPy: meta.py_strehl[i],
    msA, msB,
  };
  const dA = dev(predA, predPy, mask), dB = dev(predB, predPy, mask);
  r.npix = dA.npix;
  r.A = dA; r.B = dB;

  // residual()/correctedPsf() must be exactly consistent with strehlOf()
  if (i < 5) {
    const res = net.residual(gt, predB, mask);
    const psf = net.correctedPsf(mask, res);
    let pk = 0; for (let p = 0; p < N2; p++) if (psf[p] > pk) pk = psf[p];
    r.strehlViaPsf = pk / Math.max(net.referencePsfPeak(mask), 1e-4);
    // and piston-invariance of that path, numerically
    const resP = net.residual(gt, predB, mask, { removePiston: true });
    const psfP = net.correctedPsf(mask, resP);
    let pkP = 0; for (let p = 0; p < N2; p++) if (psfP[p] > pkP) pkP = psfP[p];
    r.strehlPistonRemoved = pkP / Math.max(net.referencePsfPeak(mask), 1e-4);
  }
  rows.push(r);
  if (i % 25 === 0) console.log(`  ${i}/${meta.n}  ${((performance.now() - t0) / 1000).toFixed(0)}s`);
}
const totalS = (performance.now() - t0) / 1000;

writeFileSync(OUT_JSON, JSON.stringify({
  ortVersion: net.ORT_VERSION,
  model: handle.url, modelBytes: handle.bytes,
  timing: { ortImportMs, downloadMs: handle.timing.downloadMs, initMs: handle.timing.initMs,
    firstInferenceMs: firstInf, steadyInferenceMs: warm / warmN, totalS },
  node: process.version,
  meta: { n: meta.n, ckpt: meta.ckpt, torch: meta.torch },
  rows,
}, null, 1));
console.log(`\nwrote ${OUT_JSON}  (${totalS.toFixed(0)}s, steady ${(warm / warmN).toFixed(1)} ms/inference)`);
server.close();
