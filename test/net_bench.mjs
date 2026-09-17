/**
 * net_bench.mjs -- cold-load and steady-state numbers for src/net.js.
 *
 *   node test/net_bench.mjs
 *
 * Serves the project over loopback HTTP so the fetch path is the real one, then
 * separates the three costs a visitor pays:
 *   1. transfer      -- bytes on the wire (raw and gzip), for both the model and
 *                       the ORT wasm binary
 *   2. ORT init      -- instantiating the 14 MB wasm module (paid once)
 *   3. session build -- parsing the ONNX graph + optimisation (paid per model)
 *   4. inference     -- first (cold arena/JIT) and steady state
 *
 * ORT gives no hook between wasm instantiation and graph build, so init is
 * separated by creating a SECOND session on the already-instantiated runtime:
 * the second create is graph-build only, and the difference is the wasm init.
 */
import { createReadStream, statSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { gzipSync } from 'node:zlib';
import { performance } from 'node:perf_hooks';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

const ROOT = normalize(join(fileURLToPath(new URL('.', import.meta.url)), '..'));
const MIME = { '.mjs': 'text/javascript', '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream', '.json': 'application/json' };
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

const net = await import(new URL('../src/net.js', import.meta.url).href);
const optics = await import(new URL('../src/optics.js', import.meta.url).href);
net.configure({ wasmPaths: `${BASE}/vendor/`, numThreads: 1, logLevel: 'error' });

/* ---- 1. transfer sizes ---- */
const files = {
  'vendor/ort.wasm.bundle.min.mjs': null,
  'vendor/ort-wasm-simd-threaded.wasm': null,
  'assets/unet_fp16.onnx': null,
  'src/net.js': null,
  'src/optics.js': null,
};
let rawTot = 0, gzTot = 0;
for (const k of Object.keys(files)) {
  const b = readFileSync(join(ROOT, k));
  const gz = gzipSync(b, { level: 9 }).length;
  files[k] = { raw: b.length, gzip: gz };
  rawTot += b.length; gzTot += gz;
}
console.log('--- transfer (cold, uncached) ---');
for (const [k, v] of Object.entries(files))
  console.log(`  ${k.padEnd(38)} ${String(v.raw).padStart(10)} B raw  ${String(v.gzip).padStart(10)} B gzip`);
console.log(`  ${'TOTAL'.padEnd(38)} ${String(rawTot).padStart(10)} B raw  ${String(gzTot).padStart(10)} B gzip`);

/* ---- 2/3. init ---- */
const t0 = performance.now();
await net.getOrt();
const importMs = performance.now() - t0;

const h1 = await net.loadModel(`${BASE}/assets/unet_fp16.onnx`);
const h2 = await net.loadModel(`${BASE}/assets/unet_fp16.onnx?second`); // fresh session, warm wasm
const wasmInitMs = h1.timing.initMs - h2.timing.initMs;

console.log('\n--- cold load (loopback HTTP; transfer time is ~0 here, see table above) ---');
console.log(`  ORT module import              ${importMs.toFixed(1)} ms`);
console.log(`  model fetch (loopback)         ${h1.timing.downloadMs.toFixed(1)} ms`);
console.log(`  first InferenceSession.create  ${h1.timing.initMs.toFixed(1)} ms  (wasm instantiate + graph build)`);
console.log(`  second create, warm runtime    ${h2.timing.initMs.toFixed(1)} ms  (graph build only)`);
console.log(`  => ORT wasm instantiate        ${wasmInitMs.toFixed(1)} ms`);

/* ---- idempotency ---- */
const again = await net.loadModel(`${BASE}/assets/unet_fp16.onnx`);
console.log(`  loadModel() idempotent:        ${again === h1 ? 'YES (same handle, no refetch)' : 'NO'}`);

/* ---- 4. inference ---- */
const G = JSON.parse(readFileSync(join(ROOT, 'assets/golden.json'), 'utf8'));
const ladder = await import(new URL('../src/ladder.js', import.meta.url).href);
const mask = ladder.decodeMaskBase64(G.masks[2].bits_b64);
const gt = optics.maskedPhase(mask, optics.synthPhase(G.phases[2].coeffs_f64, 128, 19), 128);
const meas = optics.simulateMeasurement(mask, gt, { N: 128, radius: 19 });

const tf = performance.now();
const pred0 = await net.predictPhase(h1, meas, mask);
const firstMs = performance.now() - tf;

const R = 20, times = [];
for (let i = 0; i < R; i++) {
  const t = performance.now();
  await net.predictPhase(h1, meas, mask);
  times.push(performance.now() - t);
}
times.sort((a, b) => a - b);
const mean = times.reduce((a, b) => a + b, 0) / R;

/* the optics the page runs around each inference */
const tOpt = performance.now();
for (let i = 0; i < 20; i++) {
  const m2 = optics.simulateMeasurement(mask, gt, { N: 128, radius: 19 });
  net.strehlOf(mask, gt, pred0);
  net.correctedPsf(mask, net.residual(gt, pred0, mask));
  if (m2[0] === -1) throw 0;
}
const opticsMs = (performance.now() - tOpt) / 20;

console.log('\n--- inference (single-threaded wasm EP) ---');
console.log(`  first inference (cold)         ${firstMs.toFixed(1)} ms`);
console.log(`  steady state, n=${R}            mean ${mean.toFixed(1)} ms  median ${times[R >> 1].toFixed(1)} ms  min ${times[0].toFixed(1)}  max ${times[R - 1].toFixed(1)}`);
console.log(`  optics.js per frame            ${opticsMs.toFixed(1)} ms  (measurement + Strehl + corrected PSF)`);
console.log(`\n  node ${process.version}  |  ${cpus()[0].model}  |  ort ${net.ORT_VERSION}  |  numThreads 1`);

console.log('\nJSON ' + JSON.stringify({
  files, rawTot, gzTot, importMs, downloadMs: h1.timing.downloadMs,
  createColdMs: h1.timing.initMs, createWarmMs: h2.timing.initMs, wasmInitMs,
  firstMs, steadyMeanMs: mean, steadyMedianMs: times[R >> 1], steadyMinMs: times[0],
  steadyMaxMs: times[R - 1], opticsMs, cpu: cpus()[0].model, node: process.version,
}));
server.close();
