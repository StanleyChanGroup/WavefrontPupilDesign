#!/usr/bin/env node
/**
 * verify.mjs -- validates src/optics.js against assets/golden.json (and, when present,
 * assets/golden.npz for whole-array coverage).
 *
 * Walks the eight stages of GOLDEN.md §11 in order, accumulating the worst deviation
 * per quantity, printing a table, and exiting non-zero on any failure.
 *
 *   node test/verify.mjs [--no-npz] [--bench]
 *
 * Tolerances are the ones GOLDEN.md §10.2 states. Where §10.2 states none, the
 * tolerance used is derived from a number GOLDEN.md does state (§10.1's measured
 * float32 noise floor, or the golden record's own reported f32-vs-f64 spread) and is
 * annotated in the `src` column of the report.
 */

import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import * as os from 'node:os';

import {
  N_DEFAULT, RADIUS_DEFAULT, CIRCLE_PSF_PEAK, PAPER_MODES,
  referenceCircle, zernikeDisk, synthPhase, fft2, ifft2,
  zernikeRadial, zernike, psfFromPupil, psfPeak, simulateMeasurement,
  strehlRatio, maskedPistonCorrectedMse, asymmetry, asymmetryCorrected,
  autoconvArgmax, throughput, syntheticPredictor, maskedPhase, sumMask, runPipeline,
} from '../src/optics.js';
import { decodeMaskBase64, decodeFloat32Base64, readLadder, base64ToBytes } from '../src/ladder.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = join(HERE, '..', 'assets');
const N = N_DEFAULT, N2 = N * N, RADIUS = RADIUS_DEFAULT;

const argv = new Set(process.argv.slice(2));
const USE_NPZ = !argv.has('--no-npz');

/* ================================================================== *
 * Reporting
 * ================================================================== */

const KINDS = {
  exact:   { label: 'exact',          fmt: (d) => (d === 0 ? '0 (bit-exact)' : `${d} mismatch(es)`) },
  abs:     { label: 'abs',            fmt: (d) => d.toExponential(3) },
  rel:     { label: 'rel',            fmt: (d) => d.toExponential(3) },
  relpeak: { label: 'rel-to-peak',    fmt: (d) => d.toExponential(3) },
};

const rows = [];
const byName = new Map();

function track(stage, name, kind, tol, src, gate = true) {
  const r = { stage, name, kind, tol, src, gate, worst: 0, where: '-', n: 0, notes: [] };
  rows.push(r);
  byName.set(name, r);
  return r;
}

function dev(kind, got, want, scale) {
  if (kind === 'exact') return Object.is(got, want) || got === want ? 0 : 1;
  const d = Math.abs(got - want);
  if (!Number.isFinite(d)) return Number.isFinite(got) === Number.isFinite(want) && got === want ? 0 : Infinity;
  if (kind === 'abs') return d;
  if (kind === 'rel') { const den = Math.abs(want); return den > 0 ? d / den : d; }
  if (kind === 'relpeak') return scale > 0 ? d / scale : d;
  throw new Error(`bad kind ${kind}`);
}

function obs(name, got, want, where, scale = 1) {
  const r = byName.get(name);
  if (!r) throw new Error(`untracked quantity "${name}"`);
  r.n++;
  const d = dev(r.kind, got, want, scale);
  if (r.kind === 'exact') {
    if (d !== 0) { r.worst += 1; if (r.where === '-') r.where = `${where} (got ${fmtv(got)}, want ${fmtv(want)})`; }
  } else if (d > r.worst) { r.worst = d; r.where = where; }
}

function obsArray(name, got, want, where, scale = 1) {
  const r = byName.get(name);
  if (!r) throw new Error(`untracked quantity "${name}"`);
  const n = want.length;
  let worst = 0, at = -1, bad = 0;
  for (let i = 0; i < n; i++) {
    const d = dev(r.kind, got[i], want[i], scale);
    if (r.kind === 'exact') { if (d !== 0) { bad++; if (at < 0) at = i; } }
    else if (d > worst) { worst = d; at = i; }
  }
  r.n += n;
  if (r.kind === 'exact') {
    if (bad > 0) { r.worst += bad; if (r.where === '-') r.where = `${where}[${at}] (got ${fmtv(got[at])}, want ${fmtv(want[at])})`; }
  } else if (worst > r.worst) { r.worst = worst; r.where = `${where}[${at}]`; }
}

/**
 * Gate a float64-vs-golden-float32 comparison on the EXCESS over the bound the golden
 * file itself reports for that record's f32-vs-f64 spread. §10.1's summary table is
 * rounded to two significant figures and so is not usable as a bound; the per-record
 * bound is exact. A correct f64 port sits at or below it, i.e. excess <= 0.
 */
function obsExcess(name, got, want, bound, where, scale = 1) {
  const r = byName.get(name);
  if (!r) throw new Error(`untracked quantity "${name}"`);
  if (r.n === 0) r.worst = -Infinity;   // an excess is expected to be NEGATIVE
  r.n++;
  const d = (Math.abs(got - want) - bound) / scale;
  if (d > r.worst) { r.worst = d; r.where = where; }
}

function obsExcessArray(name, got, want, bound, where, scale = 1) {
  const r = byName.get(name);
  if (!r) throw new Error(`untracked quantity "${name}"`);
  if (r.n === 0) r.worst = -Infinity;   // an excess is expected to be NEGATIVE
  let worst = -Infinity, at = -1;
  for (let i = 0; i < want.length; i++) {
    const d = (Math.abs(got[i] - want[i]) - bound) / scale;
    if (d > worst) { worst = d; at = i; }
  }
  r.n += want.length;
  if (worst > r.worst) { r.worst = worst; r.where = `${where}[${at}]`; }
}

function fmtv(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(12);
}

/* ================================================================== *
 * Numeric helpers
 * ================================================================== */

/** Pairwise summation (numpy-like); a naive loop differs in the last ULPs. */
function psum(a, f = (v) => v, lo = 0, hi = a.length) {
  const n = hi - lo;
  if (n <= 128) { let s = 0; for (let i = lo; i < hi; i++) s += f(a[i]); return s; }
  const mid = lo + (n >> 1);
  return psum(a, f, lo, mid) + psum(a, f, mid, hi);
}

function stats(a) {
  let mn = Infinity, mx = -Infinity, am = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (v < mn) mn = v;
    if (v > mx) { mx = v; am = i; }   // strict >: first flat-index maximum (§10.3)
  }
  return {
    min: mn, max: mx, argmax_flat: am, argmax_yx: [Math.floor(am / N), am % N],
    sum: psum(a), sum_sq: psum(a, (v) => v * v),
  };
}

const GRID16 = { rows: Array.from({ length: 16 }, (_, i) => i * 8), cols: Array.from({ length: 16 }, (_, i) => i * 8) };
const CORE16 = { rows: Array.from({ length: 16 }, (_, i) => 56 + i), cols: Array.from({ length: 16 }, (_, i) => 56 + i) };

function sample(a, set) {
  const out = new Float64Array(256);
  for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) out[i * 16 + j] = a[set.rows[i] * N + set.cols[j]];
  return out;
}

function toF32(a) { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i]; return o; }
function sha256Hex(u8) { return createHash('sha256').update(Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength)).digest('hex'); }
function f64leBytes(a) { const b = new Uint8Array(a.length * 8); const dv = new DataView(b.buffer); for (let i = 0; i < a.length; i++) dv.setFloat64(i * 8, a[i], true); return b; }

/** Exact integer autoconvolution of a binary mask at one shift: sum_y,x m[y][x]*m[(a-y)%N][(b-x)%N]. */
function autoconvExactAt(m, a, b, n = N) {
  let s = 0;
  for (let y = 0; y < n; y++) {
    const yy = ((a - y) % n + n) % n;
    for (let x = 0; x < n; x++) {
      if (m[y * n + x] === 0) continue;
      s += m[yy * n + (((b - x) % n + n) % n)];
    }
  }
  return s;
}

/**
 * Flat indices whose float autoconvolution is within 0.5 of the maximum. For a binary
 * mask the true values are integers spaced by >= 1 while the FFT noise is ~1e-12, so
 * this window captures exactly the candidates that could win the argmax, and their
 * exact integer values decide whether a disagreement is a genuine tie.
 */
function autoconvCandidates(m, n = N) {
  const n2 = n * n;
  const re = new Float64Array(n2), im = new Float64Array(n2);
  for (let p = 0; p < n2; p++) re[p] = m[p];
  fft2(re, im, n, false);
  for (let p = 0; p < n2; p++) { const a = re[p], b = im[p]; re[p] = a * a - b * b; im[p] = 2 * a * b; }
  ifft2(re, im, n);
  let best = -Infinity;
  for (let p = 0; p < n2; p++) { const v = Math.hypot(re[p], im[p]); re[p] = v; if (v > best) best = v; }
  const out = [];
  for (let p = 0; p < n2; p++) if (re[p] > best - 0.5) out.push({ p, y: Math.floor(p / n), x: p % n, exact: autoconvExactAt(m, Math.floor(p / n), p % n, n) });
  const exactMax = Math.max(...out.map((c) => c.exact));
  return { candidates: out, exactMax, tied: out.filter((c) => c.exact === exactMax) };
}

const FAILS = [];
function assert(cond, msg) { if (!cond) FAILS.push(msg); return !!cond; }

/* ================================================================== *
 * Load golden data
 * ================================================================== */

const G = JSON.parse(readFileSync(join(ASSETS, 'golden.json'), 'utf8'));
let NPZ = null;
const npzPath = join(ASSETS, 'golden.npz');
if (USE_NPZ && existsSync(npzPath)) {
  const { readNpz } = await import('./npz.mjs');
  const t0 = performance.now();
  NPZ = readNpz(npzPath);
  console.log(`[npz] loaded ${Object.keys(NPZ).length} arrays in ${(performance.now() - t0).toFixed(0)} ms`);
}

console.log(`[golden] schema=${G.schema}  generated_by=${G.generated_by}`);
console.log(`[golden] ${G.masks.length} masks x ${G.phases.length} phases = ${G.cases.length} cases, ` +
            `${G.noise_probe.results.length} noise probes, ${G.zernike_spot_checks.modes.length} Zernike spot checks\n`);

/* ================================================================== *
 * Tracked quantities  (stage, name, kind, tolerance, source of tolerance)
 * ================================================================== */

const T = { // §10.2 "must pass" column
  exact: 0,
  zern: 1e-12, phase: 1e-11, psf: 1e-9, meas: 1e-9, noisy: 1e-9,
  stat: 1e-11, strehl: 1e-9, mse: 1e-9, alpha: 1e-12, unet: 1e-3,
};
const S102 = 'GOLDEN.md §10.2';
const S101 = 'GOLDEN.md §10.1 (f32 floor)';
const SREC = 'record’s own f32_vs_f64 bound';

// Stage 0 -- geometry
track(0, 'geometry.integer_counts',            'exact', 0,          S102);
track(0, 'geometry.circle_bits_vs_mask0',      'exact', 0,          S102);
track(0, 'config/paper_modes/sampling echo',   'exact', 0,          S102);
// Stage 1 -- Zernike
track(1, 'zernike.R_at_r',                     'rel',   T.zern,     S102);
track(1, 'zernike.Z_at_r_theta',               'rel',   T.zern,     S102);
// Stage 2 -- phases
track(2, 'phase.disk_px',                      'exact', 0,          S102);
track(2, 'phase.stats.min/max',                'abs',   T.phase,    S102);
track(2, 'phase.stats.argmax',                 'exact', 0,          S102);
track(2, 'phase.stats.sum/sum_sq',             'rel',   T.stat,     S102);
track(2, 'phase.samples.grid16',               'abs',   T.phase,    S102);
track(2, 'phase.samples.core16',               'abs',   T.phase,    S102);
track(2, 'phase f32-gap excess over record',   'abs',   T.phase,    'my |f64-f32| minus the record’s own f32_vs_f64.max_abs; §10.2 f64 gate');
track(2, 'phase vs npz phases_f64',            'abs',   T.phase,    S102);
track(2, 'phase npz-f32 excess over record',   'abs',   T.phase,    'same, against npz phases_f32');
track(2, 'phase.coeffs vs npz',                'exact', 0,          S102);
// Stage 3 -- masks / asymmetry
track(3, 'mask.sum_mask',                      'exact', 0,          S102);
track(3, 'mask.throughput',                    'abs',   T.alpha,    S102);
track(3, 'mask.sha256_uint8',                  'exact', 0,          'byte-exact mask image');
track(3, 'mask.alpha_corrected.f64',           'abs',   T.alpha,    S102);
track(3, 'mask.alpha_published.f64 (argmax OK)','abs',  T.alpha,    S102 + ' — gated only where the argmax agrees');
track(3, 'mask.autoconv argmax: match-or-tie', 'exact', 0,          'GOLDEN.md §8: a differing argmax must be a PROVEN exact integer tie');
track(3, 'mask.dl_psf_peak == sum_mask^2',     'exact', 0,          S102 + ' (exact)');
track(3, 'mask bits vs npz masks_uint8',       'exact', 0,          S102);
// Stage 4 -- PSF
track(4, 'psf.peak_before_normalization',      'relpeak', T.psf,    S102);
track(4, 'psf.argmax_flat (intermediate)',     'exact', 0,          S102);
track(4, 'psf.stats.min/max',                  'relpeak', T.psf,    S102);
track(4, 'psf.stats.argmax',                   'exact', 0,          S102);
track(4, 'psf.stats.sum/sum_sq',               'rel',   T.stat,     S102);
track(4, 'psf.parseval sum==16384*sum_mask',   'rel',   T.stat,     S102 + ' (§4.5)');
track(4, 'psf.samples.grid16',                 'relpeak', T.psf,    S102);
track(4, 'psf.samples.core16',                 'relpeak', T.psf,    S102);
track(4, 'psf f32-gap excess over case bound', 'relpeak', T.psf,    'my |f64-f32| minus case.psf.f32.max_rel_to_peak; §10.2 f64 gate');
track(4, 'psf vs npz psf_f64 (full array)',    'relpeak', T.psf,    S102);
track(4, 'psf npz-f32 excess over case bound', 'relpeak', T.psf,    'same, against npz psf_f32 (full array)');
track(4, 'circle_psf vs npz (full array)',     'relpeak', T.psf,    S102);
// Stage 5 -- measurement
track(5, 'meas.peak_after_circle_divisor',     'abs',   T.meas,     S102);
track(5, 'meas.renorm_clamp_active',           'exact', 0,          S102);
track(5, 'meas.stats.min/max',                 'abs',   T.meas,     S102);
track(5, 'meas.stats.argmax',                  'exact', 0,          S102);
track(5, 'meas.stats.sum/sum_sq',              'rel',   T.stat,     S102);
track(5, 'meas.samples.grid16',                'abs',   T.meas,     S102);
track(5, 'meas.samples.core16',                'abs',   T.meas,     S102);
track(5, 'meas f32-gap excess over case bound','abs',   T.meas,     'my |f64-f32| minus case.measurement.f32.max_abs; §10.2 f64 gate');
track(5, 'meas vs npz measurement_f64 (full)', 'abs',   T.meas,     S102);
track(5, 'meas npz-f32 excess over case bound','abs',   T.meas,     'same, against npz measurement_f32 (full array)');
// Stage 6 -- noise probe
track(6, 'noise field vs sha256/npz',          'exact', 0,          'byte-exact noise field');
track(6, 'noisy.stats.min/max',                'abs',   T.noisy,    S102);
track(6, 'noisy.stats.argmax',                 'exact', 0,          S102);
track(6, 'noisy.stats.sum/sum_sq',             'rel',   T.stat,     S102);
track(6, 'noisy.samples.grid16',               'abs',   T.noisy,    S102);
track(6, 'noisy.samples.core16',               'abs',   T.noisy,    S102);
track(6, 'noisy f32-gap excess over record',   'abs',   T.noisy,    "my |f64-f32| minus the record's own f32.max_abs; §10.2 f64 gate");
track(6, 'noisy vs npz noisy_f64 (full)',      'abs',   T.noisy,    S102);
track(6, 'noisy npz-f32 excess over record',   'abs',   T.noisy,    'same, against npz measurement_noisy_f32');
// Stage 7 -- metrics
track(7, 'strehl vs .f64',                     'abs',   T.strehl,   S102);
track(7, 'mse vs .f64',                        'rel',   T.mse,      S102);
track(7, 'strehl f32-gap excess over golden',  'abs',   T.strehl,   'my |f64-f32| minus golden’s own |f64-f32|; §10.2 f64 gate');
track(7, 'mse f32-gap excess over golden',     'abs',   T.mse,      'my |f64-f32| minus golden’s own |f64-f32|; §10.2 f64 gate');
track(7, 'pred from GOLDEN phase vs npz',      'exact', 0,          S102 + ' (§9: bit-for-bit from golden.json)');
track(7, 'pred from MY phase vs npz',          'abs',   T.phase,    S102 + ' phase 1e-11 (pred = 0.9*gt + const)');
track(7, 'piston-invariance (0.3 -> 7.0)',     'abs',   1e-9,       S102 + ' §11 stage 7');
track(7, 'unet strehl vs torch_f32',           'rel',   T.unet,     S102);
track(7, 'unet mse vs torch_f32',              'rel',   T.unet,     S102);
track(7, 'unet pred stats/core16 (npz echo)',  'abs',   1e-6,       'f32 round-trip of npz pred_unet_f32');

/* ================================================================== *
 * Stage 0 -- geometry
 * ================================================================== */

console.log('--- Stage 0: geometry -------------------------------------------------');

const circle = referenceCircle(N, RADIUS);
const disk = zernikeDisk(N, RADIUS);

let cSum = 0, dSum = 0, both = 0, cOnly = 0, dOnly = 0;
for (let p = 0; p < N2; p++) {
  const c = circle[p], d = disk.mask[p];
  cSum += c; dSum += d;
  if (c && d) both++; else if (c) cOnly++; else if (d) dOnly++;
}
obs('geometry.integer_counts', cSum, G.geometry.reference_circle_area_px, 'reference_circle_area_px');
obs('geometry.integer_counts', dSum, G.geometry.zernike_disk_area_px, 'zernike_disk_area_px');
obs('geometry.integer_counts', both, G.geometry.shared_px, 'shared_px');
obs('geometry.integer_counts', cOnly, G.geometry.circle_only_px, 'circle_only_px');
obs('geometry.integer_counts', dOnly, G.geometry.zernike_disk_only_px, 'zernike_disk_only_px');
obs('geometry.integer_counts', G.geometry.bbox, N, 'bbox');
obs('geometry.integer_counts', G.geometry.radius, RADIUS, 'radius');
console.log(`  circle=${cSum}  disk=${dSum}  shared=${both}  circle-only=${cOnly}  disk-only=${dOnly}`);

// masks[0] must be element-wise identical to the computed circle.
const mask0Bits = decodeMaskBase64(G.masks[0].bits_b64);
obsArray('geometry.circle_bits_vs_mask0', mask0Bits, circle, 'masks[0].bits_b64 vs referenceCircle()');

// Config / mode-order / sampling echo.
obs('config/paper_modes/sampling echo', G.paper_modes.length, PAPER_MODES.length, 'paper_modes length');
for (let k = 0; k < G.paper_modes.length; k++) {
  obs('config/paper_modes/sampling echo', PAPER_MODES[k][0], G.paper_modes[k][0], `paper_modes[${k}].n`);
  obs('config/paper_modes/sampling echo', PAPER_MODES[k][1], G.paper_modes[k][1], `paper_modes[${k}].m`);
}
obs('config/paper_modes/sampling echo', G.geometry.circle_psf_peak_exact_int, CIRCLE_PSF_PEAK, 'circle_psf_peak_exact_int');
for (let i = 0; i < 16; i++) {
  obs('config/paper_modes/sampling echo', GRID16.rows[i], G.sampling.grid16.rows[i], `grid16.rows[${i}]`);
  obs('config/paper_modes/sampling echo', GRID16.cols[i], G.sampling.grid16.cols[i], `grid16.cols[${i}]`);
  obs('config/paper_modes/sampling echo', CORE16.rows[i], G.sampling.core16.rows[i], `core16.rows[${i}]`);
  obs('config/paper_modes/sampling echo', CORE16.cols[i], G.sampling.core16.cols[i], `core16.cols[${i}]`);
}
// The published configuration this port implements as its defaults.
const cfg = G.config;
assert(cfg.normalize_by_peak === true && cfg.renormalize_after_noise === true &&
       cfg.with_noise === false && cfg.noise_sigma === 0 && cfg.coeff_scale === 2 &&
       cfg.bounding_box_size === N && cfg.radius === RADIUS,
       `config drift: ${JSON.stringify(cfg)}`);

// §11.1 smoke test, stated explicitly.
const circlePsf = psfFromPupil(circle, null, N);
const cps = stats(circlePsf);
assert(cSum === 1124, 'smoke: sum(circle) !== 1124');
assert(dSum === 1129, 'smoke: sum(disk) !== 1129');
assert(cps.max === 1263376, `smoke: circlePSF.max is ${cps.max}, must be exactly 1263376`);
assert(cps.argmax_flat === 8256, `smoke: circlePSF.argmax is ${cps.argmax_flat}, must be 8256`);
// Parseval (§4.5). GOLDEN.md §10.3: full-array reductions are NOT bit-equal across
// summation orders -- compare relatively, as the §10.2 stats tolerance (1e-11) does.
assert(Math.abs(cps.sum - 16384 * 1124) / (16384 * 1124) <= 1e-11,
       `smoke: sum(circlePSF) = ${cps.sum}, must be ${16384 * 1124} to 1e-11 rel`);
obs('geometry.integer_counts', cps.max, G.reference_circle_psf.peak.exact_int, 'reference_circle_psf.peak (EXACT)');
obs('geometry.integer_counts', cps.argmax_flat, G.reference_circle_psf.argmax_flat, 'reference_circle_psf.argmax_flat');
obs('geometry.integer_counts', cps.argmax_yx[0], G.reference_circle_psf.argmax_yx[0], 'reference_circle_psf.argmax_y');
obs('geometry.integer_counts', cps.argmax_yx[1], G.reference_circle_psf.argmax_yx[1], 'reference_circle_psf.argmax_x');
console.log(`  circlePSF.peak=${cps.max} (exact int ${cps.max === 1263376})  argmax=${cps.argmax_flat}  sum=${cps.sum}`);

if (NPZ) {
  const a = NPZ.circle_psf_f64.data;
  obsArray('circle_psf vs npz (full array)', circlePsf, a, 'circle_psf_f64', cps.max);
}

/* ================================================================== *
 * Stage 1 -- Zernike spot checks
 * ================================================================== */

console.log('--- Stage 1: Zernike --------------------------------------------------');
const rProbe = G.zernike_spot_checks.r_probe;
const thProbe = G.zernike_spot_checks.theta_probe;
for (let k = 0; k < G.zernike_spot_checks.modes.length; k++) {
  const m = G.zernike_spot_checks.modes[k];
  obs('config/paper_modes/sampling echo', m.n, PAPER_MODES[k][0], `spot_check[${k}].n`);
  obs('config/paper_modes/sampling echo', m.m, PAPER_MODES[k][1], `spot_check[${k}].m`);
  for (let i = 0; i < rProbe.length; i++) {
    obs('zernike.R_at_r', zernikeRadial(m.n, Math.abs(m.m), rProbe[i]), m.R_at_r[i], `mode ${k} (${m.n},${m.m}) r=${rProbe[i]}`);
    obs('zernike.Z_at_r_theta', zernike(m.n, m.m, rProbe[i], thProbe[i]), m.Z_at_r_theta[i], `mode ${k} (${m.n},${m.m}) i=${i}`);
  }
}
console.log(`  checked ${G.zernike_spot_checks.modes.length} modes x ${rProbe.length} probes (R and Z)`);

/* ================================================================== *
 * Stage 2 -- phases
 * ================================================================== */

console.log('--- Stage 2: phases ---------------------------------------------------');
const phases = [];
for (let i = 0; i < G.phases.length; i++) {
  const rec = G.phases[i];
  const where = `phase${rec.id}`;
  const ph = synthPhase(rec.coeffs_f64, N, RADIUS);
  phases.push(ph);

  obs('phase.disk_px', dSum, rec.disk_px, `${where}.disk_px`);
  const st = stats(ph);
  obs('phase.stats.min/max', st.min, rec.stats_f64.min, `${where}.min`);
  obs('phase.stats.min/max', st.max, rec.stats_f64.max, `${where}.max`);
  obs('phase.stats.argmax', st.argmax_flat, rec.stats_f64.argmax_flat, `${where}.argmax_flat`);
  obs('phase.stats.argmax', st.argmax_yx[0], rec.stats_f64.argmax_yx[0], `${where}.argmax_y`);
  obs('phase.stats.argmax', st.argmax_yx[1], rec.stats_f64.argmax_yx[1], `${where}.argmax_x`);
  obs('phase.stats.sum/sum_sq', st.sum, rec.stats_f64.sum, `${where}.sum`);
  obs('phase.stats.sum/sum_sq', st.sum_sq, rec.stats_f64.sum_sq, `${where}.sum_sq`);
  obsArray('phase.samples.grid16', sample(ph, GRID16), rec.samples_f64.grid16, `${where}.grid16`);
  obsArray('phase.samples.core16', sample(ph, CORE16), rec.samples_f64.core16, `${where}.core16`);

  // float32 image: compare my float64 against the golden float32 bytes, bounded by
  // the record's own reported f32-vs-f64 spread, and report the bit-cast mismatch count.
  const gF32 = decodeFloat32Base64(rec.float32_b64, N2);
  obsExcessArray('phase f32-gap excess over record', ph, gF32, rec.f32_vs_f64.max_abs, `${where}.float32_b64`);
  const myF32 = toF32(ph);
  let bitDiff = 0;
  for (let p = 0; p < N2; p++) if (myF32[p] !== gF32[p]) bitDiff++;
  byName.get('phase f32-gap excess over record').notes.push(
    `${where}: ${bitDiff}/16384 pixels differ by 1 f32 ULP after casting my f64; ` +
    `max |my f64 - golden f32| = ${(() => { let m = 0; for (let q = 0; q < N2; q++) m = Math.max(m, Math.abs(ph[q] - gF32[q])); return m.toExponential(4); })()} ` +
    `vs the record's own f32_vs_f64.max_abs = ${rec.f32_vs_f64.max_abs.toExponential(4)}`);
  assert(Math.abs(rec.f32_vs_f64.max_abs) >= 0, 'f32_vs_f64 present');

  // sha256 provenance (NOT a gate; §10.3). Reported only.
  const shaF64 = sha256Hex(f64leBytes(ph));
  const shaF32 = sha256Hex(new Uint8Array(myF32.buffer, myF32.byteOffset, myF32.byteLength));
  byName.get('phase f32-gap excess over record').notes.push(
    `${where}: sha256 f64 ${shaF64 === rec.sha256_float64_le ? 'match' : 'differ'}, f32 ${shaF32 === rec.sha256_float32_le ? 'match' : 'differ'}`);

  if (NPZ) {
    const g64 = NPZ.phases_f64.data.subarray(i * N2, (i + 1) * N2);
    const g32 = NPZ.phases_f32.data.subarray(i * N2, (i + 1) * N2);
    obsArray('phase vs npz phases_f64', ph, g64, `${where}`);
    obsExcessArray('phase npz-f32 excess over record', ph, g32, rec.f32_vs_f64.max_abs, `${where}`);
    const gc = NPZ.phase_coeffs_f64.data.subarray(i * 33, (i + 1) * 33);
    obsArray('phase.coeffs vs npz', rec.coeffs_f64, gc, `${where}.coeffs`);
  }
}
console.log(`  synthesised ${phases.length} phases from coeffs_f64`);

/* ================================================================== *
 * Stage 3 -- masks and asymmetry
 * ================================================================== */

console.log('--- Stage 3: masks and asymmetry --------------------------------------');
const masks = [];
const maskById = new Map();
for (let i = 0; i < G.masks.length; i++) {
  const rec = G.masks[i];
  const m = decodeMaskBase64(rec.bits_b64);
  masks.push(m);
  maskById.set(rec.id, m);
  const where = rec.id;

  obs('mask.sum_mask', sumMask(m), rec.sum_mask, `${where}.sum_mask`);
  obs('mask.throughput', throughput(m, N, RADIUS), rec.throughput, `${where}.throughput`);
  obs('mask.sha256_uint8', sha256Hex(m) === rec.sha256_uint8 ? 1 : 0, 1, `${where}.sha256_uint8`);

  const aC = asymmetryCorrected(m, N);
  const aP = asymmetry(m, 'published', N);
  obs('mask.alpha_corrected.f64', aC, rec.alpha_corrected.f64, `${where}.alpha_corrected`);

  // Autoconvolution argmax. GOLDEN.md §8 warns that the peak is a TIE for some masks and
  // that float64 and float32 FFTs break it differently. A disagreement is only acceptable
  // if the two candidates are EXACTLY tied in integer arithmetic -- that is what is gated.
  const am = autoconvArgmax(m, N);
  const gm = rec.autoconv_argmax_yx;
  const argmaxMatches = am[0] === gm[0] && am[1] === gm[1];
  if (argmaxMatches) {
    obs('mask.autoconv argmax: match-or-tie', 1, 1, `${where}.autoconv_argmax (exact match)`);
    obs('mask.alpha_published.f64 (argmax OK)', aP, rec.alpha_published.f64, `${where}.alpha_published`);
  } else {
    const mine = autoconvExactAt(m, am[0], am[1]);
    const theirs = autoconvExactAt(m, gm[0], gm[1]);
    obs('mask.autoconv argmax: match-or-tie', mine === theirs ? 1 : 0, 1,
        `${where}.autoconv_argmax mine=(${am}) exact=${mine} vs golden=(${gm}) exact=${theirs}`);
    byName.get('mask.autoconv argmax: match-or-tie').notes.push(
      `${where}: argmax tie -- mine (${am}) and golden-f64 (${gm}) BOTH have exact autoconv ${mine}` +
      ` (=${theirs}); alpha_published differs by ${Math.abs(aP - rec.alpha_published.f64).toExponential(3)}` +
      ` (mine ${aP}, golden f64 ${rec.alpha_published.f64}, golden f32 ${rec.alpha_published.torch_f32}).` +
      ` alpha_corrected still matches exactly. Documented in GOLDEN.md §8; only alpha_corrected ships.`);
  }

  const dl = stats(psfFromPupil(m, null, N)).max;
  obs('mask.dl_psf_peak == sum_mask^2', dl, rec.diffraction_limited_psf_peak.sum_mask_squared, `${where}.dl_peak`);
  obs('mask.dl_psf_peak == sum_mask^2', dl, rec.diffraction_limited_psf_peak.f64, `${where}.dl_peak.f64`);
  obs('mask.dl_psf_peak == sum_mask^2', dl, rec.sum_mask * rec.sum_mask, `${where}.dl_peak == sum^2`);

  if (NPZ) obsArray('mask bits vs npz masks_uint8', m, NPZ.masks_uint8.data.subarray(i * N2, (i + 1) * N2), `${where}`);
  console.log(`  ${rec.id.padEnd(10)} sum=${String(rec.sum_mask).padStart(4)}  alpha_corr=${aC.toFixed(12)}  alpha_pub=${aP.toFixed(12)}  autoconv=(${am})  DL peak=${dl}`);
}
assert(asymmetryCorrected(circle, N) === 0, 'smoke: asymmetryCorrected(circle) must be exactly 0');

/* ================================================================== *
 * Stages 4, 5, 7 -- the 24 cases
 * ================================================================== */

console.log('--- Stages 4/5/7: 24 cases (PSF, measurement, metrics) ----------------');
const caseOrder = [];
for (let ci = 0; ci < G.cases.length; ci++) {
  const c = G.cases[ci];
  caseOrder.push(c.case_id);
  const mask = maskById.get(c.mask_id);
  const phase = phases[c.phase_id];
  const gt = maskedPhase(mask, phase, N);
  const where = c.case_id;

  /* --- Stage 4: PSF ------------------------------------------------ */
  const psf = psfFromPupil(mask, gt, N);
  const ps = stats(psf);
  const peak = ps.max;
  obs('psf.peak_before_normalization', peak, c.intermediate.psf_peak_before_normalization.f64, `${where}.peak`, c.intermediate.psf_peak_before_normalization.f64);
  obs('psf.argmax_flat (intermediate)', ps.argmax_flat, c.intermediate.psf_argmax_flat.f64, `${where}.psf_argmax_flat`);
  obs('psf.argmax_flat (intermediate)', c.intermediate.sum_mask, sumMask(mask), `${where}.sum_mask`);
  obs('psf.argmax_flat (intermediate)', c.intermediate.circle_peak_divisor.exact_int, CIRCLE_PSF_PEAK, `${where}.divisor`);
  obs('psf.stats.min/max', ps.min, c.psf.stats_f64.min, `${where}.psf.min`, peak);
  obs('psf.stats.min/max', ps.max, c.psf.stats_f64.max, `${where}.psf.max`, peak);
  obs('psf.stats.argmax', ps.argmax_flat, c.psf.stats_f64.argmax_flat, `${where}.psf.argmax_flat`);
  obs('psf.stats.argmax', ps.argmax_yx[0], c.psf.stats_f64.argmax_yx[0], `${where}.psf.argmax_y`);
  obs('psf.stats.argmax', ps.argmax_yx[1], c.psf.stats_f64.argmax_yx[1], `${where}.psf.argmax_x`);
  obs('psf.stats.sum/sum_sq', ps.sum, c.psf.stats_f64.sum, `${where}.psf.sum`);
  obs('psf.stats.sum/sum_sq', ps.sum_sq, c.psf.stats_f64.sum_sq, `${where}.psf.sum_sq`);
  obs('psf.parseval sum==16384*sum_mask', ps.sum, 16384 * c.intermediate.sum_mask, `${where}.parseval`);
  obsArray('psf.samples.grid16', sample(psf, GRID16), c.psf.samples_f64.grid16, `${where}.psf.grid16`, peak);
  obsArray('psf.samples.core16', sample(psf, CORE16), c.psf.samples_f64.core16, `${where}.psf.core16`, peak);
  const psfF32Bound = c.psf.f32.max_rel_to_peak * peak;
  obsExcess('psf f32-gap excess over case bound', peak, c.psf.f32.peak, psfF32Bound, `${where}.psf.f32.peak`, peak);
  obsExcessArray('psf f32-gap excess over case bound', sample(psf, CORE16), c.psf.f32.core16, psfF32Bound, `${where}.psf.f32.core16`, peak);

  /* --- Stage 5: measurement ---------------------------------------- */
  const det = simulateMeasurement(mask, gt, { N, radius: RADIUS, returnDetail: true });
  const meas = det.measurement;
  const ms = stats(meas);
  obs('meas.peak_after_circle_divisor', det.psfPeakAfterCircleDivisor, c.intermediate.psf_peak_after_circle_divisor.f64, `${where}.after_divisor`);
  obs('meas.renorm_clamp_active', det.renormClampActive, c.intermediate.renorm_clamp_active, `${where}.renorm_clamp_active`);
  obs('meas.stats.min/max', ms.min, c.measurement.stats_f64.min, `${where}.meas.min`);
  obs('meas.stats.min/max', ms.max, c.measurement.stats_f64.max, `${where}.meas.max`);
  obs('meas.stats.argmax', ms.argmax_flat, c.measurement.stats_f64.argmax_flat, `${where}.meas.argmax_flat`);
  obs('meas.stats.argmax', ms.argmax_yx[0], c.measurement.stats_f64.argmax_yx[0], `${where}.meas.argmax_y`);
  obs('meas.stats.argmax', ms.argmax_yx[1], c.measurement.stats_f64.argmax_yx[1], `${where}.meas.argmax_x`);
  obs('meas.stats.sum/sum_sq', ms.sum, c.measurement.stats_f64.sum, `${where}.meas.sum`);
  obs('meas.stats.sum/sum_sq', ms.sum_sq, c.measurement.stats_f64.sum_sq, `${where}.meas.sum_sq`);
  obsArray('meas.samples.grid16', sample(meas, GRID16), c.measurement.samples_f64.grid16, `${where}.meas.grid16`);
  obsArray('meas.samples.core16', sample(meas, CORE16), c.measurement.samples_f64.core16, `${where}.meas.core16`);
  const measF32Bound = c.measurement.f32.max_abs;
  obsExcess('meas f32-gap excess over case bound', ms.max, c.measurement.f32.peak, measF32Bound, `${where}.meas.f32.peak`);
  obsExcessArray('meas f32-gap excess over case bound', sample(meas, CORE16), c.measurement.f32.core16, measF32Bound, `${where}.meas.f32.core16`);

  /* --- Stage 7: metrics -------------------------------------------- */
  const pred = syntheticPredictor(gt, N, 0.3);
  const sr = strehlRatio(mask, gt, pred, N);
  const mse = maskedPistonCorrectedMse(mask, pred, gt);
  obs('strehl vs .f64', sr, c.metrics_synthetic_pred.strehl_ratio.f64, `${where}.strehl`);
  obs('mse vs .f64', mse, c.metrics_synthetic_pred.masked_piston_corrected_mse.f64, `${where}.mse`);
  // The §10.1 f32 spread table is rounded to two significant figures, so it is not a
  // usable bound by itself. Gate instead on the EXCESS of my distance to the golden f32
  // value over the golden file's own f64-vs-f32 distance: that isolates my contribution.
  {
    const g64 = c.metrics_synthetic_pred.strehl_ratio.f64, g32 = c.metrics_synthetic_pred.strehl_ratio.torch_f32;
    obs('strehl f32-gap excess over golden', Math.abs(sr - g32) - Math.abs(g64 - g32), 0, `${where}.strehl.f32`);
    const m64 = c.metrics_synthetic_pred.masked_piston_corrected_mse.f64, m32 = c.metrics_synthetic_pred.masked_piston_corrected_mse.torch_f32;
    obs('mse f32-gap excess over golden', (Math.abs(mse - m32) - Math.abs(m64 - m32)) / Math.abs(m64), 0, `${where}.mse.f32`);
  }

  // Piston invariance: +0.3 -> +7.0 must change neither metric (§9, §11 stage 7).
  const pred7 = syntheticPredictor(gt, N, 7.0);
  obs('piston-invariance (0.3 -> 7.0)', strehlRatio(mask, gt, pred7, N), sr, `${where}.strehl piston`);
  obs('piston-invariance (0.3 -> 7.0)', maskedPistonCorrectedMse(mask, pred7, gt), mse, `${where}.mse piston`);

  // Case-level echoes of the mask quantities.
  obs('mask.throughput', throughput(mask, N, RADIUS), c.throughput, `${where}.throughput`);
  obs('mask.alpha_corrected.f64', asymmetryCorrected(mask, N), c.alpha_corrected_f64, `${where}.alpha_corrected`);
  {
    const mrec = G.masks.find((r) => r.id === c.mask_id);
    const am2 = autoconvArgmax(mask, N);
    if (am2[0] === mrec.autoconv_argmax_yx[0] && am2[1] === mrec.autoconv_argmax_yx[1]) {
      obs('mask.alpha_published.f64 (argmax OK)', asymmetry(mask, 'published', N), c.alpha_published_f64, `${where}.alpha_published`);
    }
  }

  if (NPZ) {
    obsArray('psf vs npz psf_f64 (full array)', psf, NPZ.psf_f64.data.subarray(ci * N2, (ci + 1) * N2), `${where}`, peak);
    obsExcessArray('psf npz-f32 excess over case bound', psf, NPZ.psf_f32.data.subarray(ci * N2, (ci + 1) * N2), psfF32Bound, `${where}`, peak);
    obsArray('meas vs npz measurement_f64 (full)', meas, NPZ.measurement_f64.data.subarray(ci * N2, (ci + 1) * N2), `${where}`);
    obsExcessArray('meas npz-f32 excess over case bound', meas, NPZ.measurement_f32.data.subarray(ci * N2, (ci + 1) * N2), measF32Bound, `${where}`);
    // §9 claims bit-for-bit reproducibility FROM golden.json's own phase. Verify exactly
    // that, then separately verify the predictor built on my own (1e-14-accurate) phase.
    const gPhase = NPZ.phases_f64.data.subarray(c.phase_id * N2, (c.phase_id + 1) * N2);
    const gGt = maskedPhase(mask, gPhase, N);
    const gPred = syntheticPredictor(gGt, N, 0.3);
    const refPred = NPZ.pred_synth_f64.data.subarray(ci * N2, (ci + 1) * N2);
    obsArray('pred from GOLDEN phase vs npz', gPred, refPred, `${where}`);
    obsArray('pred from MY phase vs npz', pred, refPred, `${where}`);

    // metrics_unet_pred: the network output comes from the npz, so the metrics can be
    // validated without an ONNX runtime. f32 inference -> rel 1e-3 per §10.2.
    const up32 = NPZ.pred_unet_f32.data.subarray(ci * N2, (ci + 1) * N2);
    const up = new Float64Array(N2);
    for (let p = 0; p < N2; p++) up[p] = up32[p];
    const us = stats(up);
    const u = c.metrics_unet_pred;
    obs('unet strehl vs torch_f32', strehlRatio(mask, gt, up, N), u.strehl_ratio_torch_f32, `${where}.unet.strehl`);
    obs('unet mse vs torch_f32', maskedPistonCorrectedMse(mask, up, gt), u.masked_piston_corrected_mse_torch_f32, `${where}.unet.mse`);
    obs('unet pred stats/core16 (npz echo)', us.min, u.pred_stats_torch_f32.min, `${where}.unet.min`);
    obs('unet pred stats/core16 (npz echo)', us.max, u.pred_stats_torch_f32.max, `${where}.unet.max`);
    obs('unet pred stats/core16 (npz echo)', us.argmax_flat, u.pred_stats_torch_f32.argmax_flat, `${where}.unet.argmax`);
    obs('unet pred stats/core16 (npz echo)', us.sum, u.pred_stats_torch_f32.sum, `${where}.unet.sum`);
    obs('unet pred stats/core16 (npz echo)', us.sum_sq, u.pred_stats_torch_f32.sum_sq, `${where}.unet.sum_sq`);
    obsArray('unet pred stats/core16 (npz echo)', sample(up, CORE16), u.pred_core16_torch_f32, `${where}.unet.core16`);
  }
}
if (NPZ) {
  const ids = NPZ.case_ids.data;
  for (let i = 0; i < ids.length; i++) obs('config/paper_modes/sampling echo', caseOrder[i] === ids[i] ? 1 : 0, 1, `case_ids[${i}]`);
  for (let i = 0; i < NPZ.mask_ids.data.length; i++) obs('config/paper_modes/sampling echo', G.masks[i].id === NPZ.mask_ids.data[i] ? 1 : 0, 1, `mask_ids[${i}]`);
}
console.log(`  ${G.cases.length} cases done`);

/* ================================================================== *
 * Stage 6 -- noise probe
 * ================================================================== */

console.log('--- Stage 6: noise probe (the only test of order + divisor) -----------');
const noiseF32 = decodeFloat32Base64(G.noise_probe.noise_float32_b64, N2);
const noiseSha = sha256Hex(new Uint8Array(noiseF32.buffer, noiseF32.byteOffset, noiseF32.byteLength));
obs('noise field vs sha256/npz', noiseSha === G.noise_probe.noise_sha256_float32_le ? 1 : 0, 1, 'noise sha256');
const noise = new Float64Array(N2);              // widen to f64 before use, as the reference does
for (let p = 0; p < N2; p++) noise[p] = noiseF32[p];
if (NPZ) {
  const gn = NPZ.noise_f32.data;
  let bad = 0;
  for (let p = 0; p < N2; p++) if (gn[p] !== noiseF32[p]) bad++;
  obs('noise field vs sha256/npz', bad === 0 ? 1 : 0, 1, 'noise vs npz noise_f32');
}
{
  const ns = stats(noise);
  console.log(`  noise max=${ns.max} at (${ns.argmax_yx}) -- 0.05*max = ${(0.05 * ns.max).toFixed(6)}`);
}

for (let i = 0; i < G.noise_probe.results.length; i++) {
  const r = G.noise_probe.results[i];
  const mask = maskById.get(r.mask_id);
  const gt = maskedPhase(mask, phases[r.phase_id], N);
  const where = `${r.mask_id}__phase${r.phase_id}(sigma=${r.noise_sigma})`;
  const meas = simulateMeasurement(mask, gt, {
    N, radius: RADIUS, withNoise: true, noiseSigma: r.noise_sigma, noise,
    normalizeByPeak: true, renormalizeAfterNoise: true,
  });
  const ms = stats(meas);
  obs('noisy.stats.min/max', ms.min, r.stats_f64.min, `${where}.min`);
  obs('noisy.stats.min/max', ms.max, r.stats_f64.max, `${where}.max`);
  obs('noisy.stats.argmax', ms.argmax_flat, r.stats_f64.argmax_flat, `${where}.argmax_flat`);
  obs('noisy.stats.argmax', ms.argmax_yx[0], r.stats_f64.argmax_yx[0], `${where}.argmax_y`);
  obs('noisy.stats.argmax', ms.argmax_yx[1], r.stats_f64.argmax_yx[1], `${where}.argmax_x`);
  obs('noisy.stats.sum/sum_sq', ms.sum, r.stats_f64.sum, `${where}.sum`);
  obs('noisy.stats.sum/sum_sq', ms.sum_sq, r.stats_f64.sum_sq, `${where}.sum_sq`);
  obsArray('noisy.samples.grid16', sample(meas, GRID16), r.samples_f64.grid16, `${where}.grid16`);
  obsArray('noisy.samples.core16', sample(meas, CORE16), r.samples_f64.core16, `${where}.core16`);
  obsExcess('noisy f32-gap excess over record', ms.max, r.f32.peak, r.f32.max_abs, `${where}.f32.peak`);
  assert(ms.argmax_yx[0] === 65 && ms.argmax_yx[1] === 112,
         `noise probe ${where}: argmax is (${ms.argmax_yx}), must be (65,112) -- the 1263376 divisor is missing`);
  if (NPZ) {
    obsArray('noisy vs npz noisy_f64 (full)', meas, NPZ.measurement_noisy_f64.data.subarray(i * N2, (i + 1) * N2), where);
    obsExcessArray('noisy npz-f32 excess over record', meas, NPZ.measurement_noisy_f32.data.subarray(i * N2, (i + 1) * N2), r.f32.max_abs, where);
  }
  console.log(`  ${where}: argmax=(${ms.argmax_yx}) sum=${ms.sum.toFixed(6)}`);
}

// Negative control for §5.4: dropping the divisor must break the noisy argmax.
{
  const r = G.noise_probe.results[0];
  const mask = maskById.get(r.mask_id);
  const gt = maskedPhase(mask, phases[r.phase_id], N);
  const wrong = simulateMeasurement(mask, gt, {
    N, radius: RADIUS, withNoise: true, noiseSigma: r.noise_sigma, noise,
    normalizeByPeak: true, renormalizeAfterNoise: true, circlePeakDivisor: 1,
  });
  const ws = stats(wrong);
  assert(!(ws.argmax_yx[0] === 65 && ws.argmax_yx[1] === 112),
         'negative control: divisor=1 still gives argmax (65,112) -- the probe is not discriminating');
  console.log(`  [negative control] divisor=1 -> argmax=(${ws.argmax_yx}) (correctly NOT (65,112))`);
}

/* ================================================================== *
 * Ladder reader
 * ================================================================== */

console.log('--- ladder.bin reader + pupils.json ----------------------------------');
// ladder.json / pupils.json store alpha and throughput ROUNDED TO 6 DECIMALS, and store
// the torch-float32 alpha (see pupils.json stats: min 0.014388501644134521 is the f32
// value). Tolerance = 5e-7 (6-decimal rounding) + 5e-8 (measured f32-vs-f64 spread of
// alpha_corrected across the 6 golden masks: max 2.5e-8) = 5.5e-7.
const ALPHA6 = 5.25e-7;   // 5e-7 (6-decimal rounding) + 2.5e-8 (max |f32-f64| of alpha_corrected over the 6 golden masks)
track(8, 'ladder.bin layout vs ladder.json',   'exact', 0,       'ladder.json layout + per-frame area');
track(8, 'ladder alpha_corrected',             'abs',   ALPHA6,  '6-decimal rounding 5e-7 + measured f32 spread 2.5e-8');
track(8, 'ladder alpha_published (argmax OK)', 'abs',   ALPHA6,  'same; gated only where the argmax is unambiguous');
track(8, 'ladder autoconv argmax: unambiguous','exact', 0,       'a mismatched alpha_published must come from a PROVEN exact integer tie');
track(8, 'ladder throughput',                  'abs',   ALPHA6,  '6-decimal rounding');
{
  const meta = JSON.parse(readFileSync(join(ASSETS, 'ladder.json'), 'utf8'));
  const bin = readFileSync(join(ASSETS, 'ladder.bin'));
  const L = readLadder(bin, meta);
  obs('ladder.bin layout vs ladder.json', L.frameCount, meta.layout.frame_count, 'frameCount');
  obs('ladder.bin layout vs ladder.json', L.bytesPerFrame, meta.layout.bytes_per_frame, 'bytesPerFrame');
  obs('ladder.bin layout vs ladder.json', bin.length, meta.layout.total_bytes, 'total_bytes');
  obs('ladder.bin layout vs ladder.json', L.width, meta.layout.width, 'width');
  obs('ladder.bin layout vs ladder.json', L.height, meta.layout.height, 'height');
  obs('ladder.bin layout vs ladder.json', meta.reference_circle_area_px, cSum, 'reference_circle_area_px');

  let ties = 0;
  for (const fr of meta.frames) {
    const m = L.mask(fr.slot);
    const w = `frame ${fr.slot}${fr.pool_id != null ? ' (pool ' + fr.pool_id + ')' : ''}`;
    obs('ladder.bin layout vs ladder.json', L.area(fr.slot), fr.area, `${w}.area`);
    obs('ladder.bin layout vs ladder.json', L.pixel(fr.slot, 3, 5), m[5 * 128 + 3], `${w}.pixel(3,5) vs mask()`);
    obs('ladder.bin layout vs ladder.json', L.frameInfo(fr.slot).slot, fr.slot, `${w}.frameInfo`);
    obs('ladder alpha_corrected', asymmetryCorrected(m, N), fr.alpha_corrected, `${w}.alpha_corrected`);
    obs('ladder throughput', throughput(m, N, RADIUS), fr.throughput, `${w}.throughput`);

    const aP = asymmetry(m, 'published', N);
    if (Math.abs(aP - fr.alpha_published) <= ALPHA6) {
      obs('ladder alpha_published (argmax OK)', aP, fr.alpha_published, `${w}.alpha_published`);
      obs('ladder autoconv argmax: unambiguous', 1, 1, `${w}.autoconv (agrees)`);
    } else {
      // ladder.json does not store the argmax, so prove ambiguity directly: the exact
      // integer autoconvolution must be attained at more than one shift.
      const { tied, exactMax } = autoconvCandidates(m);
      obs('ladder autoconv argmax: unambiguous', tied.length > 1 ? 1 : 0, 1,
          `${w}.autoconv argmax ambiguous? tied=${tied.length} at exact value ${exactMax}`);
      if (tied.length > 1) ties++;
      byName.get('ladder autoconv argmax: unambiguous').notes.push(
        `${w}: alpha_published ${aP.toFixed(6)} vs ladder.json ${fr.alpha_published} — ` +
        `exact autoconv max ${exactMax} attained at ${tied.length} shifts ` +
        `(${tied.map((t) => `(${t.y},${t.x})`).join(' ')}); documented tie-break, GOLDEN.md §8. ` +
        `alpha_corrected agrees.`);
    }
  }
  console.log(`  ${L.frameCount} frames x ${L.bytesPerFrame} B verified against ladder.json (${ties} documented autoconv ties)`);

  // pupils.json holds all 1500 test pupils as metadata; cross-check the ladder frames
  // and the golden pool masks against it by pool id.
  if (existsSync(join(ASSETS, 'pupils.json'))) {
    const P = JSON.parse(readFileSync(join(ASSETS, 'pupils.json'), 'utf8'));
    const byId = new Map(P.pupils.map((r) => [r.id, r]));
    track(8, 'pupils.json cross-check (by id)', 'abs', ALPHA6, 'pupils.json stores 6-decimal f32 alphas');
    let n = 0;
    for (const fr of meta.frames) {
      if (fr.pool_id == null || !byId.has(fr.pool_id)) continue;
      const rec = byId.get(fr.pool_id), m = L.mask(fr.slot);
      obs('pupils.json cross-check (by id)', sumMask(m), rec.area, `pool ${fr.pool_id}.area`);
      obs('pupils.json cross-check (by id)', asymmetryCorrected(m, N), rec.alpha_corrected, `pool ${fr.pool_id}.alpha_corrected`);
      obs('pupils.json cross-check (by id)', throughput(m, N, RADIUS), rec.throughput, `pool ${fr.pool_id}.throughput`);
      n++;
    }
    for (const mrec of G.masks) {
      if (mrec.pool_index == null || !byId.has(mrec.pool_index)) continue;
      const rec = byId.get(mrec.pool_index), m = maskById.get(mrec.id);
      obs('pupils.json cross-check (by id)', sumMask(m), rec.area, `${mrec.id} (pool ${mrec.pool_index}).area`);
      obs('pupils.json cross-check (by id)', asymmetryCorrected(m, N), rec.alpha_corrected, `${mrec.id}.alpha_corrected`);
      obs('pupils.json cross-check (by id)', throughput(m, N, RADIUS), rec.throughput, `${mrec.id}.throughput`);
      n++;
    }
    obs('pupils.json cross-check (by id)', P.pupils.length, P.count, 'pupils count');
    obs('pupils.json cross-check (by id)', P.reference_circle_area_px, cSum, 'reference_circle_area_px');
    console.log(`  pupils.json: ${P.pupils.length} pupils indexed, ${n} cross-checked against decoded bitmaps`);
  }
}

/* ================================================================== *
 * Stage 9 -- branches the golden set carries no vectors for
 * ================================================================== */

console.log('--- Stage 9: un-goldened branches (self-consistency) ------------------');
track(9, 'S5.4 degeneracy: meas == psf/psf.max',  'abs', 1e-15, 'GOLDEN.md §5.4 states this holds to 1.1e-16');
track(9, 'legacy §5.3 collapses to the same',     'abs', 1e-15, 'noise off + renormalise on ⇒ every positive prefactor cancels');
track(9, 'psfFromPupil(normalize) == psf/peak',   'abs', 1e-15, '§4.6 definition, evaluated directly');
track(9, 'clamp 1e-6 (renorm) fires when tiny',   'exact', 0,   '§5.2: floor must engage below 1e-6');
track(9, 'clamp 1e-4 (strehl denom) fires',       'rel', 1e-12, '§6: eps floors the denominator only');
track(9, 'clamp 0.0 (post-noise) truncation',     'exact', 0,   '§5.2: 8036/16384 (circle), 8099/16384 (triangle)');
{
  // (a) §5.4: with noise off, the 1263376 divisor and the renormalisation cancel.
  // (b) The legacy §5.3 branch differs only by positive scalar prefactors, so with
  //     renormalise-after-noise on and noise off it must collapse to the same array.
  // (c) §4.6's self-normalisation is checked against its own definition.
  for (const c of G.cases) {
    const mask = maskById.get(c.mask_id);
    const gt = maskedPhase(mask, phases[c.phase_id], N);
    const psf = psfFromPupil(mask, gt, N);
    const pk = stats(psf).max;
    const pub = simulateMeasurement(mask, gt, { N, radius: RADIUS });
    const legT = simulateMeasurement(mask, gt, { N, radius: RADIUS, normalizeByPeak: false, useThroughput: true });
    const legF = simulateMeasurement(mask, gt, { N, radius: RADIUS, normalizeByPeak: false, useThroughput: false });
    const selfN = psfFromPupil(mask, gt, N, true);
    for (let p = 0; p < N2; p++) {
      obs('S5.4 degeneracy: meas == psf/psf.max', pub[p], psf[p] / pk, `${c.case_id}[${p}]`);
      obs('legacy §5.3 collapses to the same', legT[p], pub[p], `${c.case_id} thr=on [${p}]`);
      obs('legacy §5.3 collapses to the same', legF[p], pub[p], `${c.case_id} thr=off [${p}]`);
      obs('psfFromPupil(normalize) == psf/peak', selfN[p], psf[p] / Math.max(pk, 1e-4), `${c.case_id}[${p}]`);
    }
  }

  // (d) The 1e-6 renormalisation floor: drive the post-divisor peak below 1e-6 with a
  //     single-pixel pupil (peak = 1, /1263376 = 7.9e-7 < 1e-6), so the floor engages.
  const dot = new Uint8Array(N2); dot[64 * N + 64] = 1;
  const d1 = simulateMeasurement(dot, null, { N, radius: RADIUS, returnDetail: true });
  obs('clamp 1e-6 (renorm) fires when tiny', d1.renormClampActive, true, 'single-pixel pupil');
  obs('clamp 1e-6 (renorm) fires when tiny', Math.abs(stats(d1.measurement).max - (1 / CIRCLE_PSF_PEAK) / 1e-6) < 1e-9, true,
      'single-pixel peak = (1/1263376)/1e-6');
  const d2 = simulateMeasurement(circle, null, { N, radius: RADIUS, returnDetail: true });
  obs('clamp 1e-6 (renorm) fires when tiny', d2.renormClampActive, false, 'reference circle (must NOT fire)');

  // (e) The 1e-4 Strehl denominator floor: an empty mask gives peak_ref = 0 -> eps binds.
  const empty = new Uint8Array(N2);
  obs('clamp 1e-4 (strehl denom) fires', strehlRatio(empty, new Float64Array(N2), new Float64Array(N2), N), 0, 'empty mask');
  obs('clamp 1e-4 (strehl denom) fires', maskedPistonCorrectedMse(empty, new Float64Array(N2), new Float64Array(N2)), 0, 'empty mask MSE (1e-9 mask_sum floor)');

  // (f) §5.2's stated post-noise truncation counts, an independent probe of the 0.0 clamp.
  for (const [mid, want] of [['circle', 8036], ['triangle', 8099]]) {
    const mask = maskById.get(mid);
    const gt = maskedPhase(mask, phases[2], N);
    const m = simulateMeasurement(mask, gt, { N, radius: RADIUS, withNoise: true, noiseSigma: 0.05, noise });
    let zeros = 0;
    for (let p = 0; p < N2; p++) if (m[p] === 0) zeros++;
    obs('clamp 0.0 (post-noise) truncation', zeros, want, `${mid}: ${zeros} pixels truncated to 0.0`);
  }
  console.log('  degeneracy, legacy branch, and all three clamps verified');
}

/* ================================================================== *
 * Timing
 * ================================================================== */

console.log('--- timing ------------------------------------------------------------');
const benchMask = maskById.get('pool_101');
const benchCoeffs = G.phases[2].coeffs_f64;
// Timings are the MINIMUM over repeated batches (the sample least contaminated by
// descheduling). Two calibration probes establish how much of a CPU this process is
// actually getting, so the numbers can be read on any machine:
//   * sha256 throughput -- NATIVE code, so it is independent of V8's JIT tier and
//     measures the machine's raw compute share directly;
//   * a scalar multiply-add loop -- the same language and tier as the code under test.
// A healthy, unshared modern x86 core does ~2000 MB/s of sha256 and ~0.5-1.0 ns per
// scalar multiply-add. Memory bandwidth is measured too, to tell compute starvation
// (both probes slow, bandwidth fine) apart from a memory-bound machine.
const SHA_REF_MBPS = 2000;
function calibrate() {
  // Short buffer + many reps: a long single measurement is certain to be preempted on a
  // shared box, which would understate the CPU share available in a ~1 ms burst.
  const buf = Buffer.alloc(1024 * 1024);
  let bestSha = Infinity;
  for (let r = 0; r < 200; r++) {
    const t = performance.now();
    createHash('sha256').update(buf).digest();
    const d = performance.now() - t;
    if (d < bestSha) bestSha = d;
  }
  const shaMBps = 1 / (bestSha / 1000);

  const step = () => { let s = 0; for (let i = 0; i < 1e5; i++) s += i * 0.5; return s; };
  for (let i = 0; i < 600; i++) step();                        // same warm-up as timeit()
  let bestLoop = Infinity;
  for (let b = 0; b < 9; b++) {
    const t = performance.now();
    for (let i = 0; i < 40; i++) step();
    const ms = (performance.now() - t) / 40;
    if (ms < bestLoop) bestLoop = ms;
  }
  const nsPerFlop = (bestLoop * 1e6) / 1e5;

  const big = new Float64Array(4 * 1024 * 1024);
  let bestFill = Infinity;
  for (let r = 0; r < 20; r++) {
    const t = performance.now();
    big.fill(1.5);
    const d = performance.now() - t;
    if (d < bestFill) bestFill = d;
  }
  return { shaMBps, nsPerFlop, fillGBps: 32 / (bestFill / 1000) / 1024 };
}
const CAL = calibrate();
console.log(`  [machine] native sha256 ${CAL.shaMBps.toFixed(0)} MB/s (unshared core ~${SHA_REF_MBPS}), ` +
            `memcpy ${CAL.fillGBps.toFixed(1)} GB/s (healthy 5-15), ` +
            `load avg ${os.loadavg()[0].toFixed(0)} on ${os.cpus().length} cpus.`);
console.log(`  [machine] sha256 is native code, so it measures the CPU share this process actually gets, ` +
            `independent of V8's JIT tier: ${(CAL.shaMBps / SHA_REF_MBPS * 100).toFixed(0)}% of an unshared core.`);

// Minimum over many SINGLE-call timings. On a time-shared box a long timing batch is
// certain to be preempted, so a batch minimum is biased upward by however long the batch
// runs; single-call minima give every workload the same chance of landing in a clean
// scheduling window. The calibration below is measured exactly the same way.
const timeit = (label, fn, reps = 400) => {
  for (let i = 0; i < Math.max(100, reps); i++) fn();      // deep warm-up so V8 tiers up
  let best = Infinity;
  const all = new Float64Array(reps);
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    fn();
    const ms = performance.now() - t0;
    all[i] = ms;
    if (ms < best) best = ms;
  }
  all.sort();
  const med = all[reps >> 1];
  console.log(`  ${label.padEnd(46)} ${best.toFixed(3)} ms best   ${med.toFixed(3)} ms median`);
  return { best, med };
};
// Same-methodology calibration: one call ~= one pipeline's worth of scalar work.
const calFn = () => { let s = 0; for (let i = 0; i < 2e6; i++) s += i * 0.5; return s; };
const tCal = timeit('CALIBRATION 2e6 scalar mul-adds', calFn, 400).best;
const NS_PER_FLOP = (tCal * 1e6) / 2e6;
const SLOWDOWN = Math.max(1, NS_PER_FLOP / 0.85);
console.log(`  -> ${NS_PER_FLOP.toFixed(2)} ns per scalar mul-add; a healthy unshared core is ~0.85 ns, so these ` +
            `best-case timings are ${SLOWDOWN <= 1.3 ? 'representative of a real core' : SLOWDOWN.toFixed(1) + 'x inflated'}.\n`);

const tPhase = timeit('synthPhase (33 modes, 1129 px)', () => synthPhase(benchCoeffs, N, RADIUS), 400);
const phaseB = synthPhase(benchCoeffs, N, RADIUS);
const gtB = maskedPhase(benchMask, phaseB, N);
const tPsf = timeit('psfFromPupil (2-D FFT 128x128 + |.|^2)', () => psfFromPupil(benchMask, gtB, N), 400);
const tMeas = timeit('simulateMeasurement (from gt)', () => simulateMeasurement(benchMask, gtB, { N, radius: RADIUS }), 400);
const tFull = timeit('FULL pipeline: coeffs -> phase -> PSF -> meas', () => runPipeline(benchMask, benchCoeffs, { N, radius: RADIUS }), 400);
const tAsym = timeit('asymmetry(corrected) (2 FFTs)', () => asymmetryCorrected(benchMask, N), 200);
const tStrehl = timeit('strehlRatio (2 PSFs)', () => strehlRatio(benchMask, gtB, syntheticPredictor(gtB, N), N), 200);

/* ================================================================== *
 * Report
 * ================================================================== */

console.log('\n================================ RESULTS ================================');
const hdr = ['St', 'quantity', 'kind', 'tolerance', 'worst deviation', 'n', 'ok'];
const table = rows.filter((r) => r.n > 0).map((r) => {
  const ok = r.kind === 'exact' ? r.worst === 0 : r.worst <= r.tol;
  return {
    r, ok,
    cells: [
      String(r.stage),
      r.name,
      KINDS[r.kind].label,
      r.kind === 'exact' ? 'exact' : r.tol.toExponential(1),
      KINDS[r.kind].fmt(r.worst),
      String(r.n),
      ok ? 'PASS' : 'FAIL',
    ],
  };
});
const w = hdr.map((h, i) => Math.max(h.length, ...table.map((t) => t.cells[i].length)));
const line = (cs) => cs.map((c, i) => (i === 1 || i === 2 ? c.padEnd(w[i]) : c.padStart(w[i]))).join('  ');
console.log(line(hdr));
console.log(w.map((x) => '-'.repeat(x)).join('  '));
for (const t of table) console.log(line(t.cells) + (t.ok ? '' : `   <- ${t.r.where}`));

const failedRows = table.filter((t) => !t.ok);

/* A tracked row that never ran is a silently lost stage, not a pass. Without this,
   emptying a golden section (e.g. G.zernike_spot_checks.modes) drops its rows from
   BOTH the report and the verdict, and the harness still prints ALL CHECKS PASSED
   with a quietly smaller comparison count. Audit finding P-1. */
const emptyRows = rows.filter((r) => r.n === 0);
if (emptyRows.length) {
  console.log('\n--- tracked quantities that never executed ---');
  for (const r of emptyRows) console.log(`  stage ${r.stage}  ${r.name}   (expected comparisons, got 0)`);
}
console.log('\n--- tolerance provenance ---');
for (const t of table) console.log(`  ${t.r.name.padEnd(38)} ${t.r.kind === 'exact' ? 'exact' : t.r.tol.toExponential(1).padStart(8)}   ${t.r.src}`);

const notes = rows.flatMap((r) => r.notes);
if (notes.length) { console.log('\n--- informational notes (not gated) ---'); for (const n of notes) console.log('  ' + n); }

console.log('\n--- timing summary (Node ' + process.version + ', N=128) ---');
const trow = (l, v) => console.log(`  ${l.padEnd(28)} ${v.best.toFixed(3).padStart(7)} ms best   ${v.med.toFixed(3).padStart(7)} ms median   (${(1000 / v.med).toFixed(0).padStart(5)} /s at the median)`);
trow('phase synth', tPhase);
trow('PSF (FFT)', tPsf);
trow('measurement (incl. PSF)', tMeas);
trow('FULL pipeline', tFull);
trow('asymmetry(corrected)', tAsym);
trow('strehlRatio', tStrehl);
console.log(`  -> ONE FULL PIPELINE (phase synth -> PSF -> measurement) at N=128: ` +
            `${tFull.best.toFixed(2)} ms best, ${tFull.med.toFixed(2)} ms median ` +
            `= ${(1000 / tFull.med).toFixed(0)} updates/s.`);
console.log(`     A 60 fps slider has a 16.7 ms frame budget, so the demo has ${(16.7 / tFull.med).toFixed(0)}x headroom.`);
console.log(`     Machine calibration: ${NS_PER_FLOP.toFixed(2)} ns/mul-add, sha256 ${CAL.shaMBps.toFixed(0)} MB/s ` +
            `(unshared-core reference: ~0.85 ns and ~${SHA_REF_MBPS} MB/s).`);

if (FAILS.length) { console.log('\n--- assertion failures ---'); for (const f of FAILS) console.log('  ' + f); }

const allPass = failedRows.length === 0 && FAILS.length === 0 && emptyRows.length === 0;
console.log(`\n${allPass ? 'ALL CHECKS PASSED'
              : 'FAILURES: ' + (failedRows.length + FAILS.length + emptyRows.length) +
                (emptyRows.length ? ` (${emptyRows.length} never executed)` : '')}` +
            `  (${table.length} tracked quantities, ${table.reduce((a, t) => a + t.r.n, 0).toLocaleString()} individual comparisons` +
            `${NPZ ? ', golden.npz whole-array coverage ON' : ', golden.npz NOT read'})`);
process.exit(allPass ? 0 : 1);
