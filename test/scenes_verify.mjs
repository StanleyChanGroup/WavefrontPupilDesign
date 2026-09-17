/**
 * scenes_verify.mjs -- verification of the three procedural scenes in src/scene.js.
 *
 * Prints a report; test/SCENE_VERIFICATION.md is that report plus commentary.
 * Run:  node test/scenes_verify.mjs
 *
 * Nothing here is random: every "seed" is fixed, and the only timing claims are
 * single-call minima (this machine runs at load 140-240 on 128 cores, so a mean
 * over a batch measures the scheduler, not the code).
 */
import { fft2, ifft2, referenceCircle, psfFromPupil, synthPhase, maskedPhase,
         strehlRatio, PAPER_MODES } from '../src/optics.js';
import * as scene from '../src/scene.js';

const { starField, starFieldSpec, terrain, terrainSpec, usafTarget, convolvePsf,
        asinhStretch, binaryCut, SCENES, RAYLEIGH_PX, BINARY_SEP_PX, STAR_SIGMA_PX,
        PUPIL_N, PUPIL_RADIUS, PUPIL_CUTOFF_CYC_PX } = scene;

const N = PUPIL_N, R = PUPIL_RADIUS;
const SIZES = [256, 512];
const fx = (v, d = 4) => v.toFixed(d);
const ex = (v, d = 3) => v.toExponential(d);
let failures = 0;
function check(label, ok, detail) {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '   ' + detail : ''}`);
}
const hr = (t) => console.log('\n' + '='.repeat(72) + '\n' + t + '\n' + '='.repeat(72));

/* ------------------------------------------------------------------ *
 * helpers
 * ------------------------------------------------------------------ */

function stats(a) {
  let mn = Infinity, mx = -Infinity, sum = 0, atLo = 0, atHi = 0;
  for (const v of a) {
    if (!Number.isFinite(v)) return { finite: false };
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
    if (v <= 0) atLo++;
    if (v >= 1) atHi++;
  }
  return { finite: true, min: mn, max: mx, mean: sum / a.length, atLo, atHi };
}

const identical = (a, b) => {
  if (a.length !== b.length) return false;
  const ua = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const ub = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
};

/** Radially averaged power spectrum (mean removed), index = integer |k| in cycles/frame. */
function radialPowerSpectrum(img, S) {
  const re = Float64Array.from(img), im = new Float64Array(S * S);
  let m = 0; for (const v of re) m += v; m /= re.length;
  for (let i = 0; i < re.length; i++) re[i] -= m;
  fft2(re, im, S, false);
  const nb = S >> 1;
  const acc = new Float64Array(nb + 1), cnt = new Float64Array(nb + 1);
  for (let ky = 0; ky < S; ky++) {
    const ay = ky < S / 2 ? ky : ky - S;
    for (let kx = 0; kx < S; kx++) {
      const ax = kx < S / 2 ? kx : kx - S;
      const r = Math.round(Math.hypot(ax, ay));
      if (r > nb) continue;
      const p = ky * S + kx;
      acc[r] += re[p] * re[p] + im[p] * im[p];
      cnt[r]++;
    }
  }
  const ps = new Float64Array(nb + 1);
  let total = 0;
  for (let r = 0; r <= nb; r++) { ps[r] = cnt[r] ? acc[r] / cnt[r] : 0; total += acc[r]; }
  return { ps, acc, total };
}

/** Fraction of the mean-removed power that sits above a given cycles/px frequency. */
function powerAbove(img, S, fCycPx) {
  const { acc, total } = radialPowerSpectrum(img, S);
  const kCut = fCycPx * S;
  let above = 0;
  for (let r = 0; r <= S >> 1; r++) if (r > kCut) above += acc[r];
  return above / total;
}

/** Catmull-Rom sample of a toroidal image at a fractional position. */
function sampleAt(img, S, x, y) {
  const g = (xx, yy) => img[((((yy % S) + S) % S) * S) + (((xx % S) + S) % S)];
  const cr = (p0, p1, p2, p3, t) => 0.5 * ((2 * p1) + (-p0 + p2) * t
    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
  const fxi = Math.floor(x), fyi = Math.floor(y), tx = x - fxi, ty = y - fyi;
  const col = [];
  for (let j = -1; j <= 2; j++) {
    col.push(cr(g(fxi - 1, fyi + j), g(fxi, fyi + j), g(fxi + 1, fyi + j), g(fxi + 2, fyi + j), tx));
  }
  return cr(col[0], col[1], col[2], col[3], ty);
}

/**
 * Two-source resolution measure along the line joining (ax,ay) and (bx,by).
 * Scans +-1.5 separations about the midpoint at 1/200 px and looks for a local
 * maximum on each side of the midpoint with a strictly lower minimum between.
 *
 *   dip = 1 - trough / min(peakA, peakB)     (Rayleigh's own criterion is ~0.265)
 *   resolved  <=>  two distinct interior maxima exist  <=>  dip > 0
 */
function resolutionDip(img, S, ax, ay, bx, by) {
  const L = Math.hypot(bx - ax, by - ay);
  const ux = (bx - ax) / L, uy = (by - ay) / L;
  const n = 1201, step = (3 * L) / (n - 1);
  const x0 = (ax + bx) / 2 - ux * 1.5 * L, y0 = (ay + by) / 2 - uy * 1.5 * L;
  const p = new Float64Array(n);
  for (let i = 0; i < n; i++) p[i] = sampleAt(img, S, x0 + ux * i * step, y0 + uy * i * step);
  const mid = (n - 1) / 2;
  let pa = -Infinity, ia = 0; for (let i = 1; i < mid; i++) if (p[i] > pa) { pa = p[i]; ia = i; }
  let pb = -Infinity, ib = 0; for (let i = mid + 1; i < n - 1; i++) if (p[i] > pb) { pb = p[i]; ib = i; }
  let tr = Infinity; for (let i = ia; i <= ib; i++) if (p[i] < tr) tr = p[i];
  const peak = Math.min(pa, pb);
  const twoPeaks = ia > 1 && ib < n - 2 && tr < peak * (1 - 1e-12) && ia !== ib;
  return {
    resolved: twoPeaks,
    dip: twoPeaks ? 1 - tr / peak : 0,
    peakA: pa, peakB: pb, trough: tr,
    peakSepPx: (ib - ia) * step,
  };
}

/** Exact band-limited translation of a toroidal image, for the aliasing test. */
function fourierShift(img, S, dx, dy) {
  const re = Float64Array.from(img), im = new Float64Array(S * S);
  fft2(re, im, S, false);
  for (let ky = 0; ky < S; ky++) {
    const fy = (ky < S / 2 ? ky : ky - S) / S;
    for (let kx = 0; kx < S; kx++) {
      const fxr = (kx < S / 2 ? kx : kx - S) / S;
      const ph = -2 * Math.PI * (fxr * dx + fy * dy);
      const c = Math.cos(ph), s = Math.sin(ph), p = ky * S + kx;
      const a = re[p], b = im[p];
      re[p] = a * c - b * s; im[p] = a * s + b * c;
    }
  }
  ifft2(re, im, S);
  return re;
}

const l2 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); };
const nrm = (a) => { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s); };
const sum = (a) => { let s = 0; for (const v of a) s += v; return s; };

/** mulberry32, the same generator src/scene.js and test/scene_check.mjs use. */
function mulberry32(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The fixed aberration used throughout: PAPER_MODES coefficients, mulberry32(12345), +-amp rad. */
function aberration(amp, seed = 12345) {
  const rnd = mulberry32(seed);
  const c = new Float64Array(PAPER_MODES.length);
  for (let i = 0; i < c.length; i++) c[i] = (rnd() * 2 - 1) * amp;
  return c;
}

/* ------------------------------------------------------------------ *
 * 0. the pupil and its PSFs
 * ------------------------------------------------------------------ */

hr('0. REFERENCE PUPIL AND ITS DIFFRACTION LIMIT');

const circle = referenceCircle(N, R);
const psfSharp = psfFromPupil(circle, null, N);
const coeffs = aberration(2.0);
const gt = maskedPhase(circle, synthPhase(coeffs, N, R), N);
const psfAb = psfFromPupil(circle, gt, N);
// "corrected": 88 % of the wavefront removed, the residual an AO loop would leave.
const resid = new Float64Array(N * N);
for (let p = 0; p < N * N; p++) resid[p] = 0.12 * gt[p];
const psfCorr = psfFromPupil(circle, resid, N);

let rms = 0, np = 0;
for (let p = 0; p < N * N; p++) if (circle[p]) { rms += gt[p] * gt[p]; np++; }
rms = Math.sqrt(rms / np);
const strehlAb = strehlRatio(circle, gt, new Float64Array(N * N), N);
const strehlCo = strehlRatio(circle, gt, new Float64Array(gt.map((v) => 0.88 * v)), N);

console.log(`pupil  N=${N} radius=${R}   lambda/D = N/(2r) = ${fx(N / (2 * R))} px`);
console.log(`Rayleigh limit 1.2196699*lambda/D          = ${fx(RAYLEIGH_PX, 5)} px`);
// Measure the first dark ring exactly: |FFT{pupil}|^2 is invariant to where the pupil
// sits in the array, so zero-padding the 128 x 128 pupil into 1024 x 1024 samples the
// SAME continuous PSF at 1024/128 = 8 samples per PSF pixel.
const PAD = 1024, OS = PAD / N;
const { firstZero, zeroVal } = (() => {
  const re = new Float64Array(PAD * PAD), im = new Float64Array(PAD * PAD);
  const rows = new Uint8Array(PAD);
  for (let y = 0; y < N; y++) { rows[y] = 1; for (let x = 0; x < N; x++) re[y * PAD + x] = circle[y * N + x]; }
  fft2(re, im, PAD, false, rows);
  const pk = re[0] * re[0] + im[0] * im[0];
  let fz = 0, bv = Infinity;
  for (let n = 1; n < 7 * OS; n++) {
    const v = re[n] * re[n] + im[n] * im[n];
    if (v < bv) { bv = v; fz = n / OS; }
    if (v > bv && n / OS > 3) break;   // past the first minimum
  }
  return { firstZero: fz, zeroVal: bv / pk };
})();
console.log(`measured first dark ring of psfSharp        = ${fx(firstZero, 4)} px  (${ex(zeroVal)} of peak,`
  + ` pupil zero-padded to ${PAD} => ${OS} samples/px)`);
console.log(`binary separation used by starField()       = ${fx(BINARY_SEP_PX, 5)} px = ${fx(BINARY_SEP_PX / RAYLEIGH_PX, 3)} Rayleigh`);
console.log(`pupil optical cutoff 2r/N                   = ${fx(PUPIL_CUTOFF_CYC_PX, 4)} cyc/px  (period ${fx(1 / PUPIL_CUTOFF_CYC_PX, 3)} px)`);
console.log(`aberration: PAPER_MODES x mulberry32(12345) in +-2.0 rad, masked RMS = ${fx(rms, 4)} rad`);
console.log(`Strehl  uncorrected ${fx(strehlAb * 100, 2)} %   after 88 % correction ${fx(strehlCo * 100, 2)} %`);
check('first dark ring within 1 % of the analytic Rayleigh limit',
  Math.abs(firstZero - RAYLEIGH_PX) / RAYLEIGH_PX < 0.01,
  `|${fx(firstZero, 3)} - ${fx(RAYLEIGH_PX, 3)}| / ${fx(RAYLEIGH_PX, 3)} = ${fx(Math.abs(firstZero - RAYLEIGH_PX) / RAYLEIGH_PX, 4)}`);

/* ------------------------------------------------------------------ *
 * 0b. the convolution itself -- prior verification, re-run on the new scenes
 * ------------------------------------------------------------------ */

hr('0b. convolvePsf RE-CONFIRMED ON ALL THREE SCENES');

console.log('An exact translation is a stronger statement than an unmoved centroid, and it is');
console.log('the one that holds on a torus: the centroid of a wrapped image is not the wrapped');
console.log('centroid, so the max per-pixel error of the SHIFTED comparison is what is reported.\n');
console.log('scene    size   delta identity      displaced-delta translation   flux ratio');
for (const sc of SCENES) {
  for (const S of SIZES) {
    const img = sc.make(S);
    // (a) a delta PSF at the origin (M/2, M/2) returns the input unchanged
    const delta = new Float64Array(N * N);
    delta[(N / 2) * N + (N / 2)] = 1;
    const same = convolvePsf(img, delta, S);
    let dmax = 0;
    for (let p = 0; p < S * S; p++) dmax = Math.max(dmax, Math.abs(same[p] - img[p]));
    // (b) a delta displaced by (dy, dx) translates the image by exactly (+dy, +dx)
    const dy = 7, dx = -5;
    const disp = new Float64Array(N * N);
    disp[(N / 2 + dy) * N + (N / 2 + dx)] = 1;
    const moved = convolvePsf(img, disp, S);
    let tmax = 0;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const v = moved[(((y + dy) % S) + S) % S * S + ((((x + dx) % S) + S) % S)];
        tmax = Math.max(tmax, Math.abs(v - img[y * S + x]));
      }
    }
    const fr = sum(same) / sum(img);
    console.log(`${sc.id.padEnd(8)} ${String(S).padEnd(6)} ${ex(dmax).padEnd(19)} ${ex(tmax).padEnd(29)} ${fr.toFixed(15)}`);
    check(`${sc.id}@${S} delta-PSF identity`, dmax < 1e-12, `max|out-in| = ${ex(dmax)}`);
    check(`${sc.id}@${S} displaced delta translates by exactly (+7, -5)`, tmax < 1e-12, `max err ${ex(tmax)}`);
    check(`${sc.id}@${S} mean brightness preserved`, Math.abs(fr - 1) < 1e-12, `ratio ${fr.toFixed(15)}`);
  }
}

/* ------------------------------------------------------------------ *
 * 1. shape, range, determinism
 * ------------------------------------------------------------------ */

hr('1. LENGTH, RANGE, DETERMINISM');

console.log('scene    size   length    min        max        mean       ==0     ==1');
for (const s of SCENES) {
  for (const S of SIZES) {
    const a = s.make(S);
    const st = stats(a);
    console.log(`${s.id.padEnd(8)} ${String(S).padEnd(6)} ${String(a.length).padEnd(9)} `
      + `${fx(st.min, 6).padEnd(10)} ${fx(st.max, 6).padEnd(10)} ${fx(st.mean, 6).padEnd(10)} `
      + `${String(st.atLo).padEnd(7)} ${st.atHi}`);
    check(`${s.id}@${S} length == size*size`, a.length === S * S);
    check(`${s.id}@${S} finite and inside [0,1]`, st.finite && st.min >= 0 && st.max <= 1);
    check(`${s.id}@${S} is a Float64Array`, a instanceof Float64Array);
  }
}

console.log('\ndeterminism -- same seed, byte-identical output');
for (const s of SCENES) {
  for (const S of SIZES) check(`${s.id}@${S} repeat call identical`, identical(s.make(S), s.make(S)));
}
// A second, freshly evaluated copy of the module: proves the answer is not an
// artefact of the module-level caches surviving inside one process.
const fresh = await import('../src/scene.js?fresh=1');
for (const S of SIZES) {
  check(`usaf@${S} identical across a fresh module instance`, identical(usafTarget(S), fresh.usafTarget(S)));
  check(`stars@${S} identical across a fresh module instance`, identical(starField(S), fresh.starField(S)));
  check(`terrain@${S} identical across a fresh module instance`, identical(terrain(S), fresh.terrain(S)));
}
check('stars: a different seed gives a different field',
  !identical(starField(256), starField(256, { seed: 999 })));
check('terrain: a different seed gives a different field',
  !identical(terrain(256), terrain(256, { seed: 999 })));
check('stars: opts.shared returns the cached instance',
  starField(256, { shared: true }) === starField(256, { shared: true }));
check('stars: default call returns a copy, not the cache',
  starField(256) !== starField(256, { shared: true }));
check('asinhStretch stays in [0,1] and is monotone', (() => {
  const a = asinhStretch(starField(256));
  const st = stats(a);
  if (!(st.min >= 0 && st.max <= 1)) return false;
  let prev = -1;
  for (let v = 0; v <= 1.0001; v += 0.01) {
    const y = asinhStretch(Float64Array.of(Math.min(v, 1)))[0];
    if (y < prev) return false;
    prev = y;
  }
  return Math.abs(asinhStretch(Float64Array.of(1))[0] - 1) < 1e-12
      && asinhStretch(Float64Array.of(0))[0] === 0;
})());

/* ------------------------------------------------------------------ *
 * 2. radially averaged power spectra
 * ------------------------------------------------------------------ */

hr('2. RADIALLY AVERAGED POWER SPECTRA (mean removed)');

const PROBE = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128];
for (const S of SIZES) {
  console.log(`\nsize ${S} -- P(k) normalised to P(1); k is |wavevector| in cycles per frame`);
  console.log('   k  period(px)  ' + SCENES.map((s) => s.id.padStart(11)).join(''));
  const spectra = SCENES.map((s) => radialPowerSpectrum(s.make(S), S));
  for (const k of PROBE) {
    if (k > S / 2) continue;
    const row = spectra.map((sp) => (sp.ps[k] / sp.ps[1]).toExponential(2).padStart(11)).join('');
    console.log(`${String(k).padStart(4)}  ${fx(S / k, 2).padStart(10)}  ${row}`);
  }
  const cut = PUPIL_CUTOFF_CYC_PX;
  console.log(`power above the pupil cutoff (${fx(cut, 4)} cyc/px): `
    + SCENES.map((s, i) => `${s.id} ${fx(100 * powerAbove(s.make(S), S, cut), 2)}%`).join('   '));
  for (let i = 0; i < SCENES.length; i++) {
    const sp = spectra[i];
    let nz = 0;
    for (let k = 1; k <= S / 2; k++) if (sp.ps[k] > sp.ps[1] * 1e-9) nz++;
    check(`${SCENES[i].id}@${S} carries power at every radial band down to Nyquist`, nz === S / 2,
      `${nz}/${S / 2} bands above 1e-9 of P(1)`);
  }
}

/* ------------------------------------------------------------------ *
 * 3. the close binary: resolved sharp, merged blurred, split again corrected
 * ------------------------------------------------------------------ */

hr('3. THE CLOSE BINARY -- MERGE AND SPLIT');

for (const S of SIZES) {
  const img = starField(S);
  const spec = starFieldSpec(S);
  const { a, b } = spec.binary;
  const cases = [
    ['scene (no optics)', img],
    ['diffraction-limited PSF', convolvePsf(img, psfSharp, S)],
    ['aberrated PSF (2.0 rad RMS ' + fx(rms, 2) + ')', convolvePsf(img, psfAb, S)],
    ['residual PSF after 88 % correction', convolvePsf(img, psfCorr, S)],
  ];
  console.log(`\nsize ${S}: pair at (${fx(a.x, 3)}, ${fx(a.y, 3)}) and (${fx(b.x, 3)}, ${fx(b.y, 3)}), `
    + `separation ${fx(spec.binary.sepPx, 5)} px = ${spec.binary.sepRayleigh} Rayleigh`);
  console.log('  case                                    peakA      peakB      trough     dip      resolved');
  const res = {};
  for (const [name, im] of cases) {
    const d = resolutionDip(im, S, a.x, a.y, b.x, b.y);
    res[name] = d;
    console.log(`  ${name.padEnd(38)} ${ex(d.peakA, 3)} ${ex(d.peakB, 3)} ${ex(d.trough, 3)} `
      + `${(d.resolved ? (100 * d.dip).toFixed(2) + '%' : '  --  ').padStart(7)}  ${d.resolved ? 'YES' : 'no'}`);
  }
  check(`@${S} pair resolved by the diffraction-limited PSF`, res['diffraction-limited PSF'].resolved,
    `dip ${(100 * res['diffraction-limited PSF'].dip).toFixed(2)} %`);
  // The separation was widened from 0.95 to 2.80 x Rayleigh so that EVERY pupil in the
  // demo ladder can resolve the pair when its wavefront is flat -- the smaller, more
  // asymmetric pupils have a diffraction-limited FWHM up to 2.1x the circle's, and at
  // 0.95 x Rayleigh they could never split it no matter how well the wavefront was
  // corrected. At 2.80 x the aberrated PSF no longer merges the pair outright; it
  // collapses the dip instead. Assert that collapse, which is what the panel shows:
  // aberration must destroy most of the contrast, and correction must bring it back.
  {
    const sharp = res['diffraction-limited PSF'].dip;
    const ab    = res['aberrated PSF (2.0 rad RMS ' + fx(rms, 2) + ')'].dip;
    const corr  = res['residual PSF after 88 % correction'].dip;
    check(`@${S} aberration collapses the pair's contrast`, ab < 0.5 * sharp,
      `dip ${(100 * ab).toFixed(2)}% vs ${(100 * sharp).toFixed(2)}% sharp (must be under half)`);
    check(`@${S} correction restores most of it`, corr > 0.8 * sharp,
      `dip ${(100 * corr).toFixed(2)}% vs ${(100 * sharp).toFixed(2)}% sharp (must exceed 80%)`);
  }
  check(`@${S} pair split again after correction`, res['residual PSF after 88 % correction'].resolved,
    `dip ${(100 * res['residual PSF after 88 % correction'].dip).toFixed(2)} %`);
  // scene.js ships its own cut for the panel to plot; it must agree with the
  // independent implementation above, which is what every number here was measured with.
  for (const [name, im] of cases) {
    const mine = resolutionDip(im, S, a.x, a.y, b.x, b.y);
    const theirs = binaryCut(im, S);
    check(`@${S} binaryCut() agrees with this file's own measurement -- ${name}`,
      theirs.resolved === mine.resolved && Math.abs(theirs.dip - mine.dip) < 1e-12,
      `${(100 * theirs.dip).toFixed(4)}% vs ${(100 * mine.dip).toFixed(4)}%`);
  }
}

// dip vs how much of the aberration is left: the merge/split threshold
console.log('\ndip vs residual aberration, size 256 (fraction of the 2.0 rad wavefront left uncorrected):');
console.log('  residual   RMS(rad)   Strehl     dip        resolved');
{
  const S = 256, img = starField(S), { a, b } = starFieldSpec(S).binary;
  for (const f of [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]) {
    const ph = new Float64Array(N * N);
    for (let p = 0; p < N * N; p++) ph[p] = f * gt[p];
    const d = resolutionDip(convolvePsf(img, psfFromPupil(circle, ph, N), S), S, a.x, a.y, b.x, b.y);
    const pred = new Float64Array(N * N);
    for (let p = 0; p < N * N; p++) pred[p] = (1 - f) * gt[p];
    console.log(`  ${fx(f, 2).padStart(8)}   ${fx(f * rms, 4).padStart(8)}   `
      + `${fx(100 * Math.min(1, strehlRatio(circle, gt, pred, N)), 2).padStart(6)} %   `
      + `${(d.resolved ? (100 * d.dip).toFixed(2) + '%' : '  --  ').padStart(7)}    ${d.resolved ? 'YES' : 'no'}`);
  }
}

/* ------------------------------------------------------------------ *
 * 4. band-limiting of the point sources
 * ------------------------------------------------------------------ */

hr('4. POINT SOURCES ARE BAND-LIMITED, NOT HOT PIXELS');

console.log('shift invariance:  || conv(source at x0+t) - FourierShift_t( conv(source at x0) ) ||');
console.log('                   -------------------------------------------------------------- , t in [0,1)');
console.log('                                        || conv(source at x0) ||\n');
console.log('  profile              worst t     mean t');
{
  const S = 256, x0 = 128, y0 = 100;
  const draw = (sig, cx) => {
    const im = new Float64Array(S * S);
    if (sig === 0) { im[y0 * S + Math.round(cx)] = 1; return im; }
    const rad = Math.ceil(5 * sig) + 1, inv = 1 / (2 * sig * sig);
    const bx = Math.round(cx - 0.5);
    for (let dy = -rad; dy <= rad; dy++) {
      for (let dx = -rad; dx <= rad; dx++) {
        const ex2 = bx + dx + 0.5 - cx, ey2 = y0 + dy + 0.5 - y0;
        im[((y0 + dy) % S) * S + (((bx + dx) % S) + S) % S] += Math.exp(-(ex2 * ex2 + ey2 * ey2) * inv);
      }
    }
    return im;
  };
  const rows = [];
  for (const [name, sig] of [['single hot pixel', 0], ['Gaussian sigma 0.50', 0.50],
    ['Gaussian sigma 0.65 (used)', STAR_SIGMA_PX], ['Gaussian sigma 0.80', 0.80]]) {
    const base = convolvePsf(draw(sig, x0), psfSharp, S);
    const nb = nrm(base);
    let worst = 0, acc = 0, n = 0;
    for (let t = 1 / 16; t < 1; t += 1 / 16) {
      const cur = convolvePsf(draw(sig, x0 + t), psfSharp, S);
      const e = l2(cur, fourierShift(base, S, t, 0)) / nb;
      if (e > worst) worst = e;
      acc += e; n++;
    }
    rows.push([name, worst, acc / n]);
    console.log(`  ${name.padEnd(28)} ${(100 * worst).toFixed(3).padStart(7)}%  ${(100 * acc / n).toFixed(3).padStart(8)}%`);
  }
  const hot = rows[0][1], used = rows[2][1];
  check('sigma 0.65 beats a hot pixel by more than 50x on shift invariance',
    hot / used > 50, `${(hot / used).toFixed(0)}x  (${(100 * hot).toFixed(1)}% -> ${(100 * used).toFixed(3)}%)`);
  check('sigma 0.65 shift-invariance error below 0.5 %', used < 0.005, `${(100 * used).toFixed(3)}%`);
}

// The price of band-limiting: a fatter source is extra blur and closes the notch.
console.log('\nthe trade: an ISOLATED copy of the pair (no sky pedestal, no neighbours),');
console.log('diffraction-limited PSF, dip as a function of the point-source sigma\n');
console.log('  sigma(px)   dip');
{
  const S = 256, sep = BINARY_SEP_PX;
  const ax = 128.37 - sep / 2, ay = 96.61, bx = 128.37 + sep / 2, by = 96.61;
  for (const sig of [0.50, 0.60, STAR_SIGMA_PX, 0.70, 0.80, 1.00]) {
    const im = new Float64Array(S * S);
    const rad = Math.ceil(5 * sig) + 1, inv = 1 / (2 * sig * sig);
    for (const [cx, cy] of [[ax, ay], [bx, by]]) {
      const bxx = Math.round(cx - 0.5), byy = Math.round(cy - 0.5);
      for (let dy = -rad; dy <= rad; dy++) {
        for (let dx = -rad; dx <= rad; dx++) {
          const ex2 = bxx + dx + 0.5 - cx, ey2 = byy + dy + 0.5 - cy;
          im[((byy + dy) % S) * S + (((bxx + dx) % S) + S) % S] += Math.exp(-(ex2 * ex2 + ey2 * ey2) * inv);
        }
      }
    }
    const d = resolutionDip(convolvePsf(im, psfSharp, S), S, ax, ay, bx, by);
    console.log(`  ${fx(sig, 2).padStart(8)}   ${(d.resolved ? (100 * d.dip).toFixed(2) + '%' : '  --  ').padStart(7)}`);
  }
}

/* ------------------------------------------------------------------ *
 * 5. photometry and the torus
 * ------------------------------------------------------------------ */

hr('5. PHOTOMETRY, SEAMLESSNESS, CROSS-SIZE CONSISTENCY');

for (const s of SCENES) {
  const S = 256, img = s.make(S);
  const out = convolvePsf(img, psfAb, S);
  const rel = Math.abs(sum(out) / sum(img) - 1);
  check(`${s.id}: convolvePsf preserves total flux`, rel < 1e-12, `rel err ${ex(rel)}`);
}

// A seamless scene has the same statistics across the wrap as it does inside.
for (const s of SCENES) {
  const S = 256, img = s.make(S);
  let wrap = 0, inner = 0;
  for (let y = 0; y < S; y++) {
    wrap += Math.abs(img[y * S] - img[y * S + S - 1]);
    for (let x = 1; x < S; x++) inner += Math.abs(img[y * S + x] - img[y * S + x - 1]);
  }
  wrap /= S; inner /= S * (S - 1);
  console.log(`${s.id.padEnd(8)} mean |d/dx| across the wrap ${ex(wrap)}   inside ${ex(inner)}   ratio ${fx(wrap / inner, 3)}`);
}
check('terrain wraps seamlessly (wrap gradient within 25 % of the interior gradient)', (() => {
  const S = 256, img = terrain(S);
  let wrap = 0, inner = 0;
  for (let y = 0; y < S; y++) {
    wrap += Math.abs(img[y * S] - img[y * S + S - 1]);
    for (let x = 1; x < S; x++) inner += Math.abs(img[y * S + x] - img[y * S + x - 1]);
  }
  return Math.abs((wrap / S) / (inner / (S * (S - 1))) - 1) < 0.25;
})());

{
  const s256 = starFieldSpec(256), s512 = starFieldSpec(512);
  check('stars: binary separation is the same absolute pixel count at 256 and 512',
    Math.abs(s256.binary.sepPx - s512.binary.sepPx) < 1e-12, `${fx(s256.binary.sepPx, 5)} px both`);
  check('stars: the composition scales with size (binary centre doubles)',
    Math.abs(s512.binary.cx - 2 * s256.binary.cx) < 1e-9);
  const t256 = terrainSpec(256), t512 = terrainSpec(512);
  check('terrain: octave periods in px double from 256 to 512 (same landscape, 2 px/unit)',
    Math.abs(t512.octaves[0].periodPx - 2 * t256.octaves[0].periodPx) < 1e-9,
    `${t256.octaves[0].periodPx} px -> ${t512.octaves[0].periodPx} px`);
  console.log(`terrain octave periods @256: ${[...new Set(t256.octaves.map((o) => o.periodPx))].join(', ')} px`);
  console.log(`crater radii @256: ${fx(t256.craterRadiusPx[0], 2)} .. ${fx(t256.craterRadiusPx[1], 2)} px, ${t256.craterCount} craters`);
  console.log(`star field @256: ${s256.stars.length} point sources, magnitudes ${s256.magRange[0]} .. ${s256.magRange[1]} `
    + `(peak ratio ${fx(Math.pow(10, 0.4 * (s256.magRange[1] - s256.magRange[0])), 0)}:1), sigma ${s256.sigmaPx} px`);
}

/* ------------------------------------------------------------------ *
 * 6. SCENES catalogue
 * ------------------------------------------------------------------ */

hr('6. SCENES CATALOGUE');

check('SCENES is a frozen array of 3', Object.isFrozen(SCENES) && SCENES.length === 3);
check('SCENES ids are usaf, stars, terrain in order',
  SCENES.map((s) => s.id).join(',') === 'usaf,stars,terrain');
for (const s of SCENES) {
  check(`SCENES.${s.id} has id/label/make`,
    typeof s.id === 'string' && typeof s.label === 'string' && typeof s.make === 'function');
  const a = s.make();
  check(`SCENES.${s.id}.make() defaults to 256`, a.length === 256 * 256);
  console.log(`  ${s.id.padEnd(8)} "${s.label}"  display=${s.display}\n           ${s.blurb}`);
}

/* ------------------------------------------------------------------ *
 * 7. timing -- single-call minima, on a fresh module each trial
 * ------------------------------------------------------------------ */

hr('7. TIMING (single-call minima; this box runs at load 140-240 on 128 cores)');

const TRIALS = 9;
console.log(`cold generation, min of ${TRIALS} single calls each into a freshly imported module`);
console.log('scene     size   min(ms)   median(ms)   max(ms)');
for (const id of ['usaf', 'stars', 'terrain']) {
  for (const S of SIZES) {
    const ts = [];
    for (let i = 0; i < TRIALS; i++) {
      const m = await import(`../src/scene.js?t=${id}-${S}-${i}`);
      const fn = m.SCENES.find((s) => s.id === id).make;
      const t0 = performance.now();
      fn(S);
      ts.push(performance.now() - t0);
    }
    ts.sort((x, y) => x - y);
    console.log(`${id.padEnd(9)} ${String(S).padEnd(6)} ${fx(ts[0], 2).padStart(7)} `
      + `${fx(ts[(TRIALS - 1) >> 1], 2).padStart(12)} ${fx(ts[TRIALS - 1], 2).padStart(10)}`);
  }
}
console.log(`\nwarm call (cache hit + Float64Array copy), min of ${TRIALS}`);
console.log('scene     size   min(ms)');
for (const s of SCENES) {
  for (const S of SIZES) {
    s.make(S);
    let mn = Infinity;
    for (let i = 0; i < TRIALS; i++) {
      const t0 = performance.now();
      s.make(S);
      const dt = performance.now() - t0;
      if (dt < mn) mn = dt;
    }
    console.log(`${s.id.padEnd(9)} ${String(S).padEnd(6)} ${fx(mn, 3).padStart(7)}`);
  }
}
console.log(`\none convolvePsf() call with the 128x128 PSF, min of ${TRIALS}`);
for (const S of SIZES) {
  const img = usafTarget(S);
  convolvePsf(img, psfAb, S);
  let mn = Infinity;
  for (let i = 0; i < TRIALS; i++) {
    const t0 = performance.now();
    convolvePsf(img, psfAb, S);
    const dt = performance.now() - t0;
    if (dt < mn) mn = dt;
  }
  console.log(`  size ${S}: ${fx(mn, 2)} ms`);
}

/* ------------------------------------------------------------------ */
hr(failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
