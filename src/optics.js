/**
 * optics.js -- dependency-free JavaScript port of the published forward optical model.
 *
 * Normative reference: assets/GOLDEN.md (schema "pupil-optics-golden/1").
 * Section numbers in the comments below refer to that document.
 *
 * Everything is float64 (Number / Float64Array). Arrays are row-major, flat index
 * p = y*N + x, y = row (0 = top), x = column (0 = left).
 *
 * No imports, no dependencies: usable in Node (ESM) and in the browser.
 */

/* ------------------------------------------------------------------ *
 * Constants
 * ------------------------------------------------------------------ */

/** Default bounding box (GOLDEN.md preamble). */
export const N_DEFAULT = 128;
/** Default pupil radius in pixels. */
export const RADIUS_DEFAULT = 19;
/** Reference-circle area in pixels for (N=128, radius=19) -- GOLDEN.md §2. */
export const REFERENCE_CIRCLE_AREA_PX = 1124;
/** Zernike-disk area in pixels for (N=128, radius=19) -- GOLDEN.md §2. */
export const ZERNIKE_DISK_AREA_PX = 1129;
/** Peak of the reference circle's PSF; the simulate_measurement divisor -- GOLDEN.md §5.1. */
export const CIRCLE_PSF_PEAK = 1263376; // === 1124*1124, exact integer

/**
 * The 33 PAPER_MODES as [n, m], in evaluation order -- GOLDEN.md §3.1.
 * coeffs[k] multiplies mode k. Piston (n=0) and tilt (n=1) are deliberately absent.
 */
export const PAPER_MODES = Object.freeze([
  [2, 0], [2, -2], [2, 2],
  [3, -1], [3, 1], [3, -3], [3, 3],
  [4, 0], [4, -2], [4, 2], [4, -4], [4, 4],
  [5, -1], [5, 1], [5, -3], [5, 3], [5, -5], [5, 5],
  [6, 0], [6, -2], [6, 2], [6, -4], [6, 4], [6, -6], [6, 6],
  [7, -1], [7, 1], [7, -3], [7, 3], [7, -5], [7, 5], [7, -7], [7, 7],
].map((m) => Object.freeze(m)));

/* ------------------------------------------------------------------ *
 * §2.1 Reference circle  (linspace(-1,1,N) grid, centre at (N-1)/2)
 * ------------------------------------------------------------------ */

const _circleCache = new Map();

/**
 * Reference circle on the linspace(-1,1,N) grid -- GOLDEN.md §2.1.
 *
 * coords[i] = -1 + 2i/(N-1);  circle = (coords[x]^2 + coords[y]^2 <= (radius/(N/2))^2).
 * Evaluated in exact integer arithmetic so the boundary carries no float risk:
 *   N^2 * ((2x-(N-1))^2 + (2y-(N-1))^2)  <=  (2*radius*(N-1))^2
 * For N=128, radius=19 this is 4096*(...) <= 2413^2 = 5822569 and yields exactly 1124 px.
 *
 * @returns {Uint8Array} length N*N, 1 = transmissive.
 */
export function referenceCircle(N = N_DEFAULT, radius = RADIUS_DEFAULT) {
  const key = N + ':' + radius;
  const hit = _circleCache.get(key);
  if (hit) return hit;
  const out = new Uint8Array(N * N);
  const exact = Number.isInteger(N) && Number.isInteger(radius);
  const lhsScale = N * N;
  const rhs = exact ? (2 * radius * (N - 1)) * (2 * radius * (N - 1)) : 0;
  const thr = radius / (N / 2);
  const thr2 = thr * thr;
  for (let y = 0; y < N; y++) {
    const dy = 2 * y - (N - 1);
    const cy = -1 + (2 * y) / (N - 1);
    for (let x = 0; x < N; x++) {
      const dx = 2 * x - (N - 1);
      let inside;
      if (exact) {
        inside = lhsScale * (dx * dx + dy * dy) <= rhs;
      } else {
        const cx = -1 + (2 * x) / (N - 1);
        inside = cx * cx + cy * cy <= thr2;
      }
      out[y * N + x] = inside ? 1 : 0;
    }
  }
  _circleCache.set(key, out); // cached; treat the returned array as read-only

  return out;
}

/** Area (pixel count) of the reference circle for this geometry. */
export function referenceCircleArea(N = N_DEFAULT, radius = RADIUS_DEFAULT) {
  const c = referenceCircle(N, radius);
  let s = 0;
  for (let i = 0; i < c.length; i++) s += c[i];
  return s;
}

/* ------------------------------------------------------------------ *
 * §2.2 Zernike disk  (integer pixel offsets / radius, centre on pixel N/2)
 * ------------------------------------------------------------------ */

const _diskCache = new Map();

/**
 * Zernike-disk polar grid -- GOLDEN.md §2.2.
 *
 *   y = i - N/2  (ROW offset), x = j - N/2 (COLUMN offset)
 *   r = sqrt(x^2 + y^2) / radius,  theta = atan2(y, x),  disk = (r <= 1)
 *
 * The membership test is the exact integer one: (j-N/2)^2 + (i-N/2)^2 <= radius^2,
 * which gives exactly 1129 px at N=128, radius=19.
 *
 * @returns {{r: Float64Array, theta: Float64Array, mask: Uint8Array, idx: Int32Array}}
 *          `idx` lists the flat indices inside the disk (convenience / speed).
 */
export function zernikeDisk(N = N_DEFAULT, radius = RADIUS_DEFAULT) {
  const key = N + ':' + radius;
  const hit = _diskCache.get(key);
  if (hit) return hit;
  const n2 = N * N;
  const c = N >> 1;
  const r = new Float64Array(n2);
  const theta = new Float64Array(n2);
  const mask = new Uint8Array(n2);
  const exact = Number.isInteger(radius);
  const r2max = radius * radius;
  const idxList = [];
  for (let i = 0; i < N; i++) {
    const dy = i - c;
    for (let j = 0; j < N; j++) {
      const dx = j - c;
      const p = i * N + j;
      const rr = Math.sqrt(dx * dx + dy * dy) / radius;
      r[p] = rr;
      theta[p] = Math.atan2(dy, dx); // y = ROW offset, x = COLUMN offset (§3.3)
      const inside = exact ? (dx * dx + dy * dy <= r2max) : (rr <= 1);
      mask[p] = inside ? 1 : 0;
      if (inside) idxList.push(p);
    }
  }
  const out = { r, theta, mask, idx: Int32Array.from(idxList) };
  _diskCache.set(key, out);
  return out;
}

/* ------------------------------------------------------------------ *
 * §3.2 / §3.3 Zernike polynomials -- UNNORMALISED (no Noll/OSA factor)
 * ------------------------------------------------------------------ */

// Exact integer factorials; n <= 7 in PAPER_MODES, table goes further for safety.
const FACT = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800];

const _radialCache = new Map();

/**
 * Integer coefficients of the UNNORMALISED radial polynomial R(n, |m|, r):
 *   R = sum_{k=0}^{(n-|m|)/2} (-1)^k (n-k)! / [k! ((n+|m|)/2-k)! ((n-|m|)/2-k)!] r^(n-2k)
 * There is NO sqrt(2(n+1)) factor -- GOLDEN.md §3.2.
 *
 * @returns {{pow: Int32Array, coef: Float64Array}} terms in ascending k (descending power).
 */
export function zernikeRadialTerms(n, mAbs) {
  const key = n * 100 + mAbs;
  const hit = _radialCache.get(key);
  if (hit) return hit;
  if ((n - mAbs) % 2 !== 0 || mAbs > n || n < 0 || mAbs < 0) {
    throw new Error(`invalid Zernike indices n=${n}, |m|=${mAbs}`);
  }
  const half = (n - mAbs) / 2;
  const pow = new Int32Array(half + 1);
  const coef = new Float64Array(half + 1);
  for (let k = 0; k <= half; k++) {
    // (n-k)! / (k! * ((n+|m|)/2 - k)! * ((n-|m|)/2 - k)!) is an exact integer.
    const c = FACT[n - k] / (FACT[k] * FACT[(n + mAbs) / 2 - k] * FACT[half - k]);
    pow[k] = n - 2 * k;
    coef[k] = (k & 1) ? -c : c;
  }
  const out = { pow, coef };
  _radialCache.set(key, out);
  return out;
}

/** Unnormalised Zernike radial polynomial R(n, |m|, r). GOLDEN.md §3.2. */
export function zernikeRadial(n, mAbs, r) {
  const { pow, coef } = zernikeRadialTerms(n, Math.abs(mAbs));
  let s = 0;
  for (let k = 0; k < pow.length; k++) s += coef[k] * Math.pow(r, pow[k]);
  return s;
}

/**
 * Zernike mode value -- GOLDEN.md §3.3.
 *   m >= 0 : R(n,  m, r) * cos( m*theta)
 *   m <  0 : R(n, -m, r) * sin(-m*theta)   (i.e. sin(|m|*theta))
 */
export function zernike(n, m, r, theta) {
  const a = m < 0 ? -m : m;
  const R = zernikeRadial(n, a, r);
  return m >= 0 ? R * Math.cos(m * theta) : R * Math.sin(a * theta);
}

/* ------------------------------------------------------------------ *
 * §3.4 Wavefront synthesis
 * ------------------------------------------------------------------ */

/**
 * Synthesise a wavefront from Zernike coefficients -- GOLDEN.md §3.4.
 *
 *   phase[y][x] = ( sum_k coeffs[k] * Z(n_k, m_k, r, theta) ) * disk[y][x]
 *
 * The disk multiplication is last. Evaluation is restricted to the disk, which is
 * bit-identical to evaluating over the full frame and then masking (0 * finite = 0);
 * the only difference is that the reference produces -0.0 at some outside pixels,
 * and -0.0 === 0.0 (§3.4).
 *
 * @param {ArrayLike<number>} coeffs  radians, one per mode, in PAPER_MODES order
 * @returns {Float64Array} phase in radians, length N*N
 */
export function synthPhase(coeffs, N = N_DEFAULT, radius = RADIUS_DEFAULT, modes = PAPER_MODES) {
  const K = modes.length;
  if (coeffs.length !== K) {
    throw new Error(`expected ${K} coefficients, got ${coeffs.length}`);
  }
  const { idx, basis } = zernikeBasis(N, radius, modes);
  const out = new Float64Array(N * N);
  for (let t = 0, nd = idx.length; t < nd; t++) {
    const base = t * K;
    let acc = 0;
    for (let k = 0; k < K; k++) {
      const c = coeffs[k];
      if (c !== 0) acc += c * basis[base + k];   // same order as the reference sum
    }
    out[idx[t]] = acc;
  }
  return out;
}

const _basisCache = new WeakMap();

/**
 * Per-pixel Zernike basis on the disk, built once and cached: for disk pixel t and
 * mode k, `basis[t*K + k] = Z(n_k, m_k, r_t, theta_t)`. Transposed (mode-contiguous)
 * so that `synthPhase` is a cache-friendly dot product with no transcendentals in the
 * hot path -- the demo re-synthesises on every slider move.
 *
 * The values are exactly what the elementwise §3.2/§3.3 formulas produce, so caching
 * changes nothing numerically.
 */
export function zernikeBasis(N = N_DEFAULT, radius = RADIUS_DEFAULT, modes = PAPER_MODES) {
  let perModes = _basisCache.get(modes);
  if (!perModes) { perModes = new Map(); _basisCache.set(modes, perModes); }
  const key = N + ':' + radius;
  const hit = perModes.get(key);
  if (hit) return hit;

  const { r, theta, idx } = zernikeDisk(N, radius);
  const K = modes.length;

  let maxPow = 0, maxM = 0;
  const terms = new Array(K);
  const mAbsArr = new Int32Array(K);
  const isSin = new Uint8Array(K);
  for (let k = 0; k < K; k++) {
    const n = modes[k][0], m = modes[k][1];
    const a = m < 0 ? -m : m;
    terms[k] = zernikeRadialTerms(n, a);
    mAbsArr[k] = a;
    isSin[k] = m < 0 ? 1 : 0;
    if (n > maxPow) maxPow = n;
    if (a > maxM) maxM = a;
  }
  const rp = new Float64Array(maxPow + 1);
  const cosm = new Float64Array(maxM + 1);
  const sinm = new Float64Array(maxM + 1);
  const basis = new Float64Array(idx.length * K);

  for (let t = 0; t < idx.length; t++) {
    const p = idx[t];
    const rr = r[p], th = theta[p];
    for (let q = 0; q <= maxPow; q++) rp[q] = Math.pow(rr, q);
    for (let q = 0; q <= maxM; q++) { cosm[q] = Math.cos(q * th); sinm[q] = Math.sin(q * th); }
    const base = t * K;
    for (let k = 0; k < K; k++) {
      const { pow, coef } = terms[k];
      let R = 0;
      for (let j = 0; j < pow.length; j++) R += coef[j] * rp[pow[j]];
      basis[base + k] = R * (isSin[k] ? sinm[mAbsArr[k]] : cosm[mAbsArr[k]]);
    }
  }
  const out = { idx, basis, K };
  perModes.set(key, out);
  return out;
}

/* ------------------------------------------------------------------ *
 * §4.1 FFT -- radix-2 Cooley-Tukey, UNNORMALISED forward (norm='backward')
 * ------------------------------------------------------------------ */

const _fftTables = new Map();

function fftTables(n) {
  let t = _fftTables.get(n);
  if (t) return t;
  if (n < 1 || (n & (n - 1)) !== 0) throw new Error(`FFT size ${n} is not a power of two`);
  let bits = 0;
  while ((1 << bits) < n) bits++;
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let v = i, r = 0;
    for (let b = 0; b < bits; b++) { r = (r << 1) | (v & 1); v >>= 1; }
    rev[i] = r;
  }
  const half = n >> 1 || 1;
  const cosT = new Float64Array(half);
  const sinT = new Float64Array(half);
  for (let k = 0; k < half; k++) {
    const a = (-2 * Math.PI * k) / n;
    cosT[k] = Math.cos(a);
    sinT[k] = Math.sin(a);
  }
  t = { n, bits, rev, cosT, sinT };
  _fftTables.set(n, t);
  return t;
}

/** In-place 1-D FFT over a strided slice. `inverse` conjugates the twiddles (no 1/n). */
function fft1d(re, im, off, stride, T, inverse) {
  const { n, rev, cosT, sinT } = T;
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      const a = off + i * stride, b = off + j * stride;
      let tmp = re[a]; re[a] = re[b]; re[b] = tmp;
      tmp = im[a]; im[a] = im[b]; im[b] = tmp;
    }
  }
  const sgn = inverse ? -1 : 1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const w = k * step;
        const wr = cosT[w], wi = sgn * sinT[w];
        const a = off + (i + k) * stride;
        const b = a + half * stride;
        const xr = re[b], xi = im[b];
        const tr = xr * wr - xi * wi;
        const ti = xr * wi + xi * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
      }
    }
  }
}

/**
 * In-place 2-D FFT, UNNORMALISED forward transform (numpy/torch norm='backward'):
 *   X[ky][kx] = sum_y sum_x f[y][x] exp(-2*pi*i (ky*y + kx*x) / N)
 * No 1/N or 1/sqrt(N) anywhere -- GOLDEN.md §4.1.
 *
 * @param {Float64Array} re  real part, length N*N, modified in place
 * @param {Float64Array} im  imaginary part, length N*N, modified in place
 * @param {boolean} [inverse=false]  when true, computes the 1/N^2-scaled inverse
 * @param {Uint8Array|null} [activeRows=null]  optimisation hint: rows with a 0 entry are
 *        known to be entirely zero, so their row transform is skipped. Bit-identical
 *        to passing null, because FFT(0) = 0; wrong hints silently give wrong answers.
 * @returns {{re: Float64Array, im: Float64Array}} the same arrays
 */
export function fft2(re, im, N = N_DEFAULT, inverse = false, activeRows = null) {
  const T = fftTables(N);
  for (let y = 0; y < N; y++) {                                          // rows
    if (activeRows && !activeRows[y]) continue;  // an all-zero row transforms to zeros
    fft1d(re, im, y * N, 1, T, inverse);
  }
  for (let x = 0; x < N; x++) fft1d(re, im, x, N, T, inverse);           // columns
  if (inverse) {
    const s = 1 / (N * N);
    for (let p = 0, n2 = N * N; p < n2; p++) { re[p] *= s; im[p] *= s; }
  }
  return { re, im };
}

/** Inverse 2-D FFT, norm='backward' (i.e. scaled by 1/N^2) -- GOLDEN.md §4.1. */
export function ifft2(re, im, N = N_DEFAULT) {
  return fft2(re, im, N, true);
}

/**
 * Circular roll by N/2 along both axes. For even N this is `ifftshift`; for
 * N = 128 it is also `fftshift`, because -64 === +64 (mod 128) -- GOLDEN.md §4.2.
 * Provided for completeness; the PSF path folds the shift into its index maths.
 */
export function shiftHalf(a, N = N_DEFAULT) {
  const h = N >> 1;
  const out = (a instanceof Float64Array) ? new Float64Array(N * N) : new Array(N * N);
  for (let y = 0; y < N; y++) {
    const sy = (y + h) % N;
    for (let x = 0; x < N; x++) out[y * N + x] = a[sy * N + ((x + h) % N)];
  }
  return out;
}

export const ifftshift = shiftHalf;
export const fftshift = shiftHalf;

/* ------------------------------------------------------------------ *
 * §4 PSF formation
 * ------------------------------------------------------------------ */

// Reusable scratch for the PSF path (avoids two 128 KB allocations per call).
const _psfScratch = new Map();
function psfScratch(n2, N) {
  let s = _psfScratch.get(n2);
  if (!s) { s = { re: new Float64Array(n2), im: new Float64Array(n2), rows: new Uint8Array(N) }; _psfScratch.set(n2, s); }
  return s;
}

/**
 * Core PSF evaluation -- GOLDEN.md §4.1.
 *
 *   complex_pupil = P * exp(i*phi)
 *   F   = fftshift( fft2( ifftshift( complex_pupil ) ) )
 *   psf = |F|^2
 *
 * At N = 128 both shifts are the same circular roll by 64 (§4.2); they are folded
 * into the gather/scatter indices, so no intermediate arrays are built.
 *
 * @param {ArrayLike<number>} mask   pupil amplitude P (0/1 for a binary pupil)
 * @param {ArrayLike<number>|null} phase  wavefront phi in radians (null = 0)
 * @param {Float64Array|null} out    optional destination (length N*N)
 * @returns {{psf: Float64Array|null, peak: number, argmax: number}}
 */
function psfCore(mask, phase, N, out) {
  const n2 = N * N;
  const h = N >> 1;
  const { re, im, rows } = psfScratch(n2, N);

  // ifftshift + build the complex pupil in one pass, recording the non-zero rows.
  re.fill(0); im.fill(0); rows.fill(0);
  for (let y = 0; y < N; y++) {
    const ty = (y + h) % N;
    const dy = ty * N;
    const sy = y * N;
    for (let x = 0; x < N; x++) {
      const a = mask[sy + x];
      if (a === 0) continue;
      rows[ty] = 1;
      const d = dy + ((x + h) % N);
      const ph = phase ? phase[sy + x] : 0;
      if (ph === 0) { re[d] = a; im[d] = 0; }
      else { re[d] = a * Math.cos(ph); im[d] = a * Math.sin(ph); }
    }
  }

  // A radius-19 pupil occupies ~39 of 128 rows, so skipping the empty row transforms
  // removes ~35% of the 2-D FFT's work. Exact: the transform of a zero row is zero.
  fft2(re, im, N, false, rows);

  // fftshift + |.|^2 in one pass.
  let peak = -Infinity, argmax = 0;
  for (let y = 0; y < N; y++) {
    const sy = ((y + h) % N) * N;
    const dy = y * N;
    for (let x = 0; x < N; x++) {
      const s = sy + ((x + h) % N);
      const v = re[s] * re[s] + im[s] * im[s];
      const d = dy + x;
      if (out) out[d] = v;
      if (v > peak) { peak = v; argmax = d; } // strict > : first maximum wins (§10.3)
    }
  }
  return { psf: out, peak, argmax };
}

/**
 * PSF from a pupil -- GOLDEN.md §4.
 *
 * @param {ArrayLike<number>} mask   pupil amplitude
 * @param {ArrayLike<number>|null} phase  wavefront in radians
 * @param {number} [N=128]
 * @param {boolean} [normalize=false]  legacy self-normalisation psf/max(peak,1e-4) (§4.6);
 *                                     NEVER used by the published pipeline.
 * @returns {Float64Array} PSF, length N*N
 */
export function psfFromPupil(mask, phase, N = N_DEFAULT, normalize = false) {
  const out = new Float64Array(N * N);
  const { peak } = psfCore(mask, phase, N, out);
  if (normalize) {
    const d = Math.max(peak, 1e-4); // §4.6 floor, legacy path only
    for (let p = 0, n2 = N * N; p < n2; p++) out[p] /= d;
  }
  return out;
}

/** Peak of |FFT{P e^{i phi}}|^2 without materialising the PSF (used by strehlRatio). */
export function psfPeak(mask, phase, N = N_DEFAULT) {
  return psfCore(mask, phase, N, null).peak;
}

/* ------------------------------------------------------------------ *
 * §5 simulate_measurement
 * ------------------------------------------------------------------ */

/**
 * The published measurement chain -- GOLDEN.md §5.
 *
 * normalizeByPeak = true (published):
 *   1. psf = |FFT{ P exp(i*gt) }|^2                      (§4, normalize = false)
 *   2. psf /= circlePeakDivisor                          (exact integer 1263376)
 *   3. if withNoise:  psf = max(psf + sigma*noise, 0)
 *   4. if renormalizeAfterNoise:  psf /= max(psf.max(), 1e-6)
 *
 * normalizeByPeak = false (legacy ablation, §5.3):
 *   psf = psfFromPupil(P, gt, normalize = true)          (self-peak, floor 1e-4)
 *   if useThroughput: psf *= sum(P) / (pi * radius^2)
 *   then the same steps 3 and 4.
 *
 * @param {ArrayLike<number>} mask  pupil mask P
 * @param {ArrayLike<number>|null} phase  the (already masked) wavefront gt, radians
 * @param {object} [opts]
 * @param {number}  [opts.N=128]
 * @param {number}  [opts.radius=19]
 * @param {boolean} [opts.withNoise=false]
 * @param {number}  [opts.noiseSigma=0]
 * @param {ArrayLike<number>|null} [opts.noise=null]  fixed N(0,1) field, required if withNoise
 * @param {boolean} [opts.normalizeByPeak=true]
 * @param {boolean} [opts.useThroughput=true]   legacy path only
 * @param {boolean} [opts.renormalizeAfterNoise=true]
 * @param {number}  [opts.circlePeakDivisor=1263376]
 * @param {boolean} [opts.returnDetail=false]  return {measurement, ...intermediates}
 * @returns {Float64Array|object}
 */
export function simulateMeasurement(mask, phase, opts = {}) {
  const {
    N = N_DEFAULT,
    radius = RADIUS_DEFAULT,
    withNoise = false,
    noiseSigma = 0,
    noise = null,
    normalizeByPeak = true,
    useThroughput = true,
    renormalizeAfterNoise = true,
    circlePeakDivisor = CIRCLE_PSF_PEAK,
    returnDetail = false,
  } = opts;

  const n2 = N * N;
  const psf = new Float64Array(n2);
  const core = psfCore(mask, phase, N, psf);
  const peakBefore = core.peak;
  const argmaxBefore = core.argmax;

  let peakAfterDivisor;
  if (normalizeByPeak) {
    // Step 2: fixed absolute scale set by the reference circle's PSF peak (§5.1).
    for (let p = 0; p < n2; p++) psf[p] /= circlePeakDivisor;
    peakAfterDivisor = peakBefore / circlePeakDivisor;
  } else {
    // Legacy path (§5.3): divide by own peak with the 1e-4 floor, then throughput.
    const d = Math.max(peakBefore, 1e-4);
    for (let p = 0; p < n2; p++) psf[p] /= d;
    peakAfterDivisor = peakBefore / d;
    if (useThroughput) {
      let sm = 0;
      for (let p = 0; p < n2; p++) sm += mask[p];
      const f = sm / (Math.PI * radius * radius); // NOTE: pi*r^2, not the 1124 px area
      for (let p = 0; p < n2; p++) psf[p] *= f;
      peakAfterDivisor *= f;
    }
  }

  if (withNoise) {
    if (!noise) throw new Error('simulateMeasurement: withNoise=true requires opts.noise');
    // Step 3: additive i.i.d. Gaussian, then a hard floor at exactly 0.0 (§5.2).
    for (let p = 0; p < n2; p++) {
      const v = psf[p] + noiseSigma * noise[p];
      psf[p] = v > 0 ? v : 0;
    }
  }

  let renormClampActive = false;
  if (renormalizeAfterNoise) {
    let mx = -Infinity;
    for (let p = 0; p < n2; p++) if (psf[p] > mx) mx = psf[p];
    const d = Math.max(mx, 1e-6); // §5.2 floor; never fires in the golden set
    renormClampActive = mx < 1e-6;
    for (let p = 0; p < n2; p++) psf[p] /= d;
  }

  if (!returnDetail) return psf;
  return {
    measurement: psf,
    psfPeakBeforeNormalization: peakBefore,
    psfArgmaxFlat: argmaxBefore,
    circlePeakDivisor: normalizeByPeak ? circlePeakDivisor : null,
    psfPeakAfterCircleDivisor: peakAfterDivisor,
    renormClampActive,
  };
}

/* ------------------------------------------------------------------ *
 * §6 Strehl ratio
 * ------------------------------------------------------------------ */

/**
 * Strehl ratio -- GOLDEN.md §6. Returns the RAW, unclamped value; callers that
 * aggregate should apply Math.min(1, .) as evaluate.py does.
 *
 *   residual  = (gt - pred) * mask
 *   peak_corr = max |FFT{ mask * exp(i*residual) }|^2
 *   peak_ref  = max |FFT{ mask }|^2                    (same mask, zero phase)
 *   strehl    = peak_corr / max(peak_ref, 1e-4)
 */
export function strehlRatio(mask, gtPhase, predPhase, N = N_DEFAULT) {
  const n2 = N * N;
  const residual = new Float64Array(n2);
  for (let p = 0; p < n2; p++) {
    const m = mask[p];
    if (m !== 0) residual[p] = (gtPhase[p] - predPhase[p]) * m;
  }
  const peakCorr = psfPeak(mask, residual, N);
  const peakRef = psfPeak(mask, null, N);
  return peakCorr / Math.max(peakRef, 1e-4); // eps floors the DENOMINATOR only
}

/* ------------------------------------------------------------------ *
 * §7 Masked piston-corrected MSE
 * ------------------------------------------------------------------ */

/**
 * Masked, piston-corrected MSE per sample -- GOLDEN.md §7. Units: rad^2.
 * Every division is by sum(mask), never by H*W.
 */
export function maskedPistonCorrectedMse(mask, pred, target) {
  const n = mask.length;
  let sumMask = 0, sp = 0, st = 0;
  for (let p = 0; p < n; p++) {
    const m = mask[p];
    if (m === 0) continue;
    sumMask += m;
    sp += pred[p] * m;
    st += target[p] * m;
  }
  const ms = Math.max(sumMask, 1e-9); // guards an empty mask only
  const mp = sp / ms, mt = st / ms;
  let acc = 0;
  for (let p = 0; p < n; p++) {
    const m = mask[p];
    if (m === 0) continue; // outside terms are exactly 0 after the re-mask
    const pc = (pred[p] * m - mp) * m;
    const tc = (target[p] * m - mt) * m;
    const d = pc - tc;
    acc += d * d;
  }
  return acc / ms;
}

/* ------------------------------------------------------------------ *
 * §8 Asymmetry
 * ------------------------------------------------------------------ */

/**
 * asymmetry(mask, align) -- GOLDEN.md §8.
 *
 *   mass    = max(sum x, 1e-12)
 *   xflip   = x flipped on both axes
 *   X       = fft2(x)            <- RAW mask, no shifts anywhere in this routine
 *   corr    = |ifft2(X * X)|     <- AUTOCONVOLUTION (X*X), not X*conj(X)
 *   (sy,sx) = argmax(corr) as sy = k // W, sx = k % W    (W = last-axis length)
 *   align === 'corrected' : sy += 1, sx += 1
 *   aligned = circular roll of xflip by (sy, sx)
 *   alpha   = 1 - sum(x * aligned) / mass
 *
 * @returns {number} alpha. Exactly 0 for the reference circle.
 */
export function asymmetry(mask, align = 'corrected', N = N_DEFAULT) {
  const n2 = N * N;
  let mass = 0;
  for (let p = 0; p < n2; p++) mass += mask[p];
  mass = Math.max(mass, 1e-12);

  const re = new Float64Array(n2);
  const im = new Float64Array(n2);
  const rows = new Uint8Array(N);
  for (let p = 0; p < n2; p++) { if (mask[p] !== 0) { re[p] = mask[p]; rows[(p / N) | 0] = 1; } }
  fft2(re, im, N, false, rows);                 // raw mask, no ifftshift
  for (let p = 0; p < n2; p++) {                // X * X, the autoconvolution
    const a = re[p], b = im[p];
    re[p] = a * a - b * b;
    im[p] = 2 * a * b;
  }
  ifft2(re, im, N);

  let best = -Infinity, k = 0;
  for (let p = 0; p < n2; p++) {
    const v = Math.hypot(re[p], im[p]);
    if (v > best) { best = v; k = p; }          // strict >: first maximum (§10.3)
  }

  const W = N;                                   // both axes divide by W, as written (§8)
  let sy = Math.floor(k / W);
  let sx = k % W;
  if (align === 'corrected') { sy += 1; sx += 1; }
  else if (align !== 'published') throw new Error(`unknown align "${align}"`);
  sy %= N; sx %= N;                              // sy/sx can reach N, which wraps to 0

  // aligned[y][x] = xflip[(y-sy) mod N][(x-sx) mod N], xflip[y][x] = mask[N-1-y][N-1-x]
  let overlap = 0;
  for (let y = 0; y < N; y++) {
    const fy = N - 1 - ((y - sy + N) % N);
    for (let x = 0; x < N; x++) {
      const m = mask[y * N + x];
      if (m === 0) continue;
      overlap += m * mask[fy * N + (N - 1 - ((x - sx + N) % N))];
    }
  }
  return 1 - overlap / mass;
}

/** Convenience wrappers. `asymmetryCorrected(circle)` is exactly 0 (§8). */
export function asymmetryCorrected(mask, N = N_DEFAULT) { return asymmetry(mask, 'corrected', N); }
export function asymmetryPublished(mask, N = N_DEFAULT) { return asymmetry(mask, 'published', N); }

/** The autoconvolution argmax as [y, x] -- useful for diagnosing tie-breaks (§8). */
export function autoconvArgmax(mask, N = N_DEFAULT) {
  const n2 = N * N;
  const re = new Float64Array(n2), im = new Float64Array(n2);
  const rows = new Uint8Array(N);
  for (let p = 0; p < n2; p++) { if (mask[p] !== 0) { re[p] = mask[p]; rows[(p / N) | 0] = 1; } }
  fft2(re, im, N, false, rows);
  for (let p = 0; p < n2; p++) { const a = re[p], b = im[p]; re[p] = a * a - b * b; im[p] = 2 * a * b; }
  ifft2(re, im, N);
  let best = -Infinity, k = 0;
  for (let p = 0; p < n2; p++) { const v = Math.hypot(re[p], im[p]); if (v > best) { best = v; k = p; } }
  return [Math.floor(k / N), k % N];
}

/* ------------------------------------------------------------------ *
 * Throughput
 * ------------------------------------------------------------------ */

/** throughput = sum(mask) / reference-circle area in pixels (1124) -- GOLDEN.md §8. */
export function throughput(mask, N = N_DEFAULT, radius = RADIUS_DEFAULT) {
  let s = 0;
  for (let p = 0, n = mask.length; p < n; p++) s += mask[p];
  return s / referenceCircleArea(N, radius);
}

/* ------------------------------------------------------------------ *
 * §9 Synthetic predictor + convenience helpers
 * ------------------------------------------------------------------ */

/**
 * The deterministic pseudo-prediction used for the golden metric targets -- GOLDEN.md §9.
 *   pred[y][x] = 0.9*gt[y][x] + 0.3 + 0.15*((x-64)*(y-64))/4096
 * No transcendentals, so any IEEE-754 double implementation reproduces it bit-for-bit.
 *
 * @param {ArrayLike<number>} gt  the MASKED ground truth (mask * phase)
 * @param {number} [piston=0.3]   exposed so the piston-invariance test can vary it
 */
export function syntheticPredictor(gt, N = N_DEFAULT, piston = 0.3) {
  const out = new Float64Array(N * N);
  const c = N >> 1;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const p = y * N + x;
      out[p] = 0.9 * gt[p] + piston + (0.15 * ((x - c) * (y - c))) / 4096;
    }
  }
  return out;
}

/** gt = mask * phase (the masked ground-truth wavefront). */
export function maskedPhase(mask, phase, N = N_DEFAULT) {
  const n2 = N * N;
  const out = new Float64Array(n2);
  for (let p = 0; p < n2; p++) out[p] = mask[p] * phase[p];
  return out;
}

/** sum(mask) as an integer count. */
export function sumMask(mask) {
  let s = 0;
  for (let p = 0, n = mask.length; p < n; p++) s += mask[p];
  return s;
}

/**
 * One full published pipeline: coefficients -> phase -> masked gt -> measurement.
 * This is the demo's hot path.
 *
 * @returns {{phase: Float64Array, gt: Float64Array, measurement: Float64Array,
 *            psfPeak: number, throughput: number}}
 */
export function runPipeline(mask, coeffs, opts = {}) {
  const N = opts.N ?? N_DEFAULT;
  const radius = opts.radius ?? RADIUS_DEFAULT;
  const phase = synthPhase(coeffs, N, radius, opts.modes ?? PAPER_MODES);
  const gt = maskedPhase(mask, phase, N);
  const detail = simulateMeasurement(mask, gt, { ...opts, N, radius, returnDetail: true });
  return {
    phase,
    gt,
    measurement: detail.measurement,
    psfPeak: detail.psfPeakBeforeNormalization,
    throughput: throughput(mask, N, radius),
  };
}
