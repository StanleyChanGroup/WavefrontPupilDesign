# GOLDEN.md — normative conventions for the JavaScript port

**Status: normative.** `golden.json` points here. If this document and any Python
source disagree, the disagreement is a bug — but implement against *this* document.
You do not need to read `pupil_design/*.py` to write a correct port.

Scope: the forward optical model behind the published U-Net — pupil geometry, Zernike
wavefront synthesis, PSF formation, sensor-measurement simulation, and the two scoring
metrics (Strehl, masked piston-corrected MSE), plus the asymmetry metric.

Everything below is fixed at **N = 128** (`bounding_box_size`), **radius = 19**.
Arrays are **row-major**, indexed `[y][x]` with `y` = row (0 = top), `x` = column
(0 = left), flat index `p = y*128 + x`.

Published configuration for the shipped weights (`config` in `golden.json`):

```
noise_sigma = 0.0      with_noise = false       normalize_by_peak = true
use_throughput = true  renormalize_after_noise = true
coeff_scale = 2.0      modes = PAPER_MODES (33)  binary_pupils = false
```

---

## 1. The golden artifacts

Two files, same directory, generated together from the published package.

### 1.1 `golden.json` (1,710,645 B)

Self-sufficient: it carries every input (mask bits, phase coefficients, phase float32
bits, the noise field) and enough of every output (sampled values + whole-array
statistics) to validate a port **without** `golden.npz`. Top-level keys:

| key | contents |
|---|---|
| `schema` | `"pupil-optics-golden/1"` |
| `generated_by` | `pupil_design 0.1.0, torch 2.8.0+cu126, numpy 2.3.3` |
| `geometry` | the constants of §2 |
| `config` | the published `ExperimentConfig` |
| `paper_modes` | 33 `[n, m]` pairs, **in evaluation order** |
| `sampling` | the `grid16` / `core16` index sets (§1.3) |
| `encodings` | the base64 layouts (§1.4) |
| `synthetic_predictor` | the deterministic pseudo-prediction (§9) |
| `reference_circle_psf` | peak `1263376`, argmax flat `8256` = `(64,64)` |
| `phase_generation` | which RNG produced the coefficients — **do not reproduce it** |
| `masks` | 6 records |
| `phases` | 4 records |
| `cases` | 24 records = 6 masks × 4 phases |
| `noise_probe` | 2 noisy-measurement records + the fixed noise field |
| `zernike_spot_checks` | `R` and `Z` at 5 probe points for all 33 modes |

**The 6 masks**, in array order (`masks[i]` ↔ `golden.npz:masks_uint8[i]`):

| i | `id` | kind | `sum_mask` | throughput | `alpha_corrected.f64` | `alpha_published.f64` | DL PSF peak |
|---|------|------|-----------:|-----------:|----------------------:|----------------------:|------------:|
| 0 | `circle`    | analytic  | 1124 | 1.000000 | **0.0** | 0.047153024911032015 | 1263376 |
| 1 | `triangle`  | analytic  |  493 | 0.438612 | 0.3326572008113591  | 0.3427991886409736   |  243049 |
| 2 | `pool_101`  | pool test |  973 | 0.865658 | 0.014388489208633115| 0.054470709146968166 |  946729 |
| 3 | `pool_620`  | pool test |  383 | 0.340747 | 0.12793733681462138 | 0.164490861618799    |  146689 |
| 4 | `pool_1088` | pool test |  428 | 0.380783 | 0.24065420560747663 | 0.26635514018691586  |  183184 |
| 5 | `pool_1473` | pool test |  356 | 0.316726 | 0.3539325842696629  | 0.3567415730337079   |  126736 |

`throughput = sum_mask / 1124` (the **reference-circle area in pixels**, not πr²).
`DL PSF peak` = diffraction-limited peak = `sum_mask²` **exactly** (see §4.4).
The `pool_*` masks come from the published test split of
`dataset_jan2026_bbox_128_radius_19/pupil_pool_cache_threeway.pth` at the stated
`pool_index`; they are reproduced verbatim in `bits_b64`, so you never need the pool.

**The 4 phases**: ids `0,1` drawn at `coeff_scale = 1.0`, ids `2,3` at `2.0`. Each
record carries `coeffs_f64` (33 doubles, the only input you need), the float32 bits,
sha256 of both float32-LE and float64-LE byte images, `stats_f64`, and `samples_f64`.
All four have `disk_px = 1129`.

**The 24 cases**: `case_id = "{mask_id}__phase{phase_id}"`, ordered mask-major
(all four phases of `circle`, then `triangle`, …). Each case carries

- `intermediate` — `sum_mask`, `psf_peak_before_normalization`, `psf_argmax_flat`,
  `circle_peak_divisor` (`exact_int: 1263376`), `psf_peak_after_circle_divisor`,
  and `renorm_clamp_active` (**false in all 24 cases** — the 1e-6 floor never fires here);
- `psf` and `measurement` — each with `stats_f64`, `samples_f64` (`grid16`+`core16`),
  and an `f32` block (`peak`, `core16`, and the float32-vs-float64 deviation);
- `metrics_synthetic_pred` — Strehl and MSE against the §9 predictor, in both
  `torch_f32` and `f64`. **These are your metric targets.**
- `metrics_unet_pred` — the same metrics for the real network's output. Informational;
  reproducing them requires running `unet_fp32.onnx`, and they are float32-only.

`stats_f64` is always `{min, max, argmax_flat, argmax_yx, sum, sum_sq}` over the full
128×128 array in float64. `argmax` is the **first** flat-index maximum (numpy
tie-break, and the convention throughout these files); `sum_sq = Σ v²`.

### 1.2 `golden.npz` (11,776,592 B)

A standard `.npz` (ZIP of `.npy` members) for whole-array comparison. 17 arrays:

| name | shape | dtype |
|---|---|---|
| `case_ids` | (24,) | `<U17` |
| `mask_ids` | (6,) | `<U9` |
| `masks_uint8` | (6,128,128) | uint8 |
| `phases_f64` / `phases_f32` | (4,128,128) | float64 / float32 |
| `phase_coeffs_f64` | (4,33) | float64 |
| `psf_f64` / `psf_f32` | (24,128,128) | float64 / float32 |
| `measurement_f64` / `measurement_f32` | (24,128,128) | float64 / float32 |
| `pred_synth_f64` | (24,128,128) | float64 |
| `pred_unet_f32` | (24,128,128) | float32 |
| `noise_f32` | (128,128) | float32 |
| `measurement_noisy_f64` / `measurement_noisy_f32` | (2,128,128) | float64 / float32 |
| `circle_psf_f64` / `circle_psf_f32` | (128,128) | float64 / float32 |

Row order of every 24-long axis matches `case_ids`, which matches `cases` in the JSON.
The 2-long noisy axis is `[circle, triangle]`, both with `phase_id = 2`, `sigma = 0.05`.

### 1.3 Sampling index sets

Two 16×16 samplings are stored for every array, both as **flat 256-element arrays in
row-major order over `rows × cols`**: element `k = i*16 + j` is `array[rows[i]][cols[j]]`.

- **`grid16`** — `rows = cols = [0, 8, 16, …, 120]` (i.e. `range(0,128,8)`).
  A coarse whole-field sample. For phases this is mostly `0.0` (and sometimes `-0.0`)
  because the grid barely touches the radius-19 disk.
- **`core16`** — `rows = cols = [56, 57, …, 71]` (i.e. `range(56,72)`).
  The 16×16 block centred on the PSF peak at `(64,64)`. This is the discriminating
  sample: it covers the core and first rings where all the numerical content is.

### 1.4 Byte encodings

**`mask bits_b64`** — base64 of exactly **2048 bytes**. Pixel `p = y*128 + x` lives in
byte `p >> 3` at bit `p & 7`, **LSB-first** (bit 0 has value 1). A set bit means
transmissive (`mask == 1`). Identical convention to `ladder.bin`.

```js
function decodeMaskBits(b64) {                 // -> Float64Array(16384) of 0.0/1.0
  const bin = atob(b64), m = new Float64Array(16384);
  for (let p = 0; p < 16384; p++)
    m[p] = (bin.charCodeAt(p >> 3) >> (p & 7)) & 1;
  return m;
}
```

**`phase float32_b64`** — base64 of exactly **65536 bytes** = 128·128 IEEE-754
binary32, **little-endian**, row-major (`index = y*128 + x`).

```js
function decodeF32(b64) {                      // -> Float32Array(16384)
  const bin = atob(b64), buf = new Uint8Array(65536);
  for (let i = 0; i < 65536; i++) buf[i] = bin.charCodeAt(i);
  return new Float32Array(buf.buffer);         // browsers/Node are little-endian
}
```

**`noise_probe.noise_float32_b64`** — same layout as `phase float32_b64`.
Source: `numpy.random.default_rng(777).standard_normal((128,128)).astype(float32)`.
sha256 of the float32-LE bytes: `ba05dd50c1637d86d8f4baa81b87c36f82b47b70702c4b3847253f347175df42`.
Do **not** try to regenerate it — decode the bytes.

The same rule applies to `phase_generation.rng`: it records which numpy PCG64 stream
produced the coefficients so the provenance is auditable. **A JS port must not
reproduce the RNG.** Take `coeffs_f64` as input.

---

## 2. Geometry: THE TWO GRID CONVENTIONS

> **This is the top trap of the whole port.** Two different discretisations of "the
> radius-19 disk" coexist in the pipeline, they differ by 53 pixels, and they are
> **not interchangeable.** Using one where the other belongs silently changes the
> throughput scale, the Strehl denominator, and the wavefront support.

| | **reference circle** (pupil / amplitude) | **Zernike disk** (wavefront support) |
|---|---|---|
| used by | `reference_circle()` → the throughput and PSF-normalisation reference | `random_zernike_phase()` → where the phase is non-zero |
| coordinates | `linspace(-1, 1, 128)`, normalised | integer pixel offsets, divided by radius |
| centre | **(63.5, 63.5)** — between pixels | **(64, 64)** — on a pixel |
| effective pixel radius | **18.8515625** | **19** |
| area | **1124 px** | **1129 px** |

Shared: **1100 px**. Circle-only: **24 px**. Disk-only: **29 px**.

### 2.1 Reference circle — exact construction

```
coords[i] = -1 + 2*i/127                      for i = 0..127   (linspace(-1,1,128))
dist[y][x] = sqrt(coords[x]^2 + coords[y]^2)
circle[y][x] = (dist[y][x] <= 19/64) ? 1 : 0             // 19/64 = 0.296875 exactly
```

Because `coords[i] = (i - 63.5) * (2/127)`, this is exactly

```
circle[y][x] = ( (x-63.5)^2 + (y-63.5)^2 <= 18.8515625^2 ) ? 1 : 0
```

and `18.8515625 = 0.296875 * 127/2 = 2413/128`.

**Integer-exact form (recommended for JS — no floating-point boundary risk):**

```js
// (2x-127)^2 + (2y-127)^2 <= (2413/64)^2   <=>   4096*((2x-127)^2 + (2y-127)^2) <= 2413^2
const inCircle = (y, x) =>
  4096 * ((2*x - 127)**2 + (2*y - 127)**2) <= 5822569;   // 2413^2
```

All operands are exact integers below 2^53. The boundary is comfortable: the nearest
pixel sits 0.0234 px from it (the admissible values of `(2x-127)²+(2y-127)²` jump
1418 → 1450 across the threshold 1421.525634765625), so the naive float form is also
safe — but the integer form removes the question. Either yields **exactly 1124** pixels.

The half-integer centre is why the circle is **exactly 180°-rotation symmetric** about
the array, which is why it scores exactly 0 under `asymmetry(align="corrected")` (§8).

### 2.2 Zernike disk — exact construction

```
y[i][j] = i - 64        // ROW offset,    integer, range -64..63
x[i][j] = j - 64        // COLUMN offset, integer, range -64..63
r[i][j]     = sqrt(x^2 + y^2) / 19
theta[i][j] = atan2(y, x)
disk[i][j]  = (r[i][j] <= 1)
```

**Integer-exact form:** `disk[i][j] = ((j-64)² + (i-64)² <= 361)`. Boundary is clean:
the largest included `x²+y²` is exactly 361, the smallest excluded is 362. This yields
**exactly 1129** pixels. (`sqrt(361)/19` is exactly `1.0` in IEEE-754 double, so the
float form agrees, but use the integer form.)

### 2.3 Which one goes where

| quantity | grid |
|---|---|
| pupil amplitude `P` in `psf_from_pupil` | whichever mask you are given — for the reference it is the **circle** (§2.1) |
| the throughput / normalisation reference and its PSF peak `1263376` | **circle** |
| `throughput = sum_mask / 1124` | **circle** area in the denominator |
| support of the synthesised wavefront `φ` | **Zernike disk** (§2.2) |
| `mask` in Strehl and in the MSE | the **pupil mask** of the case, never the Zernike disk |

Note the consequence: for the `circle` mask, `gt = mask * phase` is non-zero on the
1100 shared pixels only. The 29 disk-only pixels carry phase but no light; the 24
circle-only pixels carry light but zero phase. That is correct and intended — do not
"fix" it.

---

## 3. Zernike synthesis

### 3.1 The 33 PAPER_MODES, in order

Index `k` (0-based) → `(n, m)`. **Order matters**: `coeffs_f64[k]` multiplies mode `k`.

```
 0:(2, 0)   1:(2,-2)   2:(2, 2)
 3:(3,-1)   4:(3, 1)   5:(3,-3)   6:(3, 3)
 7:(4, 0)   8:(4,-2)   9:(4, 2)  10:(4,-4)  11:(4, 4)
12:(5,-1)  13:(5, 1)  14:(5,-3)  15:(5, 3)  16:(5,-5)  17:(5, 5)
18:(6, 0)  19:(6,-2)  20:(6, 2)  21:(6,-4)  22:(6, 4)  23:(6,-6)  24:(6, 6)
25:(7,-1)  26:(7, 1)  27:(7,-3)  28:(7, 3)  29:(7,-5)  30:(7, 5)  31:(7,-7)  32:(7, 7)
```

All have `n - |m|` even. Tilt (`n=1`) and piston (`n=0`) are deliberately absent.

### 3.2 Radial polynomial — UNNORMALISED

```
R(n, m, r) = Σ_{k=0}^{(n-|m|)/2}  (-1)^k · (n-k)! /
                                   [ k! · ((n+|m|)/2 - k)! · ((n-|m|)/2 - k)! ] · r^(n-2k)
```

**There is no `sqrt(2(n+1))` (or `sqrt(n+1)` for m=0) Noll/OSA normalisation factor.**
The polynomials are the raw ones, so `R(n,|m|,1) = 1` for every mode and `R(2,0,0) = -1`
(since `R_2^0 = 2r² - 1`). Adding a normalisation factor is a silent, plausible-looking
bug that changes every coefficient's meaning. Check against
`zernike_spot_checks[k].R_at_r`, e.g. mode 0 `(2,0)` gives
`[-1.0, -0.9422, -0.5, 0.34906738000000015, 1.0]` at `r = [0, 0.17, 0.5, 0.8213, 1.0]`.

Factorials are small (`n ≤ 7`, so `7! = 5040`); compute them exactly in integers and
divide once, in double.

### 3.3 Angular part — the sign convention

```
Z(n, m, r, theta) =  R(n,  m, r) · cos( m · theta)      if m >= 0
Z(n, m, r, theta) =  R(n, -m, r) · sin(-m · theta)      if m <  0
```

Both branches call the radial polynomial with `|m|`. For `m < 0`, `-m = |m|`, so the
angular factor is `sin(|m|·theta)` — **positive** `|m|` inside the sine. `m = 0` gives
`cos(0) = 1`, so `Z = R`.

`theta = atan2(y, x)` with **`y` the ROW offset** and `x` the column offset (§2.2).
Row indices increase downward, so this is a *y-down* frame: `theta` increases clockwise
on screen. `theta = 0` points at `(row 64, col 83)`; `theta = +π/2` points at
`(row 83, col 64)`. Getting the argument order backwards (`atan2(x, y)`) or flipping the
row sign mirrors every odd-`m` mode and is not caught by any symmetric test case.

### 3.4 Assembling a wavefront

```
phase[y][x] = ( Σ_{k=0}^{32} coeffs[k] · Z(n_k, m_k, r[y][x], theta[y][x]) ) · disk[y][x]
```

- Coefficients are **in radians**, drawn `~ U(-coeff_scale, +coeff_scale)`. The
  published training set uses `coeff_scale = 2.0`. Golden phases 0 and 1 use 1.0,
  phases 2 and 3 use 2.0 (see each `phases[i].coeff_scale`).
- **The disk mask is applied LAST**, as a multiplication, after the whole sum.
  The reference implementation evaluates the polynomials over the full 128×128 frame
  (where `r` reaches ≈ 4.73 at the corners and `r^7` reaches ≈ 5.3e4) and only then
  multiplies by the boolean disk. You may restrict evaluation to the disk instead —
  the results are bit-identical, since `0 × finite = 0` — with one cosmetic
  difference: the reference produces `-0.0` at outside pixels where the sum was
  negative, and `samples_f64.grid16` reflects that. `-0.0 === 0.0` in JS and the sign
  of zero never propagates anywhere (`exp(i·(-0)) = 1`), so ignore it — but if you
  compare with `Object.is` you will see spurious failures.
- Accumulate in **float64**. Then, and only if you need the network input, cast to
  float32; the golden `float32_b64` / `phases_f32` are that cast. Observed
  float32-vs-float64 deviation: ≤ 2.4e-7 absolute, ≤ 4.5e-8 relative to peak.
- The result is `phase`, in radians. `gt = mask * phase` (§9) applies the **pupil**
  mask on top; the two maskings are different (§2.3).

---

## 4. `psf_from_pupil` — PSF formation

### 4.1 The pipeline

```
complex_pupil[y][x] = P[y][x] · exp(i · phi[y][x])     // P real amplitude, phi radians
F                   = fftshift( fft2( ifftshift( complex_pupil ) ) )
psf                 = |F|^2                            // = re^2 + im^2
```

`fft2` is **UNNORMALISED** — numpy/torch `norm='backward'`:

```
X[ky][kx] = Σ_{y=0}^{127} Σ_{x=0}^{127} f[y][x] · exp( -2πi (ky·y + kx·x) / 128 )
```

with **no** `1/N` or `1/sqrt(N)` anywhere. If your JS FFT library normalises the
forward transform, multiply the result by `128*128 = 16384` (or the amplitude by 128
per axis) — otherwise every PSF peak is off by `16384²` and the `1263376` divisor
check fails immediately.

The inverse transform, where it is used (§8 only), is `norm='backward'` too:
`f = (1/16384) Σ X · exp(+2πi(…)/128)`.

### 4.2 The shifts — for N = 128 they are the same operation

`roll(a, s)[i] = a[(i - s) mod N]`. `ifftshift` rolls by `-(N//2) = -64`;
`fftshift` rolls by `+(N//2) = +64`. Modulo 128, **`-64 ≡ +64`**, so for this even,
power-of-two size **both `ifftshift` and `fftshift` are the identical circular roll
by 64 along each axis** — a plain quadrant swap:

```js
function shift64(a) {                 // works as BOTH ifftshift and fftshift at N=128
  const out = new Float64Array(16384);
  for (let y = 0; y < 128; y++)
    for (let x = 0; x < 128; x++)
      out[y*128 + x] = a[(((y + 64) & 127) * 128) + ((x + 64) & 127)];
  return out;
}
```

(Do not carry this simplification to other sizes; at odd or non-power-of-two `N` the
two shifts differ.)

Applied to **both** spatial axes (`dim=(-2,-1)`). The order is
`ifftshift → fft2 → fftshift`; the pre-shift moves the pupil's centre to index 0
(so the transform sees no linear phase ramp), the post-shift moves DC from index
`(0,0)` to the array centre.

### 4.3 Where the peak lands

With `phi = 0`, the DC bin is `Σ P`, and after `fftshift` it sits at

```
argmax_yx = (64, 64)      argmax_flat = 64*128 + 64 = 8256
```

This is the aberration-free focus. With a non-zero wavefront the peak generally moves —
see the per-case `psf.stats_f64.argmax_yx`, e.g. `circle__phase2` peaks at `(69, 66)`.

### 4.4 Exactness at the peak

For a binary mask `M` and `phi = 0`, the DC bin is exactly `Σ M` (an integer), so

```
diffraction-limited PSF peak = (Σ M)^2       exactly
```

For the reference circle: `1124² = 1263376`. Every intermediate in a radix-2 FFT of a
0/1 array along the DC path is an integer `≤ 16384`, so this comes out **bit-exact in
float32 as well as float64** (`1263376 < 2^24`). Your port must produce exactly
`1263376.0` — not `1263375.9999997`. If it does not, your FFT is normalised wrongly or
your circle mask is not the §2.1 one. All six masks are checked this way in
`masks[i].diffraction_limited_psf_peak.sum_mask_squared`.

### 4.5 Parseval check (free self-test)

For any binary mask `M` and any phase,

```
Σ_{y,x} psf[y][x] = N² · Σ_{y,x} M[y][x] = 16384 · sum_mask
```

exactly (to rounding). All 24 golden cases satisfy it: `circle` → 18415616,
`triangle` → 8077312. A wrong FFT normalisation fails this instantly.

### 4.6 The `normalize` flag and its 1e-4 clamp

`psf_from_pupil` takes an optional `normalize` flag which, when set, divides the PSF by
its own peak with a floor:

```
psf = psf / max(peak, 1e-4)
```

**In the published configuration this branch is never taken.** `normalize_by_peak=True`
calls `psf_from_pupil(..., normalize=False)`, and `strehl_ratio` also calls it with the
default `False`. The `1e-4` floor therefore applies **only** in the
`normalize_by_peak=False` legacy path (§5.3). Do not confuse it with the `eps=1e-4` in
`strehl_ratio` (§6), which is a different clamp on a different quantity.

---

## 5. `simulate_measurement` — what the sensor records

Inputs: a pupil mask `P`, an **already-masked** phase `gt = P · phase`, and the
reference circle's PSF (only its peak is used). Output: the intensity image the network
consumes as channel 0.

### 5.1 The published path (`normalize_by_peak = true`)

```
1.  psf = |FFT{ P · exp(i·gt) }|^2                        // §4, normalize=False
2.  psf = psf / 1263376                                   // exact integer divisor
3.  if (with_noise):                                      // published weights: false
        psf = max( psf + sigma * noise , 0.0 )            // element-wise floor at 0
4.  if (renormalize_after_noise):                         // published: true
        psf = psf / max( psf.max() , 1e-6 )
```

Step 2's divisor is **the peak of the reference *circle*'s* PSF**, the same number for
every pupil: the exact integer `1263376 = 1124²`. It is *not* the current pupil's peak
and *not* `π·19²`. Fixing one absolute intensity scale across pupils is the whole point:
a small or spindly pupil passes less light, so its normalised PSF is dimmer and a single
`noise_sigma` means the same photon count everywhere. That is where the throughput
penalty of asymmetry enters the physics; there is no separate throughput factor in this
branch.

`psf_peak_after_circle_divisor` in each case's `intermediate` is the value after step 2
(e.g. `circle__phase0` → 0.5530811104083088, `pool_1088__phase3` → 0.013084256729…).

### 5.2 The clamps — three different floors, three different places

| floor | where | when it fires |
|---|---|---|
| `1e-4` | inside `psf_from_pupil`, on the self-peak divisor | **only** when `normalize=True`, i.e. the `normalize_by_peak=False` path (§5.3). Never in the published pipeline. |
| `0.0` | step 3, `clamp_min(0.0)` on the post-noise intensity | only when noise is on. Negative Gaussian excursions are truncated to exactly `0.0`. In the noise probe this fires on 8036/16384 (circle) and 8099/16384 (triangle) pixels. |
| `1e-6` | step 4, on the renormalisation divisor `psf.max()` | never in any golden case (`renorm_clamp_active` is `false` in all 24). Implement it anyway. |

Noise is **additive i.i.d. Gaussian**, `psf + sigma * noise` with `noise ~ N(0,1)` drawn
independently per pixel and scaled by `sigma`. It is **not** Poisson / shot noise, and
it is **not** signal-dependent. It is added *after* the circle-peak division, so its
magnitude is fixed on the absolute scale set by the reference circle.

### 5.3 The legacy path (`normalize_by_peak = false`) — not used by the shipped weights

```
psf = psf_from_pupil(P, gt, normalize=True)      // divide by own peak, floor 1e-4
if (use_throughput):
    psf = psf * ( sum(P) / (pi * radius^2) )     // NOTE: pi*19^2 = 1134.1149479459152
```
then the same noise and renormalisation steps. Note the divisor here is `π·r²`, a
different number from the 1124-pixel circle area used for the reported `throughput`
field. Implement this branch only if you need the ablation; the published model does not.

### 5.4 The noiseless degeneracy — and why the noise probe exists

With `noise_sigma = 0`, step 4 divides by the peak of the array produced by step 2, so
**the `1263376` divisor cancels exactly**. Verified: for all 24 cases,
`max | measurement − psf/psf.max() | = 1.1e-16`. Consequently:

> **Passing all 24 noiseless cases does NOT prove your divisor, your operation order,
> or your clamps are right.** A port that skips step 2 entirely reproduces all 24
> measurements to the last bit.

`noise_probe` is the only test that pins the order down. It supplies a fixed float32
noise field so you never have to match an RNG:

```
psf = |FFT{ P e^{i·gt} }|^2
psf = psf / 1263376.0
psf = max( psf + 0.05 * noise , 0.0 )
psf = psf / max( psf.max() , 1e-6 )
```

Two results are stored (`circle` and `triangle`, both with `phase_id = 2`,
`noise_sigma = 0.05`), as `stats_f64` + `samples_f64` in the JSON and full arrays in
`golden.npz:measurement_noisy_f64 / _f32`. Dropping the divisor changes the circle
result by 0.9999 — a total failure, not a rounding difference.

Sanity landmark: the noise field's maximum is `3.9563052654266357` at `(65, 112)`, so
`0.05 × 3.9563 = 0.19782` exceeds `circle__phase2`'s post-divisor signal peak of
`0.13312`. Both noisy measurements therefore have `argmax_yx = (65, 112)`, *not*
`(64, 64)`. If your noisy argmax is at the optical peak, your divisor is missing.

### 5.5 Network input

Channel 0 = the measurement above. Channel 1 = the binary mask, values exactly `0.0`
and `1.0`. Input tensor `[B, 2, 128, 128]` float32; output `[B, 1, 128, 128]` float32,
a dense phase map **in radians**, unmasked and unactivated.

---

## 6. `strehl_ratio`

```
residual = (gt_phase - pred_phase) · mask
peak_corr = max( |FFT{ mask · exp(i·residual) }|^2 )     // psf_from_pupil, normalize=False
peak_ref  = max( |FFT{ mask · exp(i·0)        }|^2 )     // SAME mask, zero phase
strehl    = peak_corr / max(peak_ref, 1e-4)
```

Points that matter:

- The **same pupil mask `M`** is used both as the amplitude in the numerator PSF and as
  the reference in the denominator. Not the circle. Not the Zernike disk.
- The residual is masked **before** it enters the exponent. Outside the pupil
  `residual = 0`, so `exp(i·0) = 1` and the (zero) amplitude kills it anyway — but mask
  first, so that an unmasked prediction's wild out-of-pupil values cannot leak in.
- `eps = 1e-4` floors the **denominator only**. For real masks `peak_ref = (Σ M)² ≥ 126736`,
  so it never binds; implement it regardless.
- Both PSFs use the full §4 pipeline (unnormalised FFT, both shifts). The shifts do not
  change a maximum, so you may omit them *here* if you want — but keep them, so there is
  one PSF routine.
- **Callers clamp the result to `≤ 1.0`.** `evaluate.py` computes
  `strehl_ratio(...).clamp(max=1.0)` before aggregating. The raw function can exceed 1
  for a pathological prediction. The golden `metrics_synthetic_pred.strehl_ratio` values
  are the **raw, unclamped** returns (all fall in `[0.9398870089629154,
  0.9985438725345253]`, so the clamp is inactive). Any UI or aggregate you build should
  apply `min(1, ·)`.
- No clamp at the low end. The value is `≥ 0` by construction.
- Piston-invariance: adding a constant to `pred` multiplies the residual field by a
  global phase `exp(-i·c)`, leaving `|·|²` unchanged. The §9 predictor's `+0.3` term
  must therefore leave the Strehl untouched — a useful internal check.

---

## 7. `masked_piston_corrected_mse_per_sample`

```
pred_m   = pred   · mask
target_m = target · mask
mask_sum = max( Σ mask , 1e-9 )

pred_c   = ( pred_m   - (Σ pred_m)   / mask_sum ) · mask
target_c = ( target_m - (Σ target_m) / mask_sum ) · mask

mse      = ( Σ (pred_c - target_c)^2 ) / mask_sum
```

Exact arithmetic, in order:

1. Mask both fields.
2. Compute each field's **in-pupil mean** as `sum / mask_sum` — divided by
   **`Σ mask`**, *not* by `H*W = 16384`. This is the single most common porting error
   here; using 16384 scales every MSE by `sum_mask/16384` (a factor 0.022–0.069 for
   these masks) and looks superficially plausible.
3. Subtract that mean from the *masked* field, then **re-apply the mask**. The
   re-masking matters: without it every outside pixel carries `-mean` and contributes
   to the sum.
4. Square the difference, sum over all 16384 pixels (outside terms are exactly 0),
   and divide by **`mask_sum`** again — not by 16384.

`clamp_min(1e-9)` applies to `mask_sum` only, guarding an empty mask. It never binds
here (`mask_sum ≥ 356`). Sums are float64 in the reference; in float32 torch the
agreement is ≤ 1.7e-8 absolute / 4.9e-7 relative.

Golden targets: `cases[i].metrics_synthetic_pred.masked_piston_corrected_mse.f64`,
spanning `0.0014571107031723294` … `0.06205361200096626`. Units are rad².

The batch-mean variant (`masked_piston_corrected_mse`) is just the mean over `B` of the
above; the per-sample form is what you should port.

---

## 8. `asymmetry(mask, align="corrected")`

The paper's central quantity: `1 − overlap` between the mask and its 180°-rotated copy,
after aligning the two by the peak of their cross-correlation.

```
1.  mass  = max( Σ x , 1e-12 )
2.  xflip = flip(x) on BOTH axes:  xflip[y][x] = x[127-y][127-x]
3.  X     = fft2(x)                    // RAW mask, NO ifftshift/fftshift anywhere here
4.  corr  = | ifft2( X * X ) |         // X * X  — AUTOCONVOLUTION, not X * conj(X)
5.  k     = argmax over the flattened 16384 values of corr   (first maximum)
6.  sy = floor(k / W) ; sx = k mod W                          // W = 128, the LAST axis length
7.  sy += 1 ; sx += 1                                         // align="corrected"
8.  aligned = circular roll of xflip by (sy, sx):  aligned[y][x] = xflip[(y-sy) mod 128][(x-sx) mod 128]
9.  overlap = Σ ( x · aligned )
10. alpha   = 1 - overlap / mass
```

Detail by detail:

- **Step 3 uses the raw mask.** No `fftshift`, no `ifftshift`. This is the one FFT in
  the codebase that is *not* wrapped in shifts.
- **Step 4 squares the transform** (`X * X`), it does not multiply by the conjugate.
  Correlating `x` with `flip(x)` *is* the autoconvolution of `x`, which is why. Using
  `X * conj(X)` gives the autocorrelation, whose peak is always at `(0,0)`, and the
  metric collapses to `1 − Σ(x·flip(x))/mass` — wrong for every off-centre pupil.
  The `ifft2` is the `1/16384`-scaled inverse; since only the argmax is used, the
  scale is irrelevant, but keep it if you want `corr`'s values to match.
- **Step 6 divides by `W` (the last-axis length) for BOTH axes** — `sy = k // W`,
  `sx = k % W`. For the square 128×128 arrays here this is ordinary row-major divmod
  and is correct. It is written against `W` rather than `H`, so it would be wrong for a
  non-square array; the pipeline is square everywhere, so reproduce it as written.
- **Step 7, the `+1` per axis**, is the correction. `flip` reflects about index
  `(N-1)/2`, so the autoconvolution peaks at `N-1` while the roll that undoes the flip
  is `N`; the raw argmax therefore lands one pixel short on each axis. The `"published"`
  mode omits this `+1` and reproduces arXiv:2604.00225 exactly, at the cost of a
  shape-dependent positive floor (a perfect circle scores 0.047 instead of 0).
  **Port `"corrected"`.** Use `"published"` only when you must reproduce a paper figure
  or a cached pool value.
- The roll is **circular** (wrap-around), applied to `xflip`, along both axes.
  `sy`/`sx` can be 128 after the `+1`, which wraps to 0 — handle it with `mod 128`.
- `alpha` is a ratio of integers for binary masks (`overlap` and `mass` are both integer
  counts), so it is exactly representable and fully reproducible.

**The circle scores exactly 0.** Its autoconv argmax is `(127, 127)`; `+1` gives
`(128, 128) ≡ (0, 0)`, so `aligned = flip(circle) = circle` (the §2.1 half-integer
centring makes the circle exactly 180°-symmetric), `overlap = mass = 1124`,
`alpha = 0.0`. Use this as your first assertion.

**Portability warning on `"published"`.** The autoconvolution peak is a *tie* for some
masks, and float32 and float64 FFTs break the tie differently:

| mask | argmax (float64) | argmax (torch float32) | `alpha_published` f64 | `alpha_published` f32 |
|---|---|---|---|---|
| `triangle`   | (124, 0) | (126, 0) | 0.3427991886409736 | 0.33468562364578247 |
| `pool_1473`  | (127, 7) | (127, 6) | 0.3567415730337079 | 0.3595505356788635  |

`alpha_corrected` happens to agree to ~1e-8 for both of these, but that is luck, not
guarantee. The stored `autoconv_argmax_yx` is the **float64** one. A double-precision
JS port should match the `f64` column; if it lands on the float32 tie instead, that is a
tie-break difference, not a bug in your FFT — but only ever ship `alpha_corrected`.

`throughput` is reported alongside: `sum_mask / 1124`, the reference-circle pixel area.

---

## 9. The synthetic predictor

The Strehl and MSE golden values are computed against a deterministic pseudo-prediction,
so you can validate the metrics **without running the network**:

```
gt[y][x]   = mask[y][x] * phase[y][x]                              // the masked ground truth
pred[y][x] = 0.9 * gt[y][x] + 0.3 + 0.15 * ((x - 64) * (y - 64)) / 4096
```

with `x` the column index and `y` the row index, both `0..127`, and all arithmetic in
IEEE-754 double. No transcendental functions appear, so **any conforming double
implementation reproduces `pred` bit-for-bit** (verified: recomputing it from
`golden.json` alone matches `golden.npz:pred_synth_f64` with max difference exactly 0,
and the metrics recompute to max difference exactly 0).

Structure of the three terms:
- `0.9 * gt` — a plausible slightly-shrunk estimate;
- `+0.3` — **pure piston**. It must cancel in `masked_piston_corrected_mse` (both fields
  get their in-pupil mean removed) and in `strehl_ratio` (a global phase). If either
  metric shifts when you change `0.3` to another constant, that metric is mis-ported.
- `0.15 * (x-64)(y-64)/4096` — a saddle-shaped residual that is **not** removed by
  piston correction, so the metrics are non-trivial.

`pred` is *not* masked before being handed to the metrics; both metrics mask internally.
That is deliberate — it exercises the internal masking, exactly as the U-Net's dense,
unmasked output does.

`metrics_unet_pred` in each case uses the real network's float32 output instead
(`golden.npz:pred_unet_f32`). Treat those as informational: matching them requires
`unet_fp32.onnx` and float32 inference, and an ONNX runtime is allowed to differ in the
last few float32 digits. Do not gate a port on them.

---

## 10. Tolerances

### 10.1 Compare against the `f64` column

The golden files store two parallel evaluations: a **float64 reference** (`*_f64`,
`stats_f64`, `samples_f64`, `.f64`) and the **torch float32** path the training code
actually ran (`*_f32`, `.torch_f32`). A JavaScript port using `Number` / `Float64Array`
must be compared against the **`f64`** values. The `f32` values are there to tell you
how much of any discrepancy is merely precision.

Measured float32-vs-float64 spread across the golden set — this is the *noise floor of
the float32 path*, not a target:

| quantity | max deviation (f32 vs f64) |
|---|---|
| phase | 2.4e-7 abs, 4.5e-8 rel-to-peak |
| PSF | 3.9e-7 rel-to-peak |
| measurement | 6.0e-7 rel-to-peak |
| Strehl | 1.2e-7 abs |
| MSE | 1.7e-8 abs, 4.9e-7 rel |

### 10.2 Required tolerances for a float64 JS port

| golden quantity | compare | **must pass** | typical for a good port |
|---|---|---|---|
| mask bits / `masks_uint8` | element-wise | **exact** (bit-identical) | exact |
| `sum_mask`, `disk_px`, `reference_circle_area_px`, `shared_px` | integer | **exact** | exact |
| `circle_psf_peak_exact_int` = 1263376 and every `sum_mask_squared` | scalar | **exact** (`=== 1263376`) | exact |
| `argmax_flat` / `argmax_yx` (psf, measurement, phase) | integer | **exact** | exact |
| `zernike_spot_checks` `R_at_r`, `Z_at_r_theta` | rel. | **1e-12** | ~1e-15 |
| phase vs `phases_f64` / `stats_f64` / `samples_f64` | abs. | **1e-11** | ~1e-14 |
| PSF vs `psf_f64` / `samples_f64` | **rel-to-peak** | **1e-9** | ~1e-13 |
| measurement vs `measurement_f64` | abs. (peak is 1) | **1e-9** | ~1e-13 |
| noisy measurement vs `measurement_noisy_f64` | abs. | **1e-9** | ~1e-13 |
| `stats_f64.sum`, `.sum_sq` | rel. | **1e-11** | ~1e-13 |
| Strehl vs `.f64` | abs. | **1e-9** | ~1e-13 |
| MSE vs `.f64` | rel. | **1e-9** | ~1e-13 |
| `alpha_corrected.f64`, `throughput` | abs. | **1e-12** | exact (rational) |
| `alpha_published.f64` | abs. | 1e-12 **if** your argmax matches; otherwise expect a tie-break mismatch (§8) | — |
| `metrics_unet_pred.*` (float32 ONNX) | rel. | **1e-3** | ~1e-5 |

The "must pass" column is deliberately loose enough to accommodate any reasonable
double-precision FFT while still being ~100× tighter than the float32 floor, so it
cannot be satisfied by an algorithmically wrong port.

### 10.3 Comparison rules

- **Never compare PSF pixels with a plain relative tolerance.** The dynamic range is
  about 10 decades (`circle__phase0`: peak 698749.4, min 0.002; measurements go down to
  2.1e-10). Use **`|a − b| ≤ tol × peak`** — relative *to the array peak*, not to the
  individual value. Otherwise the faint outer rings fail on values that are physically
  irrelevant.
- `stats_f64.sum` and `sum_sq` are full-array reductions. The reference uses numpy's
  pairwise summation; a naive left-to-right JS loop differs in the last few ULPs. Use a
  relative tolerance and do not expect bit-equality. (Kahan or pairwise summation in JS
  gets you to ~1e-15.)
- `argmax` ties break to the **lowest flat index**. `Math.max`-style scans must use
  strict `>` (not `>=`) when updating the running best.
- `-0.0` vs `0.0`: treat as equal. Do not use `Object.is` or `===` on sign-of-zero.
- Do not compare the sha256 fields unless you are byte-reproducing the arrays; they
  cover the float64-LE and float32-LE byte images and will differ by one ULP for a port
  that is otherwise perfectly acceptable. They are provenance, not a test.

---

## 11. How to run the check

A JS test should be a single script that loads `golden.json` (and optionally
`golden.npz`) and walks the following seven stages, accumulating the worst deviation per
stage and failing on the first stage that exceeds its §10.2 bound. Run them **in this
order** — each stage's inputs are the previous stage's verified outputs, so the first
failure localises the bug.

**Stage 0 — geometry.** Build the reference circle (§2.1) and the Zernike disk (§2.2).
Assert `Σ circle === 1124`, `Σ disk === 1129`, `|circle ∧ disk| === 1100`,
`|circle ∧ ¬disk| === 24`, `|¬circle ∧ disk| === 29`. Then decode `masks[0].bits_b64`
and assert it is **element-wise identical** to your computed circle. A failure here
means every later stage is meaningless.

**Stage 1 — Zernike.** For each of the 33 `zernike_spot_checks.modes`, evaluate `R` at
`r_probe = [0, 0.17, 0.5, 0.8213, 1.0]` and `Z` at `(r_probe, theta_probe)` elementwise,
`theta_probe = [0.0, 0.7, 1.9, -2.6, 3.0]`. Compare to `R_at_r` / `Z_at_r_theta`.
This isolates the normalisation (§3.2) and the sign convention (§3.3) before any FFT.

**Stage 2 — phases.** For each of the 4 `phases`, synthesise from `coeffs_f64` (§3.4)
and compare against `stats_f64` and `samples_f64.core16`. Optionally decode
`float32_b64` and check your float32 cast against it. `samples_f64.grid16` is nearly all
zeros — use `core16` as the real test.

**Stage 3 — masks and asymmetry.** For each of the 6 masks: decode `bits_b64`, check
`sum_mask` and `throughput`, compute `asymmetry(..., "corrected")` (§8) and compare to
`alpha_corrected.f64`. Assert the circle gives exactly `0`. Compute the
diffraction-limited PSF peak and assert it equals `sum_mask²` **exactly**.

**Stage 4 — PSF.** For each of the 24 `cases`: form `gt = mask · phase` (mask from the
case's `mask_id`, phase from its `phase_id`), run §4, and check
`intermediate.psf_peak_before_normalization.f64`, `intermediate.psf_argmax_flat.f64`,
`psf.stats_f64` (including the Parseval identity `sum === 16384 * sum_mask`), and
`psf.samples_f64.core16`. Compare rel-to-peak.

**Stage 5 — measurement.** Same 24 cases through §5.1 with `with_noise = false`; compare
`measurement.stats_f64` and `measurement.samples_f64.core16`. Remember this stage
**cannot** validate the `1263376` divisor (§5.4).

**Stage 6 — noise probe.** *Do not skip.* Decode `noise_probe.noise_float32_b64` into a
128×128 float32 field (widen to float64 before use, as the reference does). For each of
the two `noise_probe.results`, run the §5.4 recipe with `sigma = 0.05` and compare
`stats_f64` and `samples_f64.core16`. Assert `argmax_yx === [65, 112]` for both. This is
the only stage that proves your operation order, your divisor, and your `0.0` clamp.

**Stage 7 — metrics.** For each of the 24 cases, build the §9 synthetic predictor from
the verified `gt`, compute Strehl (§6) and masked piston-corrected MSE (§7), and compare
to `metrics_synthetic_pred.*.f64`. Then re-run both metrics with the `+0.3` constant
changed to `+7.0` and assert **both** results are unchanged to 1e-9 — the piston-
invariance test. This catches a missing re-mask in step 3 of §7 and a missing mask in
§6, which the plain comparison can miss.

Report, per stage, the worst absolute and worst relative-to-peak deviation and the
`case_id` where it occurred. If you also want whole-array coverage rather than the
256-point `core16` sample, read `golden.npz` (a ZIP of `.npy` members; a small reader is
~40 lines, or convert it once to raw `.bin` files with numpy) and diff
`psf_f64`, `measurement_f64`, `measurement_noisy_f64`, `phases_f64` in full.

### 11.1 Quick smoke test (30 seconds, catches most errors)

```
Σ circle            === 1124            // §2.1 grid
Σ disk              === 1129            // §2.2 grid, and they differ
circlePSF.max()     === 1263376         // exactly; FFT normalisation + circle
circlePSF.argmax    === 8256            // = (64,64); shift convention
Σ circlePSF         === 16384 * 1124    // Parseval
asymmetry(circle,"corrected") === 0     // the +1 and the X*X
noisyCircle.argmax  === [65,112]        // the 1263376 divisor is actually applied
```

Every one of those seven is a different bug class. If all seven pass, the remaining
failures are almost certainly Zernike sign/normalisation (Stage 1) or an MSE denominator
(§7, `Σ mask` vs 16384).

---

## Appendix A — per-case reference table

`f64` values. `peak` = PSF peak before any normalisation; `after ÷` = peak after the
`1263376` divisor; `Strehl`/`MSE` = against the §9 synthetic predictor.

| case_id | sum_mask | peak | after ÷ | psf argmax | Strehl | MSE |
|---|---:|---:|---:|---|---:|---:|
| circle__phase0 | 1124 | 698749.400943 | 0.553081110408 | (64,64) | 0.993447333 | 0.006580821 |
| circle__phase1 | 1124 | 579670.213953 | 0.458826362027 | (64,64) | 0.991335509 | 0.008711370 |
| circle__phase2 | 1124 | 168183.528979 | 0.133122307990 | (69,66) | 0.957501578 | 0.043449675 |
| circle__phase3 | 1124 | 139167.149132 | 0.110154972971 | (59,61) | 0.955930768 | 0.045086098 |
| triangle__phase0 | 493 | 185116.103770 | 0.146524948843 | (64,64) | 0.997018087 | 0.002990153 |
| triangle__phase1 | 493 | 139100.748878 | 0.110102415178 | (64,64) | 0.994481748 | 0.005533407 |
| triangle__phase2 | 493 | 46069.658553 | 0.036465516642 | (69,66) | 0.966709360 | 0.033825254 |
| triangle__phase3 | 493 | 43943.552350 | 0.034782639808 | (60,60) | 0.960368571 | 0.040398618 |
| pool_101__phase0 | 973 | 567021.168917 | 0.448814263463 | (64,64) | 0.994585166 | 0.005432812 |
| pool_101__phase1 | 973 | 459064.703137 | 0.363363482556 | (64,64) | 0.992383872 | 0.007648986 |
| pool_101__phase2 | 973 | 167554.137558 | 0.132624125801 | (69,66) | 0.958072956 | 0.042858091 |
| pool_101__phase3 | 973 | 143931.067133 | 0.113925756966 | (59,61) | 0.955481332 | 0.045547046 |
| pool_620__phase0 | 383 | 95330.869270 | 0.075457242555 | (64,64) | 0.995421946 | 0.004592074 |
| pool_620__phase1 | 383 | 72006.538782 | 0.056995335341 | (63,66) | 0.991201395 | 0.008842375 |
| pool_620__phase2 | 383 | 41062.983007 | 0.032502582768 | (69,62) | 0.961806384 | 0.038878898 |
| pool_620__phase3 | 383 | 34387.379758 | 0.027218642556 | (59,60) | 0.951516303 | 0.049591261 |
| pool_1088__phase0 | 428 | 155765.804569 | 0.123293306640 | (64,64) | 0.998375842 | 0.001625284 |
| pool_1088__phase1 | 428 | 121589.585842 | 0.096241804373 | (65,64) | 0.995021769 | 0.004989855 |
| pool_1088__phase2 | 428 | 40332.723735 | 0.031924560649 | (69,68) | 0.965230351 | 0.035437260 |
| pool_1088__phase3 | 428 | 16530.335930 | 0.013084256729 | (69,66) | 0.953665091 | 0.047382103 |
| pool_1473__phase0 | 356 | 109578.118811 | 0.086734367924 | (64,64) | 0.998543873 | 0.001457111 |
| pool_1473__phase1 | 356 | 74709.703841 | 0.059134971569 | (65,64) | 0.994156390 | 0.005858587 |
| pool_1473__phase2 | 356 | 35956.503202 | 0.028460650829 | (69,69) | 0.962656769 | 0.038063081 |
| pool_1473__phase3 | 356 | 20776.886490 | 0.016445528876 | (58,58) | 0.939887009 | 0.062053612 |

The measurement's `argmax` equals the PSF's in every noiseless case (the divisor and the
renormalisation are both positive scalars). `renorm_clamp_active` is `false` throughout.

## Appendix B — traps, in the order people hit them

1. **Using one grid for both jobs** (§2). 1124 vs 1129 pixels, different centres.
2. **A normalised FFT** (§4.1). Multiply by 16384 if your library divides.
3. **`sqrt(2(n+1))` on the Zernike radials** (§3.2). There is none.
4. **`atan2(x, y)` instead of `atan2(y, x)`**, or `y` measured upward (§3.3).
5. **Dividing the MSE by 16384 instead of `Σ mask`** (§7), or forgetting the re-mask
   after subtracting the mean.
6. **Skipping the `1263376` divisor** (§5.1) — invisible in all 24 noiseless cases, fatal
   the moment noise is on. Run the noise probe (§5.4).
7. **`X * conj(X)` in the asymmetry** (§8) instead of `X * X`, or omitting the `+1`.
8. **Comparing PSF pixels with a plain relative tolerance** (§10.3) instead of
   relative-to-peak.
9. **Reproducing the RNG** instead of reading `coeffs_f64` / `noise_float32_b64` (§1.4).
10. **Trusting `alpha_published`** (§8). Ship `alpha_corrected`.
