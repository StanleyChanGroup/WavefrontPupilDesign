# Scene verification — `src/scene.js`

Verification of the three procedural scenes the adaptive-optics panel can choose
between: **`usaf`** (the existing USAF-1951 chart), **`stars`** (new) and
**`terrain`** (new). Everything below is produced by

```
node test/scenes_verify.mjs
```

which exits non-zero if any check fails. **92 checks, 0 failures** on the run
recorded here.

| | |
|---|---|
| Date | 2026-09-16 19:23 UTC |
| Node | v22.14.0, linux x64 |
| Machine | 128 cores, load average 232–245 during the run |
| Scenes | `SCENES` = `[usaf, stars, terrain]`, each `{id, label, blurb, make(size), display, spec(size)}` |
| Sizes | 256 (default) and 512 |
| Source | `src/scene.js`; optics from `src/optics.js` (`referenceCircle`, `psfFromPupil`, `synthPhase`, `maskedPhase`, `strehlRatio`, `PAPER_MODES`) |

Because this box runs at load 140–245 on 128 cores, **every timing below is the
minimum of nine independent single calls**, never a mean over a batch: a batch
mean here measures the scheduler, not the code.

---

## Prior verification carried forward (`convolvePsf`, `usafTarget`)

These were established before this work and are **not** re-derived here; they are
recorded because the new scenes depend on exactly the same convolution path, and
because nothing in this change touches `convolvePsf()` or `usafTarget()`'s output.

| Property | Prior result |
|---|---|
| Delta-PSF identity — `convolvePsf(img, delta)` returns the input | max absolute output−input difference = 4.9e-16 |
| Displaced-delta centring — a delta at `(M/2+dy, M/2+dx)` translates by exactly `(+dy, +dx)` | exact to `~1e-28` |
| Mean brightness preserved — unit-sum kernel on the torus | relative error `2.1e-15` |

Re-running the pre-existing `test/scene_check.mjs` during this work reproduced the
first and third of those to the quoted digits (`4.947e-16`, `2.109e-15`).

> Note, unchanged by this work: `scene_check.mjs` check **(b)** prints FAIL. It
> asserts that an *aberrated* PSF leaves the image centroid unmoved, which the
> `scene.js` module header explicitly documents as **false and correct physics** —
> "an aberrated PSF DOES move it, by precisely the PSF's own centroid offset …
> that is physics (coma shifts the image), not an indexing error". The assertion
> in that script is mislabelled; the behaviour is right. It was already so before
> this change and is outside this task's scope.

Section 0b below re-confirms all three properties independently, on all three
scenes at both sizes.

---

## 0. Reference pupil and its diffraction limit

The pupil is the GOLDEN.md geometry, `N = 128`, `radius = 19`.

```
lambda/D = N/(2r)                          = 3.3684 px
Rayleigh = 1.2196699 * lambda/D            = 4.10836 px
optical cutoff 2r/N                        = 0.2969 cyc/px  (period 3.368 px)
binary separation used by starField()      = 3.90294 px = 0.950 Rayleigh
```

The analytic Rayleigh limit is checked against the actual PSF. `|FFT{pupil}|²` is
invariant to where the pupil sits in its array, so zero-padding the 128×128 pupil
into 1024×1024 samples the *same* continuous PSF at 8 samples per PSF pixel:

| quantity | value |
|---|---|
| measured first dark ring | **4.1250 px** (1.131e-5 of peak) |
| analytic Rayleigh limit | 4.10836 px |
| agreement | **0.40 %** |

The aberration used everywhere below is the fixed set already used by
`test/scene_check.mjs`: `PAPER_MODES` coefficients from `mulberry32(12345)`,
uniform in ±2.0 rad.

| wavefront | masked RMS | Strehl |
|---|---|---|
| uncorrected | 2.0340 rad | 9.43 % |
| residual after 88 % correction | 0.2441 rad | 94.36 % |

---

## 0b. `convolvePsf` re-confirmed on all three scenes

An exact translation is a stronger statement than an unmoved centroid, and it is
the one that is well defined on a torus (the centroid of a wrapped image is not
the wrapped centroid), so what is reported is the maximum per-pixel error after
undoing the expected shift. Displacement used: `(dy, dx) = (+7, −5)`.

| scene | size | delta identity | displaced-delta translation | flux ratio |
|---|---|---|---|---|
| usaf | 256 | 4.947e-16 | 8.882e-16 | 1.000000000000000 |
| usaf | 512 | 6.775e-16 | 1.110e-15 | 1.000000000000000 |
| stars | 256 | 1.123e-16 | 1.574e-16 | 1.000000000000000 |
| stars | 512 | 1.158e-16 | 1.266e-16 | 1.000000000000000 |
| terrain | 256 | 7.772e-16 | 7.772e-16 | 1.000000000000000 |
| terrain | 512 | 8.882e-16 | 8.327e-16 | 1.000000000000000 |

All 18 checks pass at a 1e-12 threshold.

---

## 1. Length, range, determinism

| scene | size | length | min | max | mean | pixels = 0 | pixels = 1 |
|---|---|---|---|---|---|---|---|
| usaf | 256 | 65536 | 0.000000 | 1.000000 | 0.404771 | 26917 | 14260 |
| usaf | 512 | 262144 | 0.000000 | 1.000000 | 0.404977 | 110168 | 59900 |
| stars | 256 | 65536 | 0.003500 | 0.922918 | 0.003976 | 0 | 0 |
| stars | 512 | 262144 | 0.003500 | 0.739954 | 0.003910 | 0 | 0 |
| terrain | 256 | 65536 | 0.000000 | 1.000000 | 0.551666 | 21 | 39 |
| terrain | 512 | 262144 | 0.000000 | 1.000000 | 0.550700 | 235 | 263 |

- `length == size*size`, every value finite and inside `[0, 1]`, every return a
  `Float64Array` — **18/18 pass**.
- `stars` never reaches either endpoint: its floor is the 0.0035 sky pedestal and
  its ceiling is the mag-0 star's peak sample (0.9229 at 256; at 512 the brightest
  star happens to sit further from a pixel centre, hence 0.7400 — a consequence of
  honest sub-pixel placement, not of clipping).
- `terrain` clips 60 pixels of 65536 at 256 (0.09 %) and 498 of 262144 at 512
  (0.19 %), by design: the shading is mapped by the *fixed* affine
  `1.70·v − 0.296`, so the picture does not change when the frame does.

**Determinism.** Byte-identical output (`Uint8Array` view comparison over the
whole buffer) for:

- the same call repeated — usaf/stars/terrain at 256 and 512, 6/6;
- a **freshly evaluated copy of the module** (`import('../src/scene.js?fresh=1')`),
  which proves the answer is not an artefact of the module-level caches surviving
  inside one process — 6/6;
- a different seed produces a different field for both seeded scenes (2/2), so the
  seed is actually being used;
- `opts.shared` returns the cached instance, the default call returns a copy (2/2);
- `asinhStretch` is monotone, maps 0 → 0 and 1 → 1 exactly, and stays in `[0, 1]`.

---

## 2. Radially averaged power spectra

Mean removed, `|k|` binned to the nearest integer radius in cycles per frame,
power averaged within each bin, normalised to `P(1)`.

**size 256**

| k | period (px) | usaf | stars | terrain |
|---:|---:|---:|---:|---:|
| 1 | 256.00 | 1.00e+0 | 1.00e+0 | 1.00e+0 |
| 2 | 128.00 | 2.21e-1 | 1.06e+0 | 5.77e-2 |
| 4 | 64.00 | 4.97e-2 | 5.62e-1 | 5.63e-2 |
| 8 | 32.00 | 4.12e-3 | 1.28e-1 | 4.91e-2 |
| 16 | 16.00 | 8.43e-3 | 1.93e-2 | 2.84e-2 |
| 32 | 8.00 | 2.46e-3 | 1.14e-2 | 1.20e-2 |
| 48 | 5.33 | 4.21e-4 | 8.11e-3 | 2.70e-3 |
| 64 | 4.00 | 1.76e-4 | 5.37e-3 | 9.13e-4 |
| 96 | 2.67 | 2.88e-5 | 1.54e-3 | 1.11e-4 |
| 128 | 2.00 | 1.59e-5 | 4.53e-4 | 1.48e-5 |

**size 512**

| k | period (px) | usaf | stars | terrain |
|---:|---:|---:|---:|---:|
| 1 | 512.00 | 1.00e+0 | 1.00e+0 | 1.00e+0 |
| 2 | 256.00 | 2.21e-1 | 8.85e-1 | 5.66e-2 |
| 4 | 128.00 | 4.97e-2 | 4.80e-1 | 5.66e-2 |
| 8 | 64.00 | 4.16e-3 | 9.12e-2 | 4.93e-2 |
| 16 | 32.00 | 8.47e-3 | 4.90e-3 | 2.93e-2 |
| 32 | 16.00 | 2.44e-3 | 1.01e-3 | 1.32e-2 |
| 48 | 10.67 | 4.20e-4 | 7.62e-4 | 3.51e-3 |
| 64 | 8.00 | 1.66e-4 | 6.50e-4 | 1.45e-3 |
| 96 | 5.33 | 2.40e-5 | 4.66e-4 | 3.07e-4 |
| 128 | 4.00 | 1.59e-5 | 3.06e-4 | 6.61e-5 |

Checked and passing: **every one of the 128 (resp. 256) radial bands from k = 1 to
Nyquist carries more than 1e-9 of `P(1)`**, for all three scenes at both sizes.
None of them is band-limited to a handful of scales; none of them is flat, which
is what white noise ("TV static") would give.

Reading the three columns:

- **usaf** is a drawing of hard-edged bars, so its spectrum is spiky rather than
  smooth — the local bump at k = 16 over k = 8 is the bar-group comb, not noise.
- **stars** is the flattest, as point sources must be: a Gaussian of σ = 0.65 px
  only starts rolling off near Nyquist.
- **terrain** falls roughly as a power law from k = 8 onward — the fBm signature —
  with the knee near k = 48–64 (period 5.3–4 px) where the finest noise octave
  (4 chart units) lives.

Fraction of mean-removed power **above the pupil's optical cutoff** (0.2969 cyc/px,
i.e. structure the telescope cannot pass even when perfect):

| size | usaf | stars | terrain |
|---|---|---|---|
| 256 | 4.43 % | 15.77 % | 2.98 % |
| 512 | 2.45 % | 7.42 % | 0.40 % |

The star field's 15.77 % is the point sources being genuinely sharper than the
telescope; the terrain's 0.40 % at 512 confirms that its finest authored octave
(4 chart units = 8 px at 512) sits comfortably inside the passband.

---

## 3. The close binary — merged by blur, split by correction

The pair: two equal mag-1.6 components, position angle 0°, separated by
**3.90294 px = 0.95 × the Rayleigh limit** — between the Sparrow limit
(≈ 0.77 Rayleigh, where the notch vanishes even for a perfect pupil) and Rayleigh,
which is exactly the regime where the correction has to earn the split.

Measured with a Catmull-Rom cut along the pair axis, ±1.5 separations either side
of the midpoint, 1201 samples. "Resolved" means Rayleigh's own criterion: two
distinct interior maxima with a strictly lower minimum between them.

```
dip = 1 − trough / min(peakA, peakB)
```

**size 256** — pair at (174.339, 84.610) and (178.241, 84.610)

| case | peak A | peak B | trough | dip | resolved |
|---|---|---|---|---|---|
| scene (no optics) | 1.512e-1 | 1.451e-1 | 5.587e-3 | 96.15 % | YES |
| **diffraction-limited PSF** | 3.759e-2 | 3.731e-2 | 3.397e-2 | **8.95 %** | **YES** |
| **aberrated PSF (2.03 rad RMS, Strehl 9.4 %)** | 6.597e-3 | 6.876e-3 | 6.597e-3 | — | **no — one peak** |
| **residual PSF after 88 % correction** | 3.546e-2 | 3.535e-2 | 3.254e-2 | **7.95 %** | **YES** |

**size 512** — pair at (350.629, 169.220) and (354.531, 169.220), same 3.90294 px

| case | peak A | peak B | trough | dip | resolved |
|---|---|---|---|---|---|
| scene (no optics) | 1.832e-1 | 1.863e-1 | 7.362e-3 | 95.98 % | YES |
| diffraction-limited PSF | 3.774e-2 | 3.787e-2 | 3.418e-2 | **9.45 %** | YES |
| aberrated PSF | 6.642e-3 | 6.940e-3 | 6.642e-3 | — | no |
| residual after 88 % correction | 3.566e-2 | 3.590e-2 | 3.267e-2 | **8.38 %** | YES |

**This is the headline: resolved when sharp (8.95 %), a single merged peak under
aberration, split again after correction (7.95 %), at both sizes.**

Note that the aberrated row's `trough == peakA`: the scan finds no interior minimum
at all, so the "peak" it reports on the low side is the scan's own endpoint. That
is the numerical signature of a genuinely merged pair, not a shallow notch.

The dip measured on the scene as shipped (8.95 %) is about 1.4 points lower than
on an isolated copy of the pair (10.32 %, section 4): the uniform sky pedestal
survives the convolution and lifts the trough, which lowers the *ratio* without
touching the *contrast*. Both numbers are quoted in `scene.js`.

### Where the merge/split threshold sits

Convolving with the PSF of a partially corrected wavefront, `residual = f × gt`:

| residual fraction f | RMS (rad) | Strehl | dip | resolved |
|---:|---:|---:|---:|---|
| 0.00 | 0.0000 | 100.00 % | 8.95 % | YES |
| 0.05 | 0.1017 | 98.98 % | 8.69 % | YES |
| 0.10 | 0.2034 | 96.02 % | 8.20 % | YES |
| 0.15 | 0.3051 | 91.41 % | 7.55 % | YES |
| 0.20 | 0.4068 | 85.53 % | 6.75 % | YES |
| 0.30 | 0.6102 | 71.65 % | 4.88 % | YES |
| 0.50 | 1.0170 | 43.15 % | 1.62 % | YES |
| 1.00 | 2.0340 | 9.43 % | — | **no** |

The pair survives down to roughly Strehl 40 % and is gone by Strehl 10 %, so the
panel's correction quality maps monotonically onto a visible, quantified change.

`scene.js` exports `binaryCut(image, size)` so the page can plot exactly this cut.
It is cross-checked against this file's own independent implementation on all four
cases at both sizes — **8/8 agree to better than 1e-12 in dip**.

---

## 4. The point sources are band-limited, not hot pixels

A star written as one bright pixel is a delta on the sampling lattice: it cannot be
placed between pixels, and everything above the pupil's cutoff folds back into the
passband. The operational test is **shift invariance**: imaging a source at `x0+t`
must equal the exact Fourier translation by `t` of the same source imaged at `x0`.

```
            || conv(source at x0+t) − FourierShift_t( conv(source at x0) ) ||
error(t) =  ───────────────────────────────────────────────────────────────
                          || conv(source at x0) ||
```

over `t ∈ [0, 1)` in sixteenths, convolved with the diffraction-limited PSF:

| profile | worst t | mean t |
|---|---:|---:|
| single hot pixel | **25.400 %** | 13.599 % |
| Gaussian σ = 0.50 px | 4.194 % | 2.589 % |
| **Gaussian σ = 0.65 px (used)** | **0.294 %** | 0.195 % |
| Gaussian σ = 0.80 px | 0.016 % | 0.011 % |

σ = 0.65 px is **86× better than a hot pixel**. Both checks pass (>50× improvement;
worst-case error < 0.5 %).

The cost of going wider is the notch. On an isolated copy of the pair (no sky
pedestal, no neighbours), diffraction-limited:

| σ (px) | dip |
|---:|---:|
| 0.50 | 14.06 % |
| 0.60 | 11.51 % |
| **0.65** | **10.32 %** |
| 0.70 | 9.17 % |
| 0.80 | 6.93 % |
| 1.00 | 2.95 % |

0.65 px is the knee: below it the aliasing error climbs by an order of magnitude
per 0.15 px, above it all a wider profile buys is a shallower notch.

**How it is done, concretely:** each source is an isotropic Gaussian evaluated at
*pixel centres* `(x+0.5, y+0.5)` about a *fractional* centre `(cx, cy)`, summed over
a ±(⌈5σ⌉+1) px box with toroidal wrap. The centre is never rounded, so position is
sub-pixel exact; the profile rolls off smoothly, so the spectrum is small where it
would otherwise alias. The galaxy is the same construction with an inclined
elliptical Gaussian plus a compact core.

---

## 5. Photometry, seamlessness, cross-size consistency

**Photometry.** `sum(convolvePsf(scene, psf)) / sum(scene)` for the aberrated PSF:
`usaf` 2.109e-15, `stars` 2.528e-13, `terrain` 7.661e-15 relative error. 3/3 pass.

**Seamlessness.** `convolvePsf` is circular, so a scene that does not tile shows a
seam under a large blur. Mean `|d/dx|` across the wrap column versus inside the
frame:

| scene | across the wrap | inside | ratio |
|---|---|---|---|
| usaf | 0.000e+0 | 5.029e-2 | 0.000 |
| stars | 1.808e-5 | 1.455e-4 | 0.124 |
| terrain | 3.331e-2 | 3.458e-2 | **0.963** |

`usaf` is exactly flat at the border (the grey field, as its header documents) and
`stars` is nearly so (sky pedestal). `terrain` is the interesting one: it is full of
structure right up to the edge, and its wrap gradient is within **3.7 %** of its
interior gradient — the wrap is statistically invisible. That is deliberate: every
noise octave is hashed modulo its own lattice period, and craters are drawn with
toroidal distance, so the field is exactly periodic over the frame.

**Cross-size consistency.**

- `stars`: the binary separation is **3.90294 px at both 256 and 512** — it is the
  one quantity held in absolute pixels, because it only means anything against the
  fixed 128×128 PSF grid. The composition around it scales (`binary.cx` doubles).
- `terrain`: octave periods double, 128 px → 256 px for the coarsest — 512 is the
  same landscape at 2 px per chart unit, matching `usafTarget`'s convention.

**Scene inventory at 256**

| | |
|---|---|
| terrain octave periods | 128, 64, 32, 16, 8, 4 px |
| craters | 110, radii 3.20 … 26.00 px |
| star field | 73 point sources, magnitudes 0 … 6.5 (peak ratio 398:1), σ = 0.65 px |
| galaxy | elliptical Gaussian, σ 13.0 × 6.4 px, PA 35°, peak 0.045 + a 2.6 px core |

---

## 6. The `SCENES` catalogue

`SCENES` is a frozen, ordered array; `SCENES.map(s => s.id)` is exactly
`usaf, stars, terrain`. Each entry carries `{id, label, blurb, make(size), display,
spec(size)}` and `make()` defaults to 256. 8/8 checks pass.

| id | label | display | blurb |
|---|---|---|---|
| `usaf` | USAF-1951 target | linear | Calibrated bar groups and Siemens stars: read the resolution off the chart. |
| `stars` | Star field | asinh | Bright star, a 3.90 px binary at 0.95 × Rayleigh, 6.5 mag of field stars, one galaxy. |
| `terrain` | Cratered terrain | linear | Hill-shaded crater and ridge field: broadband natural structure, seamless on the torus. |

`display` is advice for the canvas, not part of the data: `stars` is returned as
**linear intensity**, which is what a PSF convolves, and needs `asinhStretch()` (or
any other tone map) to show six magnitudes at once.

---

## 7. Timing

Minimum of nine independent single calls. Cold generation imports a **fresh copy of
the module each trial** (`../src/scene.js?t=…`) so no cache is warm.

**Cold generation (ms)**

| scene | size | min | median | max |
|---|---|---:|---:|---:|
| usaf | 256 | **15.34** | 17.27 | 24.77 |
| usaf | 512 | **35.22** | 35.81 | 38.10 |
| stars | 256 | **5.76** | 5.93 | 7.28 |
| stars | 512 | **8.08** | 8.42 | 9.34 |
| terrain | 256 | **68.07** | 69.64 | 87.97 |
| terrain | 512 | **234.37** | 238.82 | 243.47 |

**Warm call — cache hit plus the `Float64Array` copy (ms)**

| scene | 256 | 512 |
|---|---:|---:|
| usaf | 0.240 | 0.956 |
| stars | 0.241 | 0.868 |
| terrain | 0.229 | 0.450 |

**For scale — one `convolvePsf()` with the 128×128 PSF:** 15.14 ms at 256,
138.17 ms at 512.

Every scene is generated once per `(size, seed)` and cached, so the cold cost is
paid on the first switch to a scene and never again; `opts.shared` skips even the
copy. `terrain` is the expensive one — 512×512 × (3+3 warp + 6 fBm + 5 ridged +
3 albedo) ≈ 5.2 M gradient-noise evaluations — but at 234 ms it is still under two
`convolvePsf` calls at that size, i.e. under the cost of a single slider tick, and
it happens once.

---

## Summary

| requirement | result |
|---|---|
| every scene generated in code, no image data, no licence question | yes — 3/3 |
| output length `size*size`, `Float64Array` | 6/6 pass |
| range inside `[0, 1]` | 6/6 pass |
| determinism, same seed → identical bytes (incl. a fresh module instance) | 12/12 pass |
| radially averaged power spectrum shows multi-scale content | 6/6 pass, every radial band populated |
| binary pair resolved by the diffraction-limited PSF | 8.95 % dip at 256, 9.45 % at 512 |
| binary pair merged by a strongly aberrated PSF | single peak, no interior minimum, both sizes |
| binary pair split again after correction | 7.95 % at 256, 8.38 % at 512 |
| separation stated, and just inside the Rayleigh limit | 3.90294 px = 0.95 × 4.10836 px |
| point sources sub-pixel and band-limited | 0.294 % worst-case shift-invariance error vs 25.4 % for a hot pixel |
| terrain has real detail at several scales, not TV static | power law from k = 8 to Nyquist, 6 octaves 128 → 4 px, plus 110 craters |
| timing at 256 and 512 | reported above, single-call minima |
| `convolvePsf` unchanged and still exact | 18/18 re-confirmations at 1e-12 |

**92 checks, 0 failures.**
