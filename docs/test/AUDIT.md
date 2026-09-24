# Adversarial audit of `src/optics.js` against `assets/GOLDEN.md`

**Auditor:** independent review, assuming the port is wrong until proven otherwise.
**Date:** 2026-09-15 · **Node** v22.14.0 · **Python** 3.11.11 (numpy 2.3.3, torch 2.8.0+cu126, scipy 1.15.3)

## Verdict

**CORRECT.** The port reproduces an independently written float64 reference — built on the real
`pupil_design` package and validated bit-for-bit against `golden.npz` first — on **29 fresh pupils
× 42 fresh phases** that appear nowhere in `golden.json`, to **1e-15 … 3.6e-14**, i.e. 300×–10^6×
inside every GOLDEN.md §10.2 bound. No discrepancy attributable to the port was found.

Nine problems *are* recorded below. All nine are in the **harness**, the **spec**, or the
**implementer's report** — none is a defect in `optics.js`.

---

## 1. Line-by-line review of `optics.js` against GOLDEN.md

Every trap named in GOLDEN.md Appendix B was checked in the source and then re-checked by injecting
the corresponding bug (§3).

| GOLDEN.md requirement | `optics.js` | verdict |
|---|---|---|
| §2.1 circle: `4096·((2x−127)²+(2y−127)²) ≤ 2413²` → 1124 px | L61-83: `N*N*(dx²+dy²) ≤ (2·r·(N−1))²`, i.e. both sides ×4 of the spec form | ✔ algebraically identical; max LHS 5.3e8 ≪ 2^53 |
| §2.2 disk: `(j−64)²+(i−64)² ≤ 361` → 1129 px | L124-133 | ✔ 1129, and **not** conflated with the circle |
| §2.2/§3.3 `theta = atan2(y_ROW, x_COL)` | L131 `Math.atan2(dy, dx)`, `dy = i − 64` (row) | ✔ landmarks verified: θ(row 64,col 83)=0, θ(row 83,col 64)=+π/2, θ(row 45,col 64)=−π/2 |
| §3.2 **no** `sqrt(2(n+1))` Noll factor | L162-165 raw integer factorials, one division in double | ✔ `R_at_r` matches to 0.000e+0 |
| §3.3 `m<0 ⇒ R(n,|m|)·sin(|m|θ)` | L189 `R * Math.sin(a * theta)`, `a = |m|` | ✔ |
| §3.4 disk applied last | L215-226 evaluation restricted to the disk (bit-identical, `0×finite=0`) | ✔ |
| §4.1 unnormalised forward FFT | L353-377, no 1/N anywhere | ✔ circle PSF peak `=== 1263376` exactly |
| §4.2 both shifts = roll by 64 | folded into gather/scatter indices L432-436, L451-455 | ✔ |
| §4.4 DL peak `= (ΣM)²` exactly | | ✔ exact for all 6 golden + 26 ladder + 26 of 29 fresh masks (see P-3) |
| §4.5 Parseval `Σpsf = 16384·ΣM` | | ✔ worst 4.0e-16 rel over 52 fresh cases |
| §5.1 order: PSF → ÷1263376 → +σ·noise → clamp 0 → ÷max(peak,1e-6) | L536-586 in that order | ✔ pinned by the noise probe and by mutations M26/M27/M28 |
| §6 same mask both sides, residual masked first, eps floors denominator only | L613-622 | ✔ |
| §7 divide by **Σmask** twice, re-mask after mean subtraction | L636-652 | ✔ |
| §8 raw mask, **X·X** not X·conj(X), `sy=k//W, sx=k%W` for *both* axes, **+1**, circular roll, mod N | L674-716 | ✔ circle scores exactly `0` |
| §9 predictor `0.9·gt + 0.3 + 0.15·(x−64)(y−64)/4096` | L764-774, same grouping | ✔ bit-for-bit vs `pred_synth_f64` |
| §10.3 argmax = **first** flat index, strict `>` | L459, L702, L730 | ✔ (but untested — see P-2) |

**All six normative clamp floors are literally correct** (`1e-4` §4.6 L487/L558, `0.0` §5.2 L575,
`1e-6` §5.2 L583, `1e-4` §6 L621, `1e-9` §7 L642, `1e-12` §8 L678).

Two things worth knowing that the source gets right but that no test could ever see:

* **The pre-FFT `ifftshift` is unobservable.** Only `|F|²` is consumed, and a circular shift of the
  input changes only a linear phase. Injecting a one-row offset into the pre-shift changes the PSF by
  **6.1e-16 rel-to-peak** over 145 fresh cases — i.e. nothing. Only the *post*-FFT `fftshift` is testable
  (mutation M16 is caught instantly). "fftshift direction errors" are half a non-problem here.
* **`zernikeBasis` (the cached hot path) is bit-identical to the elementwise `zernike()`** that Stage 1
  spot-checks: worst difference **0** over all 37,257 (pixel, mode) values. So Stage 1's coverage does
  transfer to the phase path; they are not two implementations that could drift apart.

### Residual traced to its source

The ~2e-14 phase deviation vs numpy is **not** algorithmic. On the 1129 disk pixels, `r` is
**bit-identical** to numpy; `theta` differs on **166/1129 pixels by exactly 1 ULP** (4.44e-16) — V8's
`Math.atan2` vs glibc's. Amplified through 33 modes and coefficients up to 5 that is 2e-14, i.e. 500×
inside the §10.2 phase bound of 1e-11.

---

## 2. Is the harness vacuous?

Re-run: `node test/verify.mjs` → **exit 0**, `ALL CHECKS PASSED (76 tracked quantities,
4,373,471 individual comparisons, golden.npz whole-array coverage ON)`. Both numbers verified:
76 `track()` calls, 76 rows printed, so **no row was silently dropped** and every tracked quantity
really executed.

Coverage is genuine, not padded:

* 7 of the 10 stages compare **full 128×128 arrays** from `golden.npz`, not just the 256-point `core16`
  sample (393,216 comparisons per 24-case quantity).
* No `try`/`catch` anywhere in `verify.mjs`; no failure can be swallowed.
* Only three `continue` statements exist and none skips a hard case (one is an inner-loop zero skip,
  two skip ladder frames that have no `pool_id` in `pupils.json`).
* Stage 6 (noise probe) runs and carries its own **negative control**: re-running with
  `circlePeakDivisor = 1` is asserted to *break* the `(65,112)` argmax. It does.
* Stage 9 exercises three branches the golden set has no vectors for (§5.4 degeneracy, the §5.3 legacy
  path, and the clamps).

But see **P-1**: the harness has no guard that all 76 rows actually ran.

---

## 3. Independent mutation testing — 33 mutations, 24 caught, 9 survived

Done on a **copy** in a scratch directory (`mut/`), never on the project tree. 18 in batch 1,
15 in batch 2. At least 12 are mutations the implementer's list could not have contained (clamp-constant
perturbations, the argmax tie-break rule, the shift-invariance probe, the boundary `<`/`<=` pair, the
non-conjugated inverse FFT, the activeRows-hint corruption).

### Caught (24)

| # | mutation | caught by |
|---|---|---|
| M1 | `atan2(y,x)` → `atan2(x,y)` | `phase.stats.argmax`, `phase.samples.core16` |
| M2 | add `sqrt(2(n+1))` Noll normalisation | `zernike.R_at_r` (Stage 1) |
| M3 | asymmetry `X·conj(X)` instead of `X·X` | `mask.alpha_corrected.f64`, circle-is-0 assertion |
| M4 | drop the `+1` in `align="corrected"` | `mask.alpha_corrected.f64`, circle-is-0 assertion |
| M5 | MSE ÷ 16384 instead of ÷Σmask | `mse vs .f64` |
| M6 | skip the 1263376 divisor | `noisy.stats.argmax` (Stage 6 only — as §5.4 warns) |
| M7 | reference circle built on the disk rule (1129) | `geometry.integer_counts` |
| M8 | row/column transposition of the pupil in `psfCore` | `psf.stats.argmax`, `psf.peak_before_normalization` |
| M9 | MSE: forget the re-mask after mean subtraction | `mse vs .f64`, **`piston-invariance`** |
| M13 | renorm floor `1e-6` → `1e-7` | `clamp 1e-6 (renorm) fires when tiny` (Stage 9) |
| M14 | swap `sy`/`sx` when decoding the autoconv argmax | `mask.alpha_corrected.f64` |
| M16 | post-FFT `fftshift` off by one row | `psf.stats.argmax` |
| M17 | wavefront support = circle (1124) instead of disk (1129) | `phase.stats.min/max`, `phase.stats.argmax` |
| M18 | `sin(m·θ)` instead of `sin(|m|·θ)` for m<0 | `zernike.Z_at_r_theta` |
| M19 | `activeRows` hint set on the pre-shift row | `geometry.integer_counts` (circle PSF peak ≠ 1263376) |
| M22 | predictor divisor 4096 → 4095 | `strehl vs .f64`, `mse vs .f64` |
| M23 | throughput ÷1129 instead of ÷1124 | `mask.throughput` |
| M25 | complex pupil `exp(−iφ)` instead of `exp(+iφ)` | `psf.stats.argmax` |
| M26 | noise added **before** the divisor | `noisy vs npz noisy_f64` |
| M27 | drop `clamp_min(0.0)` after noise | `noisy.stats.min/max` |
| M28 | noise added without the σ factor | `noisy.stats.sum/sum_sq` |
| M34 | Zernike disk boundary `<=` → `<` | `phase.disk_px` (1129 → fewer) |
| M36 | inverse-FFT twiddle not conjugated | `mask.alpha_corrected.f64` |
| M37 | renorm divisor perturbed by 1 part in 1e12 | `S5.4 degeneracy` (Stage 9) |

### Survived — four are provably null changes

| # | mutation | why it survives |
|---|---|---|
| M15 | Strehl: do **not** mask the residual | `psfCore` skips every `mask===0` pixel *before* it reads `phase[]`, so out-of-pupil residual is never evaluated. GOLDEN.md §6's "mask first" is enforced structurally. **Null.** |
| M24 | pre-FFT `ifftshift` off by one row | `\|F\|²` is invariant under a circular shift of the input. Measured: **6.1e-16 rel-to-peak** over 145 fresh cases. **Null.** |
| M33 | reference-circle boundary `<=` → `<` | no pixel attains the threshold exactly (§2.1: values jump 1418 → 1450 across 1421.5256). **Null.** Contrast M34, which *is* caught, because `x²+y² = 361` **is** attained. |
| M31 | `Math.hypot` → `sqrt(re²+im²)` in the asymmetry argmax | no argmax moved on any of the 61 masks tested. **Benign**, though it sits on the same knife-edge as the documented ties. |

### Survived — five are real harness blind spots (P-2)

| # | mutation | normative source that is never exercised |
|---|---|---|
| M10 | argmax tie-break `>` → `>=` | **GOLDEN.md §10.3** "`argmax` ties break to the lowest flat index … must use strict `>`" |
| M11 | §4.6 self-peak floor `1e-4` → `1e-6` | **§4.6** |
| M12 | §6 Strehl denominator floor `1e-4` → `1e-2` | **§6** |
| M20 | §8 mass floor `1e-12` → `1e-9` | **§8 step 1** |
| M21 | §7 `mask_sum` floor `1e-9` → `1e-3` | **§7** |

The shipped code has all five **right**; the harness simply could not tell.

---

## 4. Fresh cases — the real defence against overfitting

**Method.** An independent float64 reference was written on top of the real `pupil_design` package
(`optics.zernike`, `random_zernike_phase`, `pupils.generate_polygon_pupils`,
`pupils.generate_binary_pupils`, `asymmetry.asymmetry`, `losses.masked_piston_corrected_mse_per_sample`),
with the PSF/measurement chain in numpy complex128 (the package's own `psf_from_pupil` hard-casts to
`complex64`, so it cannot produce the `f64` column). **That reference was first validated against
`golden.npz`: phases, PSFs, measurements, noisy measurements and Strehl all reproduce to max difference
exactly `0`; MSE to 6.3e-16 rel; all six `alpha_corrected` / `alpha_published` / autoconv argmaxes exact.**

**Inputs — none of them in `golden.json`:**

* **29 fresh pupils** — 8 convex-polygon pupils (`generate_polygon_pupils`, seed 20260915), 8 threshold
  pupils (`generate_binary_pupils`, torch seed 20260915), and 13 hand-built adversarial masks (the
  1129-px Zernike disk as a *pupil*, an off-centre circle, column-split vs row-split half-disks, a
  5×20 non-square rectangle, a chiral L, an annulus, a comma, one pixel, two pixels, a 3-px sliver, a
  checkerboard, a diagonal bar). **`alpha_corrected` spans 0.000 → 0.638**, well past the golden set's
  0.000 → 0.354.
* **42 fresh phases** — all 33 single-mode unit-coefficient wavefronts (the sharpest possible test of
  §3.2 normalisation and §3.3 sign), 6 random draws at `coeff_scale` 0.5/1/2/2/3/5, all-ones,
  alternating ±1, and an `n=7, |m|=7` pair.
* 52 full-array PSF/measurement cases, **1,218 scalar cross cases** (29×42), 58 noisy cases with a
  **fresh** noise field (`default_rng(20260917)`) at σ = 0.05 and 0.20.

**Result — 7,297 comparisons, worst deviations:**

| quantity | worst deviation | §10.2 bound | margin |
|---|---|---|---|
| phase vs reference (full array, abs) | **3.55e-14** | 1e-11 | 280× |
| phase `sum_sq` (rel) | 5.36e-16 | 1e-11 | 2×10^4 |
| PSF vs reference (full array, rel-to-peak) | **1.13e-15** | 1e-9 | 9×10^5 |
| PSF peak (rel), 1,218 cross cases | 1.75e-15 | 1e-9 | 6×10^5 |
| Parseval `Σpsf = 16384·ΣM` (rel) | 4.03e-16 | 1e-11 | 2×10^4 |
| measurement vs reference (abs) | **1.22e-15** | 1e-9 | 8×10^5 |
| noisy measurement vs reference (abs) | 7.77e-16 | 1e-9 | 1×10^6 |
| Strehl (abs), 1,218 cases | **7.77e-16** | 1e-9 | 1×10^6 |
| MSE (rel), 1,218 cases | **5.80e-15** | 1e-9 | 2×10^5 |
| piston-invariance 0.3 → 7.0, 1,218 cases | 7.77e-16 | 1e-9 | — |
| `alpha_corrected`, 29 masks | **0 (exact)** | 1e-12 | exact |
| `throughput`, 29 masks | **0 (exact)** | 1e-12 | exact |
| DL peak `= (ΣM)²`, 26 masks with a unique DC max | **0 (bit-exact)** | exact | exact |
| PSF argmax, 1,092 cross cases with a unique max | **0 mismatches** | exact | exact |
| noisy argmax + `clamp_min(0)` zero-counts, 58 cases | **0 mismatches** | exact | exact |

**Six apparent failures on the first pass were all tie / cancellation artefacts, each run to ground:**

* `two_pixels` DL peak `4.000000000000001` vs "exactly 4": that mask has **256 exactly-tied maxima**,
  and numpy itself returns `4.000000000000002` — the JS is marginally *closer* to 4 (see P-3).
* `mode8_4_-2` phase argmax: 4 exactly-symmetric maxima at (48,54), (54,48), (74,80), (80,74) separated
  by 2e-16 of float noise, while the full arrays agree to 3.6e-14. This is the case that most *looks*
  like a transposition bug and is not one.
* `mode24_6_6` "sum relative error 6.0": the true sum is 0 and numpy's is 2.66e-15 — pure cancellation.
  Against an RMS scale the deviation is 2.1e-17.

**Anti-transposition / anti-mirror discrimination.** The element-wise agreements above are only
meaningful if the arrays are not accidentally symmetric. They are not:

| case | \|JS − reference\| / peak | \|JS − JSᵀ\| / peak | \|JS − flipY\| / peak |
|---|---|---|---|
| `half_disk_rows__alternating` (PSF) | 1.1e-15 | **9.6e-1** | 3.0e-1 |
| `comma__rand2_s2.0` (PSF) | 8.8e-16 | **8.6e-1** | 9.6e-1 |
| `rect_5x20__mode14_5_-3` (PSF) | 9.2e-16 | **8.6e-1** | 4.9e-2 |
| `mode31_7_-7` (phase, abs / range 2.0) | 2.3e-15 | **1.39** | 1.69 |
| `rand2_s2.0` (phase, abs / range 14.3) | 1.6e-14 | **11.4** | 8.9 |

A transposed or mirrored port would be off by 10–100 % of peak. It is off by 1e-15.

**Bonus: non-binary pupil amplitude**, a path the golden set never exercises (all six masks are binary).
An apodised pupil `circle × U(0.2,1.0)` agrees with the package reference on every quantity to ~1e-15
relative (`sum_amp` 671.5506979132913 vs …917; Strehl 0.980789884880569 vs …5688; MSE 0.018811437181641077
vs …4099; `alpha_corrected` 0.4089206469714898 vs …8955).

---

## 5. The argmax-tie claim — verified, and the reported magnitude is understated

The implementer's claim is **true but incomplete**.

**(a) The ties are genuine exact integer ties.** Computing the circular autoconvolution in exact
integer arithmetic (`rint` of the float64 FFT, cross-checked against a brute-force integer shift sum):

| mask | tied shifts | exact autoconv value | torch f64 floats |
|---|---|---|---|
| `golden:pool_1473` | (127,6), (127,7) | **230** | 229.99999999999994 / 229.99999999999997 |
| `golden:triangle` | (124,0), (126,0) | **329** | 329.0 / 328.9999999999999 |
| `ladder:20` | (126,7), (127,7) | **272** | 272.00000000000006 / 272.0 |
| `ladder:22` | (122,121), (123,121), (123,122) | **236** | all 236.000000000000xx |
| `ladder:23` | (127,6), (127,7) | **230** | 230.0 / 230.00000000000009 |

Across 61 masks (29 fresh + 6 golden + 26 ladder), **14 have a multi-way exact tie**. The JS argmax
disagrees with the package's float64 argmax on **4 of 61**, and in every one of those four the two
candidates are exactly tied integers. It is a tie-break coin-flip on ~3e-14 of FFT rounding, exactly as
GOLDEN.md §8 documents. Not a bug.

**(b) `alpha_corrected` is not merely "unaffected" — it is structurally incapable of being affected.**
With `sy = k_y+1, sx = k_x+1`, the overlap in §8 step 9 evaluates to `C[(sy−1) mod N][(sx−1) mod N] =
C[k_y][k_x]`, which is the autoconvolution **maximum** itself. Hence

> `alpha_corrected = 1 − max(C) / Σmask`

with no dependence on *which* maximiser wins. Verified numerically: **worst |alpha_corrected − closed
form| = 0 across all 61 masks**, and `alpha_corrected` from the JS matches the package's float64
`asymmetry(..., "corrected")` with **worst deviation exactly 0** on all 61. `alpha_published` instead
reads `C[(k−1) mod N]`, a neighbour that is *not* tied — which is the whole story.

**(c) Nothing else consumes the tie-broken argmax.** `autoconvArgmax` is a diagnostic export with no
internal caller; `asymmetry()` is called only by the two wrappers. The PSF / measurement / Strehl / MSE
chain never touches the asymmetry.

**(d) The reported magnitude is 6× too small — see P-6.** The implementer reports "up to 2.8e-3".
Measured against the package's float64 `asymmetry(..., "published")` over 61 masks, the worst divergence
is **1.681e-2** (ladder frame 22). Consequence for the shipped product: none —
GOLDEN.md §8 and App. B #10 say to ship `alpha_corrected` only, and the source does.

---

## 6. Performance — claim confirmed

Measured in a separate process with single-call minima plus two calibration probes, three independent
runs, on a box at load ≈147/128 cores.

Calibration: native **sha256 1701 MB/s** (unshared-core reference ≈2000 → this process is getting ~85 %
of a core) and **0.83 ns per scalar mul-add** (healthy ≈0.85). The timings are therefore representative,
not load-inflated.

| op | best (ms) | p10 | median | p90 |
|---|---|---|---|---|
| `synthPhase` | 0.060–0.074 | 0.062–0.078 | 0.081–0.103 | 0.115–0.117 |
| `psfFromPupil` | 0.658–0.763 | 0.686–0.777 | 0.698–0.789 | 0.748–0.809 |
| `simulateMeasurement` | 0.714–0.813 | 0.736–0.833 | 0.781–0.845 | 0.800–0.863 |
| **FULL `runPipeline`** | **0.811–0.874** | 0.831–0.892 | **0.847–0.903** | 0.869–0.989 |
| `asymmetryCorrected` | 1.845–1.860 | — | 1.913–1.935 | 1.966–1.985 |

**The "0.84 ms median" claim stands** (independently 0.85–0.90 ms median, 0.81–0.87 ms best). It is a
*warm, cached steady state*; see **P-7** for the cold-start caveat.

---

## 7. Problems found

None of these is a numerical defect in `optics.js`.

### P-1 — The harness can lose a whole stage and still print "ALL CHECKS PASSED" · severity: MEDIUM (harness)

`verify.mjs:936` — `const table = rows.filter((r) => r.n > 0)` — drops any tracked row that never
executed, from the report **and** from the pass/fail decision. Nothing asserts that all 76 rows ran.

Demonstrated: injecting `G.zernike_spot_checks.modes = []` right after the golden load yields

```
ALL CHECKS PASSED  (74 tracked quantities, 4,373,075 individual comparisons, ...)   exit 0
```

Stage 1 — the stage GOLDEN.md §11 singles out as isolating the Noll normalisation and the sign
convention *before any FFT* — vanishes without a word. A truncated or partially regenerated
`golden.json` would silently cut coverage while still reporting success. (Emptying `cases`, `masks` or
`noise_probe.results` happens to crash instead, so only some stages have this exposure.)

*Fix:* one line — `assert(rows.every(r => r.n > 0), 'a tracked quantity never executed')`, plus pinning
the expected row count.

### P-2 — Five normative constants/rules are never exercised · severity: MEDIUM (harness)

Mutations M10, M11, M12, M20, M21 all pass the full harness (§3). The source has all five right, but the
harness would not notice if a future edit got them wrong:

* **§10.3 first-index argmax tie-break** (M10). Nothing in the golden set has a tied PSF or measurement
  maximum. A decisive one-line test exists: a single-pixel pupil gives a PSF that is **exactly `1.0` in
  all 16,384 bins** (verified), so `argmax` must be `0` and a `>=` scan returns `16383`.
* **§4.6 `1e-4`** (M11) — Stage 9 hard-codes `1e-4` on *both* sides of its comparison, and the smallest
  golden PSF peak is 16,530, so the floor is never the operative term.
* **§6 `1e-4`** (M12) — see P-3 below; the row that claims to test it does not.
* **§8 `1e-12`** (M20) and **§7 `1e-9`** (M21) — never bind (`mass ≥ 1`, `mask_sum ≥ 356`).

### P-3 — The harness row `clamp 1e-4 (strehl denom) fires` does not test `1e-4` · severity: LOW (harness, misleading label)

`verify.mjs:824` probes `strehlRatio(empty, 0, 0)`. With an empty mask the **numerator** peak is 0, so
the result is `0 / max(0, eps) = 0` for *any* eps > 0. Verified: the row passes with eps = 1e-4, 1e-2 and
1e+9 alike. The row name asserts coverage it does not provide. A real test needs a mask whose
`peak_ref` is below the floor, e.g. a non-binary amplitude of 0.005 on one pixel.

### P-4 — GOLDEN.md §5.4's "1.1e-16" is unattainable · severity: LOW (spec inaccuracy)

§5.4 states `max |measurement − psf/psf.max()| = 1.1e-16` over the 24 cases. Recomputed with the numpy
float64 reference: **2.220446e-16** (one ULP at 1.0, not a half). The harness gates this at `1e-15`,
which its comment presents as GOLDEN.md's number — a silent ~9× loosening. The loosening is *justified*
(the reference itself exceeds the stated figure), but it is undocumented, and it is the one place in the
harness where a stated tolerance was relaxed rather than met. The port's own value is 2.22e-16, identical
to numpy's.

### P-5 — GOLDEN.md §4.4's "(ΣM)² exactly" is conditional · severity: LOW (spec scope)

§4.4 asserts the diffraction-limited peak is `(ΣM)²` bit-exactly. That holds only when the DC bin is the
**unique** maximum. For a 2-pixel pupil at (40,30) and (90,100), 256 bins tie exactly at 4 and numpy
returns `4.000000000000002` (the JS returns `4.000000000000001`). 26 of my 29 fresh masks have a unique
DC max and are bit-exact in both implementations; the other 3 are degenerate by construction. Worth a
sentence in §4.4 so a future porter does not chase it.

### P-6 — The implementer understates the `alpha_published` divergence by 6× · severity: LOW (report accuracy)

Reported: "up to 2.8e-3". Actual worst against the package's float64 asymmetry over 61 masks:
**1.681e-2** (ladder frame 22), with 4/61 masks disagreeing (`pool_1473` 2.809e-3, `ladder:20` 7.614e-3,
`ladder:22` 1.681e-2, `ladder:23` 2.809e-3). The implementer saw only the two frames the harness happens
to flag, because the ladder branch compares against `ladder.json`, which stores the **torch float32**
alpha — a different reference from the float64 column §10.2 actually specifies. No product impact
(`alpha_published` is not shipped), but the number in the report is wrong.

### P-7 — The 0.84 ms figure is warm-only; first call is ~22 ms · severity: LOW (claim scope)

The first `runPipeline` in a fresh process takes **21.8–23.2 ms** (three runs), ~26× the steady state:
`zernikeBasis` builds 1129 × 33 = 37,257 entries with `Math.pow`/`cos`/`sin`, and V8 has not tiered up.
Correct to amortise for a slider, but a first paint costs ~20 ms, and the harness's own benchmark warms
up 400 times before measuring so it never shows this.

### P-8 — `zernikeBasis` cache is keyed on array *identity* · severity: LOW (performance foot-gun)

`_basisCache` is a `WeakMap` keyed on the `modes` array (`optics.js:243`). `PAPER_MODES` is frozen and
module-level, so the default path is fine — but a caller passing a fresh array literal (e.g.
`runPipeline(mask, c, { modes: [[2,0], ...] })`) rebuilds the 37k-entry basis **on every call**, turning
0.85 ms into ~20 ms with no warning.

### P-9 — Shared mutable module state · severity: LOW (robustness)

* `_psfScratch` (`optics.js:406`) is a module-level scratch shared by every `psfCore` call. Safe today
  (no `await`/callback between allocation and use, and the returned arrays are fresh), but any future
  re-entrancy — a generator, a worker that yields, an `await` inserted into the PSF path — silently
  corrupts results.
* `referenceCircle` returns a **cached** `Uint8Array` (`optics.js:80`, commented "treat as read-only").
  An exported API that hands out a shared mutable buffer will eventually be mutated by a caller, which
  would poison `referenceCircleArea` and therefore every `throughput`.

### Not problems (checked and cleared)

`obsExcess` / `obsExcessArray` gate the port's float64 output against the golden **float32** data on the
excess over each record's own reported f32-vs-f64 spread. This is a *derived* tolerance not in §10.2, and
it is weak by construction (it can only fail if the port is further from the f32 value than the f64
reference is). But it is an **additional** row, redundant with the strict §10.2 f64 comparisons that run
alongside it — not a relaxation of any gate.

The ladder/pupils tolerance `5.25e-7` looks suspiciously close to its observed worst (5.108e-7, only 3 %
headroom), so it was checked independently: `ladder.json` stores the **torch float32** `alpha_corrected`
rounded to 6 decimals. For the worst frame (slot 0), `|f32 − json| = 4.984e-7` (6-dp rounding) and
`|f32 − f64| = 1.24e-8`, summing to exactly the observed 5.108e-7. The decomposition is exact and the
stated provenance is honest. In any case it is moot: the JS `alpha_corrected` matches the package's
float64 value **exactly (0)** on all 61 masks tested.

Only **28 of the 1,500** pupils in `pupils.json` are cross-checked against a decoded bitmap — the other
1,472 have no bitmap in the assets, so this is a limit of the data, not of the harness.

---

## 8. What would have caught it, had it been wrong

Recorded so a future reviewer does not have to re-derive the coverage map:

* **Zernike normalisation / sign** (§3.2, §3.3) → `zernike.R_at_r`, `zernike.Z_at_r_theta` (Stage 1),
  *before any FFT*. M2 and M18 die here.
* **`atan2` argument order** (§3.3) → the 33 single-mode fresh phases, where odd-|m| modes differ from
  their own transpose by 1.3–1.4 rad out of a 2.0 range while matching the reference to 2e-15. M1 also
  dies on `phase.stats.argmax`.
* **The two grids** (§2) → `geometry.integer_counts` (1124 / 1129 / 1100 / 24 / 29) and `phase.disk_px`.
  M7, M17, M34 die here.
* **FFT normalisation and the post-shift** (§4.1, §4.2) → `circlePSF.max() === 1263376` exactly and
  `argmax === 8256`, plus Parseval. M16, M19 die here.
* **Row/column transposition** → `psf.stats.argmax` on any asymmetric pupil, plus the fresh
  `half_disk_cols` / `half_disk_rows` / `rect_5x20` pair whose PSFs differ from their transposes by
  86–96 % of peak. M8 dies here.
* **The 1263376 divisor and the operation order** (§5.1) → *only* the Stage 6 noise probe, exactly as
  §5.4 warns. M6, M26, M27, M28 die there and nowhere else.
* **MSE denominator and the re-mask** (§7) → `mse vs .f64` and the piston-invariance check. M5, M9 die.
* **`X·X` vs `X·conj(X)`, and the `+1`** (§8) → `asymmetryCorrected(circle) === 0` exactly. M3, M4,
  M14, M36 die on that single assertion.
* **Overfitting to the golden set** → the 29 × 42 fresh cross-check of §4, whose pupils span
  `alpha_corrected` 0.000–0.638 (vs the golden set's 0.000–0.354) and whose phases include every single
  Zernike mode in isolation.

The five things that would **not** have been caught are M10, M11, M12, M20, M21 (P-2), and the harness
would not have told you a stage went missing (P-1).

---

## 9. Reproducing this audit

Everything ran in a scratch directory; **`src/`, `test/` and `assets/` were not modified**
(md5sums re-verified after all mutation work: `ladder.js`, `optics.js`, `npz.mjs`, `verify.mjs` all OK),
and `/depot/chan129/users/aalmuall/pupil-design` was never written to. Mutations were applied to a copy
under `scratchpad/mut/` with `assets/` symlinked. `test/AUDIT.md` (this file) is the only addition.

*(Note: `pupil-page/content/` and `pupil-page/figures/` appeared during this session at 11:25–11:36 from
a concurrent process; they are not part of this audit and were not touched.)*

Final confirmation: `node test/verify.mjs` → **exit 0**, `ALL CHECKS PASSED (76 tracked quantities,
4,373,471 individual comparisons, golden.npz whole-array coverage ON)`.
