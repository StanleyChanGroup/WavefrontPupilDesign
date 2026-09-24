# fp16 ONNX verification -- `unet_fp16.onnx`

Date: 2026-09-14 | Verifier: Claude Code | Status: **PASS (with one backend caveat)**

## 1. Provenance / characterisation of the shipped file

`/depot/chan129/users/aalmuall/pupil-page/assets/unet_fp16.onnx` (15,403,693 B)

| property | unet_fp32.onnx | unet_fp16.onnx |
|---|---|---|
| producer | pytorch 2.8.0, ir 8, opset 17 | pytorch 2.8.0, ir 8, opset 17 |
| graph input `input` | FLOAT [batch,2,128,128] | **FLOAT** [batch,2,128,128] |
| graph output `phase` | FLOAT [batch,1,128,128] | **FLOAT** [batch,1,128,128] |
| nodes | 38 | 40 |
| op types | Conv 15, Relu 14, ConvTranspose 3, MaxPool 3, Concat 3 | same + **Cast 2** |
| initializers | 36, all FLOAT | 36, all **FLOAT16** |
| internal value_info | - | 39, all FLOAT16 |

The two Cast nodes sit exactly at the boundary:

```
Cast input_cast_to_/enc1/conv/conv.0/Conv : input  FLOAT -> FLOAT16
Cast /final/Conv_cast_to_phase            : ...    FLOAT16 -> FLOAT  -> phase
```

**Verdict on provenance: correct, no regeneration needed.** This is exactly the shape
`onnxconverter_common.float16.convert_float_to_float16(..., keep_io_types=True)` produces:
fp32 graph I/O, fp16 weights and interior, casts only at the boundary. It is *not* a naive
`.half()` export with fp16 I/O (which would show FLOAT16 graph input/output and no Cast nodes).

Weight check against `unet_fp32.onnx`: all 36 initializer names match; 7,696,735 of 7,696,769
values are the exact fp16 rounding of the fp32 value. The 34 exceptions are
`onnxconverter_common`'s `min_positive_val=1e-7` flush applied to weights of magnitude ~1e-8
(e.g. 3.76e-08 -> 1.19e-07). No infinities were produced (max |weight| is 68.48, far under the
fp16 ceiling of 65504). 17,466 values land in the fp16 subnormal range.

Note: the file differs by SHA-256 from the copy in the working scratchpad, but only because the
shipped one carries full shape inference (39 value_info entries vs 2). Node graph, initializer
dtypes and weight values are equivalent.

## 2. Verification on real data

300 (pupil, phase) pairs from the **TEST** split of
`dataset_jan2026_bbox_128_radius_19`: 300 **distinct** pupils x 300 **distinct** phases.
Measurements built with `data.simulate_measurement` under the published config
(`noise_sigma=0.0`, `normalize_by_peak=True`, `use_throughput=True`,
`renormalize_after_noise=True`, `coeff_scale=2.0`, 33 PAPER_MODES).

Sampling spans asymmetry deliberately. Corrected alpha (`asymmetry(mask, align="corrected")`)
was computed for all 1500 test pupils; note the pool is **not** stored in asymmetry order, so
index-linspace would not span it. Pupils were split into terciles of corrected alpha and 100
drawn evenly across each band:

| band | corrected alpha range | n |
|---|---|---|
| low | 0.0144 - 0.1179 | 100 |
| mid | 0.1183 - 0.2370 | 100 |
| high | 0.2372 - 0.3539 | 100 |

Reference levels (PyTorch fp32 CPU): mean masked piston-corrected MSE = 0.8448 rad^2;
mean Strehl = 0.7050 (range 0.0644 - 0.9903).

### (b) onnxruntime fp32  vs  (a) PyTorch fp32

| quantity | MEAN | WORST |
|---|---|---|
| raw predicted phase, max abs dev (rad) | 2.419e-04 | 4.798e-04 |
| raw predicted phase, RMS dev (rad) | 2.420e-05 | 3.477e-05 |
| **in-pupil** max abs dev (rad) | 6.997e-06 | 4.029e-05 |
| **in-pupil** RMS dev (rad) | 1.266e-06 | 6.295e-06 |
| d(masked piston-corr MSE), absolute | 6.094e-07 | 1.574e-05 |
| d(masked piston-corr MSE), relative | 1.166e-06 | 7.039e-06 |
| **d(Strehl), ABSOLUTE** | **1.569e-07** | **1.550e-06** |
| d(Strehl), signed mean (bias) | +3.303e-09 | - |

### (c) onnxruntime fp16  vs  (a) PyTorch fp32

| quantity | MEAN | WORST |
|---|---|---|
| raw predicted phase, max abs dev (rad) | 9.673e-02 | 2.211e-01 |
| raw predicted phase, RMS dev (rad) | 9.783e-03 | 1.663e-02 |
| **in-pupil** max abs dev (rad) | 4.034e-03 | 2.534e-02 |
| **in-pupil** RMS dev (rad) | 7.326e-04 | 5.563e-03 |
| d(masked piston-corr MSE), absolute | 3.644e-04 | 1.268e-02 |
| d(masked piston-corr MSE), relative | 6.322e-04 | 3.573e-03 |
| **d(Strehl), ABSOLUTE** | **8.348e-05** | **1.726e-03** |
| d(Strehl), signed mean (bias) | +8.352e-06 | - |

ONNX conversion error (b) is ~3 orders of magnitude below fp16 precision error (c) on every
metric -- worst Strehl deviation 1.55e-06 vs 1.73e-03. The graph conversion is effectively exact;
essentially all of (c) is fp16 precision.

Context for the raw (unmasked) numbers: outside the pupil the network output is unconstrained and
large (mean |phase| 2.04 rad, max 16.2), versus mean 1.26 rad inside. The raw deviation column is
therefore dominated by pixels the demo should be masking off; the in-pupil rows are the meaningful
ones, and there the mean worst-pixel error is 4.0e-03 rad against a ~1.26 rad signal (~0.3%).

## 3. Breakdown by asymmetry band (fp16 vs PyTorch fp32)

| band | corrected alpha | n | mean Strehl (fp32) | signed mean d(Strehl) | mean abs d | WORST abs d |
|---|---|---|---|---|---|---|
| low | 0.0144-0.1179 | 100 | 0.4067 | +1.026e-05 | 1.140e-04 | 8.016e-04 |
| mid | 0.1183-0.2370 | 100 | 0.7812 | -3.396e-06 | 9.077e-05 | 1.726e-03 |
| high | 0.2372-0.3539 | 100 | 0.9272 | +1.819e-05 | 4.564e-05 | 2.245e-04 |

| band | mean MSE (fp32) | signed mean rel d(MSE) | WORST rel d(MSE) |
|---|---|---|---|
| low | 1.9902e+00 | -8.431e-05 | 3.365e-03 |
| mid | 4.6011e-01 | -1.172e-05 | 3.573e-03 |
| high | 8.4016e-02 | -2.286e-04 | 3.229e-03 |

**fp16 does not preferentially damage high-asymmetry pupils -- the opposite.** Mean absolute
Strehl deviation falls monotonically with asymmetry (1.14e-04 low, 9.08e-05 mid, 4.56e-05 high),
worst case in the high band is 2.25e-04 (7.7x smaller than the mid-band worst), and
corr(corrected alpha, |d Strehl|) = -0.2278. The demo's central trend -- Strehl rising with
asymmetry, 0.4067 -> 0.7812 -> 0.9272 across the bands -- is preserved to within 2e-05 per band:

| band | mean Strehl fp32 | mean Strehl fp16 | shift |
|---|---|---|---|
| low | 0.406743 | 0.406753 | +1.026e-05 |
| mid | 0.781183 | 0.781179 | -3.396e-06 |
| high | 0.927214 | 0.927232 | +1.819e-05 |
| **all** | 0.705047 | 0.705055 | +8.352e-06 |

The band-to-band gaps are ~0.15-0.37 Strehl; the fp16 shift is ~1e-05. No bias is introduced.

## 4. Tail behaviour

| statistic of |d Strehl| | value |
|---|---|
| p50 | 5.105e-05 |
| p90 | 1.920e-04 |
| p99 | 3.990e-04 |
| max | 1.726e-03 |
| pairs > 0.005 (threshold) | 0 / 300 |
| pairs > 0.001 | 1 / 300 |
| pairs > 0.0005 | 2 / 300 |

Worst five pairs:

| rank | band | alpha | Strehl fp32 | Strehl fp16 | d(Strehl) | rel d(MSE) |
|---|---|---|---|---|---|---|
| 1 | mid | 0.1265 | 0.2791 | 0.2773 | -1.726e-03 | +3.57e-03 |
| 2 | low | 0.0842 | 0.6499 | 0.6508 | +8.016e-04 | -2.87e-03 |
| 3 | low | 0.0912 | 0.4731 | 0.4726 | -4.867e-04 | +1.33e-03 |
| 4 | low | 0.0265 | 0.4230 | 0.4234 | +3.982e-04 | -9.09e-04 |
| 5 | low | 0.0952 | 0.3239 | 0.3235 | -3.603e-04 | +7.25e-04 |

The single worst pair (|d| = 1.73e-03) is a mid-band pupil that is poorly corrected anyway
(Strehl 0.279); the error is a 0.6% relative perturbation of an already-low number, not a
visible change in the demo.

## 5. Caveat: what this CPU measurement does and does not cover

onnxruntime's CPU execution provider does **not** perform fp16 arithmetic. Verified directly:
an fp32 graph whose weights were rounded to fp16 and back (fp32 math, fp16 weights) produces
output matching the shipped fp16 graph to a max of **2.1e-05 rad**, while both sit ~8.9e-02 from
the true fp32 graph. The arithmetic-only component is therefore ~0.02% of the total deviation.

| path (60 pairs, raw max abs dev vs ORT fp32) | mean | max |
|---|---|---|
| fp32 graph, fp16-rounded weights (fp32 math) | 8.9125e-02 | 1.4520e-01 |
| shipped fp16 graph | 8.9125e-02 | 1.4522e-01 |
| **difference between those two** | **1.3927e-05** | **2.1040e-05** |

So every number in sections 2-4 measures **weight-rounding error only**. That is the dominant
term and it is what any fp16 backend inherits. But a backend that genuinely accumulates in fp16
(some WebGPU shader paths) would add a term this CPU test cannot see. The onnxruntime-web WASM
provider is built on the same MLAS kernels as the CPU EP tested here, so it very likely up-casts identically -- but I did not test it in a browser, so treat that as expectation, not measurement.
Given 2.9x headroom to the threshold, this is worth a spot check on the actual browser backend
before launch, not a blocker.

## 6. Verdict

**fp16 is safe to ship as the demo's only weights.**

- Worst-case absolute Strehl deviation: **1.73e-03**, against the agreed ~5e-03 threshold.
  Passes, with **2.9x headroom** -- comfortable but not enormous.
- 0 / 300 pairs exceed 0.005. 1 / 300 exceeds 0.001. p99 is 3.99e-04.
- Mean absolute Strehl deviation 8.35e-05; net bias on the headline mean Strehl +8.4e-06
  (0.705047 -> 0.705055).
- No asymmetry bias; if anything fp16 is *most* accurate on the high-asymmetry pupils that
  carry the demo's message.
- Saves 15.4 MB of the 30.8 MB download (50%).

Recommended: ship fp16 alone. Keep `unet_fp32.onnx` in the repo as the reference for
regression testing. If a WebGPU fp16 backend is used in the browser, re-run the Strehl check
on ~50 pairs there, since the 2.9x margin is not large enough to absorb an unmeasured
accumulation term without a look.

---

Reproduce: `fp16_verify.py` (main sweep), `epcheck2.py` (arithmetic isolation), `charac.py` +
`wcheck.py` (graph/weight characterisation), per-sample arrays in `fp16_results.npz`, all under
`/tmp/claude-979489/-depot-chan129-users-aalmuall/63a2bda4-40f1-435b-960c-8ea418acfc7a/scratchpad/`.
