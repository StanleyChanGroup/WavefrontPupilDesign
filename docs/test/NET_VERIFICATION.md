# Browser inference vs the published PyTorch model

Does `src/net.js` running `assets/unet_fp16.onnx` under onnxruntime-web reproduce the
published checkpoint? Yes, to well inside the fp16 budget.

## Method

`test/net_verify.mjs`. 300 (pupil, phase) pairs drawn from the **published test split**,
stratified into terciles of corrected asymmetry so the whole range 0.0144–0.3539 is
covered rather than whatever happens to sit at the head of the file. Measurements built
with `data.simulate_measurement` under the published config (`noise_sigma=0`,
`normalize_by_peak`, `use_throughput`, `renormalize_after_noise`, `coeff_scale=2.0`,
33 `PAPER_MODES`).

The harness serves the project over real HTTP on 127.0.0.1, so `net.js` takes exactly the
browser code path — `fetch` for the model, ORT's own fetch for the wasm through
`env.wasm.wasmPaths`.

Three paths compared:

| | |
|---|---|
| **PyTorch fp32, CPU** | the published `best_model.pth` — the reference |
| **A — same input** | JS inference fed the float32 measurement PyTorch itself built; isolates ONNX/fp16/WASM error |
| **B — full JS** | JS inference fed the measurement `optics.js` builds from (mask, phase); the path a visitor actually gets |

## Results

Deviation from PyTorch fp32:

| quantity | mean | p99 | worst |
|---|---|---|---|
| Strehl, path A | 8.157e-05 | 5.481e-04 | 7.018e-04 |
| **Strehl, path B (end to end)** | **8.158e-05** | **5.486e-04** | **7.021e-04** |
| MSE, absolute | 2.904e-04 | 3.294e-03 | 3.697e-03 |
| MSE, relative | 6.032e-04 | 2.513e-03 | 3.154e-03 |
| measurement, JS vs PyTorch (rel. peak) | 2.422e-07 | 4.689e-07 | 5.088e-07 |

**0 of 300** pairs exceed the 5e-3 acceptance threshold. **0 of 300** exceed even the
1.73e-3 fp16-only budget established in `assets/FP16_VERIFICATION.md`.

Paths A and B agree to 8.157e-05 vs 8.158e-05 — the JavaScript forward model contributes
essentially nothing to the total. All of the deviation is fp16 weight rounding.

By asymmetry tercile (Strehl, path B):

| band | mean | worst |
|---|---|---|
| α 0.014–0.118 | 1.111e-04 | 6.030e-04 |
| α 0.118–0.237 | 8.817e-05 | 7.021e-04 |
| α 0.237–0.354 | 4.551e-05 | 1.974e-04 |

Error *falls* with asymmetry, so the high-α pupils carrying the paper's claim are the most
accurate. The trend itself is unaffected: PyTorch Strehl runs 0.4207 (low α) → 0.9295
(high α), a gap of 0.51 against a worst-case error of 7e-04.

## Timing

Measured under Node with the wasm EP on a loaded shared node; a typical laptop is faster.

| | |
|---|---|
| model download | 15,403,693 B |
| ORT import | 6.4 ms |
| session init | 274.6 ms |
| first inference | 736.9 ms |
| steady state | 711.0 ms |

Inference is why the page debounces the network panels rather than running per slider tick.

## Caveat

onnxruntime's CPU execution provider up-casts fp16 to fp32 internally, so these numbers
measure fp16 **weight rounding**, not fp16 arithmetic. A backend that accumulates in true
fp16 (some WebGPU paths) would add a term this test cannot see. The arithmetic-only
component was isolated separately at ~0.02% of the total on CPU. The page pins the wasm EP
single-threaded, which is the path measured here.

## Reproducing

```sh
node test/net_verify.mjs <cases.json> <out.json>
```

`cases.json`/`cases.bin` come from the PyTorch side; regenerate with the companion script
recorded in the harness header.
