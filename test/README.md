# Verification

Node 18+; no npm packages. Two of the three harnesses run from a clean clone. The third
needs fixtures that are **not** in this repository -- see below.

## Forward optical model  -- runs from a clone

    node test/verify.mjs --no-npz

Checks `src/optics.js` against `assets/golden.json`, which is committed. 60 tracked
quantities, ~1.7 M comparisons.

Add whole-array coverage (76 quantities, ~4.4 M comparisons) by also providing
`assets/golden.npz` (11.8 MB, not committed) and dropping the flag:

    node test/verify.mjs

`assets/GOLDEN.md` documents every convention and the encodings both fixtures use.

## Scenes  -- runs from a clone

    node test/scenes_verify.mjs

93 checks over the USAF target, star field and cratered terrain, plus the convolution
properties (delta identity, shift invariance, brightness preservation).

## Browser inference against PyTorch  -- NOT runnable from a clone

    node test/net_verify.mjs            # reads fixtures from test/fixtures/
    NET_VERIFY_DIR=/path node test/net_verify.mjs

This compares onnxruntime-web against the published PyTorch checkpoint. It needs
`cases.json` and `cases.bin`, built on the PyTorch side from the published test split and
the checkpoint. **Neither the fixtures (83 MB) nor the generator script are in this
repository**, so this harness cannot currently be reproduced from a clone alone.

`test/NET_VERIFICATION.md` records the results of the last run (300 stratified cases;
worst-case Strehl deviation 7.0e-04 against PyTorch fp32) and the method, but reproducing
it requires regenerating the fixtures against
`optimal_pupil/paper_models_feb26/model_unet_..._normalize_by_peak/best_model.pth` and the
pools under `dataset/optimal_pupil/dataset_jan2026_bbox_128_radius_19/`. Committing a
generator script is open work.

## Page smoke test  -- runs from a clone

    node test/smoke.mjs

Boots `src/page.js` against a DOM stub and checks it initialises, the pupil track builds,
readouts populate, and the inference scheduler ends on the last-selected pupil when the
selection changes mid-prediction. No other harness loads page.js.

## Page syntax

`src/page.js` is loaded by the browser as an ES module. Check it as one -- `node --check`
parses a `.js` file as a script and will pass on module-only syntax errors:

    node --input-type=module --check < src/page.js

## Adversarial audit

`test/AUDIT.md` records an independent review of the optics port: 33 injected mutations,
29 fresh pupils and 42 fresh wavefronts generated outside the golden set.
