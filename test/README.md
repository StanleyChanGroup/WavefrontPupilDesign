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

Boots `src/page.js` against a DOM stub (initialisation, pupil track, readouts), then
drives the real scheduler from `src/netqueue.js` with a mock predictor. No other harness
loads page.js.

The scheduler checks are mutation-tested: breaking the version check, the re-run after
completion, the failure exit, or making it always re-run each causes a failure or a hang.
That sweep found two gaps, both about work the scheduler does rather than results it
shows:

* The re-run in `finally` matters only when a change lands during `commit` and its
  debounce timer is then cancelled by `flush()`; every simpler case is absorbed by the
  retry loop or the pending timer.
* The debounce timer must check `version !== settled`, not just `!busy`. With a long
  debounce and a fast predictor the retry settles first and the stale timer would
  re-predict inputs that were just committed.

Both windows are now tests, and both depend on timing: one needs the predictor to outlive
the debounce, the other the reverse. A single timing regime tests neither.

## Mobile layout and touch  -- needs playwright

    npm install && npx playwright install firefox chromium
    node test/mobile.mjs

Headless Firefox over a real HTTP server, at 320/360/390/430/520/540/620/700/768/1024:
no document-level horizontal overflow (before and after the sticky bar appears), the
sticky slider and "New aberration" on screen, the model-loading overlay fitting its panel
without covering the caption, no overflow after live resizing, the scatter selection
surviving it, tap-to-pin on the scatter, and the decomposition tooltip staying inside a
320px viewport. Each engine is guarded independently: if only one is installed, the sections needing the
other print SKIP lines naming the checks that did not run and the whole harness exits 2,
so a partial run cannot be mistaken for a pass. With neither, or with playwright itself
absent, it reports why and exits 2 without throwing a launch error.

The DOM stub in `smoke.mjs` cannot catch any of this: it has no box model, so
`getBoundingClientRect` returns whatever it was hard-coded to return. Every mobile defect
found so far was invisible there and visible here.

Two engines, deliberately. Layout runs in **Firefox**, where the reported overflow was
first reproduced (scrollWidth 693 at a 390 viewport, 800 at 768). Touch runs in
**Chromium**, because Playwright does not support mobile emulation in Firefox -- and that
is not a silent no-op:

    {hasTouch:true, isMobile:true}  ->  innerWidth 608   (600px box in a 390 viewport)
    {hasTouch:true}                 ->  innerWidth 390

With `isMobile` the Firefox layout viewport widens to fit overflowing content, which is
the opposite of what an overflow test wants, and `hasTouch` leaves `ontouchstart` false
and `maxTouchPoints` 0. Chromium honours both; the first touch check asserts the
emulation actually took, so this cannot silently regress into fake-mobile again. The page
itself branches only on `event.pointerType`, never on `ontouchstart` or a media query.

Scrolling *inside* the pupil track, the pipeline and the citation block is intentional.
The overflow check ignores anything clipped by an ancestor with `overflow-x` set, so it
measures escapes from the document rather than deliberate inner scrolling. A failing
width prints the offending elements, not just a number.

The scroll check uses a **real finger drag** through Chromium's input pipeline
(`Input.dispatchTouchEvent` over a CDP session), not `window.scrollBy`. Synthetic
TouchEvents dispatched from page script are untrusted and never scroll, and programmatic
scrolling passes even when the gesture is blocked. On a page with `touch-action:none` and
a non-passive `touchmove` handler calling `preventDefault`:

    old check (window.scrollBy):  700 -> 1200   => PASS
    new check (real finger drag): 700 -> 700    => FAIL

A companion check reports *why* panning is blocked -- an ancestor's `touch-action`, or a
cancelled `touchmove` -- so the two causes are distinguishable. Both were mutation-tested
against the live page: injecting either one fails both checks.

This still is not a physical phone. It does not cover momentum, rubber-banding, browser
chrome appearing and disappearing, or real hit-target size under a thumb.

Four checks here are mutation-tested against the defects that prompted them:

* Restoring `white-space:nowrap` on `.tip` puts the tooltip's right edge at 332px in a
  320px viewport -- the reviewer's measurement, reproduced.
* Injecting `touch-action:none` on the scatter, or a `touchmove` handler that calls
  `preventDefault`, each fails the drag check and is named by the blocker check.
* That check is only meaningful if the tap reaches the *asymmetric* label, which is the
  long one. The default pupil is a circle and has no asymmetric region, so an earlier
  version of this test tapped the default pupil, only ever saw the short symmetric
  string, and passed against the mutant. It now drives the slider to the most asymmetric
  pupil and fails if the long label was never reached.

## Page syntax

`src/page.js` is loaded by the browser as an ES module. Check it as one -- `node --check`
parses a `.js` file as a script and will pass on module-only syntax errors:

    node --input-type=module --check < src/page.js

## Adversarial audit

`test/AUDIT.md` records an independent review of the optics port: 33 injected mutations,
29 fresh pupils and 42 fresh wavefronts generated outside the golden set.
