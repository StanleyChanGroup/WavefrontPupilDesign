# Pupil Design for Computational Wavefront Estimation — project page

Project page and interactive demo for
**[Pupil Design for Computational Wavefront Estimation](https://arxiv.org/abs/2604.00225)**
— Ali Almuallem, Nicholas Chimitt, Bole Ma, Qi Guo, Stanley H. Chan (under review, Optics Express).

The page runs the paper's trained U-Net **in the browser**: pick a pupil, draw a random
aberration, and watch the network estimate the wavefront and correct the image, live.

## Viewing it

It is a static site. Open `index.html` through any web server — not `file://`, which
blocks the module imports and the model fetch:

    python3 -m http.server 8000
    # then http://localhost:8000/

All paths are relative, so it works equally at a domain root or under a project-pages
subpath such as `/WavefrontPupilDesign/`.

## What the demo does

The network is **opt-in**: the 15 MB model downloads only when you press the button in the
demo panel (about 28 MB in total with the onnxruntime WebAssembly runtime). Everything
else — the pupil gallery, the symmetric/asymmetric decomposition, the 1500-pupil
population scatter — runs without it.

Inference is `onnxruntime-web` on the WebAssembly backend, single-threaded, so the page
needs no cross-origin isolation headers and works on plain static hosting.

## Layout

    index.html            the page: markup and styles
    src/                  the demo. optics.js is the forward model (zero dependencies),
                          net.js the inference path, page.js the controller,
                          netqueue.js the scheduler that keeps a slow prediction from
                          painting under a newer pupil
    assets/               the fp16 model, pupil masks, the population data, and the
                          verification records for each
    vendor/               onnxruntime-web, vendored; see vendor/VENDOR.md
    figures/              paper figures
    content/paper.json    title, authors, abstract, bibtex
    test/                 verification harnesses; see test/README.md

## Verification

The numerical claims are checked, not asserted. From a clone, with no dependencies:

    node test/verify.mjs          # the forward model vs a PyTorch-generated golden set
    node test/scenes_verify.mjs   # convolution, PSNR, SSIM
    node test/smoke.mjs           # the page boots; the inference scheduler behaves

Browser layout and touch, which need Playwright:

    npm install && npx playwright install firefox chromium
    node test/mobile.mjs

`test/README.md` explains what each harness covers and which checks are mutation-tested.
`test/AUDIT.md`, `assets/GOLDEN.md`, `assets/FP16_VERIFICATION.md` and
`test/NET_VERIFICATION.md` record the results.

One harness does not run from a clone: `test/net_verify.mjs` compares browser inference
against the published PyTorch checkpoint and needs fixtures generated on the PyTorch side.
Its results are recorded in `test/NET_VERIFICATION.md` (worst-case Strehl deviation
7.0e-04 over 300 stratified test-split cases); committing the generator is open work.

## Citing

```bibtex
@article{almuallem2026pupil,
  title   = {Pupil Design for Computational Wavefront Estimation},
  author  = {Almuallem, Ali and Chimitt, Nicholas and Ma, Bole and Guo, Qi and Chan, Stanley H.},
  journal = {arXiv preprint arXiv:2604.00225},
  year    = {2026}
}
```

## Third-party material

See `ASSETS-LICENSE.md` (test images) and `vendor/VENDOR.md` (onnxruntime-web) for the
licences of redistributed material.
