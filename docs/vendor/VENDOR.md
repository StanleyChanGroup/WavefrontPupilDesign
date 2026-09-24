# vendor/ — pinned third-party runtime

## onnxruntime-web 1.30.0 (exact, no range)

| | |
|---|---|
| package | `onnxruntime-web` |
| version | **1.30.0** (npm `latest` on 2026-09-15) |
| tarball | https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.30.0.tgz |
| npm integrity | `sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ==` |
| npm shasum | `b88b09117992e6f486321e931a826fd2f7775377` |
| licence | MIT (Microsoft) — see `LICENSE.onnxruntime-web` |

Reproduce:

```sh
npm pack onnxruntime-web@1.30.0        # or: npm i onnxruntime-web@1.30.0
cp node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs        vendor/
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm    vendor/
```

### Files kept

| file | bytes | gzip | sha256 |
|---|---:|---:|---|
| `ort.wasm.bundle.min.mjs` | 73,054 | 24,608 | `11e64bd8ffe11bd1a2a2f0d6275fdfbbba7262f0b76b99b53d228a8a22ef3d90` |
| `ort-wasm-simd-threaded.wasm` | 14,239,897 | 3,659,955 | `3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2` |

Nothing else from `dist/` is needed. Source maps, the WebGPU/WebGL/JSPI builds
and the 26–28 MB `asyncify`/`jsep` wasm variants are all omitted.

### Why `ort.wasm.bundle.min.mjs` and not `ort.min.mjs`

* `ort.min.mjs` (368 KB) is the all-backends build: it drags in WebGPU and WebNN
  glue this page never uses.
* `ort.wasm.min.mjs` (50 KB) is WASM-only but loads its emscripten glue from a
  *second* file, `ort-wasm-simd-threaded.mjs`.
* `ort.wasm.bundle.min.mjs` (73 KB) is WASM-only **with the glue inlined**, so the
  whole runtime is two files: this module plus one `.wasm`. That is what a plain
  static server (GitHub Pages) serves most reliably — no extra MIME types, no
  extra round trip.

### Backend and threading — and why no COOP/COEP is needed

`src/net.js` sets, before the first session is created:

```js
ort.env.wasm.numThreads = 1;                 // never spawn a worker
ort.env.wasm.proxy      = false;
ort.env.wasm.wasmPaths  = new URL('../vendor/', import.meta.url).href;
```

* **Backend: `wasm` (CPU) only.** The fp16 graph is Conv / ConvTranspose /
  MaxPool / Relu / Concat / Cast — fully covered by the WASM EP, and the WASM EP
  is the path the numbers in `test/NET_VERIFICATION.md` were measured on. WebGPU
  would need its own verification pass plus a CPU fallback for the ~30 % of
  visitors without it; the model runs in well under a second on CPU, so it buys
  nothing here.
* **Threads: exactly 1.** Since 1.18, onnxruntime-web ships a single wasm binary,
  built with pthreads (`ort-wasm-simd-threaded.wasm`) — there is no separate
  non-threaded artefact to choose. Its worker pool needs `SharedArrayBuffer`
  *sharing*, which needs `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy`
  response headers, which **GitHub Pages cannot set**. Pinning `numThreads = 1`
  means the pool is never created, so the page behaves identically on an isolated
  and a non-isolated origin. (ORT clamps to 1 by itself when
  `self.crossOriginIsolated` is false; we set it explicitly so the behaviour is
  not environment-dependent and no console warning is emitted.)
* **The shared-memory question.** The wasm module *does* declare its imported
  memory as `shared` (flags `0x3`, min 256 / max 65536 pages). That is fine
  without cross-origin isolation: browsers permit
  `new WebAssembly.Memory({ shared: true })` off an isolated origin — only the
  bare `SharedArrayBuffer` constructor and cross-thread `postMessage` of a SAB are
  gated. The emscripten glue in this bundle relies on exactly that, via
  `globalThis.SharedArrayBuffer ?? new WebAssembly.Memory({initial:0,maximum:0,shared:true}).buffer.constructor`.
  So: **no COOP/COEP headers, no cross-origin isolation, no service worker.**
* **SIMD is required, not optional.** onnxruntime-web ≥ 1.18 publishes no
  non-SIMD binary, so there is no "single-threaded non-SIMD" build to prefer.
  Baseline support for WebAssembly SIMD: Chrome/Edge 91+ (May 2021), Firefox 89+,
  Safari 16.4+ (Mar 2023). Below that ORT throws
  `WebAssembly SIMD is not supported in the current environment` and `loadModel()`
  rejects — the page should catch that and show the static figures.

### The relative wasm path

`wasmPaths` is resolved from `import.meta.url` of `src/net.js`, **not** from the
document URL:

```js
ort.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
```

A bare relative string (`'vendor/'`) would be resolved by ORT against the page's
URL, so it would break the moment the site is served from a sub-path such as
`https://user.github.io/pupil-page/`. Resolving against the module keeps the
whole tree relocatable: any origin, any sub-directory, `file://` included.

Required layout (this is the only constraint):

```
<anywhere>/
  src/net.js          src/optics.js   src/ladder.js
  vendor/ort.wasm.bundle.min.mjs
  vendor/ort-wasm-simd-threaded.wasm
  assets/unet_fp16.onnx
```

`net.js` must be loaded as an ES module (`<script type="module">`); it uses a
dynamic `import()` so the 14 MB wasm is only touched when `loadModel()` is first
called, not at page parse.

### Serving notes

* `.wasm` must be served as `application/wasm` (GitHub Pages does) or the
  streaming instantiation falls back to the slower `arrayBuffer` path.
* `ort-wasm-simd-threaded.wasm` is 14.2 MB raw / **3.66 MB gzip**; the model is
  15.4 MB raw / 14.2 MB gzip (already-compressed fp16 weights). Together that is
  ≈ 17.9 MB over the wire on a cold, uncached visit. Both are immutable —
  version the paths or rely on the ETag; a repeat visit is served from cache.
