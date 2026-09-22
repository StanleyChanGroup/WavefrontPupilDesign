# Verification

Everything here reruns from a fresh clone. Node 18+ required; no npm packages.

## Forward optical model (no fixtures to fetch)

    node test/verify.mjs --no-npz

Checks `src/optics.js` against `assets/golden.json`, which is committed. 76 tracked
quantities. Drop `--no-npz` to include whole-array coverage, which additionally needs
`assets/golden.npz` (11.8 MB, not committed -- regenerate with the script named in
`assets/GOLDEN.md`).

## Scenes

    node test/scenes_verify.mjs

93 checks over the USAF target, star field and terrain, plus the convolution properties.

## Browser inference against PyTorch

    node test/net_verify.mjs

Needs PyTorch-side fixtures (`cases.json`, `cases.bin`) produced by the companion script
described in `test/NET_VERIFICATION.md`. By default they are read from `test/fixtures/`;
point elsewhere with `NET_VERIFY_DIR=/path/to/fixtures`.

## Adversarial audit

`test/AUDIT.md` records an independent review of the optics port: 33 injected mutations,
29 fresh pupils and 42 fresh wavefronts generated outside the golden set.
