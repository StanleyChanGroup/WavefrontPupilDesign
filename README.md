# Pupil Design for Computational Wavefront Estimation

Reference implementation for [*Pupil Design for Computational Wavefront Estimation*](https://arxiv.org/abs/2604.00225)
by Ali Almuallem, Nicholas Chimitt, Bole Ma, Qi Guo and Stanley H. Chan.

[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge)](https://stanleychangroup.github.io/WavefrontPupilDesign/)

The [project page](https://stanleychangroup.github.io/WavefrontPupilDesign/) runs the
trained network in your browser: pick a pupil, draw an aberration, and watch the
wavefront estimate and the corrected image update live. Its source is under `docs/`.

Recovering a wavefront from a single intensity image is degenerate for a
symmetric pupil: the measurement cannot distinguish a wavefront from its
conjugate. Breaking that symmetry lifts the degeneracy. This repository provides
a quantitative asymmetry metric, generators for asymmetry-balanced pupil
populations, and the networks and training code used to show that recoverability
improves as asymmetry increases.

## Install

Conda, matching the environment used for the published results:

```bash
conda env create -f environment.yml
conda activate pupil-design
pip install -e .
```

Or pip, after installing [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA version:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

A GPU is strongly recommended for training. Everything else, including the demo
notebook, runs on CPU.

## Train and evaluate

Two commands. The first builds the pupil and phase pools on first use and caches
them under `--data-root`, so later runs start immediately.

```bash
# Smoke test in about a minute, to check the install
pupil-design-train --demo --model unet --out runs/demo
pupil-design-eval runs/demo --bins 5

# A real run: one architecture at one noise level
pupil-design-train --model unet --noise-sigma 0.01 --epochs 4 --out runs/unet_0.01
pupil-design-eval runs/unet_0.01 --split test --bins 30

# The paper's full sweep, both architectures at every published noise level
bash scripts/reproduce_paper.sh ./data ./runs
```

Each run writes `best_model.pth`, `config.json`, `history.json`, a metrics
`.npz`, and plots of error and Strehl ratio against asymmetry. Evaluation reads
`config.json`, so a run always gets evaluated under the settings it was trained
with.

Run `pupil-design-train --help` for the full option list. On a Slurm cluster,
`scripts/slurm_train.sh` submits the noise sweep as an array job.

## Demo notebook

`notebooks/demo.ipynb` walks through the idea in a few minutes: measuring the
asymmetry of common apertures, seeing why a symmetric pupil cannot distinguish a
wavefront from its negative, running a trained model, and reproducing the trend
of error against asymmetry.

```bash
jupyter notebook notebooks/demo.ipynb
```

It runs from a fresh clone whether or not you installed the package: the first
cell adds `src/` to the path when `pupil_design` is not already importable.

It also adapts to your hardware. The GPU is used only when it has free memory,
which matters on shared machines, and the final experiment shrinks its sample
size on CPU. Expect roughly one minute on a GPU and a few minutes on CPU. Set
`DEMO_FORCE_CPU=1` to skip the GPU entirely.

## Pretrained weights

Weights and the exact simulation data behind the paper are attached to the
[latest release](../../releases/latest), not stored in the repository. From
inside a clone:

```bash
gh release download --dir .
sha256sum -c SHA256SUMS

mkdir -p runs data/dataset_jan2026_bbox_128_radius_19
tar xf unet_no_noise.tar -C runs/
tar xf mlp_no_noise.tar  -C runs/
mv *_pool_cache_threeway.pth data/dataset_jan2026_bbox_128_radius_19/
```

Then reproduce the paper's noiseless test curve without training anything:

```bash
pupil-design-eval runs/unet_no_noise --data-root ./data --split test --bins 30
```

Add `--max-pupils 40` for a spot check that finishes in under a minute.

The release currently carries the noiseless U-Net and MLP. Models at the
published noise levels of 0.001, 0.01 and 0.03 are not yet included; train them
with `scripts/reproduce_paper.sh`.

Downloading the pupil and phase pools is optional, since generation is seeded
and reproducible. They are shipped so that reproduction does not depend on that
being true, and the test suite asserts the shipped pools match the published
asymmetries exactly.

## Library use

```python
import torch
from pupil_design import asymmetry, reference_circle, generate_polygon_pupils

# Asymmetry of a shape, in [0, 1)
circle = reference_circle(bounding_box_size=128, radius=19)
print(float(asymmetry(circle, align="corrected")))   # 0.0, a circle is symmetric

# A population of convex pupils, uniformly covering the asymmetry axis
pool = generate_polygon_pupils(radius=19, bounding_box_size=128, count=100, n_bins=10)
print(pool["masks"].shape, pool["asym"].min(), pool["asym"].max())
```

## Repository layout

```
src/pupil_design/   library and command line entry points
tests/              regression tests against the published artefacts
notebooks/demo.ipynb
scripts/            reproduce_paper.sh, slurm_train.sh
environment.yml     conda environment
```

### Modules

| Module | Contents |
|---|---|
| `optics` | PSF formation, Zernike phase synthesis, Strehl ratio |
| `asymmetry` | The asymmetry metric |
| `pupils` | Convex-polygon and binary pupil generators, asymmetry-balanced sampling |
| `data` | Measurement simulation, dataset, cached pupil and phase pools |
| `models` | U-Net and MLP wavefront estimators |
| `losses` | Masked, piston-corrected MSE |
| `config` | `ExperimentConfig`, saved beside every run as `config.json` |
| `train` | Training loop and the `pupil-design-train` command |
| `evaluate` | Per-pupil evaluation and the `pupil-design-eval` command |

## Known issue: two alignment modes for the asymmetry metric

The metric aligns a pupil against its 180-degree rotation before measuring
overlap. The implementation used for the paper rolls by the autoconvolution
argmax, which lands **one pixel off in each axis**. An exactly symmetric circle
of radius 19 in a 128-pixel box therefore scores 0.047 rather than 0.

The offset is not constant. It tracks the perimeter-to-area ratio, so it grows
for smaller or more elongated pupils and cannot be removed by subtracting a
constant.

- `asymmetry(mask)` and `asymmetry(mask, align="published")` reproduce the
  paper's numbers exactly. Use these with released weights, the cached pupil
  pools, and anything compared against published figures.
- `asymmetry(mask, align="corrected")` adds the missing pixel and scores a
  symmetric pupil 0. Use this for new work.

Across the released test pupils the two rank pupils almost identically
(correlation above 0.99), so conclusions that depend on ordering rather than on
absolute values are not expected to change.

## Reproducibility

`tests/` locks the package to the published artefacts. The pupil pool ships with
its asymmetry values, and the noiseless U-Net ships with per-pupil test losses;
both are asserted against freshly computed values.

```bash
pytest -m "not slow"       # unit and metric tests, CPU
pytest -m slow             # end-to-end against published per-pupil losses, needs GPU
```

Point the tests at the artefacts with `PUPIL_DESIGN_DATA` and
`PUPIL_DESIGN_METRICS` if they are not at their default locations.

## Noise model

Intensity is simulated as `y = |F P x|^2 / max|F C 1|^2 + n`, with
`n ~ N(0, sigma^2 I)` and `C` the reference circular pupil. Normalising by the
circle's peak before adding noise fixes one absolute intensity scale, so a
single `sigma` means the same photon count for every pupil. A pupil that passes
less light produces a dimmer image and a correspondingly lower SNR. That is the
throughput cost of asymmetry, and it emerges from this normalisation rather than
from any explicit throughput factor.

Published noise levels are `sigma` in `{0, 0.001, 0.01, 0.03}`.

## Citation

```bibtex
@article{almuallem2026pupil,
  author = {Ali Almuallem and Nicholas Chimitt and Bole Ma and Qi Guo and Stanley H. Chan},
  journal = {Opt. Express},
  number = {23},
  pages = {45049--45063},
  publisher = {Optica Publishing Group},
  title = {Pupil design for computational wavefront estimation},
  volume = {34},
  month = {Nov},
  year = {2026},
  url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-34-23-45049},
  doi = {10.1364/OE.609577},
}
```
