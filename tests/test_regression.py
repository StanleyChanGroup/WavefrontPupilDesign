"""Regression tests against the published models and metrics.

These lock the refactored package to the numbers behind the paper. They need
the released weights and dataset caches; set PUPIL_DESIGN_DATA and
PUPIL_DESIGN_MODELS, or let them default to the Purdue depot paths.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from pupil_design import (
    ExperimentConfig,
    PupilPhaseDataset,
    asymmetry,
    build_model,
    masked_piston_corrected_mse_per_sample,
    reference_circle,
    simulate_measurement,
)
from pupil_design.config import parse_legacy_dirname
from pupil_design.optics import psf_from_pupil

DATA_ROOT = Path(os.environ.get("PUPIL_DESIGN_DATA", "/depot/chan129/users/aalmuall/dataset/optimal_pupil"))
DATASET = DATA_ROOT / "dataset_jan2026_bbox_128_radius_19"
UNET_DIR = DATA_ROOT / (
    "model_unet_with_pupil_no_noise_bbox_128_radius_19_using_throughput"
    "_normalized_after_throughput_not_fighting_overfitting_normalize_by_peak"
)
METRICS = Path(os.environ.get(
    "PUPIL_DESIGN_METRICS",
    "/depot/chan129/users/aalmuall/optimal_pupil/paper_models_feb26/"
    + UNET_DIR.name + "/test_loss_vs_asym_metrics.npz",
))

requires_data = pytest.mark.skipif(not DATASET.exists(), reason="dataset cache not available")
requires_metrics = pytest.mark.skipif(not METRICS.exists(), reason="published metrics not available")


@pytest.fixture(scope="module")
def pupil_pool():
    return torch.load(DATASET / "pupil_pool_cache_threeway.pth", weights_only=False)


@pytest.fixture(scope="module")
def phase_pool():
    return torch.load(DATASET / "phase_pool_cache_threeway.pth", weights_only=False)


# --------------------------------------------------------------------------
# Unit behaviour
# --------------------------------------------------------------------------

def test_circle_is_exactly_its_own_rotation():
    """Precondition for the two tests below: the circle is truly symmetric."""
    circle = reference_circle(128, 19)
    assert torch.equal(torch.flip(circle, dims=(0, 1)), circle)


def test_corrected_alignment_scores_symmetric_pupil_zero():
    circle = reference_circle(128, 19)
    assert float(asymmetry(circle, align="corrected")) == pytest.approx(0.0, abs=1e-6)


def test_published_alignment_has_a_perimeter_dependent_floor():
    """Documents the off-by-one in the released metric. See asymmetry docstring.

    The floor is not constant: it grows as the pupil shrinks, tracking the
    perimeter-to-area ratio, which is why it cannot be corrected by an offset.
    """
    floors = {r: float(asymmetry(reference_circle(128, r))) for r in (10, 19, 30, 50)}
    assert floors[19] == pytest.approx(0.0472, abs=1e-3)
    assert floors[10] > floors[19] > floors[30] > floors[50] > 0.0


def test_asymmetry_is_translation_invariant():
    """Shifting a pupil must not change its asymmetry."""
    circle = reference_circle(128, 19)
    shifted = torch.roll(circle, shifts=(7, -11), dims=(0, 1))
    assert float(asymmetry(shifted)) == pytest.approx(float(asymmetry(circle)), abs=1e-6)


def test_alignment_modes_preserve_ordering():
    """The bug inflates values but barely reorders them, which is why the
    paper's qualitative trend is expected to survive correction."""
    pool = DATASET / "pupil_pool_cache_threeway.pth"
    if not pool.exists():
        pytest.skip("dataset cache not available")
    masks = torch.load(pool, weights_only=False)["test"]["masks"][::100]
    published = asymmetry(masks, align="published").numpy()
    corrected = asymmetry(masks, align="corrected").numpy()
    assert (published >= corrected - 1e-6).all()
    assert np.corrcoef(published, corrected)[0, 1] > 0.99


def test_asymmetry_batched_matches_single():
    torch.manual_seed(0)
    masks = (torch.rand(4, 64, 64) > 0.5).float()
    batched = asymmetry(masks)
    for i in range(4):
        assert float(batched[i]) == pytest.approx(float(asymmetry(masks[i])), abs=1e-6)


def test_piston_is_not_penalised():
    """Adding a constant inside the pupil must not change the loss."""
    torch.manual_seed(0)
    mask = reference_circle(64, 20).view(1, 1, 64, 64)
    target = torch.randn(1, 1, 64, 64) * mask
    pred = target.clone()
    base = masked_piston_corrected_mse_per_sample(pred, target, mask)
    shifted = masked_piston_corrected_mse_per_sample(pred + 3.7, target, mask)
    assert float(base) == pytest.approx(float(shifted), abs=1e-6)


def test_noiseless_peak_normalization_is_a_noop():
    """Documented cancellation: with no noise the flag cannot change the input."""
    torch.manual_seed(0)
    mask = reference_circle(64, 20).view(1, 1, 64, 64)
    phase = torch.randn(1, 1, 64, 64) * mask
    circle_psf = psf_from_pupil(reference_circle(64, 20), torch.zeros(64, 64))
    on = ExperimentConfig(bounding_box_size=64, radius=20, noise_sigma=0.0, normalize_by_peak=True)
    off = ExperimentConfig(bounding_box_size=64, radius=20, noise_sigma=0.0, normalize_by_peak=False)
    a = simulate_measurement(mask, phase, circle_psf, on)
    b = simulate_measurement(mask, phase, circle_psf, off)
    assert torch.allclose(a, b, atol=1e-6)


def test_legacy_dirname_roundtrip():
    cfg = parse_legacy_dirname(UNET_DIR.name)
    assert cfg.model_type == "unet"
    assert cfg.noise_sigma == 0.0
    assert cfg.normalize_by_peak is True
    assert cfg.bounding_box_size == 128 and cfg.radius == 19
    assert cfg.narrow is False


def test_dataset_pairs_every_pupil_with_every_phase():
    pupils, phases = torch.zeros(3, 8, 8), torch.zeros(5, 8, 8)
    ds = PupilPhaseDataset(phases, pupils)
    assert len(ds) == 15
    assert [ds[i][2] for i in range(15)] == [0] * 5 + [1] * 5 + [2] * 5


# --------------------------------------------------------------------------
# Regression against published artefacts
# --------------------------------------------------------------------------

@requires_data
@requires_metrics
def test_cached_asymmetry_matches_published_metrics(pupil_pool):
    """The shipped pupils are the exact ones the paper evaluated."""
    stored = np.load(METRICS, allow_pickle=True)["target_asym"]
    cached = np.asarray(pupil_pool["test"]["asym"])
    assert np.array_equal(stored, cached)


@requires_data
def test_asymmetry_recomputes_from_masks(pupil_pool):
    """Recomputing the metric from masks reproduces the stored values."""
    masks = pupil_pool["test"]["masks"][:64]
    stored = np.asarray(pupil_pool["test"]["asym"][:64])
    recomputed = asymmetry(masks).numpy()
    assert np.allclose(recomputed, stored, atol=1e-6)


@requires_data
def test_published_unet_checkpoint_loads():
    ckpt = UNET_DIR / "best_model.pth"
    if not ckpt.exists():
        pytest.skip("U-Net checkpoint not available")
    model = build_model("unet", bounding_box_size=128, with_pupil=True, narrow=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    assert sum(p.numel() for p in model.parameters()) == 7_702_401


@requires_data
@requires_metrics
@pytest.mark.slow
def test_per_pupil_loss_matches_published(pupil_pool, phase_pool):
    """End-to-end: the package reproduces published per-pupil test losses.

    Tolerance is set by mixed-precision nondeterminism, not by algorithmic
    drift; the original evaluation ran the forward pass under fp16 autocast.
    """
    ckpt = UNET_DIR / "best_model.pth"
    if not ckpt.exists():
        pytest.skip("U-Net checkpoint not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 2 * 1024**3:
        pytest.skip(f"GPU busy: only {free_bytes / 1e9:.2f} GB free, need ~2 GB")

    n_pupils = int(os.environ.get("PUPIL_DESIGN_TEST_PUPILS", 20))
    device = torch.device("cuda")
    cfg = ExperimentConfig(noise_sigma=0.0, normalize_by_peak=True, renormalize_after_noise=True)

    model = build_model("unet", cfg.bounding_box_size, cfg.with_pupil, cfg.narrow).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    circle = reference_circle(cfg.bounding_box_size, cfg.radius, device=device)
    circle_psf = psf_from_pupil(circle, torch.zeros_like(circle))

    pupils = pupil_pool["test"]["masks"][:n_pupils].to(device)
    phases = phase_pool["test_phases"].to(device)
    published = np.load(METRICS, allow_pickle=True)["avg_loss_per_pupil"][:n_pupils]

    ours = np.zeros(n_pupils)
    chunk = 250
    with torch.no_grad():
        for pi in range(n_pupils):
            pupil = pupils[pi]
            total = 0.0
            for start in range(0, phases.shape[0], chunk):
                block = phases[start:start + chunk]
                mask = pupil.expand(block.shape[0], -1, -1).unsqueeze(1)
                target = (pupil.unsqueeze(0) * block).unsqueeze(1)
                psf = simulate_measurement(mask, target, circle_psf, cfg)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(torch.cat([psf, mask], dim=1))
                total += masked_piston_corrected_mse_per_sample(
                    pred.float(), target, mask
                ).sum().item()
            ours[pi] = total / phases.shape[0]

    rel = np.abs(ours - published) / np.maximum(np.abs(published), 1e-12)
    assert rel.max() < 1e-3, f"max relative error {rel.max():.2e}"
