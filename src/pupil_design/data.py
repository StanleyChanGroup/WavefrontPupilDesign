"""Measurement simulation, dataset, and cached pupil/phase pools."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .optics import PAPER_MODES, SMALL_MODES, psf_from_pupil, random_zernike_phase
from .pupils import generate_binary_pupils, generate_polygon_pupils

__all__ = ["simulate_measurement", "PupilPhaseDataset", "build_pools", "load_or_build_pools"]

PUPIL_CACHE_NAME = "pupil_pool_cache_threeway.pth"
PHASE_CACHE_NAME = "phase_pool_cache_threeway.pth"


def simulate_measurement(
    pupil: torch.Tensor,
    phase: torch.Tensor,
    circle_psf: torch.Tensor,
    cfg: ExperimentConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Form the noisy intensity image a sensor would record.

    The order of operations carries the physics and is easy to get wrong.

    With ``normalize_by_peak`` (the published setting) the PSF is divided by the
    peak of the *reference circle's* PSF before noise is added. That fixes one
    absolute intensity scale for every pupil, so a single ``noise_sigma`` means
    the same photon count throughout. A small or spindly pupil passes less
    light, so its normalised PSF is dimmer and its effective SNR is lower. That
    penalty is the light-throughput cost of asymmetry, and it appears here
    rather than in any explicit throughput factor.

    Without it, each PSF is first scaled to unit peak and then multiplied by its
    own area ratio, an equivalent intent but applied per-image.

    The final renormalisation to unit peak happens *after* noise, keeping the
    network's input range fixed without undoing the SNR difference above. Note
    that in the noiseless case this cancels the peak normalisation exactly, so
    ``normalize_by_peak`` is then a no-op.

    Args:
        pupil, phase: ``[B,1,H,W]``. ``phase`` must already be masked.
        circle_psf: PSF of the reference circle, from ``reference_circle``.
        generator: Optional RNG for reproducible noise.

    Returns:
        Intensity images ``[B,1,H,W]``.
    """
    if cfg.normalize_by_peak:
        psf = psf_from_pupil(pupil, phase, normalize=False)
        if circle_psf.dim() == 4:
            psf = psf / circle_psf.amax(dim=(-2, -1), keepdim=True)
        else:
            psf = psf / circle_psf.max()
    else:
        psf = psf_from_pupil(pupil, phase, normalize=True)
        if cfg.use_throughput:
            circle_area = float(np.pi * cfg.radius**2)
            psf = psf * (pupil.sum(dim=(-2, -1), keepdim=True) / circle_area)

    if cfg.with_noise:
        noise = torch.randn(psf.shape, device=psf.device, dtype=psf.dtype, generator=generator)
        psf = (psf + noise * cfg.noise_sigma).clamp_min(0.0)

    if cfg.renormalize_after_noise:
        psf = psf / psf.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    return psf


class PupilPhaseDataset(Dataset):
    """Every pupil crossed with every phase.

    Pupils and phases are stored as independent pools and paired by index
    arithmetic, so ``N_pupils * N_phases`` examples cost only ``N_pupils +
    N_phases`` in memory. Splits are made over the *pools*, so a test pupil is
    never seen during training under any phase.

    Yields ``(masked_phase, pupil, pupil_index)``. Intensity is not formed here:
    :func:`simulate_measurement` runs on the training device, which keeps noise
    resampled every epoch and avoids moving images through worker processes.
    """

    def __init__(self, phases: torch.Tensor, pupils: torch.Tensor):
        self.phases = phases.float()
        self.pupils = pupils.float()
        self.num_phases = self.phases.shape[0]
        self.num_pupils = self.pupils.shape[0]

    def __len__(self) -> int:
        return self.num_pupils * self.num_phases

    def __getitem__(self, idx: int):
        pupil_idx, phase_idx = divmod(idx, self.num_phases)
        pupil = self.pupils[pupil_idx]
        masked_phase = pupil * self.phases[phase_idx]
        return masked_phase.unsqueeze(0), pupil.unsqueeze(0), pupil_idx


def build_pools(cfg: ExperimentConfig) -> tuple[dict, dict]:
    """Generate pupil and phase pools for all three splits."""
    rng = np.random.default_rng(cfg.seed)
    generate = generate_binary_pupils if cfg.binary_pupils else generate_polygon_pupils

    pupil_pools = {"meta": {"bbox": cfg.bounding_box_size, "radius": cfg.radius}}
    for split, count in [
        ("train", cfg.train_pupils),
        ("val", cfg.val_pupils),
        ("test", cfg.test_pupils),
    ]:
        kwargs = {} if cfg.binary_pupils else {"rng": rng}
        pupil_pools[split] = generate(
            radius=cfg.radius,
            bounding_box_size=cfg.bounding_box_size,
            count=count,
            n_bins=cfg.n_bins,
            **kwargs,
        )

    modes = SMALL_MODES if cfg.small_modes else PAPER_MODES
    phase_pools = {
        "meta": {"bbox": cfg.bounding_box_size, "radius": cfg.radius, "modes": modes},
    }
    for split, count in [
        ("train", cfg.train_phases),
        ("val", cfg.val_phases),
        ("test", cfg.test_phases),
    ]:
        arr = random_zernike_phase(
            circle_radius=cfg.radius,
            image_size=cfg.bounding_box_size,
            num_phases=count,
            modes=modes,
            coeff_scale=cfg.coeff_scale,
            rng=rng,
        )
        phase_pools[f"{split}_phases"] = torch.from_numpy(arr).float()

    return pupil_pools, phase_pools


def load_or_build_pools(cfg: ExperimentConfig, data_root: str | Path) -> tuple[dict, dict]:
    """Load cached pools, generating and caching them on first use.

    The cache directory is keyed by :meth:`ExperimentConfig.dataset_key`, so all
    runs sharing a data configuration reuse one pool. This is what lets a noise
    sweep be compared pupil by pupil.
    """
    data_dir = Path(data_root) / f"dataset_jan2026_{cfg.dataset_key()}"
    pupil_path, phase_path = data_dir / PUPIL_CACHE_NAME, data_dir / PHASE_CACHE_NAME

    if pupil_path.exists() and phase_path.exists():
        return (
            torch.load(pupil_path, weights_only=False),
            torch.load(phase_path, weights_only=False),
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    pupil_pools, phase_pools = build_pools(cfg)
    torch.save(pupil_pools, pupil_path)
    torch.save(phase_pools, phase_path)
    return pupil_pools, phase_pools
