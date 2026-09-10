"""Fourier optics primitives: PSF formation, Zernike phase, Strehl ratio.

Numerics here are byte-compatible with the original research scripts
(`trainer_jan_2026.py`). Do not "clean up" the FFT shift order or the
normalisation clamps without re-running tests/test_regression.py.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import factorial

__all__ = [
    "psf_from_pupil",
    "reference_circle",
    "zernike",
    "zernike_radial",
    "random_zernike_phase",
    "strehl_ratio",
    "PAPER_MODES",
    "SMALL_MODES",
]

# Zernike (n, m) sets. PAPER_MODES is the 33-mode set behind every published
# figure; SMALL_MODES is the reduced set used only in an early ablation.
PAPER_MODES: list[tuple[int, int]] = [
    (2, 0), (2, -2), (2, 2),
    (3, -1), (3, 1), (3, -3), (3, 3),
    (4, 0), (4, -2), (4, 2), (4, -4), (4, 4),
    (5, -1), (5, 1), (5, -3), (5, 3), (5, -5), (5, 5),
    (6, 0), (6, -2), (6, 2), (6, -4), (6, 4), (6, -6), (6, 6),
    (7, -1), (7, 1), (7, -3), (7, 3), (7, -5), (7, 5), (7, -7), (7, 7),
]

SMALL_MODES: list[tuple[int, int]] = [
    (2, 0), (2, -2), (2, 2),
    (3, -1), (3, 1), (3, -3), (3, 3),
    (4, 0), (4, -2), (4, 2), (4, -4), (4, 4),
]


def psf_from_pupil(
    pupil: torch.Tensor,
    phase: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Incoherent PSF of a pupil carrying a phase aberration.

    Forms the complex pupil function ``P * exp(i*phi)`` and propagates it to the
    focal plane by FFT. Accepts any leading batch/channel dims; the last two are
    spatial.

    Args:
        pupil: Real amplitude mask, typically binary.
        phase: Phase in radians, same shape as ``pupil``.
        normalize: Divide each PSF by its own peak. The clamp floor of 1e-4
            matches the original implementation.
    """
    pupil_func = pupil.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
    field = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(pupil_func, dim=(-2, -1)),
            dim=(-2, -1),
        ),
        dim=(-2, -1),
    )
    psf = torch.abs(field) ** 2
    if normalize:
        psf = psf / psf.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    return psf


def reference_circle(bounding_box_size: int, radius: int, device=None) -> torch.Tensor:
    """Binary circular pupil of the given radius, the throughput reference.

    The circle is the unit against which every other pupil's light throughput
    and PSF peak are measured, so it defines the absolute intensity scale.
    """
    coords = torch.linspace(-1, 1, bounding_box_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist = torch.sqrt(xx**2 + yy**2)
    return (dist <= (radius / (bounding_box_size / 2))).float()


def zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Radial polynomial R_n^m."""
    R = np.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        c = (-1) ** k * factorial(n - k) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
        R += c * r ** (n - 2 * k)
    return R


def zernike(n: int, m: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Zernike polynomial Z_n^m on polar coordinates, unnormalised."""
    if m >= 0:
        return zernike_radial(n, m, r) * np.cos(m * theta)
    return zernike_radial(n, -m, r) * np.sin(-m * theta)


def random_zernike_phase(
    circle_radius: int,
    image_size: int,
    num_phases: int = 1,
    modes: list[tuple[int, int]] | None = None,
    coeff_scale: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random wavefronts as uniformly weighted sums of Zernike modes.

    Coefficients are drawn uniformly from ``[-coeff_scale, coeff_scale]``. A
    uniform draw, rather than Gaussian, keeps aberration strength evenly covered
    instead of concentrating samples near zero.

    Returns:
        Array of shape ``(num_phases, image_size, image_size)``, zero outside
        the unit circle.
    """
    if modes is None:
        modes = PAPER_MODES
    y, x = np.indices((image_size, image_size)) - image_size // 2
    r = np.sqrt(x**2 + y**2) / circle_radius
    theta = np.arctan2(y, x)
    mask = r <= 1

    draw = rng.uniform if rng is not None else np.random.uniform
    coeffs = draw(-coeff_scale, coeff_scale, size=(num_phases, len(modes)))

    phases = np.zeros((num_phases, image_size, image_size))
    r_b, theta_b = r[None, :, :], theta[None, :, :]
    for idx, (n, m) in enumerate(modes):
        phases += coeffs[:, idx][:, None, None] * zernike(n, m, r_b, theta_b)
    return phases * mask[None, :, :]


def strehl_ratio(
    gt_phase: torch.Tensor,
    pred_phase: torch.Tensor,
    pupil_mask: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Per-sample Strehl ratio of the residual wavefront.

    Measures how well a predicted phase would correct the true one: the peak of
    the PSF formed by the residual ``gt - pred``, divided by the diffraction
    limited peak of the same pupil. 1.0 means perfect correction.

    Args:
        gt_phase, pred_phase: ``[B,1,H,W]`` (2D and 3D inputs are promoted).
        pupil_mask: Broadcastable mask of the same spatial size.

    Returns:
        Tensor of shape ``[B]``.
    """
    if gt_phase.shape != pred_phase.shape:
        raise ValueError(f"gt_phase {tuple(gt_phase.shape)} != pred_phase {tuple(pred_phase.shape)}")

    if pupil_mask.dim() == 2:
        pupil_mask = pupil_mask.unsqueeze(0).unsqueeze(0)
    elif pupil_mask.dim() == 3:
        pupil_mask = pupil_mask.unsqueeze(1)
    elif pupil_mask.dim() != 4:
        raise ValueError(f"Unsupported pupil_mask shape: {tuple(pupil_mask.shape)}")

    if gt_phase.dim() == 2:
        gt_phase, pred_phase = gt_phase[None, None], pred_phase[None, None]
    elif gt_phase.dim() == 3:
        gt_phase, pred_phase = gt_phase.unsqueeze(1), pred_phase.unsqueeze(1)

    residual = (gt_phase - pred_phase) * pupil_mask
    corrected = psf_from_pupil(pupil_mask, residual)
    diffraction_limited = psf_from_pupil(pupil_mask, torch.zeros_like(residual))

    peak_corr = corrected.amax(dim=(-2, -1))
    peak_ref = diffraction_limited.amax(dim=(-2, -1)).clamp_min(eps)
    return (peak_corr / peak_ref).squeeze(1)
