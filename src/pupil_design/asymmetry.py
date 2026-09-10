"""The asymmetry metric, the paper's central quantity.

A pupil's recoverability is governed by how far it departs from 180-degree
rotational symmetry. A symmetric pupil produces a PSF that cannot distinguish a
wavefront from its conjugate, so the inverse problem is degenerate; breaking
that symmetry lifts the degeneracy.
"""
from __future__ import annotations

import torch

__all__ = ["asymmetry", "ASYM_RANGE_POLYGON", "ASYM_RANGE_BINARY"]

# Empirical spans of the metric over each pupil family. Used to place bin edges
# so that asymmetry is sampled uniformly rather than however the generator
# happens to favour.
ASYM_RANGE_POLYGON = (0.0266, 0.3600)
ASYM_RANGE_BINARY = (0.1, 0.8)


def asymmetry(pupil_mask: torch.Tensor, align: str = "published") -> torch.Tensor:
    """Asymmetry of a pupil mask, in ``[0, 1)``.

    Computes ``1 - overlap`` between the mask and its 180-degree rotation, after
    aligning the two by the peak of their cross-correlation. The alignment step
    matters: without it, a perfectly symmetric pupil that is merely off-centre
    would score as asymmetric, conflating position with shape.

    The correlation is evaluated in the Fourier domain. Correlating ``x`` with
    ``flip(x)`` is the autoconvolution of ``x``, which is why the transform is
    squared rather than multiplied by a conjugate.

    Alignment modes
    ---------------
    ``align="published"`` (default)
        Reproduces the metric used for arXiv:2604.00225 exactly. It rolls the
        flipped mask by the autoconvolution argmax. For an ``N``-point axis,
        ``flip`` reflects about index ``(N-1)/2``, so the autoconvolution peaks
        at ``N-1`` while the roll that undoes the flip is ``N``. The alignment
        therefore lands one pixel off in each axis.

        The consequence is a positive floor on the metric that scales with a
        pupil's perimeter-to-area ratio: an exactly symmetric circle of radius
        19 in a 128-pixel box scores 0.047 rather than 0. Because the offset
        depends on shape rather than being constant, it is not absorbed by a
        change of origin.

    ``align="corrected"``
        Adds the missing pixel, so an exactly symmetric mask scores 0.

    Use ``"published"`` for anything compared against released weights, cached
    pupil pools, or figures from the paper, and ``"corrected"`` for new work.

    Args:
        pupil_mask: ``[H,W]``, ``[1,H,W]``, ``[B,H,W]`` or ``[B,1,H,W]``.
        align: ``"published"`` or ``"corrected"``.

    Returns:
        Scalar for a single 2D mask, otherwise ``[B]``.
    """
    if align not in ("published", "corrected"):
        raise ValueError(f"align must be 'published' or 'corrected', got {align!r}")

    x = pupil_mask
    if not torch.is_tensor(x):
        x = torch.as_tensor(x.copy()).float()

    input_dims = x.dim()
    if input_dims == 2:
        x = x.unsqueeze(0)
        single = True
    elif input_dims == 3:
        single = x.size(0) == 1
    elif input_dims == 4:
        if x.size(1) != 1:
            raise ValueError("Expected channel=1 for [B,C,H,W] input.")
        x = x[:, 0]
        single = False
    else:
        raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

    B, _, W = x.shape
    mass = x.sum(dim=(1, 2)).clamp_min(1e-12)
    x_flip = torch.flip(x, dims=(1, 2))

    X = torch.fft.fft2(x)
    corr_map = torch.fft.ifft2(X * X).abs()

    flat_idx = corr_map.reshape(B, -1).argmax(dim=1)
    shift_y, shift_x = flat_idx // W, flat_idx % W
    if align == "corrected":
        shift_y, shift_x = shift_y + 1, shift_x + 1

    aligned = torch.empty_like(x_flip)
    for b in range(B):
        aligned[b] = torch.roll(x_flip[b], shifts=(int(shift_y[b]), int(shift_x[b])), dims=(0, 1))

    overlap = (x * aligned).sum(dim=(1, 2))
    asym = 1.0 - overlap / mass
    return asym[0] if single and input_dims == 2 else asym
