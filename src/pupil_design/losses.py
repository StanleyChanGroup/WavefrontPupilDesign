"""Loss and error metrics defined only inside the pupil."""
from __future__ import annotations

import torch

__all__ = ["masked_piston_corrected_mse", "masked_piston_corrected_mse_per_sample"]


def masked_piston_corrected_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample MSE inside the pupil, with piston removed.

    Two corrections make this the right error measure for the problem. Pixels
    outside the pupil carry no light and are excluded rather than counted as
    zeros. And a constant phase offset (piston) leaves the intensity image
    unchanged, so it is unobservable and must not be penalised: both fields have
    their in-pupil mean subtracted before comparison.

    Args:
        pred, target, mask: ``[B,1,H,W]``.

    Returns:
        Tensor of shape ``[B]``.
    """
    pred_m = pred * mask
    target_m = target * mask

    mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-9)
    pred_c = (pred_m - pred_m.sum(dim=(2, 3), keepdim=True) / mask_sum) * mask
    target_c = (target_m - target_m.sum(dim=(2, 3), keepdim=True) / mask_sum) * mask

    sq_err = ((pred_c - target_c) ** 2).sum(dim=(2, 3))
    return (sq_err / mask_sum.squeeze(-1).squeeze(-1)).squeeze(1)


def masked_piston_corrected_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Batch-mean of :func:`masked_piston_corrected_mse_per_sample`."""
    return masked_piston_corrected_mse_per_sample(pred, target, mask).mean()
