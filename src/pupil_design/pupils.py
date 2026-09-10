"""Pupil mask generation and asymmetry-balanced sampling.

Two families are supported. Random convex polygons are the family behind every
published figure. Binary threshold masks were explored but excluded from the
paper; they remain here because the generator is cheap and the comparison is
informative.

The sampling strategy matters more than the generator. Rejection sampling into
equal-width asymmetry bins gives a flat marginal over asymmetry, so trends in
error against asymmetry cannot be an artefact of uneven sampling density.
"""
from __future__ import annotations

import numpy as np
import torch

from .asymmetry import asymmetry, ASYM_RANGE_POLYGON, ASYM_RANGE_BINARY
from .optics import reference_circle

__all__ = [
    "rasterize_polygon",
    "random_convex_polygon",
    "polygon_area_ratio",
    "generate_polygon_pupils",
    "generate_binary_pupils",
]


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for polygon rasterization") from exc
    return cv2


def rasterize_polygon(bounding_box_size: int, verts: np.ndarray, radius: int) -> np.ndarray:
    """Fill a polygon into a centred binary mask.

    Vertices are centred on the image, not on a radius-dependent grid point. An
    earlier version shifted by ``int((bbox/r)/2)*r``, which decentred the pupil
    for most radii and biased the asymmetry metric.
    """
    cv2 = _require_cv2()
    shift = bounding_box_size // 2
    canvas = np.zeros((bounding_box_size, bounding_box_size), dtype=np.uint8)
    pts = verts.astype(np.int32).reshape((-1, 1, 2)) + shift
    mask = cv2.fillPoly(canvas, [pts], 1)
    # Flip to image convention. ascontiguousarray drops the negative stride the
    # slice introduces, which torch.from_numpy rejects.
    return np.ascontiguousarray(mask[::-1, :])


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace area of a simple polygon."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_area_ratio(verts: np.ndarray, radius: int) -> float:
    """Polygon area as a fraction of the reference circle's area."""
    return polygon_area(verts[:, 0], verts[:, 1]) / (np.pi * radius**2)


def random_convex_polygon(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random convex polygon with ``n`` vertices, centred at origin.

    Uses the Valtr construction: random x and y increments are split into two
    monotone chains, paired, sorted by angle, then accumulated. This yields a
    closed convex polygon directly, with no rejection step.
    """
    xy = np.sort(rng.random((n, 2)), axis=0)
    take = rng.choice([True, False], n - 2)
    pos, neg = xy[1:-1][take], xy[1:-1][~take]
    pos = np.vstack((xy[0], pos, xy[-1]))
    neg = np.vstack((xy[0], neg, xy[-1]))

    vertices = np.vstack((pos[1:] - pos[:-1], neg[:-1] - neg[1:]))
    rng.shuffle(vertices[:, 1])
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
    vertices = np.cumsum(vertices, axis=0)

    x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    vertices[:, 0] -= (x_max + np.min(vertices[:, 0])) / 2
    vertices[:, 1] -= (y_max + np.min(vertices[:, 1])) / 2
    return vertices


def random_convex_polygon_in_circle(n: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Random convex polygon rescaled to be inscribed in a circle."""
    vertices = random_convex_polygon(n, rng)
    max_r = np.max(np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2))
    if max_r > 0:
        vertices = vertices * (radius / max_r)
    return vertices


def _bin_targets(total: int, n_bins: int) -> np.ndarray:
    """Split ``total`` samples as evenly as possible across ``n_bins``."""
    base, rem = divmod(total, n_bins)
    return np.array([base + 1 if i < rem else base for i in range(n_bins)], dtype=int)


def generate_polygon_pupils(
    radius: int,
    bounding_box_size: int,
    count: int,
    n_bins: int = 30,
    asym_range: tuple[float, float] = ASYM_RANGE_POLYGON,
    min_area_ratio: float = 0.3,
    min_verts: int = 3,
    max_verts: int = 50,
    max_attempts_factor: int = 400,
    rng: np.random.Generator | None = None,
) -> dict:
    """Sample convex polygon pupils, balanced across asymmetry bins.

    Candidates below ``min_area_ratio`` of the circle's area are discarded: very
    thin slivers pass almost no light, so their PSFs are dominated by noise and
    they say nothing useful about shape.

    Returns:
        Dict with ``masks`` ``[N,H,W]``, ``verts_norm`` (list of vertex arrays
        scaled by radius, so pupils can be re-rasterized at any resolution) and
        ``asym`` ``[N]``.
    """
    rng = rng or np.random.default_rng(42)
    min_asym, max_asym = asym_range
    bin_width = (max_asym - min_asym) / n_bins

    targets = _bin_targets(count, n_bins)
    counts = np.zeros(n_bins, dtype=int)
    masks_bins: list[list] = [[] for _ in range(n_bins)]
    verts_bins: list[list] = [[] for _ in range(n_bins)]
    asym_bins: list[list] = [[] for _ in range(n_bins)]

    collected = 0
    attempts = 0
    max_attempts = count * max_attempts_factor
    vertex_choices = np.arange(min_verts, max_verts)

    while collected < count and attempts < max_attempts:
        attempts += 1
        n_verts = int(rng.choice(vertex_choices))
        verts = random_convex_polygon_in_circle(n_verts, radius, rng)
        if polygon_area_ratio(verts, radius) < min_area_ratio:
            continue

        mask = rasterize_polygon(bounding_box_size, verts, radius)
        asym = float(asymmetry(torch.from_numpy(mask).float()))
        if not (min_asym <= asym < max_asym):
            continue

        b = min(max(int((asym - min_asym) / bin_width), 0), n_bins - 1)
        if counts[b] >= targets[b]:
            continue

        masks_bins[b].append(mask.astype(np.uint8))
        verts_bins[b].append((verts / float(radius)).astype(np.float32))
        asym_bins[b].append(asym)
        counts[b] += 1
        collected += 1

    if collected < count:
        raise RuntimeError(
            f"Only collected {collected}/{count} pupils in {attempts} attempts. "
            f"Per-bin counts: {counts.tolist()}"
        )

    return {
        "masks": torch.from_numpy(np.stack([m for b in masks_bins for m in b])).float(),
        "verts_norm": [v for b in verts_bins for v in b],
        "asym": np.array([a for b in asym_bins for a in b], dtype=np.float32),
    }


def generate_binary_pupils(
    radius: int,
    bounding_box_size: int,
    count: int,
    n_bins: int = 30,
    asym_range: tuple[float, float] = ASYM_RANGE_BINARY,
    min_area_ratio: float = 0.3,
    thr_low: float = 0.1,
    thr_high: float = 0.9,
    batch_size: int = 128,
    max_attempts_factor: int = 2000,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
) -> dict:
    """Sample binary (thresholded-noise) pupils, balanced across asymmetry bins.

    Not used in the paper. Kept because these masks reach far higher asymmetry
    than convex polygons can, which is useful for probing the metric's range.
    """
    del rng  # candidate masks are drawn on-device via torch
    min_asym, max_asym = asym_range
    bin_width = (max_asym - min_asym) / n_bins

    circle = reference_circle(bounding_box_size, radius, device=device)
    circle_area = circle.sum().clamp_min(1.0)

    targets = _bin_targets(count, n_bins)
    counts = np.zeros(n_bins, dtype=int)
    masks_bins: list[list] = [[] for _ in range(n_bins)]
    asym_bins: list[list] = [[] for _ in range(n_bins)]

    collected = 0
    attempts = 0
    max_attempts = count * max_attempts_factor

    while collected < count and attempts < max_attempts:
        B = min(batch_size, max_attempts - attempts)
        attempts += B

        thresholds = torch.empty((B, 1, 1), device=device).uniform_(thr_low, thr_high)
        noise = torch.rand((B, bounding_box_size, bounding_box_size), device=device)
        candidates = (noise >= thresholds).float() * circle

        areas = candidates.sum(dim=(-2, -1)) / circle_area
        for i in torch.nonzero(areas >= min_area_ratio, as_tuple=False).squeeze(1).tolist():
            if collected >= count:
                break
            mask_i = candidates[i]
            asym = float(asymmetry(mask_i))
            if not (min_asym <= asym < max_asym):
                continue
            b = min(max(int((asym - min_asym) / bin_width), 0), n_bins - 1)
            if counts[b] >= targets[b]:
                continue
            masks_bins[b].append(mask_i.detach().cpu().to(torch.uint8).numpy())
            asym_bins[b].append(asym)
            counts[b] += 1
            collected += 1

    if collected < count:
        raise RuntimeError(
            f"Only collected {collected}/{count} binary pupils. Per-bin counts: {counts.tolist()}"
        )

    return {
        "masks": torch.from_numpy(np.stack([m for b in masks_bins for m in b])).float(),
        "verts_norm": None,
        "asym": np.array([a for b in asym_bins for a in b], dtype=np.float32),
    }
