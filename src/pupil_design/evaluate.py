"""Evaluation entry point: error and Strehl as a function of pupil asymmetry.

Run as ``pupil-design-eval`` or ``python -m pupil_design.evaluate``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ExperimentConfig, parse_legacy_dirname
from .data import load_or_build_pools, simulate_measurement
from .losses import masked_piston_corrected_mse_per_sample
from .models import build_model
from .optics import psf_from_pupil, reference_circle, strehl_ratio

__all__ = ["evaluate_per_pupil", "bin_by_asymmetry", "load_run", "main"]


def load_run(run_dir: str | Path) -> tuple[ExperimentConfig, Path]:
    """Read a run's config, falling back to its directory name for legacy runs."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = ExperimentConfig.load(config_path)
    else:
        cfg = parse_legacy_dirname(run_dir.name)
        print(f"No config.json in {run_dir}; recovered settings from directory name.")
    weights = run_dir / "best_model.pth"
    if not weights.exists():
        raise FileNotFoundError(f"No best_model.pth in {run_dir}")
    return cfg, weights


@torch.no_grad()
def evaluate_per_pupil(
    cfg: ExperimentConfig,
    weights: str | Path,
    data_root: str | Path,
    split: str = "test",
    device: torch.device | None = None,
    phase_chunk: int = 250,
    max_pupils: int | None = None,
    compute_strehl: bool = True,
) -> dict:
    """Average wavefront error and Strehl ratio for each pupil in a split.

    Every pupil is evaluated against every phase in the pool, so the per-pupil
    average isolates the effect of pupil shape from that of the particular
    aberration drawn.

    Returns:
        Dict of arrays keyed ``asym``, ``loss``, ``strehl``, ``count``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pupil_pools, phase_pools = load_or_build_pools(cfg, data_root)

    pupils = pupil_pools[split]["masks"]
    asym = np.asarray(pupil_pools[split]["asym"], dtype=np.float64)
    phases = phase_pools[f"{split}_phases"].to(device)

    if max_pupils is not None and max_pupils < pupils.shape[0]:
        # Pupils are stored in asymmetry-bin order, so the first N would all sit
        # at the low-asymmetry end and the summary below would be meaningless.
        # Take an evenly spaced subset so a quick check still spans the range.
        total = pupils.shape[0]
        idx = np.linspace(0, total - 1, max_pupils).round().astype(int)
        pupils, asym = pupils[idx], asym[idx]
        print(f"Spot check: {max_pupils} of {total} pupils, sampled evenly across "
              f"the asymmetry range. Omit --max-pupils for the full evaluation.")
    pupils = pupils.to(device)

    model = build_model(cfg.model_type, cfg.bounding_box_size, cfg.with_pupil, cfg.narrow).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    circle = reference_circle(cfg.bounding_box_size, cfg.radius, device=device)
    circle_psf = psf_from_pupil(circle, torch.zeros_like(circle))

    n_pupils, n_phases = pupils.shape[0], phases.shape[0]
    loss_out = np.zeros(n_pupils)
    strehl_out = np.full(n_pupils, np.nan)

    for pi in tqdm(range(n_pupils), desc=f"{split} pupils", dynamic_ncols=True):
        pupil = pupils[pi]
        loss_sum, strehl_sum = 0.0, 0.0
        for start in range(0, n_phases, phase_chunk):
            block = phases[start:start + phase_chunk]
            mask = pupil.expand(block.shape[0], -1, -1).unsqueeze(1)
            target = (pupil.unsqueeze(0) * block).unsqueeze(1)

            psf = simulate_measurement(mask, target, circle_psf, cfg)
            inp = torch.cat([psf, mask], dim=1) if cfg.with_pupil else psf
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                pred = model(inp)
            pred = pred.float()

            loss_sum += float(masked_piston_corrected_mse_per_sample(pred, target, mask).sum())
            if compute_strehl:
                strehl_sum += float(strehl_ratio(target, pred, mask).clamp(max=1.0).sum())

        loss_out[pi] = loss_sum / n_phases
        if compute_strehl:
            strehl_out[pi] = strehl_sum / n_phases

    return {
        "asym": asym,
        "loss": loss_out,
        "strehl": strehl_out,
        "count": np.full(n_pupils, n_phases, dtype=np.int64),
    }


def bin_by_asymmetry(asym: np.ndarray, values: np.ndarray, n_bins: int = 30) -> dict:
    """Mean and spread of ``values`` within equal-width asymmetry bins."""
    edges = np.linspace(asym.min(), asym.max(), n_bins + 1)
    bin_id = np.clip(np.digitize(asym, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = values[bin_id == b]
        count[b] = sel.size
        if sel.size:
            mean[b], std[b] = sel.mean(), sel.std()
    return {"edges": edges, "centers": centers, "mean": mean, "std": std, "count": count}


def _plot(results: dict, out_dir: Path, n_bins: int, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, label, better in [
        ("loss", "Wavefront MSE", "lower is better"),
        ("strehl", "Strehl ratio", "higher is better"),
    ]:
        values = results[key]
        if not np.isfinite(values).any():
            continue
        binned = bin_by_asymmetry(results["asym"], values, n_bins)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(results["asym"], values, s=8, alpha=0.25, color="#8899aa",
                   label="individual pupils")
        ok = np.isfinite(binned["mean"])
        ax.errorbar(binned["centers"][ok], binned["mean"][ok], yerr=binned["std"][ok],
                    fmt="o-", capsize=3, color="#c0392b", label=f"binned mean ({n_bins} bins)")
        ax.set_xlabel("Pupil asymmetry")
        ax.set_ylabel(f"{label} ({better})")
        ax.set_title(f"{label} vs pupil asymmetry, {split} split")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{split}_{key}_vs_asymmetry.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a trained model: error and Strehl vs pupil asymmetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", help="Directory holding best_model.pth and config.json")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--max-pupils", type=int, default=None,
                   help="Evaluate only the first N pupils, for a quick check")
    p.add_argument("--no-strehl", action="store_true")
    p.add_argument("--out", default=None, help="Where to write results (default: run_dir)")
    args = p.parse_args(argv)

    cfg, weights = load_run(args.run_dir)
    out_dir = Path(args.out) if args.out else Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_per_pupil(
        cfg, weights, args.data_root, split=args.split,
        max_pupils=args.max_pupils, compute_strehl=not args.no_strehl,
    )

    binned_loss = bin_by_asymmetry(results["asym"], results["loss"], args.bins)
    npz_path = out_dir / f"{args.split}_metrics.npz"
    np.savez_compressed(
        npz_path,
        asym=results["asym"], loss=results["loss"], strehl=results["strehl"],
        count=results["count"], bin_centers=binned_loss["centers"],
        loss_bin_mean=binned_loss["mean"], loss_bin_std=binned_loss["std"],
        loss_bin_n=binned_loss["count"], num_bins=args.bins,
    )
    _plot(results, out_dir, args.bins, args.split)

    lo, hi = results["asym"].min(), results["asym"].max()
    third = (hi - lo) / 3
    low_mask, high_mask = results["asym"] < lo + third, results["asym"] > hi - third
    print(f"\nMean error, least asymmetric third : {results['loss'][low_mask].mean():.4f}")
    print(f"Mean error, most asymmetric third  : {results['loss'][high_mask].mean():.4f}")
    print(f"\nWrote {npz_path} and plots to {out_dir}")


if __name__ == "__main__":
    main()
