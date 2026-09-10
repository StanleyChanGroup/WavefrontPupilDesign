"""Training entry point.

Run as ``pupil-design-train`` or ``python -m pupil_design.train``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import PupilPhaseDataset, load_or_build_pools, simulate_measurement
from .losses import masked_piston_corrected_mse
from .models import build_model
from .optics import psf_from_pupil, reference_circle

__all__ = ["train", "main"]


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loaders(cfg: ExperimentConfig, data_root: Path):
    pupil_pools, phase_pools = load_or_build_pools(cfg, data_root)
    loaders = {}
    for split in ("train", "val"):
        dataset = PupilPhaseDataset(
            phases=phase_pools[f"{split}_phases"],
            pupils=pupil_pools[split]["masks"],
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
    return loaders, pupil_pools, phase_pools


@torch.no_grad()
def _validate(model, loader, cfg, circle_psf, device) -> float:
    model.eval()
    total, batches = 0.0, 0
    for phase_gt, mask, _ in loader:
        phase_gt = phase_gt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        psf = simulate_measurement(mask, phase_gt, circle_psf, cfg)
        inp = torch.cat([psf, mask], dim=1) if cfg.with_pupil else psf
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(inp)
        total += float(masked_piston_corrected_mse(pred.float(), phase_gt, mask))
        batches += 1
    model.train()
    return total / max(batches, 1)


def train(cfg: ExperimentConfig, data_root: str | Path, out_dir: str | Path) -> Path:
    """Train one model and write weights, config and history to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.save(out_dir / "config.json")

    loaders, _, _ = _make_loaders(cfg, Path(data_root))
    circle = reference_circle(cfg.bounding_box_size, cfg.radius, device=device)
    circle_psf = psf_from_pupil(circle, torch.zeros_like(circle))

    model = build_model(cfg.model_type, cfg.bounding_box_size, cfg.with_pupil, cfg.narrow).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    print(f"Training {cfg.run_name()} on {device} | "
          f"{len(loaders['train'].dataset):,} train, {len(loaders['val'].dataset):,} val examples")

    history: list[dict] = []
    best_val = float("inf")

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch + 1}/{cfg.num_epochs}", dynamic_ncols=True)
        for phase_gt, mask, _ in pbar:
            phase_gt = phase_gt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Noise is resampled here, so every epoch sees a fresh realisation.
            psf = simulate_measurement(mask, phase_gt, circle_psf, cfg)
            inp = torch.cat([psf, mask], dim=1) if cfg.with_pupil else psf

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(inp)
            loss = masked_piston_corrected_mse(pred.float(), phase_gt, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.detach().item():.4e}")

        val_loss = _validate(model, loaders["val"], cfg, circle_psf, device)
        history.append({"epoch": epoch, "val_loss": val_loss})
        print(f"  epoch {epoch + 1}: val loss {val_loss:.6e}")

        torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch}.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print("  new best")

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"Done. Best val loss {best_val:.6e}. Weights in {out_dir}")
    return out_dir / "best_model.pth"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a wavefront estimator on asymmetry-balanced pupils.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", dest="model_type", default="unet", choices=["unet", "mlp"])
    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="Gaussian noise std. Paper uses 0, 0.001, 0.01, 0.03")
    p.add_argument("--epochs", dest="num_epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--radius", type=int, default=19)
    p.add_argument("--bbox", dest="bounding_box_size", type=int, default=128)
    p.add_argument("--binary-pupils", action="store_true", help="Not used in the paper")
    p.add_argument("--small-modes", action="store_true", help="12 Zernike modes instead of 33")
    p.add_argument("--no-pupil-input", action="store_true",
                   help="Withhold the pupil mask from the network")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data", help="Where pupil/phase pools are cached")
    p.add_argument("--out", default=None, help="Output directory (default: ./runs/<run_name>)")
    p.add_argument("--demo", action="store_true",
                   help="Tiny run for smoke-testing: small pools, one epoch")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg = ExperimentConfig(
        model_type=args.model_type,
        with_pupil=not args.no_pupil_input,
        bounding_box_size=args.bounding_box_size,
        radius=args.radius,
        noise_sigma=args.noise_sigma,
        binary_pupils=args.binary_pupils,
        small_modes=args.small_modes,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
    )
    if args.demo:
        cfg.train_pupils, cfg.val_pupils, cfg.test_pupils = 60, 30, 30
        cfg.train_phases, cfg.val_phases, cfg.test_phases = 40, 20, 20
        cfg.n_bins, cfg.num_epochs = 10, 1

    out = Path(args.out) if args.out else Path("runs") / cfg.run_name()
    train(cfg, args.data_root, out)


if __name__ == "__main__":
    main()
