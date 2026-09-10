"""Experiment configuration.

The original scripts encoded every setting in the output directory name and
recovered it later by regular expression. That coupling meant a rename silently
changed how a model was evaluated. Here a run writes ``config.json`` next to its
weights, and evaluation reads that file. :func:`parse_legacy_dirname` exists
only to migrate pre-existing result folders.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ExperimentConfig", "parse_legacy_dirname"]


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce a training run."""

    # Model
    model_type: str = "unet"
    with_pupil: bool = True
    narrow: bool = False

    # Optics
    bounding_box_size: int = 128
    radius: int = 19

    # Measurement model
    noise_sigma: float = 0.0
    normalize_by_peak: bool = True
    use_throughput: bool = True
    renormalize_after_noise: bool = True

    # Data
    binary_pupils: bool = False
    small_modes: bool = False
    n_bins: int = 30
    train_pupils: int = 6000
    val_pupils: int = 1500
    test_pupils: int = 1500
    train_phases: int = 5000
    val_phases: int = 1000
    test_phases: int = 1000
    coeff_scale: float = 2.0

    # Optimisation
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 4
    seed: int = 42

    # Provenance
    notes: str = ""

    @property
    def with_noise(self) -> bool:
        return self.noise_sigma > 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def dataset_key(self) -> str:
        """Identity of the pupil/phase pool this run needs.

        Only fields that change the generated data appear here, so runs that
        differ solely in noise or architecture share one cached dataset.
        """
        parts = [f"bbox_{self.bounding_box_size}", f"radius_{self.radius}"]
        if self.small_modes:
            parts.append("small_modes")
        if self.binary_pupils:
            parts.append("binary_pupil")
        return "_".join(parts)

    def run_name(self) -> str:
        """Short, readable directory name. Not parsed by anything."""
        noise = "no_noise" if not self.with_noise else f"noise_{self.noise_sigma:g}"
        parts = [self.model_type, noise]
        if self.binary_pupils:
            parts.append("binary")
        if self.small_modes:
            parts.append("small_modes")
        if not self.with_pupil:
            parts.append("no_pupil_input")
        return "_".join(parts)


_LEGACY_NOISE = re.compile(r"with_noise_([\d.eE+-]+?)_")


def parse_legacy_dirname(name: str) -> ExperimentConfig:
    """Recover a config from an original long-form experiment directory name.

    Migration aid for results produced before configs were written to disk.
    Prefer reading ``config.json`` wherever one exists.
    """
    if "no_noise" in name:
        noise = 0.0
    else:
        match = _LEGACY_NOISE.search(name)
        if match is None:
            raise ValueError(f"Cannot determine noise level from {name!r}")
        noise = float(match.group(1))

    return ExperimentConfig(
        model_type="mlp" if "model_mlp" in name else "unet",
        with_pupil="with_pupil" in name,
        narrow="not_fighting_overfitting" not in name,
        bounding_box_size=int(re.search(r"bbox_(\d+)", name).group(1)),
        radius=int(re.search(r"radius_(\d+)", name).group(1)),
        noise_sigma=noise,
        normalize_by_peak="normalize_by_peak" in name,
        use_throughput="using_throughput" in name,
        renormalize_after_noise="normalized_after_throughput" in name,
        binary_pupils="binary" in name,
        small_modes="small_modes" in name,
    )
