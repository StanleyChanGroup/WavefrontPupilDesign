"""Pupil design for computational wavefront estimation.

Reference implementation for Almuallem, Chimitt, Ma, Guo and Chan,
"Pupil Design for Computational Wavefront Estimation" (arXiv:2604.00225).
"""
from .asymmetry import asymmetry
from .config import ExperimentConfig
from .data import PupilPhaseDataset, load_or_build_pools, simulate_measurement
from .losses import masked_piston_corrected_mse, masked_piston_corrected_mse_per_sample
from .models import PhaseMLP, UNet, build_model
from .optics import PAPER_MODES, psf_from_pupil, random_zernike_phase, reference_circle, strehl_ratio
from .pupils import generate_binary_pupils, generate_polygon_pupils

__version__ = "0.1.0"

__all__ = [
    "asymmetry",
    "ExperimentConfig",
    "PupilPhaseDataset",
    "load_or_build_pools",
    "simulate_measurement",
    "masked_piston_corrected_mse",
    "masked_piston_corrected_mse_per_sample",
    "UNet",
    "PhaseMLP",
    "build_model",
    "PAPER_MODES",
    "psf_from_pupil",
    "random_zernike_phase",
    "reference_circle",
    "strehl_ratio",
    "generate_polygon_pupils",
    "generate_binary_pupils",
]
