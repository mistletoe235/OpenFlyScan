"""Compatibility entrypoint for the shared, versioned teacher grid adapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openflyscan.quality_predictor.teacher_grid import (
    load_standard, load_two_buildings, lookup_pixels, require, validate_grid, weighted_psnr,
)
