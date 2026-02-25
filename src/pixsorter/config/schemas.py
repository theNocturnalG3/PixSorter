from __future__ import annotations
from pathlib import Path
from typing import List
from .defaults import SortConfig


def validate_config(cfg: SortConfig) -> List[str]:
    errors: List[str] = []

    # Paths
    if not cfg.input_dir or not Path(cfg.input_dir).exists():
        errors.append("Input folder: path is missing or does not exist.")
    if not cfg.output_dir or not Path(cfg.output_dir).exists():
        errors.append("Output folder: path is missing or does not exist.")

    # Output folder name
    if cfg.output_create_new:
        name = (cfg.output_new_name or "").strip()
        if not name:
            errors.append("New output folder name is empty.")
        if "/" in name or "\\" in name:
            errors.append("New output folder name must not contain slashes.")

    # Core numeric ranges (keep aligned with UI widget ranges)
    if cfg.min_group < 2:
        errors.append("Min group size must be >= 2.")

    if cfg.knn_k < 5 or cfg.knn_k > 60:
        errors.append("Neighbors (K) must be in [5, 60].")

    if not (0.70 <= cfg.clip_sim_thresh <= 0.99):
        errors.append("Candidate similarity must be in [0.70, 0.99].")

    if cfg.orb_nfeatures < 800 or cfg.orb_nfeatures > 6000:
        errors.append("ORB features must be in [800, 6000].")

    if not (0.55 <= cfg.orb_match_thresh <= 0.95):
        errors.append("Match strictness must be in [0.55, 0.95].")

    if cfg.min_inliers < 10 or cfg.min_inliers > 80:
        errors.append("Min inliers must be in [10, 80].")

    if cfg.max_side_verify < 500 or cfg.max_side_verify > 1400:
        errors.append("Verify resize must be in [500, 1400].")

    if cfg.best_of:
        if cfg.top_k < 1 or cfg.top_k > 20:
            errors.append("TOP-K per group must be in [1, 20].")
        if cfg.diverse_inliers < 15 or cfg.diverse_inliers > 80:
            errors.append("Diversity threshold must be in [15, 80].")

        if cfg.use_eyes and not (1.0 <= cfg.eyes_weight <= 5.0):
            errors.append("Eyes penalty must be in [1.0, 5.0].")

    if cfg.mode not in ("copy", "move"):
        errors.append("Mode must be 'copy' or 'move'.")

    if cfg.clip_model not in ("vit_l_14", "vit_b_32"):
        errors.append("CLIP model must be 'vit_l_14' or 'vit_b_32'.")

    return errors