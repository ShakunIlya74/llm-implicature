"""Repo root resolution for exploratory scripts (code/pipeline_v1/exploratory -> repo)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def latest_rsa_run_dir() -> Path:
    """Newest run dir under results/rsa_results/ (by rsa_predictions.jsonl mtime)."""
    base = REPO_ROOT / "results" / "rsa_results"
    candidates = sorted(
        base.glob("*/rsa_predictions.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No rsa_predictions.jsonl under {base}; run "
            "code/pipeline_v1/scripts/rsa_probe.py first."
        )
    return candidates[0].parent


def default_rsa_run_dir() -> Path:
    """Same as latest_rsa_run_dir(); override with --rsa-run-dir on CLIs."""
    return latest_rsa_run_dir()
