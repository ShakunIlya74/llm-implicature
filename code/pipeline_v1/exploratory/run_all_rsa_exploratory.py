"""Run all exploratory RSA analyses (1, 2, 4) with shared defaults.

Usage (from repo root):
  py -3 code/pipeline_v1/exploratory/run_all_rsa_exploratory.py [--rsa-run-dir PATH]

Default rsa run dir is the newest results/rsa_results/<run>/ (by rsa_predictions.jsonl mtime).

Writes under results/exploratory_rsa/ and METHODS.txt at that root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _paths import REPO_ROOT, default_rsa_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rsa-run-dir",
        type=Path,
        default=None,
        help="Directory containing rsa_*.jsonl (default: newest under results/rsa_results/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Pass to each analysis script (skip matplotlib PNGs)",
    )
    args = parser.parse_args()

    run_dir = args.rsa_run_dir or default_rsa_run_dir()
    exploratory_root = REPO_ROOT / "results" / "exploratory_rsa"
    exploratory_root.mkdir(parents=True, exist_ok=True)

    methods = exploratory_root / "METHODS.txt"
    methods.write_text(
        "\n".join(
            [
                "Exploratory RSA analyses (post-hoc, existing rsa_probe outputs only)",
                "",
                "Prerequisites:",
                "  - rsa_vs_behavior.jsonl, rsa_alpha_fits.jsonl, rsa_predictions.jsonl",
                "    from the same rsa_probe run directory.",
                "",
                "01_bets_logprob (Analysis 1):",
                "  Script: code/pipeline_v1/exploratory/analyze_bets_logprob_agreement.py",
                "  Uses KL/MAE fields precomputed in rsa_vs_behavior.jsonl (same definitions as rsa_probe).",
                "  Only rows with has_logprob_distribution=true enter KL/MAE summaries;",
                "  n_rows_without_logprob_excluded is reported in summary_global.csv.",
                "  Cell = (model_id, experiment, access, utterance); summaries unweighted across cells.",
                "",
                "02_alpha_fit (Analysis 2):",
                "  Script: code/pipeline_v1/exploratory/analyze_alpha_rsa_fit.py",
                "  Boundary: |alpha_hat - alpha_min| <= step/2+eps or same for alpha_max;",
                "  must pass the same --alpha-min/--alpha-max/--alpha-step as rsa_probe for correct flags.",
                "  Boundary rates use denominator = count of fit_status==ok rows.",
                "",
                "03_residuals (Analysis 4):",
                "  Script: code/pipeline_v1/exploratory/analyze_bets_rsa_residuals.py",
                "  res_s = p_bets(s) - p_rsa(s); L2 = sqrt(sum_s res_s^2).",
                "  Means over cells are unweighted (each cell one aggregated posterior).",
                "",
                "Figures (matplotlib, same out-dir as CSVs):",
                "  01_bets_logprob/summary_plot.png",
                "  02_alpha_fit/summary_plot.png",
                "  03_residuals/summary_plot.png",
                "  Use --no-plots on run_all or individual scripts to skip.",
                "",
                f"Default run dir for this batch: {run_dir}",
            ]
        ),
        encoding="utf-8",
    )

    here = Path(__file__).resolve().parent
    py = sys.executable
    common = [py, "-u"]

    def run(script: str, *extra: str) -> None:
        cmd = common + [str(here / script)] + list(extra)
        if args.no_plots:
            cmd.append("--no-plots")
        print(">", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    vs = run_dir / "rsa_vs_behavior.jsonl"
    fits = run_dir / "rsa_alpha_fits.jsonl"
    pred = run_dir / "rsa_predictions.jsonl"

    run(
        "analyze_bets_logprob_agreement.py",
        "--vs-behavior",
        str(vs),
        "--out-dir",
        str(exploratory_root / "01_bets_logprob"),
    )
    run(
        "analyze_alpha_rsa_fit.py",
        "--alpha-fits",
        str(fits),
        "--out-dir",
        str(exploratory_root / "02_alpha_fit"),
    )
    run(
        "analyze_bets_rsa_residuals.py",
        "--predictions",
        str(pred),
        "--out-dir",
        str(exploratory_root / "03_residuals"),
    )

    print(f"Done. Outputs under {exploratory_root}")


if __name__ == "__main__":
    main()
