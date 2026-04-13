"""Analysis 1: Agreement between structured-output bets and logprob posteriors.

Uses only rsa_vs_behavior.jsonl. Rows with has_logprob_distribution=false are excluded
from KL/MAE summaries (documented counts).

Definitions (same as rsa_probe output):
  - kl_bets_vs_logprob: KL( p_bets || p_logprob ) over states {0,1,2,3}
  - mae_bets_vs_logprob: mean_s | p_bets(s) - p_logprob(s) |

Aggregation: each JSONL row is one (model, experiment, access, utterance) cell after
pipeline aggregation; summaries take unweighted means across cells unless noted.

Outputs (under --out-dir):
  - per_cell.csv: one row per cell with logprobs
  - summary_by_model_experiment.csv: means and SDs by model × experiment
  - summary_global.csv: single row overall (cells with logprobs only)
  - summary_plot.png: bar charts of mean KL / MAE (bets vs logprob) by model

Usage (from repo root):
  py -3 code/pipeline_v1/exploratory/analyze_bets_logprob_agreement.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT, default_rsa_run_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bets vs logprob agreement (existing rsa_vs_behavior.jsonl only)"
    )
    parser.add_argument(
        "--vs-behavior",
        type=Path,
        default=None,
        help="Path to rsa_vs_behavior.jsonl (default: results/rsa_results/<run>/...)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/exploratory_rsa/01_bets_logprob)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing summary_plot.png (requires matplotlib)",
    )
    args = parser.parse_args()

    vs_path = args.vs_behavior or (default_rsa_run_dir() / "rsa_vs_behavior.jsonl")
    if not vs_path.exists():
        raise SystemExit(f"Input not found: {vs_path}")

    out_dir = args.out_dir or (REPO_ROOT / "results" / "exploratory_rsa" / "01_bets_logprob")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(vs_path)
    total = len(rows)
    with_lp = [r for r in rows if r.get("has_logprob_distribution")]
    without_lp = total - len(with_lp)

    per_cell_fields = [
        "model_id",
        "experiment",
        "access",
        "utterance",
        "alpha_hat",
        "kl_bets_vs_logprob",
        "mae_bets_vs_logprob",
        "kl_bets_vs_rsa",
        "mae_bets_vs_rsa",
        "kl_rsa_vs_logprob",
        "mae_rsa_vs_logprob",
    ]
    per_cell_path = out_dir / "per_cell.csv"
    with per_cell_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_cell_fields, extrasaction="ignore")
        w.writeheader()
        for r in with_lp:
            w.writerow({k: r.get(k, "") for k in per_cell_fields})

    # By model × experiment
    bucket: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in with_lp:
        key = (str(r["model_id"]), str(r["experiment"]))
        bucket[key].append(
            (float(r["kl_bets_vs_logprob"]), float(r["mae_bets_vs_logprob"]))
        )

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def sd(xs: list[float]) -> float:
        if len(xs) < 2:
            return float("nan")
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    summ_path = out_dir / "summary_by_model_experiment.csv"
    summ_fields = [
        "model_id",
        "experiment",
        "n_cells_with_logprob",
        "mean_kl_bets_vs_logprob",
        "sd_kl_bets_vs_logprob",
        "mean_mae_bets_vs_logprob",
        "sd_mae_bets_vs_logprob",
        "mean_kl_bets_vs_rsa",
        "mean_mae_bets_vs_rsa",
    ]
    rsa_bucket: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in with_lp:
        k = (str(r["model_id"]), str(r["experiment"]))
        rsa_bucket[k].append((float(r["kl_bets_vs_rsa"]), float(r["mae_bets_vs_rsa"])))

    with summ_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ_fields)
        w.writeheader()
        for (mid, exp), pairs in sorted(bucket.items()):
            kls = [p[0] for p in pairs]
            maes = [p[1] for p in pairs]
            rk = rsa_bucket.get((mid, exp), [])
            w.writerow(
                {
                    "model_id": mid,
                    "experiment": exp,
                    "n_cells_with_logprob": len(pairs),
                    "mean_kl_bets_vs_logprob": round(mean(kls), 8),
                    "sd_kl_bets_vs_logprob": round(sd(kls), 8) if len(kls) > 1 else "",
                    "mean_mae_bets_vs_logprob": round(mean(maes), 8),
                    "sd_mae_bets_vs_logprob": round(sd(maes), 8) if len(maes) > 1 else "",
                    "mean_kl_bets_vs_rsa": round(mean([x[0] for x in rk]), 8) if rk else "",
                    "mean_mae_bets_vs_rsa": round(mean([x[1] for x in rk]), 8) if rk else "",
                }
            )

    all_kl = [float(r["kl_bets_vs_logprob"]) for r in with_lp]
    all_mae = [float(r["mae_bets_vs_logprob"]) for r in with_lp]
    glob_path = out_dir / "summary_global.csv"
    try:
        input_rel = str(vs_path.relative_to(REPO_ROOT))
    except ValueError:
        input_rel = str(vs_path)

    with glob_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "input_file",
                "n_rows_total",
                "n_rows_without_logprob_excluded",
                "n_cells_analyzed",
                "mean_kl_bets_vs_logprob",
                "sd_kl_bets_vs_logprob",
                "mean_mae_bets_vs_logprob",
                "sd_mae_bets_vs_logprob",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "input_file": input_rel,
                "n_rows_total": total,
                "n_rows_without_logprob_excluded": without_lp,
                "n_cells_analyzed": len(with_lp),
                "mean_kl_bets_vs_logprob": round(mean(all_kl), 8),
                "sd_kl_bets_vs_logprob": round(sd(all_kl), 8) if len(all_kl) > 1 else "",
                "mean_mae_bets_vs_logprob": round(mean(all_mae), 8),
                "sd_mae_bets_vs_logprob": round(sd(all_mae), 8) if len(all_mae) > 1 else "",
            }
        )

    meta_path = out_dir / "RUN_META.txt"
    meta_lines = [
        f"input: {vs_path}",
        f"rows_total: {total}",
        f"rows_with_logprob: {len(with_lp)}",
        f"rows_excluded_no_logprob: {without_lp}",
        "KL_bets_vs_logprob is KL(p_bets || p_logprob) as in rsa_probe.",
        "Each row is one aggregated (model, experiment, access, utterance) cell.",
        "Summaries are unweighted over cells.",
    ]
    if not args.no_plots:
        _plot_bets_logprob(bucket, out_dir)
        meta_lines.append("Figure: summary_plot.png (mean KL and MAE by model × experiment).")
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")

    print(f"Wrote -> {out_dir}")


def _short_mid(mid: str) -> str:
    return (mid.split("--")[-1] if "--" in mid else mid)[:24]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _plot_bets_logprob(
    bucket: dict[tuple[str, str], list[tuple[float, float]]],
    out_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip summary_plot.png")
        return

    experiments = ("exps_some", "exps_numerical")
    fig, axes = plt.subplots(len(experiments), 2, figsize=(14, 7), squeeze=False)
    for ri, exp in enumerate(experiments):
        models = sorted({m for (m, e) in bucket if e == exp})
        if not models:
            for ci in range(2):
                axes[ri, ci].set_visible(False)
            continue
        kls = [_mean([p[0] for p in bucket[(m, exp)]]) for m in models]
        maes = [_mean([p[1] for p in bucket[(m, exp)]]) for m in models]
        x = list(range(len(models)))
        w = 0.35
        labels = [_short_mid(m) for m in models]

        ax0 = axes[ri, 0]
        ax0.bar([i - w / 2 for i in x], kls, w, label="mean KL", color="C0")
        ax0.set_ylabel("Mean KL(bets || logprob)")
        ax0.set_title(f"{exp}: KL")
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax0.grid(True, axis="y", alpha=0.3)

        ax1 = axes[ri, 1]
        ax1.bar([i - w / 2 for i in x], maes, w, color="C1")
        ax1.set_ylabel("Mean MAE(bets, logprob)")
        ax1.set_title(f"{exp}: MAE")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax1.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Bets vs logprob agreement (mean over cells per model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_plot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
