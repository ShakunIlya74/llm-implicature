"""Analysis 4: Residuals between observed structured bets and RSA-predicted posteriors.

For each cell in rsa_predictions.jsonl:
  r(s) = p_observed_bets(s) - p_rsa_predicted(s),  s in {0,1,2,3}

L2 residual norm: sqrt(sum_s r(s)^2) (not normalized by dimension).

Aggregation tables:
  - mean residual per (model_id, experiment, state): unweighted mean of r(s) across cells
  - per_cell_residuals.csv: full audit trail
  - summary_plot.png: heatmap of mean residual (bets − RSA) by model × state

Interpretation: positive mean residual at state s means bets assign more mass to s than
RSA at fitted alpha (cell-wise alpha from same pipeline; RSA curve uses that cell's alpha).

Usage (from repo root):
  py -3 code/pipeline_v1/exploratory/analyze_bets_rsa_residuals.py
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

STATES = ("0", "1", "2", "3")


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
        description="Bets minus RSA posterior residuals (rsa_predictions.jsonl)"
    )
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    pred_path = args.predictions or (default_rsa_run_dir() / "rsa_predictions.jsonl")
    if not pred_path.exists():
        raise SystemExit(f"Input not found: {pred_path}")

    out_dir = args.out_dir or (REPO_ROOT / "results" / "exploratory_rsa" / "03_residuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(pred_path)

    # Accumulate mean residual per (model, experiment, state)
    sum_res: dict[tuple[str, str, str], float] = defaultdict(float)
    count: dict[tuple[str, str, str], int] = defaultdict(int)

    per_cell_fields = [
        "model_id",
        "experiment",
        "access",
        "utterance",
        "alpha_hat",
        "res_0",
        "res_1",
        "res_2",
        "res_3",
        "l2_residual",
    ]
    per_rows: list[dict[str, Any]] = []

    for r in rows:
        obs = r["observed_posterior_bets"]
        rsa = r["rsa_predicted_posterior"]
        mid = str(r["model_id"])
        exp = str(r["experiment"])
        res_vec: dict[str, float] = {}
        sq = 0.0
        for s in STATES:
            dv = float(obs[s]) - float(rsa[s])
            res_vec[s] = dv
            sq += dv * dv
            key = (mid, exp, s)
            sum_res[key] += dv
            count[key] += 1
        l2 = math.sqrt(sq)
        per_rows.append(
            {
                "model_id": mid,
                "experiment": exp,
                "access": r["access"],
                "utterance": r["utterance"],
                "alpha_hat": r.get("alpha_hat", ""),
                "res_0": res_vec["0"],
                "res_1": res_vec["1"],
                "res_2": res_vec["2"],
                "res_3": res_vec["3"],
                "l2_residual": round(l2, 10),
            }
        )

    per_path = out_dir / "per_cell_residuals.csv"
    with per_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_cell_fields)
        w.writeheader()
        w.writerows(per_rows)

    mean_path = out_dir / "mean_residual_by_model_experiment_state.csv"
    mf = [
        "model_id",
        "experiment",
        "state",
        "mean_residual_bets_minus_rsa",
        "n_cells",
    ]
    with mean_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mf)
        w.writeheader()
        keys_sorted = sorted(sum_res.keys(), key=lambda x: (x[0], x[1], int(x[2])))
        for key in keys_sorted:
            mid, exp, s = key
            n = count[key]
            w.writerow(
                {
                    "model_id": mid,
                    "experiment": exp,
                    "state": int(s),
                    "mean_residual_bets_minus_rsa": round(sum_res[key] / n, 8),
                    "n_cells": n,
                }
            )

    # L2 summary by model × experiment
    l2_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_rows:
        l2_bucket[(row["model_id"], row["experiment"])].append(float(row["l2_residual"]))

    l2_path = out_dir / "summary_l2_by_model_experiment.csv"
    with l2_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "experiment",
                "n_cells",
                "mean_l2_residual",
                "median_l2_residual",
            ],
        )
        w.writeheader()
        for (mid, exp), vec in sorted(l2_bucket.items()):
            vec_s = sorted(vec)
            n = len(vec)
            med = vec_s[n // 2] if n % 2 else (vec_s[n // 2 - 1] + vec_s[n // 2]) / 2
            w.writerow(
                {
                    "model_id": mid,
                    "experiment": exp,
                    "n_cells": n,
                    "mean_l2_residual": round(sum(vec) / n, 8),
                    "median_l2_residual": round(med, 8),
                }
            )

    meta_lines = [
        f"input: {pred_path}",
        "res_s = p_observed_posterior_bets(s) - p_rsa_predicted_posterior(s)",
        "l2_residual = sqrt(sum_s res_s^2) over s in {0,1,2,3}",
        "mean_residual_by_*: unweighted mean of res_s across posterior cells",
        "  (each cell = one aggregated condition; not weighted by raw trial n).",
    ]
    if not args.no_plots:
        _plot_residual_heatmap(sum_res, count, out_dir)
        meta_lines.append("Figure: summary_plot.png (mean residual heatmap, model × state).")
    meta_path = out_dir / "RUN_META.txt"
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")

    print(f"Wrote -> {out_dir}")


def _short_mid(mid: str) -> str:
    return (mid.split("--")[-1] if "--" in mid else mid)[:22]


def _plot_residual_heatmap(
    sum_res: dict[tuple[str, str, str], float],
    count: dict[tuple[str, str, str], int],
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
    fig, axes = plt.subplots(1, len(experiments), figsize=(10, 6), squeeze=False)
    state_labels = ["0", "1", "2", "3"]
    for j, exp in enumerate(experiments):
        models = sorted({m for (m, e, _) in sum_res if e == exp})
        if not models:
            axes[0, j].set_visible(False)
            continue
        mat: list[list[float]] = []
        for m in models:
            row: list[float] = []
            for s in STATES:
                key = (m, exp, s)
                if count.get(key, 0) > 0:
                    row.append(sum_res[key] / count[key])
                else:
                    row.append(float("nan"))
            mat.append(row)
        flat_abs = [abs(v) for r in mat for v in r if v == v]
        vmax = max(flat_abs) if flat_abs else 1e-9
        if vmax <= 0:
            vmax = 1e-9
        ax = axes[0, j]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("State")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_short_mid(m) for m in models], fontsize=7)
        ax.set_title(f"{exp}\nmean(bets − RSA)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Prob. mass")

    fig.suptitle("Residual heatmap: positive ⇒ bets put more mass on state than RSA", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_plot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
