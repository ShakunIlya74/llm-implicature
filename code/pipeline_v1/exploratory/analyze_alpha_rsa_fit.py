"""Analysis 2: RSA alpha fit quality and grid-boundary saturation.

Uses rsa_alpha_fits.jsonl. Boundary flags: alpha_hat within tol of alpha_min or alpha_max
(default tol = alpha_step/2, matching discrete grid semantics).

Rows with fit_status != 'ok' are listed separately (e.g. missing_prior) and excluded from
boundary-rate denominators for alpha_hat (documented in RUN_META).

Outputs:
  - per_cell.csv: all rows with fit_status, boundary flags, objective_value
  - summary_by_model_experiment.csv: n_ok, n_at_min, n_at_max, rate_at_min, rate_at_max,
      mean/median objective_value on ok rows
  - missing_prior_or_failed.csv: rows that did not obtain alpha fit
  - summary_plot.png: 2×2 panel — boundary rates and mean KL objective by experiment

Usage (from repo root):
  py -3 code/pipeline_v1/exploratory/analyze_alpha_rsa_fit.py \\
    --alpha-min 0.1 --alpha-max 8.0 --alpha-step 0.1
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
    parser = argparse.ArgumentParser(description="RSA alpha fit and boundary analysis")
    parser.add_argument(
        "--alpha-fits",
        type=Path,
        default=None,
    )
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    fits_path = args.alpha_fits or (default_rsa_run_dir() / "rsa_alpha_fits.jsonl")
    if not fits_path.exists():
        raise SystemExit(f"Input not found: {fits_path}")

    tol = args.alpha_step / 2.0 + 1e-9
    out_dir = args.out_dir or (REPO_ROOT / "results" / "exploratory_rsa" / "02_alpha_fit")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(fits_path)

    def at_boundary(a: float) -> tuple[bool, bool]:
        at_min = abs(a - args.alpha_min) <= tol
        at_max = abs(a - args.alpha_max) <= tol
        return at_min, at_max

    per_fields = [
        "model_id",
        "experiment",
        "access",
        "utterance",
        "fit_status",
        "alpha_hat",
        "at_alpha_min_boundary",
        "at_alpha_max_boundary",
        "objective_value",
        "objective",
        "num_alpha_grid_points",
    ]

    missing: list[dict[str, Any]] = []
    per_out: list[dict[str, Any]] = []

    for r in rows:
        st = str(r.get("fit_status", ""))
        if st != "ok":
            missing.append(
                {
                    "model_id": r.get("model_id"),
                    "experiment": r.get("experiment"),
                    "access": r.get("access"),
                    "utterance": r.get("utterance"),
                    "fit_status": st,
                }
            )
            per_out.append(
                {
                    "model_id": r.get("model_id", ""),
                    "experiment": r.get("experiment", ""),
                    "access": r.get("access", ""),
                    "utterance": r.get("utterance", ""),
                    "fit_status": st,
                    "alpha_hat": "",
                    "at_alpha_min_boundary": "",
                    "at_alpha_max_boundary": "",
                    "objective_value": r.get("objective_value", ""),
                    "objective": r.get("objective", ""),
                    "num_alpha_grid_points": r.get("num_alpha_grid_points", ""),
                }
            )
            continue

        a = float(r["alpha_hat"])
        at_min, at_max = at_boundary(a)
        per_out.append(
            {
                "model_id": r["model_id"],
                "experiment": r["experiment"],
                "access": r["access"],
                "utterance": r["utterance"],
                "fit_status": st,
                "alpha_hat": a,
                "at_alpha_min_boundary": int(at_min),
                "at_alpha_max_boundary": int(at_max),
                "objective_value": r.get("objective_value", ""),
                "objective": r.get("objective", ""),
                "num_alpha_grid_points": r.get("num_alpha_grid_points", ""),
            }
        )

    per_path = out_dir / "per_cell.csv"
    with per_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_fields)
        w.writeheader()
        for row in per_out:
            w.writerow(row)

    miss_path = out_dir / "missing_prior_or_failed.csv"
    with miss_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model_id", "experiment", "access", "utterance", "fit_status"],
        )
        w.writeheader()
        w.writerows(missing)

    ok_rows = [r for r in rows if r.get("fit_status") == "ok"]
    bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in ok_rows:
        bucket[(str(r["model_id"]), str(r["experiment"]))].append(r)

    def median(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        s = sorted(xs)
        n = len(s)
        mid = n // 2
        if n % 2:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2

    summ_path = out_dir / "summary_by_model_experiment.csv"
    sf = [
        "model_id",
        "experiment",
        "n_ok",
        "n_at_alpha_min",
        "n_at_alpha_max",
        "rate_at_alpha_min",
        "rate_at_alpha_max",
        "mean_objective_kl_obs_to_rsa",
        "median_objective_kl_obs_to_rsa",
        "mean_alpha_hat",
        "median_alpha_hat",
    ]
    summ_records: list[dict[str, Any]] = []
    with summ_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sf)
        w.writeheader()
        for (mid, exp), lst in sorted(bucket.items()):
            objs = [float(x["objective_value"]) for x in lst if "objective_value" in x]
            alphas = [float(x["alpha_hat"]) for x in lst]
            n_at_min = sum(
                1
                for x in lst
                if at_boundary(float(x["alpha_hat"]))[0]
            )
            n_at_max = sum(
                1
                for x in lst
                if at_boundary(float(x["alpha_hat"]))[1]
            )
            n_ok = len(lst)
            rec = {
                "model_id": mid,
                "experiment": exp,
                "n_ok": n_ok,
                "n_at_alpha_min": n_at_min,
                "n_at_alpha_max": n_at_max,
                "rate_at_alpha_min": round(n_at_min / n_ok, 6) if n_ok else "",
                "rate_at_alpha_max": round(n_at_max / n_ok, 6) if n_ok else "",
                "mean_objective_kl_obs_to_rsa": round(sum(objs) / len(objs), 8)
                if objs
                else "",
                "median_objective_kl_obs_to_rsa": round(median(objs), 8)
                if objs
                else "",
                "mean_alpha_hat": round(sum(alphas) / len(alphas), 6) if alphas else "",
                "median_alpha_hat": round(median(alphas), 6) if alphas else "",
            }
            summ_records.append(rec)
            w.writerow(rec)

    meta_lines = [
        f"input: {fits_path}",
        f"alpha_min={args.alpha_min} alpha_max={args.alpha_max} alpha_step={args.alpha_step}",
        f"boundary_tol = step/2 + 1e-9 = {tol:.12g}",
        "at_min: |alpha_hat - alpha_min| <= tol; at_max analogous.",
        "Boundary rates use denominator n_ok only (fit_status==ok).",
        "objective_value is KL(observed_bets || RSA_pred) at optimal alpha on grid.",
    ]
    if not args.no_plots:
        _plot_alpha_fit(summ_records, out_dir)
        meta_lines.append("Figure: summary_plot.png (boundaries + mean objective).")

    meta_path = out_dir / "RUN_META.txt"
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")

    print(f"Wrote -> {out_dir}")


def _short_mid(mid: str) -> str:
    return (mid.split("--")[-1] if "--" in mid else mid)[:24]


def _plot_alpha_fit(summ_records: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip summary_plot.png")
        return

    experiments = ("exps_some", "exps_numerical")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for j, exp in enumerate(experiments):
        recs = [r for r in summ_records if r["experiment"] == exp]
        recs.sort(key=lambda r: str(r["model_id"]))
        models = [_short_mid(str(r["model_id"])) for r in recs]
        x = list(range(len(recs)))
        w = 0.35

        axb = axes[0, j]
        if recs:
            rmin = [float(r["rate_at_alpha_min"] or 0) for r in recs]
            rmax = [float(r["rate_at_alpha_max"] or 0) for r in recs]
            axb.bar([i - w / 2 for i in x], rmin, w, label="rate at α_min", color="C0")
            axb.bar([i + w / 2 for i in x], rmax, w, label="rate at α_max", color="C1")
            axb.set_ylim(0, 1.05)
            axb.set_ylabel("Fraction of ok cells")
            axb.set_xticks(x)
            axb.set_xticklabels(models, rotation=55, ha="right", fontsize=7)
            axb.set_title(f"{exp}: α at grid boundary")
            axb.legend(fontsize=7)
            axb.grid(True, axis="y", alpha=0.3)
        else:
            axb.set_visible(False)

        axo = axes[1, j]
        if recs:
            objs = [float(r["mean_objective_kl_obs_to_rsa"] or 0) for r in recs]
            axo.bar(x, objs, color="C2")
            axo.set_ylabel("Mean KL(obs || RSA_pred)")
            axo.set_xticks(x)
            axo.set_xticklabels(models, rotation=55, ha="right", fontsize=7)
            axo.set_title(f"{exp}: mean fit loss at α̂")
            axo.grid(True, axis="y", alpha=0.3)
        else:
            axo.set_visible(False)

    fig.suptitle(
        "RSA α fit (ok cells only): boundary rates (top) and mean KL at α̂ (bottom)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "summary_plot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
