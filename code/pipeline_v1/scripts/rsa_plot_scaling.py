"""Scaling analysis: RSA metrics vs approximate model size (billions of parameters).

Reads rsa_vs_behavior.jsonl (and optionally rsa_alpha_fits.jsonl), aggregates mean
metrics per (model_id, experiment), joins param count, writes a summary CSV and
matplotlib figures (log-scale x-axis where appropriate).

Usage (from repo root):
  py -3 code/pipeline_v1/scripts/rsa_plot_scaling.py

Omit paths to use the newest results/rsa_results/<run>/ and write under results/rsa_plots/.

Param sizes: explicit map for known checkpoints in this repo; otherwise regex on
the HuggingFace-style id. Cross-architecture comparisons are indicative only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _latest_rsa_run_dir() -> Path:
    base = _repo_root() / "results" / "rsa_results"
    candidates = sorted(
        base.glob("*/rsa_predictions.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"No rsa_predictions.jsonl under {base}; run rsa_probe.py first."
        )
    return candidates[0].parent


# Approximate size in billions (active params); extend when adding models.
PARAM_BILLIONS_BY_ID: dict[str, float] = {
    "Qwen--Qwen3.5-0.8B": 0.8,
    "Qwen--Qwen3.5-2B": 2.0,
    "Qwen--Qwen3.5-4B": 4.0,
    "Qwen--Qwen3.5-9B": 9.0,
    "meta-llama--Llama-3.2-1B-Instruct": 1.0,
    "meta-llama--Llama-3.2-3B-Instruct": 3.0,
    "meta-llama--Llama-3.1-8B-Instruct": 8.0,
    "microsoft--Phi-4-mini-instruct": 3.8,
    "google--gemma-3-4b-it": 4.0,
    "google--gemma-3-1b-it": 1.0,
}

_PARAM_RE = re.compile(
    r"(?P<n>\d+\.?\d*)\s*[Bb](?![a-zA-Z])"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def param_billions(model_id: str) -> float | None:
    if model_id in PARAM_BILLIONS_BY_ID:
        return PARAM_BILLIONS_BY_ID[model_id]
    m = _PARAM_RE.search(model_id)
    if m:
        return float(m.group("n"))
    return None


def model_family(model_id: str) -> str:
    return model_id.split("--", 1)[0] if "--" in model_id else model_id


def aggregate_vs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, list[float]]]:
    """(model_id, experiment) -> metric -> list of values."""
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = (str(r["model_id"]), str(r["experiment"]))
        for field in (
            "kl_bets_vs_rsa",
            "mae_bets_vs_rsa",
            "kl_bets_vs_logprob",
            "mae_bets_vs_logprob",
            "kl_rsa_vs_logprob",
            "mae_rsa_vs_logprob",
        ):
            if field in r and isinstance(r[field], (int, float)):
                acc[key][field].append(float(r[field]))
    return acc


def aggregate_alpha(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("fit_status") != "ok":
            continue
        if "alpha_hat" not in r:
            continue
        key = (str(r["model_id"]), str(r["experiment"]))
        out[key].append(float(r["alpha_hat"]))
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= 0 or deny <= 0:
        return None
    return num / (denx * deny)


def print_scaling_correlations(summary: list[dict[str, Any]]) -> None:
    """Rough linear trend of log10(params) vs metrics (mixed architectures)."""
    for exp in ("exps_some", "exps_numerical"):
        xs: list[float] = []
        ys: list[float] = []
        for r in summary:
            if r["experiment"] != exp:
                continue
            p = r.get("param_billions")
            if p == "" or p is None:
                continue
            kl = r.get("mean_kl_bets_vs_rsa")
            if kl == "":
                continue
            xs.append(math.log10(float(p)))
            ys.append(float(kl))
        r_val = pearson_r(xs, ys)
        n = len(xs)
        if r_val is not None:
            print(
                f"  {exp}: n={n} models, Pearson r(log10 B, mean KL_bets_vs_rsa) = {r_val:.3f}"
            )
        else:
            print(f"  {exp}: n={n} models, correlation not defined")


def write_summary_csv(
    path: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return
    fields = list(summary_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)


def build_summary(
    vs_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    vs_acc = aggregate_vs(vs_rows)
    alpha_acc = aggregate_alpha(alpha_rows) if alpha_rows else {}

    keys = sorted(vs_acc.keys())
    out: list[dict[str, Any]] = []
    for model_id, experiment in keys:
        p = param_billions(model_id)
        mets = vs_acc[(model_id, experiment)]
        row: dict[str, Any] = {
            "model_id": model_id,
            "family": model_family(model_id),
            "experiment": experiment,
            "param_billions": p if p is not None else "",
            "log10_param_billions": round(math.log10(p), 4) if p and p > 0 else "",
            "n_cells": len(mets["kl_bets_vs_rsa"]),
            "mean_kl_bets_vs_rsa": round(mean(mets["kl_bets_vs_rsa"]), 6),
            "mean_mae_bets_vs_rsa": round(mean(mets["mae_bets_vs_rsa"]), 6),
            "mean_kl_bets_vs_logprob": round(mean(mets["kl_bets_vs_logprob"]), 6),
            "mean_mae_bets_vs_logprob": round(mean(mets["mae_bets_vs_logprob"]), 6),
            "mean_kl_rsa_vs_logprob": round(mean(mets["kl_rsa_vs_logprob"]), 6),
            "mean_mae_rsa_vs_logprob": round(mean(mets["mae_rsa_vs_logprob"]), 6),
        }
        a = alpha_acc.get((model_id, experiment), [])
        row["mean_alpha_hat"] = round(mean(a), 4) if a else ""
        row["n_alpha_cells"] = len(a)
        out.append(row)
    return out


def plot_scaling(
    summary: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def points_for_exp(exp: str, ykey: str) -> list[tuple[float, float, str, str]]:
        pts: list[tuple[float, float, str, str]] = []
        for r in summary:
            if r["experiment"] != exp:
                continue
            p = r.get("param_billions")
            if p == "" or p is None:
                continue
            yv = r.get(ykey)
            if yv == "" or yv is None or (isinstance(yv, float) and math.isnan(yv)):
                continue
            pts.append((float(p), float(yv), str(r["model_id"]), str(r["family"])))
        return pts

    experiments = ("exps_some", "exps_numerical")
    metrics = [
        ("mean_kl_bets_vs_rsa", "Mean KL(bets || RSA)"),
        ("mean_mae_bets_vs_rsa", "Mean MAE(bets, RSA)"),
        ("mean_kl_bets_vs_logprob", "Mean KL(bets || logprobs)"),
        ("mean_alpha_hat", "Mean fitted α_hat"),
    ]

    fig, axes = plt.subplots(len(metrics), len(experiments), figsize=(11, 12), squeeze=False)
    for j, exp in enumerate(experiments):
        for i, (ykey, ylabel) in enumerate(metrics):
            ax = axes[i][j]
            pts = points_for_exp(exp, ykey)
            by_fam: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
            for p, y, mid, fam in pts:
                by_fam[fam].append((p, y, mid))
            cmap = plt.cm.tab10
            for fi, (fam, triples) in enumerate(sorted(by_fam.items(), key=lambda x: x[0])):
                triples.sort(key=lambda t: t[0])
                xs = [t[0] for t in triples]
                ys = [t[1] for t in triples]
                color = cmap(fi % 10)
                ax.scatter(xs, ys, color=[color] * len(xs), s=36, label=fam if i == 0 else None)
                if len(triples) >= 2:
                    ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1)
                for p, y, mid in triples:
                    short = mid.split("--")[-1][:20]
                    ax.annotate(
                        short,
                        (p, y),
                        textcoords="offset points",
                        xytext=(3, 2),
                        fontsize=4,
                        alpha=0.9,
                    )
            ax.set_xscale("log")
            ax.set_xlabel("Params (B, log scale)")
            ax.set_ylabel(ylabel)
            ax.set_title(exp)
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Scaling: RSA / elicitation metrics vs model size (color = vendor family; labels = checkpoint)\n"
        "Cross-architecture x-axis is indicative only.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "rsa_scaling_metrics_vs_params.png", dpi=150)
    plt.close(fig)

    # Second figure: overlay both experiments (KL bets vs RSA), color = experiment
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = {"exps_some": "C0", "exps_numerical": "C1"}
    for exp in experiments:
        pts = points_for_exp(exp, "mean_kl_bets_vs_rsa")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mids = [p[2] for p in pts]
        ax2.scatter(
            xs,
            ys,
            c=colors.get(exp, "gray"),
            s=45,
            alpha=0.85,
            label=exp,
        )
        for x, y, mid in zip(xs, ys, mids):
            short = mid.split("--")[-1][:18]
            ax2.annotate(short, (x, y), textcoords="offset points", xytext=(3, 2), fontsize=5, alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Params (billions)")
    ax2.set_ylabel("Mean KL(bets || RSA)")
    ax2.set_title("Mean KL(bets || RSA) vs size (both experiments)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "rsa_scaling_kl_bets_rsa_combined.png", dpi=150)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="RSA metrics vs model parameter scaling")
    parser.add_argument(
        "--vs-behavior",
        type=Path,
        default=None,
        help="rsa_vs_behavior.jsonl (default: newest under results/rsa_results/)",
    )
    parser.add_argument(
        "--alpha-fits",
        type=Path,
        default=None,
        help="rsa_alpha_fits.jsonl (default: same run as vs-behavior)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Summary CSV (default: results/rsa_plots/rsa_scaling_summary.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figures dir (default: results/rsa_plots/figures_scaling)",
    )
    args = parser.parse_args()

    run_dir = _latest_rsa_run_dir()
    vs_path = args.vs_behavior or (run_dir / "rsa_vs_behavior.jsonl")
    alpha_path = (
        args.alpha_fits if args.alpha_fits is not None else (run_dir / "rsa_alpha_fits.jsonl")
    )
    out_csv = args.out_csv or (_repo_root() / "results" / "rsa_plots" / "rsa_scaling_summary.csv")
    out_dir = args.out_dir or (_repo_root() / "results" / "rsa_plots" / "figures_scaling")

    if not vs_path.exists():
        raise SystemExit(f"Missing vs-behavior file: {vs_path}")

    vs_rows = read_jsonl(vs_path)
    alpha_rows = read_jsonl(alpha_path) if alpha_path.exists() else None

    summary = build_summary(vs_rows, alpha_rows)
    write_summary_csv(out_csv, summary)
    print(f"Wrote {len(summary)} rows -> {out_csv}")

    plot_scaling(summary, out_dir)
    print(f"Wrote scaling figures -> {out_dir}")
    print("Exploratory correlation (mixed architectures; interpret cautiously):")
    print_scaling_correlations(summary)


if __name__ == "__main__":
    main()
