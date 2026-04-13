"""Paper-style faceted bar plots from rsa_export_long_table.py CSV output.

Layout mirrors G&S-style figures: x = state 0–3, y = bet (0–100), faceted by
access and utterance. Multiple series (LLM bets, RSA, logprobs, optional human)
are dodged in each panel.

Usage (from repo root):
  py -3 code/pipeline_v1/scripts/rsa_plot_paper_bars.py

Defaults: results/rsa_plots/rsa_long_combined.csv and results/rsa_plots/figures_bar/

Caption note: human rows (if any) use paper-style digitized posteriors; LLM rows are
aggregated as in rsa_probe (trial-level knowledge filter from the paper is not applied).

See also: rsa_plot_scaling.py for metrics vs model size (billions of parameters).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

# matplotlib imported inside main paths to allow --help without dependency

# Numerical experiment: valid (access, utterance) cells (paper-style grid).
NUMERICAL_PANELS: list[tuple[int, str]] = [
    (1, "1"),
    (2, "1"),
    (3, "1"),
    (2, "2"),
    (3, "2"),
    (3, "3"),
]


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fbet(r: dict[str, Any]) -> float:
    return float(r["bet_0_100"])


def human_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, int], float]:
    """(experiment, access, utterance, state) -> bet_0_100 for human posteriors."""
    idx: dict[tuple[str, int, str, int], float] = {}
    for r in rows:
        if r.get("source") != "human_posterior_bets":
            continue
        k = (str(r["experiment"]), int(r["access"]), str(r["utterance"]), int(r["state"]))
        idx[k] = fbet(r)
    return idx


def series_for_model(
    rows: list[dict[str, Any]],
    model_id: str,
    experiment: str,
) -> dict[tuple[int, str, int, str], float]:
    """(access, utterance, state, source) -> bet."""
    out: dict[tuple[int, str, int, str], float] = {}
    for r in rows:
        if r["model_id"] != model_id or r["experiment"] != experiment:
            continue
        key = (int(r["access"]), str(r["utterance"]), int(r["state"]), str(r["source"]))
        out[key] = fbet(r)
    return out


def sources_present(series: dict[tuple[int, str, int, str], float]) -> list[str]:
    seen: set[str] = set()
    for (*_, src) in series.keys():
        seen.add(src)
    order = [
        "human_posterior_bets",
        "llm_bets",
        "rsa_predicted",
        "logprobs_posterior",
    ]
    return [s for s in order if s in seen]


def plot_quantifier_figure(
    ax_grid: Any,
    model_id: str,
    experiment: str,
    series: dict[tuple[int, str, int, str], float],
    human_idx: dict[tuple[str, int, str, int], float],
    utterances: list[str],
    accesses: list[int],
    source_list: list[str],
    label_map: dict[str, str],
) -> None:
    import numpy as np

    n_sources = len(source_list)
    width = 0.8 / max(n_sources, 1)
    x_base = np.arange(4)

    for i, utt in enumerate(utterances):
        for j, acc in enumerate(accesses):
            ax = ax_grid[i, j] if ax_grid.ndim == 2 else ax_grid[i * len(accesses) + j]
            for si, src in enumerate(source_list):
                heights = []
                for st in range(4):
                    if src == "human_posterior_bets":
                        h = human_idx.get((experiment, acc, utt, st))
                    else:
                        h = series.get((acc, utt, st, src))
                    heights.append(h if h is not None else float("nan"))
                offset = (si - (n_sources - 1) / 2) * width
                mask = [h == h for h in heights]  # not NaN
                if not any(mask):
                    continue
                ax.bar(
                    x_base[mask] + offset,
                    [heights[k] for k in range(4) if mask[k]],
                    width * 0.95,
                    label=label_map.get(src, src) if i == 0 and j == 0 else None,
                )
            ax.set_xticks(x_base)
            ax.set_xticklabels(["0", "1", "2", "3"])
            ax.set_ylim(0, 100)
            if i == len(utterances) - 1:
                ax.set_xlabel("State")
            if j == 0:
                ax.set_ylabel("Bet / prob × 100")
            ax.set_title(f"{utt}, access {acc}")


def plot_numerical_figure(
    fig: Any,
    model_id: str,
    experiment: str,
    series: dict[tuple[int, str, int, str], float],
    human_idx: dict[tuple[str, int, str, int], float],
    source_list: list[str],
    label_map: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # 3x3: rows = utterance 1,2,3 — cols = access 1,2,3
    axes = fig.subplots(3, 3, sharey=True)
    utter_order = ["1", "2", "3"]
    acc_order = [1, 2, 3]
    width = 0.8 / max(len(source_list), 1)
    x_base = np.arange(4)

    for ri, utt in enumerate(utter_order):
        for ci, acc in enumerate(acc_order):
            ax = axes[ri, ci]
            if (acc, utt) not in {(a, u) for a, u in NUMERICAL_PANELS}:
                ax.set_visible(False)
                continue
            for si, src in enumerate(source_list):
                heights = []
                for st in range(4):
                    if src == "human_posterior_bets":
                        h = human_idx.get((experiment, acc, utt, st))
                    else:
                        h = series.get((acc, utt, st, src))
                    heights.append(h if h is not None else float("nan"))
                offset = (si - (len(source_list) - 1) / 2) * width
                mask = [h == h for h in heights]
                if not any(mask):
                    continue
                ax.bar(
                    x_base[mask] + offset,
                    [heights[k] for k in range(4) if mask[k]],
                    width * 0.95,
                    label=label_map.get(src, src) if ri == 0 and ci == 0 else None,
                )
            ax.set_xticks(x_base)
            ax.set_xticklabels(["0", "1", "2", "3"])
            ax.set_ylim(0, 100)
            if ri == 2:
                ax.set_xlabel("State")
            if ci == 0:
                ax.set_ylabel("Bet / prob × 100")
            ax.set_title(f'"{utt}", access {acc}')


def run_plotting(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_csv(args.csv)
    human_idx = human_index(rows)

    llm_models = sorted(
        {r["model_id"] for r in rows if not str(r["model_id"]).startswith("human_")}
    )
    label_map = {
        "human_posterior_bets": "Human (digitized)",
        "llm_bets": "LLM structured bets",
        "rsa_predicted": "RSA (fitted α)",
        "logprobs_posterior": "LLM logprobs",
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id in llm_models:
        for experiment in ("exps_some", "exps_numerical"):
            series = series_for_model(rows, model_id, experiment)
            if not series:
                continue
            src_llm = sources_present(series)
            # Human rows (digitized) are keyed by experiment in human_idx
            has_h = any(k[0] == experiment for k in human_idx)
            source_list = (["human_posterior_bets"] if has_h else []) + [
                s for s in src_llm if s != "human_posterior_bets"
            ]

            safe_name = model_id.replace("/", "-").replace(" ", "_")
            if experiment == "exps_some":
                seen_u = {str(r["utterance"]) for r in rows if r["experiment"] == experiment}
                pref = ("all", "none", "some")
                utterances = [u for u in pref if u in seen_u] + sorted(seen_u.difference(pref))
                accesses = sorted({int(r["access"]) for r in rows if r["experiment"] == experiment})
                fig, ax_grid = plt.subplots(
                    len(utterances),
                    len(accesses),
                    figsize=(3.2 * len(accesses), 2.8 * len(utterances)),
                    sharey=True,
                    squeeze=False,
                )
                plot_quantifier_figure(
                    ax_grid,
                    model_id,
                    experiment,
                    series,
                    human_idx,
                    utterances,
                    accesses,
                    source_list,
                    label_map,
                )
                fig.suptitle(f"{model_id} — {experiment}\n(LLM aggregate; human Fig.2c-style if provided)")
                handles, labels = ax_grid[0, 0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper right", fontsize=8)
                fig.tight_layout()
                out = out_dir / f"{safe_name}__{experiment}.png"
                fig.savefig(out, dpi=150)
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(10, 9))
                plot_numerical_figure(fig, model_id, experiment, series, human_idx, source_list, label_map)
                fig.suptitle(f"{model_id} — {experiment}\n(empty cells = not run; LLM aggregate)")
                ax0 = fig.axes[0]
                handles, labels = ax0.get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper right", fontsize=8)
                fig.tight_layout()
                out = out_dir / f"{safe_name}__{experiment}.png"
                fig.savefig(out, dpi=150)
                plt.close(fig)

    print(f"Wrote figures under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RSA long CSV -> paper-style bar plots")
    root = _repo_root()
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "results" / "rsa_plots" / "rsa_long_combined.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "results" / "rsa_plots" / "figures_bar",
    )
    args = parser.parse_args()
    run_plotting(args)


if __name__ == "__main__":
    main()
