"""Fit rational-speaker softmax parameter alpha: p(u) ∝ exp(alpha * U(u;s)).

Uses literal utilities from utility.py. Compares model distribution to observed
quantifier_probs from parse_quantifier_logprobs (speaker probes only).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pipeline_v1.rsa.utility import QUANTS, softmax_over_quants, utility_table


def cross_entropy(p_obs: dict[str, float], p_model: dict[str, float]) -> float:
    eps = 1e-12
    return -sum(p_obs[q] * math.log(p_model.get(q, eps) + eps) for q in QUANTS)


def fit_alpha_for_row(stimulus: dict[str, Any], p_obs: dict[str, float]) -> dict[str, Any]:
    s = int(stimulus["state_s"])
    util = utility_table(s)
    best_a = 1.0
    best_ce = float("inf")
    # log-spaced grid
    for i in range(-60, 61):
        a = 10 ** (i / 20.0)  # ~0.03 to ~32
        pm = softmax_over_quants(util, a)
        ce = cross_entropy(p_obs, pm)
        if ce < best_ce:
            best_ce = ce
            best_a = a
    # local refine
    for _ in range(30):
        step = best_a * 0.05
        for a in (best_a - step, best_a + step):
            if a <= 0:
                continue
            pm = softmax_over_quants(util, a)
            ce = cross_entropy(p_obs, pm)
            if ce < best_ce:
                best_ce = ce
                best_a = a
    pm = softmax_over_quants(util, best_a)
    return {
        "alpha_hat": best_a,
        "cross_entropy": best_ce,
        "p_obs": p_obs,
        "p_model": pm,
        "utility": util,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parsed",
        type=Path,
        required=True,
        help="parsed_quantifiers.jsonl from parse_quantifier_logprobs.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: same dir as --parsed / alpha_fits.jsonl",
    )
    args = parser.parse_args()
    out = args.output or (args.parsed.parent / "alpha_fits.jsonl")

    fits: list[dict[str, Any]] = []
    with args.parsed.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("probe_type") != "speaker":
                continue
            if not row.get("parse_ok"):
                continue
            st = row.get("stimulus") or {}
            p_obs = row.get("quantifier_probs") or {}
            if not p_obs:
                continue
            fit = fit_alpha_for_row(st, p_obs)
            fits.append(
                {
                    "rsa_id": row.get("rsa_id"),
                    "story_index": st.get("story_index"),
                    "k": st.get("k"),
                    "state_s": st.get("state_s"),
                    "model_id": row.get("model_id"),
                    "alpha_hat": fit["alpha_hat"],
                    "cross_entropy": fit["cross_entropy"],
                    "p_obs": p_obs,
                    "p_model": fit["p_model"],
                }
            )

    with out.open("w", encoding="utf-8") as f:
        for r in fits:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if fits:
        mean_ce = sum(r["cross_entropy"] for r in fits) / len(fits)
        print(f"Wrote {len(fits)} fits to {out} (mean CE = {mean_ce:.4f})")
    else:
        print(f"No speaker rows found; wrote empty {out}")


if __name__ == "__main__":
    main()
