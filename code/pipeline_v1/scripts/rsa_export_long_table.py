"""Export RSA probe outputs to a tidy long CSV for plotting and stats.

Reads rsa_predictions.jsonl (required) and rsa_vs_behavior.jsonl (optional) and writes
one row per (model, experiment, access, utterance, state, series).

Usage (from repo root):
  py -3 code/pipeline_v1/scripts/rsa_export_long_table.py

Omit paths to use the newest results/rsa_results/<run>/ (by rsa_predictions.jsonl mtime).
"""

from __future__ import annotations

import argparse
import csv
import json
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expand_dist(
    row_base: dict[str, Any],
    dist: dict[str, Any],
    source: str,
    inference_method: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for state in ("0", "1", "2", "3"):
        p = float(dist[state])
        out.append(
            {
                "model_id": row_base["model_id"],
                "experiment": row_base["experiment"],
                "inference_method": inference_method,
                "source": source,
                "access": row_base["access"],
                "utterance": row_base["utterance"],
                "state": int(state),
                "prob_0_1": p,
                "bet_0_100": round(p * 100.0, 6),
                "alpha_hat": row_base.get("alpha_hat", ""),
            }
        )
    return out


def load_human_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows from gs2013_csv_to_jsonl-style human_baseline.jsonl."""
    rows_out: list[dict[str, Any]] = []
    for r in read_jsonl(path):
        block = str(r.get("experiment_block", "")).lower()
        if block in ("quantifier", "exp1", "some"):
            experiment = "exps_some"
        elif block in ("numeral", "numerals", "exp2", "numerical"):
            experiment = "exps_numerical"
        else:
            experiment = block or "human_unknown"
        model_id = f"human_{r.get('source', 'baseline')}"
        access = int(r["k"])
        utterance = str(r.get("utterance", ""))
        state = int(r["state"])
        p = float(r["p_human"])
        rows_out.append(
            {
                "model_id": model_id,
                "experiment": experiment,
                "inference_method": "human_digitized",
                "source": "human_posterior_bets",
                "access": access,
                "utterance": utterance,
                "state": state,
                "prob_0_1": p,
                "bet_0_100": round(p * 100.0, 6),
                "alpha_hat": "",
            }
        )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="RSA predictions -> tidy long CSV")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="rsa_predictions.jsonl (default: newest under results/rsa_results/)",
    )
    parser.add_argument(
        "--vs-behavior",
        type=Path,
        default=None,
        help="rsa_vs_behavior.jsonl (default: same run dir as predictions)",
    )
    parser.add_argument(
        "--human-jsonl",
        type=Path,
        default=None,
        help="Optional gs2013-style human baseline JSONL (merged into same long table)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: results/rsa_plots/rsa_long_combined.csv)",
    )
    args = parser.parse_args()

    run_dir = _latest_rsa_run_dir()
    pred_path = args.predictions or (run_dir / "rsa_predictions.jsonl")
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions file: {pred_path}")

    vs_path = args.vs_behavior if args.vs_behavior is not None else (run_dir / "rsa_vs_behavior.jsonl")
    out_path = args.out or (_repo_root() / "results" / "rsa_plots" / "rsa_long_combined.csv")

    pred_rows = read_jsonl(pred_path)
    vs_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    if vs_path.exists():
        for r in read_jsonl(vs_path):
            k = (str(r["model_id"]), str(r["experiment"]), int(r["access"]), str(r["utterance"]))
            vs_by_key[k] = r

    long_rows: list[dict[str, Any]] = []
    for r in pred_rows:
        base = {
            "model_id": r["model_id"],
            "experiment": r["experiment"],
            "access": int(r["access"]),
            "utterance": str(r["utterance"]),
            "alpha_hat": r.get("alpha_hat"),
        }
        obs = r["observed_posterior_bets"]
        rsa = r["rsa_predicted_posterior"]
        long_rows.extend(
            expand_dist(base, obs, "llm_bets", "structured_output_agg")
        )
        long_rows.extend(
            expand_dist(base, rsa, "rsa_predicted", "rsa_fit")
        )
        k = (base["model_id"], base["experiment"], base["access"], base["utterance"])
        vs = vs_by_key.get(k)
        if vs and vs.get("has_logprob_distribution") and isinstance(
            vs.get("logprob_posterior"), dict
        ):
            long_rows.extend(
                expand_dist(
                    base,
                    vs["logprob_posterior"],
                    "logprobs_posterior",
                    "ftp_logprobs_agg",
                )
            )

    if args.human_jsonl and args.human_jsonl.exists():
        long_rows.extend(load_human_jsonl(args.human_jsonl))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_id",
        "experiment",
        "inference_method",
        "source",
        "access",
        "utterance",
        "state",
        "prob_0_1",
        "bet_0_100",
        "alpha_hat",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(long_rows)

    print(f"Wrote {len(long_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
