"""Minimal Phase-A analysis with optional human-baseline comparison.

Inputs:
  --run-dir .../results/runs/<run_id> (expects distributions.jsonl)
  --human-baseline optional JSONL matching baseline_points.schema.json

Outputs:
  <run-dir>/analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate_model_posteriors(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, float]]:
    # key: (model_id, k, utterance, prompt_strategy, question_key)
    buckets: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if not row.get("parse_ok"):
            continue
        dist = row.get("distribution")
        if not isinstance(dist, dict):
            continue
        key = (
            row.get("model_id"),
            int(row.get("k")),
            str(row.get("utterance")),
            str(row.get("prompt_strategy")),
            str(row.get("question_key")),
        )
        buckets[key].append({str(k): float(v) for k, v in dist.items()})

    agg: dict[tuple, dict[str, float]] = {}
    for key, values in buckets.items():
        keys = values[0].keys()
        agg[key] = {k: round(sum(v[k] for v in values) / len(values), 6) for k in keys}
    return agg


def implicature_shift_metrics(agg: dict[tuple, dict[str, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # some-posterior core metric from G&S:
    # shift = P(state=3 | k=1,some) - P(state=3 | k=3,some)
    models = sorted({k[0] for k in agg.keys()})
    for model_id in models:
        by_strat = sorted({k[3] for k in agg.keys() if k[0] == model_id})
        for strat in by_strat:
            p_k1 = agg.get((model_id, 1, "some", strat, "posterior"))
            p_k3 = agg.get((model_id, 3, "some", strat, "posterior"))
            if not p_k1 or not p_k3:
                continue
            s = round(float(p_k1.get("3", 0.0)) - float(p_k3.get("3", 0.0)), 6)
            out.append(
                {
                    "model_id": model_id,
                    "prompt_strategy": strat,
                    "metric": "some_k1_minus_k3_p_state3",
                    "value": s,
                }
            )
    return out


def compare_to_human(
    agg: dict[tuple, dict[str, float]], human_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # Compare only posterior rows for quantifier block.
    human_map: dict[tuple, float] = {}
    for row in human_rows:
        if row.get("question_key") != "posterior":
            continue
        if row.get("experiment_block") != "quantifier":
            continue
        key = (
            int(row["k"]),
            str(row["utterance"]),
            str(row["state"]),
        )
        human_map[key] = float(row["p_human"])

    out: list[dict[str, Any]] = []
    models = sorted({k[0] for k in agg.keys()})
    for model_id in models:
        by_strat = sorted({k[3] for k in agg.keys() if k[0] == model_id})
        for strat in by_strat:
            abs_errs: list[float] = []
            for k in (1, 2, 3):
                for utt in ("none", "some", "all"):
                    dist = agg.get((model_id, k, utt, strat, "posterior"))
                    if not dist:
                        continue
                    for state in ("0", "1", "2", "3"):
                        h = human_map.get((k, utt, state))
                        if h is None:
                            continue
                        abs_errs.append(abs(float(dist.get(state, 0.0)) - h))
            mae = _safe_mean(abs_errs)
            if mae is not None:
                out.append(
                    {
                        "model_id": model_id,
                        "prompt_strategy": strat,
                        "metric": "mae_vs_human_quantifier_posterior",
                        "value": round(mae, 6),
                        "num_points": len(abs_errs),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze phase-a distributions")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--human-baseline", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    dist_path = run_dir / "distributions.jsonl"
    rows = read_jsonl(dist_path)
    agg = aggregate_model_posteriors(rows)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_rows": len(rows),
        "num_parse_ok": sum(1 for r in rows if r.get("parse_ok")),
        "implicature_metrics": implicature_shift_metrics(agg),
    }

    if args.human_baseline:
        baseline_path = Path(args.human_baseline)
        if baseline_path.exists():
            human = read_jsonl(baseline_path)
            summary["human_comparison"] = compare_to_human(agg, human)
            summary["human_baseline_path"] = str(baseline_path)
        else:
            summary["human_comparison"] = []
            summary["human_baseline_path"] = str(baseline_path)
            summary["human_baseline_warning"] = "provided path does not exist"
    else:
        summary["human_comparison"] = []
        summary["human_baseline_path"] = None

    out_path = run_dir / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
