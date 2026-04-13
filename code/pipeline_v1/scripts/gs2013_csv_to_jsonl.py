"""Convert filled gs2013_exp1_some_posterior_template.csv to human_baseline JSONL.

Expects columns: k, state, mean_bet_0_100
Checks that for each k, values sum to ~100 (within tolerance) and converts to p_human = bet/100.

Usage:
  PYTHONPATH=code python code/pipeline_v1/scripts/gs2013_csv_to_jsonl.py \\
    --csv code/pipeline_v1/data/human_baseline/gs2013_exp1_some_posterior_filled.csv \\
    --out code/pipeline_v1/data/human_baseline/human_baseline.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--out",
        default="code/pipeline_v1/data/human_baseline/human_baseline.jsonl",
    )
    parser.add_argument(
        "--source",
        default="gs2013_figure2c_digitized",
        help="Value for each row's source field",
    )
    parser.add_argument("--sum-tol", type=float, default=2.0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line = {k.strip(): (v or "").strip() for k, v in row.items() if k and k.strip()}
            if not line or line.get("k", "").startswith("#"):
                continue
            if "k" not in line or "state" not in line:
                continue
            rows.append(line)

    by_k: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if not row.get("mean_bet_0_100"):
            raise SystemExit(f"Missing mean_bet_0_100 in row: {row}")
        k = int(row["k"])
        state = int(row["state"])
        bet = float(row["mean_bet_0_100"])
        by_k[k].append((state, bet))

    for k, pairs in by_k.items():
        pairs.sort(key=lambda x: x[0])
        s = sum(v for _, v in pairs)
        if abs(s - 100.0) > args.sum_tol:
            raise SystemExit(
                f"For k={k}, mean bets sum to {s}, expected ~100 (tol={args.sum_tol})"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            k = int(row["k"])
            state = int(row["state"])
            bet = float(row["mean_bet_0_100"])
            rec = {
                "source": args.source,
                "experiment_block": "quantifier",
                "story_shortname": "collapsed_all_stories",
                "k": k,
                "utterance_type": "quantifier",
                "utterance": "some",
                "question_key": "posterior",
                "state": state,
                "p_human": round(bet / 100.0, 6),
                "notes": "Collapsed across scenarios; from G&S Fig 2c digitization",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
