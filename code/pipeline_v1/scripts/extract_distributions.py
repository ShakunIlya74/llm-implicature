"""Extract normalized distributions from run_all responses.

Reads:
  code/pipeline_v1/results/runs/<run_id>/responses.jsonl

Writes:
  code/pipeline_v1/results/runs/<run_id>/distributions.jsonl

This script is intentionally permissive: it first tries JSON parsing,
then falls back to extracting number patterns from free text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


COUNT_KEYS = ("0", "1", "2", "3")
KNOWLEDGE_KEYS = ("yes", "no")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize(d: dict[str, float]) -> dict[str, float] | None:
    total = float(sum(d.values()))
    if total <= 0:
        return None
    return {k: round(v / total, 6) for k, v in d.items()}


def _from_json_blob(text: str, expected_keys: tuple[str, ...]) -> dict[str, float] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    lowered = {str(k).lower(): v for k, v in parsed.items()}
    if set(lowered.keys()) != set(expected_keys):
        return None
    if not all(isinstance(v, (int, float)) for v in lowered.values()):
        return None
    numeric = {k: float(lowered[k]) for k in expected_keys}
    return _normalize(numeric)


def _extract_numbers(text: str) -> list[float]:
    # Matches integers and decimals, including percentage suffix.
    pattern = r"(?<!\w)(\d+(?:\.\d+)?)\s*%?"
    return [float(m) for m in re.findall(pattern, text)]


def parse_count_distribution(text: str) -> dict[str, float] | None:
    parsed = _from_json_blob(text, COUNT_KEYS)
    if parsed is not None:
        return parsed

    nums = _extract_numbers(text)
    if len(nums) >= 4:
        cand = {k: nums[i] for i, k in enumerate(COUNT_KEYS)}
        return _normalize(cand)
    return None


def parse_knowledge_distribution(text: str) -> dict[str, float] | None:
    parsed = _from_json_blob(text, KNOWLEDGE_KEYS)
    if parsed is not None:
        return parsed

    nums = _extract_numbers(text)
    if len(nums) >= 2:
        cand = {"yes": nums[0], "no": nums[1]}
        return _normalize(cand)
    return None


def extract_one(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result", {})
    raw_text = str(result.get("text", ""))
    qkey = str(row.get("question_key", "posterior"))

    if qkey == "knowledge":
        dist = parse_knowledge_distribution(raw_text)
    else:
        dist = parse_count_distribution(raw_text)

    return {
        "model_id": row.get("model_id"),
        "scenario_id": row.get("scenario_id"),
        "k": row.get("k"),
        "utterance": row.get("utterance"),
        "prompt_strategy": row.get("prompt_strategy"),
        "question_key": qkey,
        "distribution": dist,
        "parse_ok": dist is not None,
        "raw_text": raw_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract distributions from responses.jsonl")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    in_path = run_dir / "responses.jsonl"
    out_path = run_dir / "distributions.jsonl"

    rows = read_jsonl(in_path)
    parsed_rows = [extract_one(r) for r in rows]

    with out_path.open("w", encoding="utf-8") as out:
        for row in parsed_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = sum(1 for r in parsed_rows if r["parse_ok"])
    print(f"Extracted {len(parsed_rows)} rows, parse_ok={ok}, parse_fail={len(parsed_rows)-ok}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
