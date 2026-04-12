"""Generate rsa_stimuli.jsonl (no API). Align keys with Phase A: story_index, k, etc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RSA_VERSION = "rsa_v1"


def build_records(
    knowledge_levels: tuple[int, ...] = (1, 2, 3),
    states: tuple[int, ...] = (0, 1, 2, 3),
    include_prior: bool = True,
) -> list[dict[str, Any]]:
    from llm_utils.prompts import STORIES

    rows: list[dict[str, Any]] = []
    n = 0
    for story in STORIES:
        sid = int(story["index"])
        if include_prior:
            n += 1
            rows.append(
                {
                    "rsa_id": f"rsa-{n:05d}",
                    "rsa_version": RSA_VERSION,
                    "probe_type": "prior",
                    "story_index": sid,
                    "story_shortname": story["shortname"],
                }
            )
        for k in knowledge_levels:
            for state_s in states:
                n += 1
                rows.append(
                    {
                        "rsa_id": f"rsa-{n:05d}",
                        "rsa_version": RSA_VERSION,
                        "probe_type": "speaker",
                        "story_index": sid,
                        "story_shortname": story["shortname"],
                        "k": k,
                        "state_s": state_s,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RSA probe stimuli JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("code/pipeline_v1/data/processed/rsa_stimuli.jsonl"),
    )
    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Only speaker probes (for smaller runs)",
    )
    args = parser.parse_args()
    rows = build_records(include_prior=not args.no_prior)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
