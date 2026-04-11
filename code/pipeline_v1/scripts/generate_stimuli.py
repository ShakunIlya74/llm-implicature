"""Generate stimuli.jsonl for G&S-style experiments (no LLM calls).

Data volume (evaluation, not this script):
  API calls ≈ num_stimulus_rows × samples_per_prompt × num_models

Run from repo root (llm-implicature-main) with PYTHONPATH=code, e.g.:
  python code/pipeline_v1/scripts/generate_stimuli.py
  python code/pipeline_v1/scripts/generate_stimuli.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pipeline_v1.stimuli.gs2013_stimuli import build_all_records, summarize_counts


def load_experiment_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stimuli JSONL from experiment.yaml (deterministic; no API)."
    )
    parser.add_argument(
        "--config",
        default="code/pipeline_v1/configs/experiment.yaml",
        help="Path to experiment YAML",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print count summary and exit without writing files",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path.resolve()}")

    cfg = load_experiment_config(cfg_path)
    records = build_all_records(cfg)
    summary = summarize_counts(cfg, len(records))

    if args.summary_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Would write {len(records)} rows (see generation.output_path in config)")
        return

    output_path = Path(cfg["generation"]["output_path"])
    write_jsonl(records, output_path)

    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "config": str(cfg_path),
                "output_path": str(output_path),
                "num_rows": len(records),
                **summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Generated {len(records)} records -> {output_path}")
    print(f"Meta -> {meta_path}")


if __name__ == "__main__":
    main()
