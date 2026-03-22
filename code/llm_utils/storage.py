"""Storage utilities for experiment results.

Directory layout:
    - data/llm-inference
    - {experiment} # exps_some | exps_numerical
    - {model} # model name (/ replaced with --)
    - {method} # structured_output | natural_language ...
    - {version} # prompting-v1 | prompting-v2 ...
    - meta.json, inference_story_1.jsonl, .... inference_story_6.jsonl
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "llm-inference"


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "--")


def get_output_dir( experiment: str, model: str, method: str, version: str) -> Path:
    return BASE_DIR / experiment / sanitize_model_name(model) / method / version


def write_meta(
    output_dir: Path,
    *,
    model: str,
    method: str,
    version: str,
    experiment: str,
    temperature: float,
    max_tokens: int,
    base_url: str,
    **extra: Any,
) -> Path:
    """Write meta.json with model parameters and run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "model": model,
        "method": method,
        "prompt_version": version,
        "experiment": experiment,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "base_url": base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # capture git commit hash to link results to code version
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            meta["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    meta.update(extra)

    path = output_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def append_jsonl(output_dir: Path, story_index: int, record: dict) -> None:
    """Append a single JSON record as one line to a story's JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"inference_story_{story_index}.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")


def load_existing_keys(output_dir: Path, story_index: int) -> set[str]:
    """Load already-completed (access|observe|key) tuples for resumption."""
    path = output_dir / f"inference_story_{story_index}.jsonl"
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            k = f"{rec.get('access')}|{rec.get('observe')}|{rec.get('key')}"
            keys.add(k)
        except json.JSONDecodeError:
            pass
    return keys


def make_record_key(access: Any, observe: Any, question_key: str) -> str:
    """Build the deduplication key matching load_existing_keys format."""
    return f"{access}|{observe}|{question_key}"
