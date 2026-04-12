"""Run RSA probes: max_tokens=1, logprobs for quantifier analysis.

Uses OpenAI-compatible client. Env (same pattern as run_all.py):
  OPENROUTER_API_KEY or LITELLM_API_KEY or OPENAI_API_KEY
  OPENROUTER_BASE_URL or LITELLM_BASE_URL or OPENAI_BASE_URL (default OpenRouter)

For local vLLM: set OPENAI_BASE_URL=http://127.0.0.1:8000/v1 and OPENAI_API_KEY=EMPTY
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install pyyaml") from exc

from pipeline_v1.rsa.messages import build_messages, load_contract


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _api_base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("LITELLM_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )


def _api_key() -> str:
    return (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )


def _normalize_model_id(model_id: str) -> str:
    if model_id.startswith("openrouter/"):
        return model_id[len("openrouter/") :]
    return model_id


def _client():
    from openai import OpenAI

    key = _api_key()
    if not key:
        raise SystemExit(
            "Set OPENROUTER_API_KEY, LITELLM_API_KEY, or OPENAI_API_KEY in .env"
        )
    return OpenAI(api_key=key, base_url=_api_base_url())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_one(
    payload: tuple[dict[str, Any], str, dict[str, Any], float, int, int, bool],
) -> dict[str, Any]:
    row, model_id, contract, temperature, top_logprobs, max_tokens, dry = payload
    messages = build_messages(row, contract)
    if dry:
        return {
            "rsa_id": row["rsa_id"],
            "probe_type": row["probe_type"],
            "story_index": row["story_index"],
            "model_id": model_id,
            "stimulus": row,
            "result": {
                "mode": "dry_run",
                "text": "Some",
                "logprobs": None,
            },
        }

    client = _client()
    resp = client.chat.completions.create(
        model=_normalize_model_id(model_id),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    raw_lp = getattr(choice, "logprobs", None)
    if raw_lp is not None and hasattr(raw_lp, "model_dump"):
        logprobs_out: Any = raw_lp.model_dump()
    else:
        logprobs_out = raw_lp

    return {
        "rsa_id": row["rsa_id"],
        "probe_type": row["probe_type"],
        "story_index": row["story_index"],
        "model_id": model_id,
        "stimulus": row,
        "result": {"mode": "live", "text": text, "logprobs": logprobs_out},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RSA logprob probes")
    parser.add_argument(
        "--stimuli",
        type=Path,
        default=Path("code/pipeline_v1/data/processed/rsa_stimuli.jsonl"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("code/pipeline_v1/rsa/contract_rsa.yaml"),
    )
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("code/pipeline_v1/results/rsa_runs"),
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    contract = load_contract(args.contract)
    rows = read_jsonl(args.stimuli)
    if args.limit is not None:
        rows = rows[: args.limit]

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "rsa_responses.jsonl"

    tasks = [
        (
            row,
            args.model,
            contract,
            args.temperature,
            args.top_logprobs,
            args.max_tokens,
            args.dry_run,
        )
        for row in rows
    ]

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        records = list(ex.map(_run_one, tasks))

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "run_dir": str(run_dir),
        "num_rows": len(records),
        "model": args.model,
        "base_url": _api_base_url(),
        "dry_run": args.dry_run,
        "contract": str(args.contract),
        "stimuli": str(args.stimuli),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
