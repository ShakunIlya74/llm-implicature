from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pipeline_v1.concurrency import DynamicConcurrencyConfig, run_adaptive_map

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def init_run_dir(run_root: Path) -> Path:
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalize_model_id(model_id: str) -> str:
    """OpenRouter via OpenAI SDK uses provider/model slugs without a litellm-style prefix."""
    if model_id.startswith("openrouter/"):
        return model_id[len("openrouter/") :]
    return model_id


def _openrouter_base_url() -> str:
    return (
        os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("LITELLM_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )


def _openrouter_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The openai package is required for live runs. Install with: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "LITELLM_API_KEY"
    )
    if not api_key:
        raise SystemExit(
            "Missing OPENROUTER_API_KEY or LITELLM_API_KEY in environment (.env)."
        )
    return OpenAI(api_key=api_key, base_url=_openrouter_base_url())


def _resolve_prompt_contract_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    cand = Path.cwd() / path_str
    if cand.is_file():
        return cand
    return p


def call_openrouter_chat(
    model_id: str,
    stimulus: dict[str, Any],
    contract: dict[str, Any],
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> dict[str, Any]:
    from pipeline_v1.prompts.gs2013_message_builder import build_openai_messages

    scenario_id = stimulus.get("scenario_id", stimulus.get("stimulus_id", "unknown"))
    messages = build_openai_messages(stimulus, contract)

    if dry_run:
        if stimulus.get("question_key") == "knowledge":
            dry_text = '{"yes": 80, "no": 20}'
        else:
            dry_text = '{"0": 10, "1": 20, "2": 30, "3": 40}'
        return {
            "mode": "dry_run",
            "text": dry_text,
            "logprobs": None,
            "note": f"DRY_RUN response for {scenario_id}",
        }

    client = _openrouter_client()
    strategy = str(stimulus.get("prompt_strategy", "baseline"))
    create_kwargs: dict[str, Any] = {
        "model": _normalize_model_id(model_id),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if strategy == "structured_output":
        create_kwargs["response_format"] = {"type": "json_object"}
    else:
        create_kwargs["logprobs"] = True
        create_kwargs["top_logprobs"] = 5

    response = client.chat.completions.create(**create_kwargs)
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    raw_lp = getattr(choice, "logprobs", None)
    if raw_lp is not None and hasattr(raw_lp, "model_dump"):
        logprobs_out: Any = raw_lp.model_dump()
    else:
        logprobs_out = raw_lp
    return {
        "mode": "live",
        "text": text,
        "logprobs": logprobs_out,
    }


def _build_record(
    model_id: str,
    stim: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "scenario_id": stim.get("scenario_id", stim.get("stimulus_id")),
        "k": stim["k"],
        "utterance": stim["utterance"],
        "prompt_strategy": stim["prompt_strategy"],
        "question_key": stim.get("question_key", "posterior"),
        "result": result,
    }


def _run_one_task(
    payload: tuple[str, dict[str, Any], float, int, bool, dict[str, Any]],
) -> dict[str, Any]:
    model_id, stim, temperature, max_tokens, run_dry, contract = payload
    result = call_openrouter_chat(
        model_id=model_id,
        stimulus=stim,
        contract=contract,
        temperature=temperature,
        max_tokens=max_tokens,
        dry_run=run_dry,
    )
    return _build_record(model_id, stim, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal pipeline-v1 skeleton")
    parser.add_argument(
        "--experiment-config",
        default="code/pipeline_v1/configs/experiment.yaml",
    )
    parser.add_argument(
        "--models-config",
        default="code/pipeline_v1/configs/models_openrouter_logprobs_initial.yaml",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Override max examples from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run mode (no external API call)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Parallelism: fixed pool size, or initial adaptive limit when --dynamic-concurrency (default: run.concurrency in YAML, else 15)",
    )
    parser.add_argument(
        "--dynamic-concurrency",
        action="store_true",
        help="429-aware adaptive pool (also set run.dynamic_concurrency: true in experiment YAML)",
    )
    parser.add_argument(
        "--fixed-concurrency",
        action="store_true",
        help="Force fixed ThreadPoolExecutor (overrides --dynamic-concurrency and YAML)",
    )
    parser.add_argument(
        "--prompt-contract",
        default=None,
        metavar="PATH",
        help="YAML prompt contract (default: prompt_contract_path in experiment YAML, else G&S wording)",
    )
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    exp_cfg = load_yaml(Path(args.experiment_config))
    models_cfg = load_yaml(Path(args.models_config))

    from pipeline_v1.prompts.gs2013_message_builder import load_contract

    default_contract = "code/pipeline_v1/prompts/phase_a_prompt_contract_gs2013_wording.yaml"
    pc_path = args.prompt_contract or exp_cfg.get("prompt_contract_path", default_contract)
    contract = load_contract(_resolve_prompt_contract_path(pc_path))

    stimuli_path = Path(exp_cfg["generation"]["output_path"])
    stimuli = read_jsonl(stimuli_path)
    if not stimuli:
        raise SystemExit(
            f"No stimuli found at {stimuli_path}. Run generate_stimuli.py first."
        )

    max_examples = (
        args.max_examples
        if args.max_examples is not None
        else int(exp_cfg["run"]["max_examples"])
    )
    run_dry = bool(exp_cfg["run"]["dry_run"]) or bool(args.dry_run)
    temperature = float(exp_cfg["run"]["temperature"])
    max_tokens = int(exp_cfg["run"]["max_tokens"])
    concurrency = (
        args.concurrency
        if args.concurrency is not None
        else int(exp_cfg["run"].get("concurrency", 15))
    )
    if concurrency < 1:
        raise SystemExit("--concurrency / run.concurrency must be >= 1")

    use_dynamic = bool(exp_cfg["run"].get("dynamic_concurrency", False)) or bool(
        args.dynamic_concurrency
    )
    if args.fixed_concurrency:
        use_dynamic = False

    dc_cfg_raw = exp_cfg["run"].get("dynamic_concurrency_config")
    dc_fields = {f.name for f in dataclasses.fields(DynamicConcurrencyConfig)}
    dc_kwargs: dict[str, Any] = {}
    if isinstance(dc_cfg_raw, dict):
        dc_kwargs = {
            k: v
            for k, v in dc_cfg_raw.items()
            if k in dc_fields and v is not None
        }
    if "initial" not in dc_kwargs:
        dc_kwargs["initial"] = concurrency
    dynamic_cfg = DynamicConcurrencyConfig(**dc_kwargs)

    run_root = Path(exp_cfg["output"]["run_root"])
    run_dir = init_run_dir(run_root)
    out_path = run_dir / "responses.jsonl"

    selected = stimuli[:max_examples]
    models = models_cfg["models"]

    tasks: list[tuple[str, dict[str, Any], float, int, bool, dict[str, Any]]] = []
    for model in models:
        model_id = model["id"]
        for stim in selected:
            tasks.append((model_id, stim, temperature, max_tokens, run_dry, contract))

    if use_dynamic:
        records, dyn_stats = run_adaptive_map(tasks, _run_one_task, dynamic_cfg)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            records = list(executor.map(_run_one_task, tasks))
        dyn_stats = None

    with out_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_models": len(models),
        "num_examples_per_model": len(selected),
        "num_tasks": len(tasks),
        "concurrency": concurrency,
        "dynamic_concurrency": use_dynamic,
        "dry_run": run_dry,
        "prompt_contract_path": str(_resolve_prompt_contract_path(pc_path)),
        "prompt_version": contract.get("prompt_version"),
        "timestamp": dt.datetime.now().isoformat(),
    }
    if use_dynamic and dyn_stats is not None:
        meta["dynamic_concurrency_stats"] = dyn_stats
        meta["final_adaptive_limit"] = dyn_stats.get("final_limit")
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Run complete. Outputs at: {run_dir}")


if __name__ == "__main__":
    main()
