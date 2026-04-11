from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def fetch_models(base_url: str, api_key: str, output_modalities: str) -> list[dict[str, Any]]:
    endpoint = f"{base_url.rstrip('/')}/models"
    params = urllib.parse.urlencode({"output_modalities": output_modalities})
    url = f"{endpoint}?{params}"
    req = urllib.request.Request(
        url=url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("data", [])


def parse_param_list(model_obj: dict[str, Any]) -> set[str]:
    supported = model_obj.get("supported_parameters", []) or []
    return {str(x).strip() for x in supported}


def infer_scale_tier(model_id: str, model_name: str) -> str:
    size_b = extract_params_b(model_id, model_name)
    if size_b is None:
        return "unknown"
    if size_b <= 4:
        return "very_small"
    if size_b <= 10:
        return "small"
    if size_b <= 40:
        return "medium"
    return "large_frontier_open"


def extract_params_b(model_id: str, model_name: str) -> float | None:
    text = f"{model_id} {model_name}".lower()
    # MoE pattern like 235b-a22b -> use total params for bucket; keep active separately.
    moe = re.search(r"(\d+(\.\d+)?)\s*b[-_ ]a(\d+(\.\d+)?)\s*b", text)
    if moe:
        return float(moe.group(1))
    match = re.search(r"(\d+(\.\d+)?)\s*b", text)
    if match:
        return float(match.group(1))
    return None


def extract_active_params_b(model_id: str, model_name: str) -> float | None:
    text = f"{model_id} {model_name}".lower()
    moe = re.search(r"(\d+(\.\d+)?)\s*b[-_ ]a(\d+(\.\d+)?)\s*b", text)
    if moe:
        return float(moe.group(3))
    return None


def provider_from_id(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return "unknown"


def is_open_source_family(model_id: str) -> bool:
    """
    Heuristic open-source family filter by model slug prefix.
    """
    prefix = provider_from_id(model_id)
    open_prefixes = {
        "qwen",
        "deepseek",
        "meta-llama",
        "google",
        "mistralai",
        "allenai",
        "microsoft",
        "tii",
        "nousresearch",
    }
    return prefix in open_prefixes


def to_model_entry(model_obj: dict[str, Any]) -> dict[str, Any]:
    model_id = str(model_obj.get("id", ""))
    model_name = str(model_obj.get("name", ""))
    return {
        "id": model_id,
        "provider": provider_from_id(model_id),
        "family": provider_from_id(model_id),
        "params_b": extract_params_b(model_id, model_name),
        "active_params_b": extract_active_params_b(model_id, model_name),
        "scale_tier": infer_scale_tier(model_id, model_name),
        "reasoning_mode": "unknown",
    }


def bucketize_models(models: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "very_small": [],
        "small": [],
        "medium": [],
        "large_frontier_open": [],
        "unknown": [],
    }
    for m in models:
        tier = m.get("scale_tier", "unknown")
        if tier not in buckets:
            tier = "unknown"
        buckets[tier].append(m)
    for key in buckets:
        buckets[key].sort(key=lambda x: str(x.get("id", "")))
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter OpenRouter models by supported parameters and export YAML list."
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenRouter API base URL. Defaults to OPENROUTER_BASE_URL or LITELLM_BASE_URL.",
    )
    parser.add_argument(
        "--require",
        nargs="+",
        default=["logprobs", "top_logprobs"],
        help="Required supported_parameters for each model.",
    )
    parser.add_argument(
        "--output-modalities",
        default="text",
        help="Modalities filter for OpenRouter models endpoint.",
    )
    parser.add_argument(
        "--out-yaml",
        default="code/pipeline_v1/configs/models_openrouter_logprobs.yaml",
        help="Output YAML path",
    )
    parser.add_argument(
        "--out-json",
        default="code/pipeline_v1/configs/models_openrouter_logprobs_raw.json",
        help="Output JSON path with metadata",
    )
    parser.add_argument(
        "--out-bucketed-yaml",
        default="code/pipeline_v1/configs/models_openrouter_logprobs_bucketed.yaml",
        help="Output YAML path for bucketed model list.",
    )
    parser.add_argument(
        "--open-source-only",
        action="store_true",
        help="Keep only open-source model families (heuristic by model slug prefix).",
    )
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    base_url = (
        args.base_url
        or os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("LITELLM_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY/LITELLM_API_KEY in environment.")

    all_models = fetch_models(base_url=base_url, api_key=api_key, output_modalities=args.output_modalities)
    req_set = {x.strip() for x in args.require}

    filtered: list[dict[str, Any]] = []
    raw_filtered: list[dict[str, Any]] = []
    for m in all_models:
        supported = parse_param_list(m)
        model_id = str(m.get("id", ""))
        if args.open_source_only and not is_open_source_family(model_id):
            continue
        if req_set.issubset(supported):
            filtered.append(to_model_entry(m))
            raw_filtered.append(
                {
                    "id": model_id,
                    "name": m.get("name"),
                    "supported_parameters": sorted(list(supported)),
                    "context_length": (m.get("top_provider") or {}).get("context_length"),
                    "max_completion_tokens": (m.get("top_provider") or {}).get("max_completion_tokens"),
                    "pricing": m.get("pricing"),
                }
            )

    filtered.sort(key=lambda x: x["id"])
    raw_filtered.sort(key=lambda x: str(x.get("id", "")))

    yaml_payload = {
        "models": filtered,
        "defaults": {
            "provider": "openrouter",
            "requires_logprobs": "logprobs" in req_set,
            "required_parameters": sorted(list(req_set)),
        },
    }

    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=False), encoding="utf-8")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "base_url": base_url,
                "required_parameters": sorted(list(req_set)),
                "total_models": len(all_models),
                "filtered_models": len(raw_filtered),
                "models": raw_filtered,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    bucketed = bucketize_models(filtered)
    bucketed_payload = {
        "defaults": yaml_payload["defaults"],
        "bucket_rule": {
            "very_small": "<=4B",
            "small": ">4B and <=10B",
            "medium": ">10B and <=40B",
            "large_frontier_open": ">40B",
            "unknown": "params missing from model slug/name",
        },
        "buckets": bucketed,
    }
    out_bucketed = Path(args.out_bucketed_yaml)
    out_bucketed.parent.mkdir(parents=True, exist_ok=True)
    out_bucketed.write_text(
        yaml.safe_dump(bucketed_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    print(
        f"Filtered {len(raw_filtered)}/{len(all_models)} models. "
        f"Wrote: {out_yaml}, {out_json}, and {out_bucketed}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
