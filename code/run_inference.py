"""Main experiment orchestrator!!!

Iterates over models (local via vLLM + remote via api), runs both experiments (exps_some, exps_numerical) with 
respective inference methods for each model.

Usage examples:
    # Run everything (local + API models)
    python run_inference.py

    # Local models only
    python run_inference.py --mode local
"""

from __future__ import annotations

import argparse
import os
import sys

from llm_utils.inference import InferenceClient
from llm_utils.prompts import METHODS
from llm_utils.vllm_server import start_vllm_server, stop_vllm_server, wait_for_vllm
from exps_some import run_some_experiment
from exps_numerical import run_numerical_experiment

# Model registries

LOCAL_MODELS = [
    # TODO: check if  model list is the sota in these families in their seizes (<=9b)
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-8B",
    # qwen 3.5
    # "Qwen/Qwen3.5-0.8B",
    # "Qwen/Qwen3.5-4B",
    # "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-9B",  
    # llamas
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct", 
    # phi
    "microsoft/Phi-4-mini-instruct",
    # Gemmas
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
]

API_MODELS = [
    # TODO: update to the best (or keep the cheapest haha) api models
    {
        "model": "gpt-5.4-nano",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    {
        "model": "gpt-5.4-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    {
        "model": "gemini-3.1-flash-lite-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
    },
]

# Methods available for each model type
ALL_METHODS = METHODS  # all 4
API_METHODS = [  # prefilling not available for API models
    "structured_output",
    "natural_language",
    "ftp_logprobs_single",
]

# Experiment registry
EXPERIMENT_RUNNERS = {
    "exps_some": run_some_experiment,
    "exps_numerical": run_numerical_experiment,
}

# Local model runner


def run_local_models(
    models: list[str] | None = None,
    methods: list[str] | None = None,
    experiments: list[str] | None = None,
    version: str = "prompting-v1",
    port: int = 8000,
    api_key: str = "token-abc123",
    vllm_extra_args: list[str] | None = None,
) -> None:
    """Run experiments on local models via vLLM.

    For each model: starts a vLLM server, runs all requested experiments
    and methods, then shuts down before loading the next model.
    """
    models = models or LOCAL_MODELS
    methods = methods or ALL_METHODS
    experiments = experiments or list(EXPERIMENT_RUNNERS.keys())
    base_url = f"http://127.0.0.1:{port}/v1"

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"LOCAL MODEL: {model}")
        print(f"{'=' * 60}")

        proc = start_vllm_server(
            model=model,
            port=port,
            api_key=api_key,
            extra_args=vllm_extra_args or [],
        )

        try:
            wait_for_vllm(
                base_url=f"http://127.0.0.1:{port}",
                api_key=api_key,
            )

            client = InferenceClient(
                base_url=base_url,
                api_key=api_key,
                model=model,
            )

            for method in methods:
                for exp_name in experiments:
                    exp_fn = EXPERIMENT_RUNNERS[exp_name]
                    print(f"\n  >>> {exp_name} / {method}")
                    try:
                        exp_fn(client, method, version, base_url)
                    except Exception as e:
                        print(f"  ERROR in {exp_name}/{method}: {e}")
                        continue

        finally:
            stop_vllm_server(proc)


# API model runner


def run_api_models(
    models: list[dict] | None = None,
    methods: list[str] | None = None,
    experiments: list[str] | None = None,
    version: str = "prompting-v1",
) -> None:
    """Run experiments on API models (OpenAI, Gemini).

    Reads API keys from environment variables. Skips models whose key
    is not set.
    """
    models = models or API_MODELS
    methods = methods or API_METHODS
    experiments = experiments or list(EXPERIMENT_RUNNERS.keys())

    for model_cfg in models:
        model = model_cfg["model"]
        base_url = model_cfg["base_url"]
        api_key = os.environ.get(model_cfg["api_key_env"], "")

        if not api_key:
            print(
                f"\nSkipping {model}: "
                f"env var {model_cfg['api_key_env']} not set"
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"API MODEL: {model}")
        print(f"{'=' * 60}")

        client = InferenceClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )

        for method in methods:
            for exp_name in experiments:
                exp_fn = EXPERIMENT_RUNNERS[exp_name]
                print(f"\n  >>> {exp_name} / {method}")
                try:
                    exp_fn(client, method, version, base_url)
                except Exception as e:
                    print(f"  ERROR in {exp_name}/{method}: {e}")
                    continue


# CLI


def main(args_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM implicature experiments across models and methods."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api", "all"],
        default="all",
        help="Which model set to run (default: all)",
    )
    parser.add_argument(
        "--version",
        default="prompting-v1",
        help="Prompt version to use (default: prompting-v1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific local model names to run (local mode only)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        help="Specific methods to run",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENT_RUNNERS.keys()),
        help="Specific experiments to run",
    )
    parser.add_argument(
        "--vllm-extra-args",
        nargs="+",
        default=[],
        help="Extra arguments passed to vLLM server (e.g. --gpu-memory-utilization 0.9)",
    )
    args = parser.parse_args(args_list)

    if args.mode in ("local", "all"):
        run_local_models(
            models=args.models,
            methods=args.methods,
            experiments=args.experiments,
            version=args.version,
            port=args.port,
            vllm_extra_args=args.vllm_extra_args,
        )

    if args.mode in ("api", "all"):
        run_api_models(
            methods=args.methods,
            experiments=args.experiments,
            version=args.version,
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # temp testing arguments when run without args from editor
        # main(["--mode", "local", "--models", "Qwen/Qwen3.5-0.8B"])
        main(["--mode", "local"])
    else:
        main()
