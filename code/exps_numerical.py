"""Numerical quantifier experiment runner.

Runs inference across 6 stories (each with one of the 6 conditions) * 3 question types 
using the specified inference method and prompt version.
"""

from __future__ import annotations

from llm_utils.inference import InferenceClient, run_experiment
from llm_utils.prompts import NUMERICAL_CONDITIONS

EXPERIMENT_NAME = "exps_numerical"


def run_numerical_experiment(
    client: InferenceClient,
    method: str,
    version: str = "prompting-v1",
    base_url: str = "",
) -> None:
    """Run the numerical quantifier experiment."""
    print(f"\n[{EXPERIMENT_NAME}] Running method={method}, version={version}")
    run_experiment(
        client=client,
        experiment_name=EXPERIMENT_NAME,
        conditions=NUMERICAL_CONDITIONS,
        method=method,
        version=version,
        base_url=base_url,
    )
