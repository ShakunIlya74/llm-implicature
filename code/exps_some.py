""""Some" quantifier experiment runner.

Runs inference across 6 stories (having 3 conditions: access=1,2,3; observe="Some")
* 3 question types (prior belief, posterior belief, speaker knowledge estimate)
using the specified inference method and prompt version.
"""

from __future__ import annotations

from llm_utils.inference import InferenceClient, run_experiment
from llm_utils.prompts import SOME_CONDITIONS

EXPERIMENT_NAME = "exps_some"


def run_some_experiment(
    client: InferenceClient,
    method: str,
    version: str = "prompting-v1",
    base_url: str = "",
) -> None:
    """Run the 'some' quantifier experiment."""
    print(f"\n[{EXPERIMENT_NAME}] Running method={method}, version={version}")
    run_experiment(
        client=client,
        experiment_name=EXPERIMENT_NAME,
        conditions=SOME_CONDITIONS,
        method=method,
        version=version,
        base_url=base_url,
    )
