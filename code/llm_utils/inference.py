"""Inference client(routing wrapper) and experiment runner.

Provides:
- InferenceClient: thin wrapper around the OpenAI SDK with 4 execution modes for now
  (structured output, natural language, FTP log probs, FTP prefilling attack)
- dispatch_inference: routes a call to the correct client method
- run_experiment: generic loop shared by both experiment types
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from openai import OpenAI


class InferenceClient:
    """Thin wrapper around the OpenAI SDK for experiment inference.

    Should work with  any base_url that speaks the OpenAI protocol.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        default_max_tokens: int = 256,
        use_max_completion_tokens: bool = False,
        supports_logprobs: bool = True,
        max_top_logprobs: int = 10,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.default_max_tokens = default_max_tokens
        self.use_max_completion_tokens = use_max_completion_tokens
        self.supports_logprobs = supports_logprobs
        self.max_top_logprobs = max_top_logprobs

    def _tokens_kwarg(self, n: int) -> dict:
        """Return the correct max-tokens kwarg for this model."""
        key = "max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"
        return {key: n}

    def structured_output(
        self,
        messages: list[dict],
        validate_fn: Callable[[dict], tuple[bool, str]],
        max_retries: int = 3,
        max_tokens: int = 128,
    ) -> dict[str, Any]:
        """Request JSON output, validate with *validate_fn*, retry on failure.

        Returns dict with keys: output, parsed, valid, attempts.
        """
        attempts: list[str] = []
        current_messages = list(messages)
        raw_output = ""

        for _ in range(max_retries):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                temperature=self.temperature,
                **self._tokens_kwarg(max_tokens),
                response_format={"type": "json_object"},
            )

            raw_output = response.choices[0].message.content or ""
            attempts.append(raw_output)

            try:
                parsed = json.loads(raw_output)
                parsed = {str(k): v for k, v in parsed.items()}
                valid, error = validate_fn(parsed)
                if valid:
                    return {
                        "output": raw_output,
                        "parsed": parsed,
                        "valid": True,
                        "attempts": attempts,
                    }
                # Retry with error feedback
                current_messages = list(messages) + [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            f"Your response was invalid: {error}. "
                            "Please try again with the correct format."
                        ),
                    },
                ]
            except json.JSONDecodeError:
                current_messages = list(messages) + [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. "
                            "Please respond with ONLY a valid JSON object."
                        ),
                    },
                ]

        return {
            "output": raw_output,
            "parsed": None,
            "valid": False,
            "attempts": attempts,
        }

    def natural_language(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Request free-form natural language output."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self._tokens_kwarg(max_tokens or self.default_max_tokens),
        )
        raw_output = response.choices[0].message.content or ""
        return {"output": raw_output}


    def ftp_logprobs(
        self,
        messages: list[dict],
        max_tokens: int = 1,
        top_logprobs: int = 10,
    ) -> dict[str, Any]:
        """Request single-token completion with top log probs.

        Returns output text + dict of {token: logprob} for the first token.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self._tokens_kwarg(max_tokens),
            logprobs=True,
            top_logprobs=min(top_logprobs, self.max_top_logprobs),
        )
        choice = response.choices[0]
        raw_output = choice.message.content or ""

        logprobs_data: dict[str, float] = {}
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                if token_info.top_logprobs:
                    logprobs_data = {
                        lp.token: round(lp.logprob, 6)
                        for lp in token_info.top_logprobs
                    }

        return {"output": raw_output, "log_probs": logprobs_data}


    def ftp_prefilling(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 10,
    ) -> dict[str, Any]:
        """Prefilling attack via the /v1/completions endpoint.

        Sends a raw prompt (with manually constructed chat template including
        a partial assistant turn) and captures first-token log probs.

        Only works with vLLM or other servers that expose /v1/completions.
        """
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
            logprobs=top_logprobs,
        )
        choice = response.choices[0]
        raw_output = choice.text or ""

        logprobs_data: dict[str, float] = {}
        if choice.logprobs and choice.logprobs.top_logprobs:
            for token_probs in choice.logprobs.top_logprobs:
                if token_probs:
                    logprobs_data = {
                        token: round(prob, 6)
                        for token, prob in token_probs.items()
                    }

        return {"output": raw_output, "log_probs": logprobs_data}


# pack all methods and validate fn depending on the method
def dispatch_inference(
    client: InferenceClient,
    method: str,
    prompt_data: dict,
    question_key: str,
) -> dict[str, Any]:
    """Route a single inference call to the appropriate client method."""
    from llm_utils.prompts import validate_count_bets, validate_knowledge_bets

    if method == "structured_output":
        validate_fn = (
            validate_knowledge_bets
            if question_key == "knowledge"
            else validate_count_bets
        )
        return client.structured_output(
            messages=prompt_data["messages"],
            validate_fn=validate_fn,
        )
    elif method == "natural_language":
        return client.natural_language(messages=prompt_data["messages"])
    elif method == "ftp_logprobs_single":
        return client.ftp_logprobs(messages=prompt_data["messages"])
    elif method == "ftp_logprobs_prefilling":
        return client.ftp_prefilling(prompt=prompt_data["prompt"])
    else:
        raise ValueError(f"Unknown method: {method}")


# experiment loop
def run_experiment(
    client: InferenceClient,
    experiment_name: str,
    conditions: list[dict],
    method: str,
    version: str = "prompting-v1",
    base_url: str = "",
) -> None:
    """Run a full experiment: (all stories + conditions) * question types.

    Shared by exps_some and exps_numerical (they differ only in the
    conditions list they pass in)
    Supports resumption: skips (story, condition, question) triples that
    already exist in the JSONL outputs
    """
    from llm_utils.prompts import STORIES, QUESTION_KEYS, PROMPT_BUILDERS
    from llm_utils.storage import (
        get_output_dir,
        write_meta,
        append_jsonl,
        load_existing_keys,
        make_record_key,
    )

    builder = PROMPT_BUILDERS[version][method]
    output_dir = get_output_dir(experiment_name, client.model, method, version)

    write_meta(
        output_dir,
        model=client.model,
        method=method,
        version=version,
        experiment=experiment_name,
        temperature=client.temperature,
        max_tokens=client.default_max_tokens,
        base_url=base_url,
    )

    for story in STORIES:
        existing = load_existing_keys(output_dir, story["index"])

        for condition in conditions:
            for q_key, q_label in QUESTION_KEYS.items():
                record_key = make_record_key(
                    condition["access"], condition["observe"], q_label
                )
                if record_key in existing:
                    continue

                # build prompt (prefilling needs model_name for chat template)
                if method == "ftp_logprobs_prefilling":
                    prompt_data = builder(
                        story, condition, q_key, client.model
                    )
                else:
                    prompt_data = builder(story, condition, q_key)

                # execute inference
                result = dispatch_inference(client, method, prompt_data, q_key)

                # build JSONL record
                record: dict[str, Any] = {
                    "story": story["shortname"],
                    "access": condition["access"],
                    "observe": condition["observe"],
                    "key": q_label,
                    "input": prompt_data["input_text"],
                    "output": result["output"],
                }
                if "log_probs" in result:
                    record["log_probs"] = result["log_probs"]
                if "attempts" in result:
                    record["attempts"] = result["attempts"]
                    record["valid"] = result.get("valid", True)

                append_jsonl(output_dir, story["index"], record)

        print(
            f"  [{method}] Story {story['index']} "
            f"({story['shortname']}) done."
        )
