"""Build OpenAI-style chat messages for G&S-style stimuli (Phase A).

Uses wording from phase_a_prompt_contract_gs2013_wording.yaml and story questions
from llm_utils.prompts.STORIES (matched by story_index).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_utils.prompts import STORIES


def get_story(story_index: int) -> dict[str, Any]:
    for s in STORIES:
        if int(s["index"]) == int(story_index):
            return s
    raise KeyError(f"Unknown story_index={story_index!r} (not in STORIES)")


def build_openai_messages(stimulus: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, str]]:
    """Return [{"role":"system",...},{"role":"user",...}] for one stimulus row."""
    strategy = str(stimulus.get("prompt_strategy", "baseline"))
    strategies = contract.get("strategies") or {}
    if strategy not in strategies:
        raise KeyError(
            f"Unknown prompt_strategy={strategy!r}; expected one of {list(strategies)}"
        )
    strat_cfg = strategies[strategy]

    qk = str(stimulus.get("question_key", "posterior"))
    qmap = contract.get("question_key_to_story_field") or {}
    if qk not in qmap:
        raise KeyError(f"Unknown question_key={qk!r}")
    story_field = qmap[qk]

    story = get_story(int(stimulus["story_index"]))
    if story_field not in story:
        raise KeyError(f"Story {story['index']} has no field {story_field!r}")
    question_text = str(story[story_field])

    rules = contract.get("utterance_block_rules") or {}
    rule = rules.get(qk, "include_utterance_after_setup")
    include_utterance = rule == "include_utterance_after_setup"

    context_prefix = (stimulus.get("context_prefix") or "").strip()
    setup = str(stimulus.get("setup", "")).strip()
    utterance_text = str(stimulus.get("utterance_text", "")).strip()

    context_parts: list[str] = []
    if context_prefix:
        context_parts.append(context_prefix)
    context_parts.append(setup)
    if include_utterance and utterance_text:
        context_parts.append(utterance_text)
    context_block = "\n\n".join(context_parts)

    betting_instructions = str(contract.get("betting_instructions", "")).strip()
    betting_reminder = str(contract.get("betting_reminder", "")).strip()

    user_parts: list[str] = [
        betting_instructions,
        context_block,
        question_text,
        betting_reminder,
    ]

    if strategy == "structured_output":
        if qk == "knowledge":
            instr = strat_cfg.get("knowledge_question_json_instruction", "")
        else:
            instr = strat_cfg.get("count_question_json_instruction", "")
        instr = str(instr).strip()
        if instr:
            user_parts.append(instr)
    elif strategy == "cot":
        suf = str(strat_cfg.get("user_suffix", "")).strip()
        if suf:
            user_parts.append(suf)

    user_content = "\n\n".join(p for p in user_parts if p)
    system_content = str(strat_cfg.get("system", "")).strip()

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def load_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt contract not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML in {path}")
    return data
