"""Build Goodman & Stuhlmüller-style stimuli from repo stories (llm_utils.prompts).

No LLM calls: all rows are deterministic cross-products of stories, conditions,
prompt strategies, paraphrase variants, and question types.
"""

from __future__ import annotations

from typing import Any

from llm_utils.prompts import (
    NUMERICAL_CONDITIONS,
    STORIES,
    UTTERANCE_TEMPLATE,
    render_utterance,
)


QUANTIFIER_WORDS = ("none", "some", "all")

# Light surface-form variants for robustness checks (no model).
PARAPHRASE_VARIANTS: list[dict[str, str]] = [
    {"id": "1", "label": "default", "context_prefix": ""},
    {
        "id": "2",
        "label": "study_frame",
        "context_prefix": "You are participating in a short reasoning study.\n\n",
    },
    {
        "id": "3",
        "label": "hypothetical",
        "context_prefix": "This is a hypothetical scenario.\n\n",
    },
    {
        "id": "4",
        "label": "careful",
        "context_prefix": "Read the scenario carefully before answering.\n\n",
    },
    {
        "id": "5",
        "label": "neutral",
        "context_prefix": "Consider only the information given.\n\n",
    },
]


def _quantifier_display(word: str) -> str:
    w = word.lower()
    if w == "none":
        return "None"
    if w == "some":
        return "Some"
    if w == "all":
        return "All"
    return word


def render_quantifier_utterance(story: dict[str, Any], access: int, quant: str) -> str:
    q = _quantifier_display(quant)
    return UTTERANCE_TEMPLATE.format(
        speaker=story["speaker"],
        access=access,
        items=story["items"],
        quantifier=q,
        havehas="have",
        property=story["property"],
    )


def _select_stories(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    indices = cfg["design"].get("story_indices")
    if not indices:
        return list(STORIES)
    wanted = set(int(i) for i in indices)
    return [s for s in STORIES if s["index"] in wanted]


def _paraphrase_list(n: int) -> list[dict[str, str]]:
    if n <= 0:
        return [PARAPHRASE_VARIANTS[0]]
    out: list[dict[str, str]] = []
    for i in range(n):
        out.append(PARAPHRASE_VARIANTS[i % len(PARAPHRASE_VARIANTS)])
    return out


def build_all_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    design = cfg["design"]
    generation = cfg.get("generation", {})
    total_items = int(design["total_items"])
    state_space = list(range(total_items + 1))

    stories = _select_stories(cfg)
    knowledge_levels = [int(k) for k in design["knowledge_levels"]]
    utterances = [str(u).lower() for u in design["utterances"]]
    strategies = list(design["prompt_strategies"])
    question_types = list(design.get("question_types", ["posterior"]))
    n_para = int(design.get("paraphrases_per_condition", 5))
    paraphrases = _paraphrase_list(n_para)

    records: list[dict[str, Any]] = []
    idx = 0

    for story in stories:
        for k in knowledge_levels:
            for utterance in utterances:
                if utterance not in QUANTIFIER_WORDS:
                    continue
                utterance_text = render_quantifier_utterance(story, k, utterance)
                for strategy in strategies:
                    for para_idx, para in enumerate(paraphrases):
                        for qk in question_types:
                            idx += 1
                            records.append(
                                {
                                    "stimulus_id": f"st-{idx:06d}",
                                    "experiment_block": "quantifier",
                                    "story_index": story["index"],
                                    "story_shortname": story["shortname"],
                                    "setup": story["setup"],
                                    "speaker": story["speaker"],
                                    "items": story["items"],
                                    "property": story["property"],
                                    "k": k,
                                    "utterance_type": "quantifier",
                                    "utterance": utterance,
                                    "utterance_text": utterance_text,
                                    "prompt_strategy": strategy,
                                    "paraphrase_id": para_idx + 1,
                                    "paraphrase_label": para["label"],
                                    "context_prefix": para["context_prefix"],
                                    "question_key": qk,
                                    "state_space": state_space,
                                }
                            )

    if generation.get("include_number_words", False):
        number_words_cfg = generation.get("number_words", ["one", "two", "three"])
        for story in stories:
            for cond in NUMERICAL_CONDITIONS:
                access = int(cond["access"])
                observe = cond["observe"]
                utterance_text = render_utterance(story, cond)
                nw = number_words_cfg
                if isinstance(observe, int) and 1 <= observe <= len(nw):
                    nw_label = str(nw[observe - 1])
                else:
                    nw_label = str(observe)
                for strategy in strategies:
                    for para_idx, para in enumerate(paraphrases):
                        for qk in question_types:
                            idx += 1
                            records.append(
                                {
                                    "stimulus_id": f"st-{idx:06d}",
                                    "experiment_block": "number_word",
                                    "story_index": story["index"],
                                    "story_shortname": story["shortname"],
                                    "setup": story["setup"],
                                    "speaker": story["speaker"],
                                    "items": story["items"],
                                    "property": story["property"],
                                    "k": access,
                                    "utterance_type": "number_word",
                                    "utterance": nw_label,
                                    "numeral": observe,
                                    "utterance_text": utterance_text,
                                    "prompt_strategy": strategy,
                                    "paraphrase_id": para_idx + 1,
                                    "paraphrase_label": para["label"],
                                    "context_prefix": para["context_prefix"],
                                    "question_key": qk,
                                    "state_space": state_space,
                                }
                            )

    return records


def summarize_counts(cfg: dict[str, Any], num_records: int) -> dict[str, Any]:
    design = cfg["design"]
    generation = cfg.get("generation", {})
    stories_n = len(_select_stories(cfg))
    k_n = len(design["knowledge_levels"])
    u_n = len([u for u in design["utterances"] if str(u).lower() in QUANTIFIER_WORDS])
    s_n = len(design["prompt_strategies"])
    p_n = int(design.get("paraphrases_per_condition", 5))
    q_n = len(design.get("question_types", ["posterior"]))
    quantifier_rows = stories_n * k_n * u_n * s_n * p_n * q_n
    number_block = 0
    if generation.get("include_number_words"):
        number_block = stories_n * len(NUMERICAL_CONDITIONS) * s_n * p_n * q_n
    samples = int(design.get("samples_per_prompt", 20))
    return {
        "num_stimulus_rows": num_records,
        "expected_quantifier_rows": quantifier_rows,
        "expected_number_word_rows": number_block,
        "formula_quantifier": "stories * k * utterances(none/some/all) * strategies * paraphrases * question_types",
        "formula_number_word": "stories * numerical_conditions * strategies * paraphrases * question_types",
        "samples_per_prompt_config": samples,
        "note_api_calls": "Stimulus generation does not call models. API calls ~ stimulus_rows * samples_per_prompt * num_models (evaluation phase).",
    }
