"""Extracted stories and filler vals + conditions, prompt templates, 
and output schemas etc.

Prompt builder functions for different inference methods, JSON schemas + validators for
structured output.
"""

from __future__ import annotations

import json
from typing import Any


TOTAL_ITEMS = 3

QUESTION_KEYS = {
    "prior": "prior belief",
    "posterior": "posterior belief",
    "knowledge": "speaker knowledge estimate",
}

METHODS = [
    "structured_output",
    "natural_language",
    "ftp_logprobs_single",
    "ftp_logprobs_prefilling",
]

# Betting instructions from html/js
BETTING_PREAMBLE = (
    'In the questions that follow you will be asked to answer questions by '
    '"betting" on the possible answers. You should imagine that you have $100 '
    'and divide your money between the answers -- the amount of money you bet '
    'on each option should correspond to how confident you are that it is '
    'correct. Your bets over all the options must sum to 100'
)

BETTING_REMINDER = (
    '(Please answer by betting on the options -- your bets should correspond '
    'to your confidence that each option is correct.)'
)

# Stories (from the original web js+html exps)
STORIES: list[dict[str, Any]] = [
    {
        "index": 1,
        "shortname": "seeds",
        "setup": (
            "Corendula seeds almost always sprout within a day when put into water. "
            "Two days ago, botanist Jim put 3 Corendula seeds into water."
        ),
        "speaker": "Jim",
        "items": "seeds",
        "property": "sprouted",
        "prior_q": "How many of the 3 seeds do you think have sprouted?",
        "posterior_q": "Now how many of the 3 seeds do you think have sprouted?",
        "knowledge_q": "Do you think Jim knows exactly how many of the 3 seeds have sprouted?",
    },
    {
        "index": 2,
        "shortname": "tickets",
        "setup": (
            "Tickets in the DoubleDay instant lottery almost always win. "
            "Joe bought 3 DoubleDay tickets yesterday."
        ),
        "speaker": "Joe",
        "items": "tickets",
        "property": "won",
        "prior_q": "How many of the 3 tickets do you think have won?",
        "posterior_q": "Now how many of the 3 tickets do you think have won?",
        "knowledge_q": "Do you think Joe knows exactly how many of the 3 tickets have won?",
    },
    {
        "index": 3,
        "shortname": "exams",
        "setup": (
            "Students in the introductory bio class almost always have passing grade "
            "on the exam. Mark's 3 intro bio students took an exam yesterday."
        ),
        "speaker": "Mark",
        "items": "exams",
        "property": "passing grade",
        "prior_q": "How many of the 3 exams do you think have passing grade?",
        "posterior_q": "Now how many of the 3 exams do you think have passing grade?",
        "knowledge_q": "Do you think Mark knows exactly how many of the 3 exams have passing grade?",
    },
    {
        "index": 4,
        "shortname": "fruits",
        "setup": (
            "Mongines are small fruits that almost always have dried out pith inside. "
            "Monica bought 3 mondrine fruits yesterday."
        ),
        "speaker": "Monica",
        "items": "fruits",
        "property": "dried out pith",
        "prior_q": "How many of the 3 fruits do you think have dried out pith?",
        "posterior_q": "Now how many of the 3 fruits do you think have dried out pith?",
        "knowledge_q": "Do you think Monica knows exactly how many of the 3 fruits have dried out pith?",
    },
    {
        "index": 5,
        "shortname": "phones",
        "setup": (
            "Broken Sigo phones almost always have burned out transistors. "
            "Ben must repair 3 broken Sigo phones."
        ),
        "speaker": "Ben",
        "items": "phones",
        "property": "burned out transistors",
        "prior_q": "How many of the 3 phones do you think have burned out transistors?",
        "posterior_q": "Now how many of the 3 phones do you think have burned out transistors?",
        "knowledge_q": "Do you think Ben knows exactly how many of the 3 phones have burned out transistors?",
    },
    {
        "index": 6,
        "shortname": "letters",
        "setup": (
            "Letters to Laura's company almost always have checks inside. "
            "Today Laura received 3 letters."
        ),
        "speaker": "Laura",
        "items": "letters",
        "property": "checks inside",
        "prior_q": "How many of the 3 letters do you think have checks inside?",
        "posterior_q": "Now how many of the 3 letters do you think have checks inside?",
        "knowledge_q": "Do you think Laura knows exactly how many of the 3 letters have checks inside?",
    },
]

SOME_CONDITIONS: list[dict[str, Any]] = [
    {"access": 1, "observe": "Some"},
    {"access": 2, "observe": "Some"},
    {"access": 3, "observe": "Some"},
]

NUMERICAL_CONDITIONS: list[dict[str, Any]] = [
    {"access": 1, "observe": 1},
    {"access": 2, "observe": 1},
    {"access": 2, "observe": 2},
    {"access": 3, "observe": 1},
    {"access": 3, "observe": 2},
    {"access": 3, "observe": 3},
]

# Utterance template for the speakers statement (ie "speach" in the original js, jajaja, funny typo)
UTTERANCE_TEMPLATE = (
    '{speaker} tells you on the phone: "I have looked at {access} of the 3 '
    '{items}. {quantifier} of the {items} {havehas} {property}."'
)

def render_utterance(story: dict, condition: dict) -> str:
    observe = condition["observe"]
    havehas = "has" if observe == 1 else "have"
    return UTTERANCE_TEMPLATE.format(
        speaker=story["speaker"],
        access=condition["access"],
        items=story["items"],
        quantifier=str(observe),
        havehas=havehas,
        property=story["property"],
    )


# JSON schemas for structured output
COUNT_BET_SCHEMA = {
    "type": "object",
    "properties": {
        "0": {"type": "integer"},
        "1": {"type": "integer"},
        "2": {"type": "integer"},
        "3": {"type": "integer"},
    },
    "required": ["0", "1", "2", "3"],
    "additionalProperties": False,
}

KNOWLEDGE_BET_SCHEMA = {
    "type": "object",
    "properties": {
        "yes": {"type": "integer"},
        "no": {"type": "integer"},
    },
    "required": ["yes", "no"],
    "additionalProperties": False,
}


def validate_count_bets(data: dict) -> tuple[bool, str]:
    """Validate count bet distribution sums to 100."""
    expected = {"0", "1", "2", "3"}
    keys = set(str(k) for k in data.keys())
    if keys != expected:
        return False, f"Expected keys {expected}, got {keys}"
    values = [data[str(k)] for k in range(4)]
    if not all(isinstance(v, (int, float)) for v in values):
        return False, "All values must be numbers"
    total = sum(values)
    if total != 100:
        return False, f"Bets must sum to 100, got {total}"
    return True, ""


def validate_knowledge_bets(data: dict) -> tuple[bool, str]:
    """Validate knowledge bet distribution sums to 100."""
    normalized = {str(k).lower(): v for k, v in data.items()}
    if set(normalized.keys()) != {"yes", "no"}:
        return False, f"Expected keys yes/no, got {set(data.keys())}"
    if not all(isinstance(v, (int, float)) for v in normalized.values()):
        return False, "All values must be numbers"
    total = sum(normalized.values())
    if total != 100:
        return False, f"Bets must sum to 100, got {total}"
    return True, ""


# Prompt builders
#
# Each builder returns a dict with:
#   "messages"?: list[dict] - for chat completions
#   "prompt"?: str - for raw completions endpoint (prefilling atack trick only)
#   "input_text": str - spec symbols escaped llm input for the JSONL "input" field logging

def _serialize_input(
    *, messages: list[dict] | None = None, prompt: str | None = None
) -> str:
    if prompt is not None:
        return prompt
    return json.dumps(messages, ensure_ascii=False)

# Method: Structured Output 

_STRUCTURED_SYSTEM = (
    "You are participating in a probability judgment study. "
    "Respond ONLY with a valid JSON object, no other text."
)

_STRUCTURED_COUNT_INST = (
    "Distribute exactly $100 across the possible outcomes (0, 1, 2, or 3). "
    "Your bets must sum to exactly 100.\n"
    'Respond with a JSON object: {"0": <bet>, "1": <bet>, "2": <bet>, "3": <bet>}'
)

_STRUCTURED_KNOWLEDGE_INST = (
    "Distribute exactly $100 between yes and no. "
    "Your bets must sum to exactly 100.\n"
    'Respond with a JSON object: {"yes": <bet>, "no": <bet>}'
)


def build_structured_prompt(
    story: dict, condition: dict, question_key: str
) -> dict[str, Any]:
    """Build messages for structured output method."""
    if question_key == "prior":
        context = story["setup"]
        question = story["prior_q"]
        instruction = _STRUCTURED_COUNT_INST
    elif question_key == "posterior":
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["posterior_q"]
        instruction = _STRUCTURED_COUNT_INST
    else:  # knowledge
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["knowledge_q"]
        instruction = _STRUCTURED_KNOWLEDGE_INST

    user_content = (
        f"{BETTING_PREAMBLE}\n\n"
        f"{context}\n\n"
        f"{question}\n\n"
        f"{BETTING_REMINDER}\n\n"
        f"{instruction}"
    )
    messages = [
        {"role": "system", "content": _STRUCTURED_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return {"messages": messages, "input_text": _serialize_input(messages=messages)}


# Method: Natural Language

_NATURAL_SYSTEM = (
    "You are participating in a probability judgment study. "
    "Answer the questions based on the given scenario."
)


def build_natural_prompt(
    story: dict, condition: dict, question_key: str
) -> dict[str, Any]:
    """Build messages for natural language method."""
    if question_key == "prior":
        context = story["setup"]
        question = story["prior_q"]
    elif question_key == "posterior":
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["posterior_q"]
    else:
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["knowledge_q"]

    user_content = (
        f"{BETTING_PREAMBLE}\n\n"
        f"{context}\n\n"
        f"{question}\n\n"
        f"{BETTING_REMINDER}"
    )
    messages = [
        {"role": "system", "content": _NATURAL_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return {"messages": messages, "input_text": _serialize_input(messages=messages)}


# Method: FTP Log Probs - Single Output


_FTP_COUNT_INST = (
    "Answer with a single character -- just the number (0, 1, 2, or 3) "
    "and nothing else."
)

_FTP_KNOWLEDGE_INST = (
    "Answer with a single character -- Y if you agree or N if you disagree, "
    "and nothing else."
)


def build_ftp_single_prompt(
    story: dict, condition: dict, question_key: str
) -> dict[str, Any]:
    """Build messages for FTP single-token output method."""
    if question_key == "prior":
        context = story["setup"]
        question = story["prior_q"]
        instruction = _FTP_COUNT_INST
    elif question_key == "posterior":
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["posterior_q"]
        instruction = _FTP_COUNT_INST
    else:
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["knowledge_q"]
        instruction = _FTP_KNOWLEDGE_INST

    user_content = f"{context}\n\n{question}\n\n{instruction}"
    messages = [{"role": "user", "content": user_content}]
    return {"messages": messages, "input_text": _serialize_input(messages=messages)}


# Method: FTP Log Probs - Prefilling attack trick

CHAT_TEMPLATES = {
    "llama": {
        "bos": "<|begin_of_text|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen": {
        "bos": "",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
    },
    "phi": {
        "bos": "",
        "user_start": "<|user|>\n",
        "user_end": "<|end|>\n",
        "assistant_start": "<|assistant|>\n",
    },
    "gemma": {
        "bos": "<bos>",
        "user_start": "<start_of_turn>user\n",
        "user_end": "<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n",
    },
}


def detect_model_family(model_name: str) -> str:
    """Detect model family from model name string."""
    name_lower = model_name.lower()
    for family in ("llama", "qwen", "phi", "gemma"):
        if family in name_lower:
            return family
    return "llama"  # default fallback


def build_prefilling_prompt(
    story: dict,
    condition: dict,
    question_key: str,
    model_name: str,
) -> dict[str, Any]:
    """Build raw prompt for FTP prefilling attack via completions endpoint.

    Constructs the full token sequence including a partial assistant turn,
    using the appropriate chat template for the model family.
    """
    if question_key == "prior":
        context = story["setup"]
        question = story["prior_q"]
        instruction = _FTP_COUNT_INST
        options = "0, 1, 2, or 3"
    elif question_key == "posterior":
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["posterior_q"]
        instruction = _FTP_COUNT_INST
        options = "0, 1, 2, or 3"
    else:
        context = story["setup"] + "\n\n" + render_utterance(story, condition)
        question = story["knowledge_q"]
        instruction = _FTP_KNOWLEDGE_INST
        options = "Y or N"

    user_content = f"{context}\n\n{question}\n\n{instruction}"
    assistant_prefix = (
        f"Given the question and the possible answers ({options}), my answer is:"
    )

    family = detect_model_family(model_name)
    tpl = CHAT_TEMPLATES[family]

    raw_prompt = (
        f"{tpl['bos']}"
        f"{tpl['user_start']}{user_content}{tpl['user_end']}"
        f"{tpl['assistant_start']}{assistant_prefix}"
    )

    return {"prompt": raw_prompt, "input_text": _serialize_input(prompt=raw_prompt)}



PROMPT_BUILDERS: dict[str, dict[str, Any]] = {
    "prompting-v1": {
        "structured_output": build_structured_prompt,
        "natural_language": build_natural_prompt,
        "ftp_logprobs_single": build_ftp_single_prompt,
        "ftp_logprobs_prefilling": build_prefilling_prompt,
    },
}
