"""Build OpenAI chat messages for RSA probes from contract + stimulus row."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_utils.prompts import STORIES


def get_story(story_index: int) -> dict[str, Any]:
    for s in STORIES:
        if int(s["index"]) == int(story_index):
            return s
    raise KeyError(f"Unknown story_index={story_index!r}")


def load_contract(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def build_messages(row: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, str]]:
    probe = str(row["probe_type"])
    story = get_story(int(row["story_index"]))

    if probe == "speaker":
        sp = contract.get("speaker_probe") or {}
        tmpl = str(sp.get("user_template", ""))
        user = tmpl.format(
            setup=story["setup"],
            state_s=int(row["state_s"]),
            items=story["items"],
            property=story["property"],
            k=int(row["k"]),
        )
        system = str(sp.get("system", "")).strip()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if probe == "prior":
        pr = contract.get("prior_probe") or {}
        tmpl = str(pr.get("user_template", ""))
        user = tmpl.format(
            setup=story["setup"],
            prior_q=story["prior_q"],
            speaker=story["speaker"],
            items=story["items"],
            property=story["property"],
        )
        system = str(pr.get("system", "")).strip()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    raise ValueError(f"Unknown probe_type={probe!r}")
