"""Parse first-token logprobs from rsa_responses.jsonl into quantifier scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pipeline_v1.rsa.utility import QUANTS


def _norm_tok(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def _match_quant(t: str) -> str | None:
    n = _norm_tok(t).strip("*\"'")
    if n in QUANTS:
        return n
    for q in QUANTS:
        if n.startswith(q):
            return q
    return None


def extract_first_position_logprobs(logprobs: Any) -> dict[str, float]:
    """Map token string -> logprob for the first generated token (best-effort)."""
    if logprobs is None:
        return {}
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
    else:
        return {}

    if not content or not isinstance(content, list):
        return {}

    first = content[0]
    if not isinstance(first, dict):
        return {}

    out: dict[str, float] = {}
    # Chosen token
    tok = first.get("token")
    lp = first.get("logprob")
    if isinstance(tok, str) and isinstance(lp, (int, float)):
        out[_norm_tok(tok)] = float(lp)

    tops = first.get("top_logprobs")
    if isinstance(tops, list):
        for item in tops:
            if isinstance(item, dict):
                t = item.get("token")
                p = item.get("logprob")
                if isinstance(t, str) and isinstance(p, (int, float)):
                    out.setdefault(_norm_tok(t), float(p))

    return out


def aggregate_quantifier_logprobs(token_lp: dict[str, float]) -> dict[str, float]:
    """Merge token strings onto none/some/all via logsumexp of matching keys."""
    buckets: dict[str, list[float]] = {q: [] for q in QUANTS}
    for raw_t, lp in token_lp.items():
        q = _match_quant(raw_t)
        if q:
            buckets[q].append(lp)

    result: dict[str, float] = {}
    for q in QUANTS:
        vals = buckets[q]
        if not vals:
            result[q] = float("-inf")
        elif len(vals) == 1:
            result[q] = vals[0]
        else:
            m = max(vals)
            result[q] = m + math.log(sum(math.exp(v - m) for v in vals))
    return result


def softmax_from_logprobs(lp: dict[str, float]) -> dict[str, float]:
    qs = list(QUANTS)
    lvs = [lp.get(q, float("-inf")) for q in qs]
    finite = [x for x in lvs if math.isfinite(x)]
    if not finite:
        u = 1.0 / len(qs)
        return {q: u for q in qs}
    m = max(finite)
    exps = [math.exp(x - m) if math.isfinite(x) else 0.0 for x in lvs]
    s = sum(exps) or 1e-30
    return {q: exps[i] / s for i, q in enumerate(qs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing rsa_responses.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <run-dir>/parsed_quantifiers.jsonl",
    )
    args = parser.parse_args()
    inp = args.run_dir / "rsa_responses.jsonl"
    out = args.output or (args.run_dir / "parsed_quantifiers.jsonl")

    rows_out: list[dict[str, Any]] = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lp_raw = (rec.get("result") or {}).get("logprobs")
            tok_lp = extract_first_position_logprobs(lp_raw)
            q_lp = aggregate_quantifier_logprobs(tok_lp)
            p_hat = softmax_from_logprobs(q_lp)
            rows_out.append(
                {
                    "rsa_id": rec.get("rsa_id"),
                    "probe_type": rec.get("probe_type"),
                    "story_index": rec.get("story_index"),
                    "model_id": rec.get("model_id"),
                    "stimulus": rec.get("stimulus"),
                    "token_logprobs": tok_lp,
                    "quantifier_logprobs": q_lp,
                    "quantifier_probs": p_hat,
                    "parse_ok": bool(tok_lp),
                }
            )

    with out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows_out)} rows to {out}")


if __name__ == "__main__":
    main()
