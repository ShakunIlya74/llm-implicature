"""RSA probing over llm-inference outputs.

This script unifies llm-inference JSONL records (structured_output + ftp_logprobs_*),
fits alpha on bets (paper-style), and writes:
  - rsa_alpha_fits.jsonl
  - rsa_predictions.jsonl
  - rsa_vs_behavior.jsonl

Usage:
  python code/pipeline_v1/scripts/rsa_probe.py --data-root code/pipeline_v1/results
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


COUNT_KEYS = ("0", "1", "2", "3")
KNOW_KEYS = ("yes", "no")
EPS = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_utterance(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "some": "some",
        "none": "none",
        "all": "all",
        "one": "1",
        "two": "2",
        "three": "3",
    }
    if text in mapping:
        return mapping[text]
    if text in {"0", "1", "2", "3"}:
        return text
    return text


def normalize_question_key(value: str) -> str:
    text = str(value).strip().lower()
    if "prior" in text:
        return "prior"
    if "posterior" in text:
        return "posterior"
    return "knowledge"


def normalize_distribution(
    raw: dict[str, Any], expected_keys: tuple[str, ...]
) -> dict[str, float] | None:
    vals: dict[str, float] = {}
    for k in expected_keys:
        if k not in raw:
            return None
        v = raw[k]
        if not isinstance(v, (int, float)):
            return None
        vals[k] = float(v)
    s = sum(vals.values())
    if s <= 0:
        return None
    return {k: vals[k] / s for k in expected_keys}


def parse_attempts_distribution(
    row: dict[str, Any], qkey: str
) -> dict[str, float] | None:
    expected = KNOW_KEYS if qkey == "knowledge" else COUNT_KEYS
    attempts = row.get("attempts")
    if isinstance(attempts, list):
        for candidate in reversed(attempts):
            if not isinstance(candidate, str):
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            lowered = {str(k).lower(): v for k, v in parsed.items()}
            dist = normalize_distribution(lowered, expected)
            if dist is not None:
                return dist

    output = row.get("output")
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            lowered = {str(k).lower(): v for k, v in parsed.items()}
            return normalize_distribution(lowered, expected)
    return None


def softmax_logprobs(logps: dict[str, float]) -> dict[str, float]:
    mx = max(logps.values())
    exps = {k: math.exp(v - mx) for k, v in logps.items()}
    s = sum(exps.values())
    return {k: exps[k] / s for k in logps}


def parse_logprob_distribution(
    row: dict[str, Any], qkey: str
) -> dict[str, float] | None:
    raw = row.get("log_probs")
    if not isinstance(raw, dict) or not raw:
        return None

    canon: dict[str, float] = {}
    for token, lp in raw.items():
        if not isinstance(lp, (int, float)):
            continue
        t = str(token).strip().lower()
        if qkey == "knowledge":
            if t in {"y", "yes"}:
                canon["yes"] = float(lp)
            elif t in {"n", "no"}:
                canon["no"] = float(lp)
        else:
            if t in {"0", "1", "2", "3"}:
                canon[t] = float(lp)

    if qkey == "knowledge":
        if set(canon) != set(KNOW_KEYS):
            return None
        return softmax_logprobs(canon)
    if not canon:
        return None
    # Allow partial state support in logprob top-k; normalize over seen states.
    probs = softmax_logprobs(canon)
    full = {k: 0.0 for k in COUNT_KEYS}
    for k, v in probs.items():
        full[k] = v
    s = sum(full.values())
    if s <= 0:
        return None
    return {k: full[k] / s for k in COUNT_KEYS}


def iter_unified_rows(data_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for file in data_root.glob("**/inference_story_*.jsonl"):
        rel = file.relative_to(data_root)
        # exps_x/model/method/version/inference_story_*.jsonl
        if len(rel.parts) < 5:
            continue
        experiment, model, method, version = rel.parts[:4]
        if method == "natural_language":
            continue
        rows = read_jsonl(file)
        for row in rows:
            qkey = normalize_question_key(str(row.get("key", "")))
            unified = {
                "experiment": experiment,
                "model_id": model,
                "method": method,
                "prompt_version": version,
                "story": row.get("story"),
                "access": int(row.get("access")),
                "utterance": normalize_utterance(row.get("observe")),
                "question_key": qkey,
            }
            if method == "structured_output":
                unified["source"] = "bets"
                unified["distribution"] = parse_attempts_distribution(row, qkey)
                unified["valid"] = bool(row.get("valid", False))
            else:
                unified["source"] = "logprobs"
                unified["distribution"] = parse_logprob_distribution(row, qkey)
                unified["valid"] = unified["distribution"] is not None
            out.append(unified)
    return out


def aggregate_distributions(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int, str, str], dict[str, float]]:
    buckets: dict[tuple[str, str, int, str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        dist = row.get("distribution")
        if not isinstance(dist, dict):
            continue
        key = (
            str(row["model_id"]),
            str(row["experiment"]),
            int(row["access"]),
            str(row["utterance"]),
            str(row["question_key"]),
        )
        buckets[key].append({str(k): float(v) for k, v in dist.items()})

    out: dict[tuple[str, str, int, str, str], dict[str, float]] = {}
    for key, vals in buckets.items():
        keys = tuple(vals[0].keys())
        out[key] = {k: sum(v[k] for v in vals) / len(vals) for k in keys}
    return out


def literal_prob(utterance: str, state: int, access: int) -> float:
    # Hypergeometric chance of seeing x positives when sampling access from 3.
    total = math.comb(3, access)
    prob = 0.0
    min_x = max(0, access - (3 - state))
    max_x = min(access, state)
    for x in range(min_x, max_x + 1):
        p_x = (math.comb(state, x) * math.comb(3 - state, access - x)) / total
        if utterance == "none" and x == 0:
            prob += p_x
        elif utterance == "some" and x > 0:
            prob += p_x
        elif utterance == "all" and x == access:
            prob += p_x
        elif utterance in {"0", "1", "2", "3"} and x == int(utterance):
            prob += p_x
    return prob


def alternatives_for_utterance(utterance: str) -> list[str]:
    if utterance in {"none", "some", "all"}:
        return ["none", "some", "all"]
    if utterance in {"1", "2", "3"}:
        return ["1", "2", "3"]
    return [utterance]


def rsa_posterior(
    prior: dict[str, float], utterance: str, access: int, alpha: float
) -> dict[str, float]:
    alts = alternatives_for_utterance(utterance)
    numer: dict[str, float] = {}
    for s in COUNT_KEYS:
        si = int(s)
        speaker_scores: dict[str, float] = {}
        for alt in alts:
            lit = max(literal_prob(alt, si, access), EPS)
            speaker_scores[alt] = math.exp(alpha * math.log(lit))
        z = sum(speaker_scores.values())
        p_u = speaker_scores[utterance] / z if z > 0 else 0.0
        numer[s] = max(float(prior.get(s, 0.0)), 0.0) * p_u
    z2 = sum(numer.values())
    if z2 <= 0:
        return {k: 0.25 for k in COUNT_KEYS}
    return {k: numer[k] / z2 for k in COUNT_KEYS}


def kl_div(p: dict[str, float], q: dict[str, float]) -> float:
    val = 0.0
    for k in p:
        pk = max(float(p[k]), EPS)
        qk = max(float(q.get(k, 0.0)), EPS)
        val += pk * math.log(pk / qk)
    return val


def mae(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(abs(float(p[k]) - float(q.get(k, 0.0))) for k in p) / len(p)


def pick_prior(
    priors: dict[tuple[str, str, int, str, str], dict[str, float]],
    model_id: str,
    experiment: str,
    access: int,
    utterance: str,
) -> dict[str, float] | None:
    k1 = (model_id, experiment, access, utterance, "prior")
    if k1 in priors:
        return priors[k1]
    cand = [v for (m, e, a, _u, q), v in priors.items() if m == model_id and e == experiment and a == access and q == "prior"]
    if cand:
        return {s: sum(x[s] for x in cand) / len(cand) for s in COUNT_KEYS}
    cand2 = [v for (m, e, _a, _u, q), v in priors.items() if m == model_id and e == experiment and q == "prior"]
    if cand2:
        return {s: sum(x[s] for x in cand2) / len(cand2) for s in COUNT_KEYS}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RSA probing over llm-inference outputs")
    parser.add_argument("--data-root", default="code/pipeline_v1/results")
    parser.add_argument(
        "--output-dir",
        default="code/pipeline_v1/results/rsa_probe",
        help="Directory where rsa_*.jsonl files are written",
    )
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data root not found: {data_root}")

    rows = iter_unified_rows(data_root)
    if not rows:
        raise SystemExit("No usable rows found under data root.")

    bets_rows = [r for r in rows if r["source"] == "bets" and r.get("valid")]
    log_rows = [r for r in rows if r["source"] == "logprobs" and r.get("valid")]

    bets_agg = aggregate_distributions(bets_rows)
    log_agg = aggregate_distributions(log_rows)

    alpha_fits: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    alpha_values: list[float] = []
    cur = args.alpha_min
    while cur <= args.alpha_max + 1e-9:
        alpha_values.append(round(cur, 10))
        cur += args.alpha_step

    for (model_id, experiment, access, utterance, qkey), obs in sorted(bets_agg.items()):
        if qkey != "posterior":
            continue
        if utterance not in {"none", "some", "all", "1", "2", "3"}:
            continue
        prior = pick_prior(bets_agg, model_id, experiment, access, utterance)
        if prior is None:
            alpha_fits.append(
                {
                    "model_id": model_id,
                    "experiment": experiment,
                    "access": access,
                    "utterance": utterance,
                    "fit_status": "missing_prior",
                }
            )
            continue

        best_alpha = None
        best_kl = None
        best_pred = None
        for alpha in alpha_values:
            pred = rsa_posterior(prior, utterance, access, alpha)
            loss = kl_div(obs, pred)
            if best_kl is None or loss < best_kl:
                best_kl = loss
                best_alpha = alpha
                best_pred = pred

        assert best_alpha is not None and best_pred is not None and best_kl is not None
        fit_row = {
            "model_id": model_id,
            "experiment": experiment,
            "access": access,
            "utterance": utterance,
            "fit_status": "ok",
            "alpha_hat": best_alpha,
            "objective": "kl_obs_to_rsa",
            "objective_value": best_kl,
            "num_alpha_grid_points": len(alpha_values),
        }
        alpha_fits.append(fit_row)

        pred_row = {
            "model_id": model_id,
            "experiment": experiment,
            "access": access,
            "utterance": utterance,
            "alpha_hat": best_alpha,
            "prior_distribution": prior,
            "observed_posterior_bets": obs,
            "rsa_predicted_posterior": best_pred,
        }
        preds.append(pred_row)

        cmp_row: dict[str, Any] = {
            "model_id": model_id,
            "experiment": experiment,
            "access": access,
            "utterance": utterance,
            "alpha_hat": best_alpha,
            "kl_bets_vs_rsa": kl_div(obs, best_pred),
            "mae_bets_vs_rsa": mae(obs, best_pred),
            "has_logprob_distribution": False,
        }
        lp_key = (model_id, experiment, access, utterance, "posterior")
        if lp_key in log_agg:
            lp_dist = log_agg[lp_key]
            cmp_row["has_logprob_distribution"] = True
            cmp_row["logprob_posterior"] = lp_dist
            cmp_row["kl_bets_vs_logprob"] = kl_div(obs, lp_dist)
            cmp_row["mae_bets_vs_logprob"] = mae(obs, lp_dist)
            cmp_row["kl_rsa_vs_logprob"] = kl_div(best_pred, lp_dist)
            cmp_row["mae_rsa_vs_logprob"] = mae(best_pred, lp_dist)
        comparisons.append(cmp_row)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "rsa_alpha_fits.jsonl", alpha_fits)
    write_jsonl(out_dir / "rsa_predictions.jsonl", preds)
    write_jsonl(out_dir / "rsa_vs_behavior.jsonl", comparisons)

    summary = {
        "data_root": str(data_root),
        "output_dir": str(out_dir),
        "num_unified_rows": len(rows),
        "num_bets_rows_valid": len(bets_rows),
        "num_logprob_rows_valid": len(log_rows),
        "num_alpha_fits": len(alpha_fits),
        "num_predictions": len(preds),
        "num_comparisons": len(comparisons),
    }
    (out_dir / "rsa_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
