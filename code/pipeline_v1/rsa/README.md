# RSA Task 2 (mechanism probes)

Separate from Phase A betting prompts (`proposal/proposal.tex`, Task 2).

## Artifacts

| Path | Role |
|------|------|
| `contract_rsa.yaml` | Prior / speaker prompt wording |
| `utility.py` | Literal \(U(u;s)\) for \(\alpha\) fitting |
| `build_rsa_stimuli.py` | Writes `data/processed/rsa_stimuli.jsonl` (78 rows: 6 prior + 72 speaker) |
| `run_rsa_probes.py` | `max_tokens=1`, `logprobs` → `results/rsa_runs/<ts>/rsa_responses.jsonl`, `meta.json` |
| `parse_quantifier_logprobs.py` | `parsed_quantifiers.jsonl` |
| `fit_speaker_alpha.py` | `alpha_fits.jsonl` (speaker, `parse_ok` only) |

**Scripts:** (1) contract YAML (2) build stimuli (3) run probes (4) parse first-token logprobs → none/some/all (5) fit \(\hat\alpha\) on speaker rows via `utility.py` (G&S literal semantics, \(N{=}3\)).

---

## Inference (vLLM / logprob-capable API)

`pip install -r requirements.txt`. Repo root: `PYTHONPATH=code`.

| Env | Use |
|-----|-----|
| `OPENAI_BASE_URL` | Local vLLM: e.g. `http://127.0.0.1:8000/v1` |
| `OPENAI_API_KEY` | Often `EMPTY` / placeholder locally |
| `OPENROUTER_API_KEY` / `LITELLM_API_KEY` | If no `OPENAI_API_KEY` |

Precedence: `OPENAI_BASE_URL` → `OPENROUTER_BASE_URL` → `LITELLM_BASE_URL` → OpenRouter. For vLLM, set `OPENAI_BASE_URL` so calls don’t use an OpenRouter URL from `.env`.

```text
set PYTHONPATH=code
python code/pipeline_v1/rsa/build_rsa_stimuli.py
python code/pipeline_v1/rsa/run_rsa_probes.py --model <served-model-name> --limit 3 --top-logprobs 10
```

Full run: drop `--limit` (78 calls); `--concurrency` default 8. `--model` must match `--served-model-name` (e.g. `Qwen/Qwen2.5-7B-Instruct`), not necessarily the OpenRouter slug.

**Output:** `code/pipeline_v1/results/rsa_runs/<timestamp>/` — commit after a live run.

---

## Parse + fit (after responses exist)

```text
python code/pipeline_v1/rsa/parse_quantifier_logprobs.py --run-dir code/pipeline_v1/results/rsa_runs/<id>
python code/pipeline_v1/rsa/fit_speaker_alpha.py --parsed code/pipeline_v1/results/rsa_runs/<id>/parsed_quantifiers.jsonl
```

Dry-run check: add `--dry-run --limit 5` to `run_rsa_probes.py` first.

## Notes

- Prior probe is coarse (one quantifier); full \(P(s)\) over \(\{0,\ldots,3\}\) may use Phase A `question_key=prior` or an extended design.
- Speaker rows use `state_s`, `k`, `story_index` for joins.
- No `logprobs` in the response → `parse_ok` false → no \(\alpha\) fit; use a backend that returns logprobs.
