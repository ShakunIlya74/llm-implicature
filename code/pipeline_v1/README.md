# pipeline_v1

Minimal runnable scaffold for:
- config fields (`experiment.yaml`, model config YAMLs)
- stimuli generation (`scripts/generate_stimuli.py`)
- unified runner (`run_all.py`) calling **OpenRouter** via the official **`openai`** Python client (no LiteLLM dependency)
- response extraction (`scripts/extract_distributions.py`)
- minimal phase-a analysis (`scripts/analyze_phase_a.py`)

## Phase A (freeze what we measure)

- Full factorial config: `code/pipeline_v1/configs/experiment.phase_a.yaml` → writes `data/processed/stimuli.phase_a.jsonl`
- Smoke config (tiny): `code/pipeline_v1/configs/experiment.phase_a_smoke.yaml` → `stimuli.phase_a_smoke.jsonl`
- Prompt contract (for runner): `code/pipeline_v1/prompts/phase_a_prompt_contract.yaml`
- Human baseline schema: `code/pipeline_v1/data/human_baseline/baseline_points.schema.json`

Generate with a chosen config:

```bash
set PYTHONPATH=code
python code/pipeline_v1/scripts/generate_stimuli.py --config code/pipeline_v1/configs/experiment.phase_a.yaml
```

## Quickstart

1. Fill root `.env` (gitignored):
   - `OPENROUTER_API_KEY=...` (or legacy `LITELLM_API_KEY=...` — same key value)
   - Optional: `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`), or `LITELLM_BASE_URL`
2. Install: `pip install openai pyyaml` (see repo `requirements.txt`).
3. Generate stimuli (no API; needs `PYTHONPATH=code` from repo root):
   - `python code/pipeline_v1/scripts/generate_stimuli.py`
   - Summary only: `python code/pipeline_v1/scripts/generate_stimuli.py --summary-only`
4. Run dry-run (parallelism defaults to `run.concurrency` in YAML, else **15**):
   - `python code/pipeline_v1/run_all.py --dry-run`
   - Override: `python code/pipeline_v1/run_all.py --dry-run --concurrency 15`
5. Filter OpenRouter models by parameter support:
   - `python code/pipeline_v1/scripts/filter_openrouter_models.py --require logprobs top_logprobs seed`
   - open-source only:
     `python code/pipeline_v1/scripts/filter_openrouter_models.py --require logprobs top_logprobs seed --open-source-only`

## Minimal end-to-end workflow

From repo root:

```bash
set PYTHONPATH=code
python code/pipeline_v1/scripts/generate_stimuli.py --config code/pipeline_v1/configs/experiment.phase_a_smoke.yaml
python code/pipeline_v1/run_all.py --experiment-config code/pipeline_v1/configs/experiment.phase_a_smoke.yaml --dry-run
python code/pipeline_v1/scripts/extract_distributions.py --run-dir code/pipeline_v1/results/runs/<timestamp>
python code/pipeline_v1/scripts/analyze_phase_a.py --run-dir code/pipeline_v1/results/runs/<timestamp>
```

Optional human comparison after adding baseline rows:

```bash
python code/pipeline_v1/scripts/analyze_phase_a.py --run-dir code/pipeline_v1/results/runs/<timestamp> --human-baseline code/pipeline_v1/data/human_baseline/human_baseline.jsonl
```

## Current behavior

- `run_all.py` reads (paths are CLI defaults; override with `--experiment-config` / `--models-config`):
  - Experiment YAML (e.g. `code/pipeline_v1/configs/experiment.yaml` or `experiment.phase_a.yaml`)
  - Models YAML (default: `code/pipeline_v1/configs/models_openrouter_logprobs_initial.yaml`)
  - `.env` (`OPENROUTER_API_KEY` or `LITELLM_API_KEY`)
- Writes run artifacts under:
  - `code/pipeline_v1/results/runs/<timestamp>/` (`responses.jsonl`, `meta.json`)
- **Concurrency:** `run.concurrency` in the experiment YAML (default **15**) or `--concurrency N`. Uses `ThreadPoolExecutor`; results stay in deterministic order (models × stimuli).

## Design status (important)

- **Stimuli** follow the G\&S-style factorial (stories, \(k\), quantifiers, paraphrases, prior/posterior/knowledge) — see `stimuli/gs2013_stimuli.py` and `configs/experiment.phase_a.yaml`.
- **`run_all.py` prompts** load **`prompts/phase_a_prompt_contract_gs2013_wording.yaml`** by default (`prompt_contract_path` in experiment YAML or `--prompt-contract`). Messages are built in `prompts/gs2013_message_builder.py`: betting instructions, story `setup`, optional `utterance_text` (per `question_key`), and questions from `llm_utils.prompts.STORIES`. Strategies `baseline` / `cot` / `structured_output` match the contract; structured runs set `response_format=json_object` and skip logprobs.
- **Live engineering run** example: full Phase A quantifier grid, single model, concurrency 15 — `configs/experiment.phase_a_live_single.yaml` + `configs/models_live_single.yaml`.

## Notes

- Live inference uses `openai.OpenAI(base_url=..., api_key=...).chat.completions.create(...)`. You only pay **OpenRouter** (or your provider); no separate LiteLLM product.
- Next implementation step: build user messages from the chosen `phase_a_prompt_contract*.yaml` (or reuse `llm_utils.prompts` builders) inside `run_all.py`.
- **RSA Task 2** (logprob / \(\alpha\) fits): see `code/pipeline_v1/rsa/README.md`.
- Initial candidate list for scaling tests:
  - `code/pipeline_v1/configs/models_openrouter_logprobs_initial.yaml`
