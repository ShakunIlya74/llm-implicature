# llm-implicature
Mod. agents team project

## Setup

1. Run the setup script from the repo root:
   ```bash
   bash setup_env.sh
   ```
   This creates the `llm-implicature` conda environment and configures `PYTHONPATH`
   so that all imports resolve relative to the `code/` directory.
   If the environment already exists, the path setup still runs.

2. Activate the environment:
   ```bash
   conda activate llm-implicature
   ```

3. Install Python dependencies (e.g. `openai`, `pyyaml` for `pipeline_v1`):
   ```bash
   pip install -r requirements.txt
   ```
