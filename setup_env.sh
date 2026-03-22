#!/bin/bash
# setup_env.sh — creates the llm-implicature conda environment and configures
# PYTHONPATH so all imports resolve relative to the /code directory.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$REPO_DIR/code"
ENV_NAME="llm-implicature"

# 1. Create conda environment 
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] Conda env '$ENV_NAME' already exists — skipping creation."
else
    echo "[setup] Creating conda env '$ENV_NAME' (Python 3.11)..."
    conda create -n "$ENV_NAME" python=3.11 -y || {
        echo "[setup] WARNING: conda env creation failed — continuing with path setup."
    }
fi

# 2 Locate the env
ENV_PREFIX="$(conda info --envs | awk -v name="$ENV_NAME" '$1 == name {print $NF}')"

if [ -z "$ENV_PREFIX" ]; then
    echo "[setup] ERROR: Could not locate conda env '$ENV_NAME'. Path hooks not installed."
    exit 1
fi

# 3 Install activate / deactivate PYTHONPATH hooks
ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/llm_implicature_path.sh" <<EOF
#!/bin/bash
# Adds the project's /code directory to PYTHONPATH so all imports are relative to it.
# To set up on a new machine: re-run setup_env.sh from the repo root.
export LLM_IMPLICATURE_CODE_PREV_PYTHONPATH="\$PYTHONPATH"
export PYTHONPATH="$CODE_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
EOF

cat > "$DEACTIVATE_DIR/llm_implicature_path.sh" <<EOF
#!/bin/bash
export PYTHONPATH="\$LLM_IMPLICATURE_CODE_PREV_PYTHONPATH"
unset LLM_IMPLICATURE_CODE_PREV_PYTHONPATH
EOF

echo "[setup] PYTHONPATH hooks installed → $CODE_DIR"
echo "[setup] Done. Run: conda activate $ENV_NAME"
