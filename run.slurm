#!/usr/bin/env bash
#SBATCH --job-name=lexcausa
#SBATCH --partition=gpus
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
# This job serves the models with Cerberus (llama.cpp) AND runs the LexCausa
# pipeline in the SAME env, so that env must have BOTH:
#   pip install -e /path/to/Cerberus            # cerberus CLI + client + openai SDK
#   pip install -r requirements_hpc.txt         # LexCausa deps (langchain, neo4j, …)
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh && conda activate cerberus

export HF_TOKEN=hf_...                 # only used by `cerberus download` (gated repos)

DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"      # SLURM runs a copy of the script: use this
cd "$DIR"
mkdir -p logs

# ── LexCausa backend → Cerberus served models ────────────────────────────────
export LLM_BACKEND=local               # 'local' → ChatCerberus
export CERBERUS_ENDPOINTS="$DIR/endpoints.json"

# ── Neo4j (assumed already running/reachable; adjust to your setup) ───────────
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jpassword}"

# ── Serve the models (llama.cpp) ─────────────────────────────────────────────
# Do NOT wrap `cerberus up` in srun — Cerberus issues its own srun to the nodes.
rm -f endpoints.json
cerberus up &                          # serves all models in models.conf, in background
UP=$!
trap 'kill -INT "$UP" 2>/dev/null || true; wait "$UP" 2>/dev/null || true' EXIT

echo "waiting for endpoints.json..."
for _ in $(seq 1 400); do [ -f endpoints.json ] && break; sleep 3; done
[ -f endpoints.json ] || { echo "servers did not start (see .cerberus/*/logs)"; exit 1; }
cerberus status

# ── Sanity: claims file must be present (it is gitignored user data) ──────────
[ -f "$DIR/claims.md" ] || { echo "claims.md not found in $DIR — provide it first"; exit 1; }

# ── Reduced Multi-DoE run: 2 models vary as Reasoner × Counter (2×2 = 4 cells) ─
# Aliases resolve via CERBERUS_ALIAS_MAP (src/config.py):
#   gpt_oss_120b→oss120 | llama_3_3_70b_instruct→llama33   (both vary: R × C)
#   llama_4_scout_17b→assistant  (fixed support/evaluator — NOT a DoE factor)
RUN_DIR="experiments/multi_doe/runs/ibisco_reduced_$(date +%Y%m%d_%H%M%S)"

python scripts/run_multi_doe_ibisco.py \
    --seed 42 \
    --replicates 1 \
    --reasoner-models gpt_oss_120b,llama_3_3_70b_instruct \
    --counter-models  gpt_oss_120b,llama_3_3_70b_instruct \
    --domains CIVILE \
    --planning-ablations on \
    --causality-ablations on \
    --fixed-model llama_4_scout_17b \
    --evaluator-model llama_4_scout_17b \
    --out "$RUN_DIR"

echo "trial finished — results in $RUN_DIR/metrics.csv"
# On exit, the trap stops the Cerberus servers.
