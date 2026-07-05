#!/bin/bash
#SBATCH --job-name=lexcausa_doe
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                  # adjust to available GPUs
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh
conda activate lexcausa

export HF_TOKEN="<your_huggingface_token>"   # only used by `cerberus download` for gated repos
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4jpassword"
export LLM_BACKEND="local"                   # 'local' → Cerberus (llama.cpp) served models

# Cerberus project dir: folder holding models.conf (labels must match the LexCausa
# aliases). `cerberus up` writes endpoints.json here; the client reads it. The
# lexcausa env must expose the Cerberus client + CLI: pip install -e /path/to/Cerberus
# GPU sizing / context / quantization are per-model in models.conf, not env vars.
export CERBERUS_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export CERBERUS_ENDPOINTS="${CERBERUS_PROJECT_DIR}/endpoints.json"

# ── Start Neo4j (Singularity) ─────────────────────────────────────────────────
# Adjust SIF path to your Singularity image location
NEO4J_SIF="/ibiscostorage/${USER}/singularity/neo4j_5.sif"

singularity instance start \
    --bind /ibiscostorage/${USER}/neo4j_data:/data \
    --bind /ibiscostorage/${USER}/neo4j_logs:/logs \
    "${NEO4J_SIF}" neo4j_inst \
    --env NEO4J_AUTH="${NEO4J_USER}/${NEO4J_PASSWORD}" &

echo "Waiting for Neo4j…"
sleep 30

# ── Load Knowledge Base (only needed first time) ──────────────────────────────
# Uncomment to (re)initialize the KB:
# python src/db/db_orchestrator.py

# ── Serve the models with Cerberus (llama.cpp) ────────────────────────────────
# `cerberus up` serves ALL models in models.conf concurrently and writes
# endpoints.json. Do NOT wrap it in srun (Cerberus issues its own srun calls).
cd "${CERBERUS_PROJECT_DIR}"
rm -f endpoints.json
cerberus up &
CERB_UP=$!
trap 'kill -INT "${CERB_UP}" 2>/dev/null || true; wait "${CERB_UP}" 2>/dev/null || true' EXIT

echo "Waiting for Cerberus endpoints.json…"
for _ in $(seq 1 400); do [ -f endpoints.json ] && break; sleep 3; done
[ -f endpoints.json ] || { echo "Cerberus servers did not start (see .cerberus/*/logs)"; exit 1; }
cerberus status

# ── Run Multi-DoE ─────────────────────────────────────────────────────────────
RUN_DIR="experiments/multi_doe/runs/ibisco_$(date +%Y%m%d_%H%M%S)"

python scripts/run_multi_doe_ibisco.py \
    --seed 42 \
    --replicates 10 \
    --reasoner-models "gpt_oss_120b,groq_llama_3_3_70b_versatile" \
    --counter-models  "gpt_oss_120b,groq_llama_3_3_70b_versatile" \
    --domains "CIVILE,PENALE,AMMINISTRATIVO,MISTO" \
    --planning-ablations on,off \
    --causality-ablations on \
    --fixed-model "groq_llama_3_3_70b_versatile" \
    --evaluator-model "groq_llama_3_3_70b_versatile" \
    --out "${RUN_DIR}"
# --fixed-model pins the retrieval/filter/AQA/evaluator model. With Cerberus every
# model is served at once, so there is no per-pair GPU load/unload.

# ── Analyze results ───────────────────────────────────────────────────────────
python scripts/analyze_multi_doe.py \
    --run-dir "${RUN_DIR}" \
    --output "experiments/multi_doe/analysis/ibisco_$(date +%Y%m%d_%H%M%S)"

# ── Cleanup Neo4j ─────────────────────────────────────────────────────────────
singularity instance stop neo4j_inst
