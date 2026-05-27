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

export HF_TOKEN="<your_huggingface_token>"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4jpassword"
export LLM_BACKEND="local"
export LOCAL_LLM_HF_CACHE_DIR="/ibiscostorage/${USER}/hf_cache"
export LOCAL_LLM_TENSOR_PARALLEL_SIZE=2
export LOCAL_LLM_GPU_MEMORY_UTILIZATION=0.90

# Optional: reduce context length if GPU memory is tight
# export LOCAL_LLM_MAX_MODEL_LEN=8192

# Optional: AWQ quantization to halve VRAM usage
# export LOCAL_LLM_QUANTIZATION=awq

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

# ── Run Multi-DoE ─────────────────────────────────────────────────────────────
RUN_DIR="experiments/multi_doe/runs/ibisco_$(date +%Y%m%d_%H%M%S)"

python scripts/run_multi_doe_ibisco.py \
    --seed 42 \
    --replicates 10 \
    --models "gpt_oss_120b,groq_llama_3_3_70b_versatile" \
    --domains "CIVILE,PENALE,AMMINISTRATIVO,MISTO" \
    --planning-ablations on,off \
    --causality-ablations on,off \
    --causality-models gpt_oss_120b \
    --fixed-model "groq_llama_3_3_70b_versatile" \
    --tensor-parallel-size 2 \
    --hf-cache-dir "${LOCAL_LLM_HF_CACHE_DIR}" \
    --out "${RUN_DIR}"
# --fixed-model keeps Llama 3.3 70B always loaded for retrieval/filter/AQA.
# DeepSeek R1 is loaded only when needed for its DoE conditions, then swapped.

# ── Analyze results ───────────────────────────────────────────────────────────
python scripts/analyze_multi_doe.py \
    --run-dir "${RUN_DIR}" \
    --output "experiments/multi_doe/analysis/ibisco_$(date +%Y%m%d_%H%M%S)"

# ── Cleanup Neo4j ─────────────────────────────────────────────────────────────
singularity instance stop neo4j_inst
