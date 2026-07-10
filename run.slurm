#!/usr/bin/env bash
#SBATCH --job-name=lexcausa
#SBATCH --partition=gpus
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

# Raise the per-user process/thread limit to the hard max (Neo4j JVM + Cerberus
# llama.cpp + pipeline spawn many threads → "fork: Resource temporarily unavailable").
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

# ── Environment (single job, two conda envs) ─────────────────────────────────
# `cerberus up` runs in `cerberus`; the LexCausa pipeline runs in `lexcausa`
# (which must ALSO have the Cerberus client: pip install -e /path/to/Cerberus).
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh && conda activate cerberus
export HF_TOKEN=hf_...                 # only used by `cerberus download` (gated repos)

DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"      # SLURM runs a copy of the script: use this
cd "$DIR"
mkdir -p logs

# ── Per-job WORK dir: FULL isolation for parallel jobs ───────────────────────
# Everything ephemeral (Cerberus endpoints + .cerberus, Neo4j store + endpoint,
# claim-context cache, DoE outputs) lives under WORK, keyed by the SLURM job id.
# So multiple `sbatch run.slurm` run concurrently without clobbering each other —
# each has its own allocation (own nodes), own endpoints, own KB, own cache.
JOB="${SLURM_JOB_ID:-$$}"
WORK="$DIR/runs/job_${JOB}"
mkdir -p "$WORK"/{neo4j,cache,results}
echo "📁 WORK: $WORK"

export LLM_BACKEND=local                          # 'local' → ChatCerberus
export CERBERUS_ENDPOINTS="$WORK/endpoints.json"  # client reads this
export CERBERUS_DEBUG=1                           # log every alias→label resolution + FALLBACKs
export NEO4J_DATA_DIR="$WORK/neo4j"               # per-job Neo4j store
export NEO4J_ENDPOINTS="$WORK/neo4j_endpoint.json"
export CLAIM_CONTEXT_CACHE_DIR="$WORK/cache"       # per-job retrieval cache (no shared-file WAL hazard)
export LEXCAUSA_LOG_DIR="$WORK/results/logs"       # per-claim pipeline logs land next to run_errors.log, not repo-root logs/

# Generous token budgets (paired with the removed SLURM time limit). The DoE
# payload sets llm_max_tokens=20000; these per-phase caps must also be raised or
# each call stays bounded by min(llm_max_tokens, cap) — the small default caps
# (planner=1600) are why gpt-oss/deepseek exhausted the budget on thinking and
# returned empty JSON. Set via env so the global (cloud) defaults are untouched.
export ANCILLARY_MAX_TOKENS_CAP=20000
export REASONER_PLANNER_MAX_TOKENS_CAP=20000
export REASONER_SUPPORT_STEP_MAX_TOKENS_CAP=20000
export REASONER_CONCLUSION_MAX_TOKENS_CAP=20000
export COUNTER_PLANNER_MAX_TOKENS_CAP=20000

# ── Model slice (override per job: sbatch --export=ALL,REASONER=...,COUNTER=...) ─
REASONER="${REASONER:-gpt_oss_120b,llama_3_3_70b_instruct}"
COUNTER="${COUNTER:-gpt_oss_120b,llama_3_3_70b_instruct}"
FIXED="${FIXED:-llama_4_scout_17b}"
DOMAINS="${DOMAINS:-CIVILE}"
REPLICATES="${REPLICATES:-1}"
SEED="${SEED:-42}"
SAME_MODEL="${SAME_MODEL:-on}"          # 'off' = skip R==C diagonal cells (slice runner)
# 'on,off' = full ablation (2 planning levels x 2 causality levels -> 4x rows/job);
# 'on' = single level (ablation not explored), kept as the standalone-run default.
PLANNING_ABLATIONS="${PLANNING_ABLATIONS:-on,off}"
CAUSALITY_ABLATIONS="${CAUSALITY_ABLATIONS:-on,off}"

# ── Optional: seed the cache from a pre-warmed sqlite (efficiency + no WAL races) ─
# Warm once over all claims in a single job, then pass its file:
#   sbatch --export=ALL,WARM_CACHE=/lustre/.../claim_context_cache.sqlite run.slurm
# The warm job must have FINISHED (SQLite checkpoints WAL on last close). Every
# retrieval then HITs the local copy → no recompute, and no cross-node WAL sharing.
if [ -n "${WARM_CACHE:-}" ] && [ -f "${WARM_CACHE}" ]; then
    echo ">> Seeding claim-context cache from ${WARM_CACHE}"
    cp -f "${WARM_CACHE}" "$WORK/cache/claim_context_cache.sqlite"
fi

# ── 1) Neo4j on node 0 (per-job store under WORK) ────────────────────────────
deploy/neo4j/neo4j_up.sh

# ── 2) Serve the models — endpoints + .cerberus written under WORK ───────────
# Do NOT wrap `cerberus up` in srun. --project-dir isolates the endpoint map.
# MODELS_CONF lets the slice runner pass a per-slice models.conf (only this
# slice's models + fixed) to save GPUs; defaults to the full models.conf.
cerberus up -c "${MODELS_CONF:-models.conf}" --project-dir "$WORK" &
UP=$!
trap 'kill -INT "$UP" 2>/dev/null || true; wait "$UP" 2>/dev/null || true; "$DIR/deploy/neo4j/neo4j_up.sh" stop 2>/dev/null || true' EXIT

echo "waiting for $WORK/endpoints.json ..."
for _ in $(seq 1 400); do [ -f "$WORK/endpoints.json" ] && break; sleep 3; done
[ -f "$WORK/endpoints.json" ] || { echo "servers did not start (see $WORK/.cerberus/*/logs)"; exit 1; }
cerberus status --project-dir "$WORK"

# ── Sanity: claims file must be present (gitignored user data) ────────────────
[ -f "$DIR/claims.md" ] || { echo "claims.md not found in $DIR — provide it first"; exit 1; }

# ── Switch to the LexCausa env for the pipeline ──────────────────────────────
conda activate lexcausa

# ── 3) Populate this job's Neo4j (always fresh; fast if WARM_CACHE covers retrieval) ─
deploy/neo4j/populate_kb.sh --clean

# ── 4) Multi-DoE run (all outputs under WORK/results) ────────────────────────
python scripts/run_multi_doe_ibisco.py \
    --seed "$SEED" \
    --replicates "$REPLICATES" \
    --reasoner-models "$REASONER" \
    --counter-models  "$COUNTER" \
    --domains "$DOMAINS" \
    --planning-ablations "$PLANNING_ABLATIONS" \
    --causality-ablations "$CAUSALITY_ABLATIONS" \
    --same-model-cells "$SAME_MODEL" \
    --fixed-model "$FIXED" \
    --evaluator-model "$FIXED" \
    --out "$WORK/results"

echo "✅ job ${JOB} finished — results in $WORK/results/metrics.csv"
# On exit, the trap stops Cerberus and Neo4j.
