#!/bin/bash
# =============================================================================
# PC "A" (macOS) — Multi-DoE OpenRouter, THESIS models, the 10 claims that the
# thesis campaign did not cover.
#
# Design = exactly the thesis 720 campaign, on the remaining claims:
#   10 claims x 2 Reasoner x 2 Counter x 3 paradigms x 5 replicates = 600 runs
#   models: gpt_oss_120b, llama_3_3_70b   (aux/evaluator: llama_4_scout)
# Split over TWO machines with a global shard-count of 8:
#   PC A -> global shards 0,1,2,3  (this file, 300 runs)
#   PC B -> global shards 4,5,6,7  (run_doe600_pcB.sh, 300 runs)
# Partition is df.iloc[i::8]: disjoint by construction.
#
# The 600 runs are meant to be MERGED with the thesis 720 into a single
# 1320-run analysis over all 22 claims, so every parameter below is pinned to
# the thesis values. Do not change them.
#
# Usage:  caffeinate -i bash scripts/run_doe600_pcA.sh
# =============================================================================
set -euo pipefail

PC_LABEL="A"
SHARD_OFFSET=0            # PC A takes global shards 0..3
GLOBAL_SHARDS=8           # 2 machines x 4 workers
WORKERS="${WORKERS:-4}"
BASE_PORT="${BASE_PORT:-8001}"

# ---- pinned to the thesis campaign; must match PC B exactly -----------------
CLAIMS="${CLAIMS:-claims_doe10_remaining.md}"
REPLICATES="${REPLICATES:-5}"
PLANNING="${PLANNING:-on,off,single}"
PAIRING="${PAIRING:-cross}"
R_MODELS="gpt_oss_120b,llama_3_3_70b"
C_MODELS="gpt_oss_120b,llama_3_3_70b"
SEED=42
MIN_KEPT=8
MAX_STATUTES=100
MAX_PRECEDENTS=5
EXPECTED_TOTAL=600
# -----------------------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-experiments/multi_doe/runs/or_doe600_pc${PC_LABEL}_${STAMP}}"

cd "$(dirname "$0")/.."

# portable sha256 (macOS: shasum, Linux: sha256sum)
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

echo "==============================================================="
echo " PC $PC_LABEL (macOS) | global shards ${SHARD_OFFSET}..$((SHARD_OFFSET+WORKERS-1)) of $GLOBAL_SHARDS"
echo "==============================================================="
echo "  claims=$CLAIMS  models=$R_MODELS  planning=$PLANNING  reps=$REPLICATES  seed=$SEED"
echo "  retrieval: min_kept=$MIN_KEPT max_statutes=$MAX_STATUTES max_precedents=$MAX_PRECEDENTS"
echo "  out=$OUT_BASE"
echo

# --- 0. PREFLIGHT -------------------------------------------------------------
echo "[0/5] preflight: verifying the run matrix..."
if [ ! -f "$CLAIMS" ]; then
    echo "ABORT: $CLAIMS not found. Copy it from PC A before running." >&2; exit 1
fi
PREFLIGHT="$(LLM_BACKEND=openrouter poetry run python scripts/run_multi_doe.py \
    --claims-file "$CLAIMS" \
    --reasoner-models "$R_MODELS" --counter-models "$C_MODELS" \
    --pairing "$PAIRING" --planning-ablations "$PLANNING" \
    --replicates "$REPLICATES" --seed "$SEED" \
    --shard-index "$SHARD_OFFSET" --shard-count "$GLOBAL_SHARDS" \
    --out "/tmp/preflight_pc${PC_LABEL}" --dry-run 2>&1)"
echo "$PREFLIGHT" | sed 's/^/    /'

TOTAL="$(echo "$PREFLIGHT" | grep -oE 'Generated [0-9]+' | grep -oE '[0-9]+' || echo 0)"
if [ "$TOTAL" != "$EXPECTED_TOTAL" ]; then
    echo "ABORT: matrix is $TOTAL runs, expected $EXPECTED_TOTAL. Config drift — do NOT run split." >&2
    exit 1
fi
CLAIMS_HASH="$(sha256_of "$CLAIMS")"
FINGERPRINT="$(printf '%s|%s|%s|%s|%s|%s|%s' \
    "$CLAIMS_HASH" "$R_MODELS" "$C_MODELS" "$PAIRING" "$PLANNING" "$REPLICATES" "$SEED" \
    | { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; } | cut -c1-12)"
echo
echo "    >>> matrix total = $TOTAL runs (this machine: $((TOTAL/GLOBAL_SHARDS*WORKERS)))"
echo "    >>> claims sha256    = $CLAIMS_HASH"
echo "    >>> CONFIG FINGERPRINT = $FINGERPRINT"
echo "    >>> both values MUST be identical on PC B."
echo
sleep 5

# --- 1. clean slate -----------------------------------------------------------
echo "[1/5] stopping previous backends..."
pkill -f "python.*src/api_server.py" 2>/dev/null || true
sleep 2

# --- 2. Neo4j -----------------------------------------------------------------
echo "[2/5] starting Neo4j..."
docker compose up -d neo4j
until docker ps --format '{{.Names}} {{.Status}}' | grep -q 'lexcausa-neo4j.*healthy'; do sleep 2; done
poetry run python src/db/db_orchestrator.py --check

# --- 3. one backend per worker ------------------------------------------------
echo "[3/5] starting $WORKERS backends (LLM_BACKEND=openrouter)..."
for i in $(seq 0 $((WORKERS-1))); do
    port=$((BASE_PORT+i))
    API_PORT=$port LLM_BACKEND=openrouter \
        poetry run python src/api_server.py > "/tmp/lexcausa_be_pc${PC_LABEL}_${i}.log" 2>&1 &
    echo "   worker $i -> :$port"
done

echo "[4/5] waiting for backends..."
for i in $(seq 0 $((WORKERS-1))); do
    port=$((BASE_PORT+i))
    until curl -sf "http://127.0.0.1:${port}/health" >/dev/null; do sleep 2; done
    echo "   worker $i healthy (:$port)"
done

# --- 4. launch this machine's slice ------------------------------------------
echo "[5/5] launching shards..."
pids=()
for i in $(seq 0 $((WORKERS-1))); do
    port=$((BASE_PORT+i))
    gshard=$((SHARD_OFFSET+i))
    poetry run python scripts/run_multi_doe.py \
        --claims-file "$CLAIMS" \
        --reasoner-models "$R_MODELS" --counter-models "$C_MODELS" \
        --pairing "$PAIRING" --planning-ablations "$PLANNING" \
        --replicates "$REPLICATES" --seed "$SEED" \
        --min-kept "$MIN_KEPT" --max-statutes "$MAX_STATUTES" --max-precedents "$MAX_PRECEDENTS" \
        --api-url "http://localhost:${port}" \
        --shard-index "$gshard" --shard-count "$GLOBAL_SHARDS" \
        --out "${OUT_BASE}_g${gshard}" > "/tmp/lexcausa_doe_pc${PC_LABEL}_g${gshard}.log" 2>&1 &
    shard_pid=$!
    pids+=("$shard_pid")
    echo "   global shard $gshard -> :$port (pid $shard_pid, log /tmp/lexcausa_doe_pc${PC_LABEL}_g${gshard}.log)"
done

echo
echo "Monitoring:  tail -f /tmp/lexcausa_doe_pc${PC_LABEL}_g${SHARD_OFFSET}.log"
echo

fail=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then echo "shard $((SHARD_OFFSET+idx)) OK"
    else echo "shard $((SHARD_OFFSET+idx)) FAILED — see its log"; fail=1; fi
done

echo
echo "=== PC $PC_LABEL done. Results: ${OUT_BASE}_g${SHARD_OFFSET}..g$((SHARD_OFFSET+WORKERS-1)) ==="
echo "Stop backends: pkill -f \"python.*src/api_server.py\""
echo
echo "Next: copy PC B's *_g4..g7 dirs here, then merge everything (new 600 + thesis 720):"
echo "  poetry run python scripts/merge_doe_shards.py \\"
echo "      --shards \"experiments/multi_doe/runs/or_doe600_pc*_g*\" \\"
echo "               \"experiments/multi_doe/runs/or_doe480_4w_s*\" \\"
echo "               \"experiments/multi_doe/runs/or_sc240_s*\" \\"
echo "      --out experiments/multi_doe/runs/or_doe1320_all22 --expect 1320"
exit $fail
