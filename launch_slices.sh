#!/usr/bin/env bash
# =============================================================================
#  launch_slices.sh — submit the full 4×4 Reasoner×Counter DoE as C(n,2)
#  two-model slice jobs (one `sbatch run.slurm` each), fully isolated via
#  per-job WORK. 4 models → 6 jobs.
#
#  Per-slice models.conf: each job serves ONLY its 2 slice models + the fixed
#  support model (≈3 models) to save GPUs, instead of the whole models.conf.
#  The reduced conf is generated here and passed via MODELS_CONF → run.slurm.
#
#  Diagonals (R==C) are computed EXACTLY ONCE: a pair runs with
#  --same-model-cells=on iff it is a "matching" pair (models i,i+1 with i even),
#  otherwise =off. For an even model count this covers all n×n cells once.
#
#  Cache: already seeded and ALWAYS ON. No warm job — each slice seeds its
#  per-job cache from SEED_CACHE (a copy, so no shared SQLite/WAL across nodes).
#
#  Usage (from the repo root):
#    ./launch_slices.sh
#    DOMAINS=CIVILE REPLICATES=1 ./launch_slices.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"            # repo root ($DIR)

# ── The 4 DoE models (aliases) ────────────────────────────────────────────────
MODELS=(gpt_oss_120b llama_3_3_70b_instruct deepseek_r1_distill_70b qwen_25_72b)
FIXED="${FIXED:-llama_4_scout_17b}"            # fixed support/evaluator (NOT a DoE factor)

# ── alias → served label (KEEP IN SYNC with CERBERUS_ALIAS_MAP in src/config.py) ─
declare -A LABEL=(
    [gpt_oss_120b]=oss120
    [llama_3_3_70b_instruct]=llama33
    [deepseek_r1_distill_70b]=deepseek70
    [qwen_25_72b]=qwen72
    [llama_4_scout_17b]=assistant
)

DOMAINS="${DOMAINS:-CIVILE,PENALE,AMMINISTRATIVO,MISTO}"
REPLICATES="${REPLICATES:-5}"
SEED="${SEED:-42}"
SEED_CACHE="${SEED_CACHE:-$PWD/cache/claim_context_cache.sqlite}"   # already-seeded cache
BASE_CONF="$PWD/models.conf"
CONF_DIR="$PWD/runs/confs"; mkdir -p "$CONF_DIR"

n=${#MODELS[@]}
(( n >= 2 )) || { echo "!! need at least 2 models"; exit 1; }
(( n % 2 == 0 )) || echo "⚠️  odd model count: one diagonal is not auto-covered."

if [ -f "$SEED_CACHE" ]; then echo "seed cache: $SEED_CACHE (copied into each job)"
else echo "⚠️  seed cache not found at $SEED_CACHE — each slice builds its own."; SEED_CACHE=""; fi
echo "Models ($n): ${MODELS[*]}  | slice jobs: $(( n * (n - 1) / 2 ))"

# Write a models.conf keeping only the [[model]] blocks whose label ∈ keep-set,
# plus the [allocation]/[defaults] header. $1=keep(csv labels) $2=out path.
gen_conf() {
    awk -v keep="$1" '
        BEGIN { split(keep, a, ","); for (i in a) want[a[i]] = 1; header = 1; inmodel = 0 }
        /^\[\[model\]\]/ {
            if (inmodel && (lbl in want)) printf "%s", block
            inmodel = 1; header = 0; block = $0 "\n"; lbl = ""; next
        }
        { if (header) { print; next }
          block = block $0 "\n"
          if ($0 ~ /^[[:space:]]*label[[:space:]]*=/) {
              l = $0; sub(/^[^=]*=[[:space:]]*"?/, "", l); sub(/".*$/, "", l); lbl = l } }
        END { if (inmodel && (lbl in want)) printf "%s", block }
    ' "$BASE_CONF" > "$2"
}

# ── All C(n,2) pair-jobs ──────────────────────────────────────────────────────
for (( i = 0; i < n; i++ )); do
    for (( j = i + 1; j < n; j++ )); do
        mi=${MODELS[$i]}; mj=${MODELS[$j]}
        if (( i % 2 == 0 && j == i + 1 )); then same=on; else same=off; fi

        # per-slice models.conf: this slice's two models + the fixed support model
        conf="$CONF_DIR/slice_${mi}_${mj}.models.conf"
        gen_conf "${LABEL[$mi]},${LABEL[$mj]},${LABEL[$FIXED]}" "$conf"

        id=$(
            export DOMAINS="$DOMAINS" REPLICATES="$REPLICATES" SEED="$SEED" SAME_MODEL="$same" \
                   REASONER="$mi,$mj" COUNTER="$mi,$mj" FIXED="$FIXED" MODELS_CONF="$conf"
            [ -n "$SEED_CACHE" ] && export WARM_CACHE="$SEED_CACHE"
            sbatch --parsable --export=ALL run.slurm
        )
        printf 'slice {%s, %s} same=%-3s conf=%s -> job %s\n' "$mi" "$mj" "$same" "$(basename "$conf")" "$id"
    done
done

echo "Done. Watch:  squeue -u \$USER    |    results: runs/job_<id>/results/"
