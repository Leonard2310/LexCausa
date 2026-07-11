#!/usr/bin/env bash
# =============================================================================
#  launch_slices_split_causality.sh — variant of launch_slices.sh that submits
#  12 jobs instead of 6: each of the C(n,2)=6 model-pair slices is split into
#  TWO jobs, one with causality_ablations pinned to "on" and one pinned to
#  "off", instead of a single job exploring both values internally.
#
#  Why: run.slurm's per-cell row count is claims × RC_combos × replicates ×
#  planning_levels × causality_levels. Pulling causality out to the job axis
#  halves the rows (and wall-clock) of EACH job while doubling the job count —
#  same total work, but if SLURM can schedule more jobs concurrently, the whole
#  sweep finishes sooner. planning_ablations still varies "on,off" WITHIN each
#  job (2x rows, not 4x).
#
#  Everything else (per-slice models.conf, diagonal-once same-model logic,
#  per-job WORK isolation, seeded cache) is identical to launch_slices.sh.
#
#  Usage (from the repo root):
#    ./launch_slices_split_causality.sh
#    DOMAINS=CIVILE REPLICATES=1 ./launch_slices_split_causality.sh
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

# ── The factor split across the job axis (fixed on/off per job) ──────────────
CAUSALITY_VALUES=(on off)

n=${#MODELS[@]}
(( n >= 2 )) || { echo "!! need at least 2 models"; exit 1; }
(( n % 2 == 0 )) || echo "⚠️  odd model count: one diagonal is not auto-covered."

if [ -f "$SEED_CACHE" ]; then echo "seed cache: $SEED_CACHE (copied into each job)"
else echo "⚠️  seed cache not found at $SEED_CACHE — each slice builds its own."; SEED_CACHE=""; fi
echo "Models ($n): ${MODELS[*]}  | slice pairs: $(( n * (n - 1) / 2 ))  | jobs: $(( n * (n - 1) / 2 * ${#CAUSALITY_VALUES[@]} )) (causality split on/off)"

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

# ── All C(n,2) pair-jobs × 2 causality values ─────────────────────────────────
for (( i = 0; i < n; i++ )); do
    for (( j = i + 1; j < n; j++ )); do
        mi=${MODELS[$i]}; mj=${MODELS[$j]}
        if (( i % 2 == 0 && j == i + 1 )); then same=on; else same=off; fi

        # per-slice models.conf: this slice's two models + the fixed support model
        # (shared by both causality jobs of this pair — the served models don't
        # depend on the ablation factor).
        conf="$CONF_DIR/slice_${mi}_${mj}.models.conf"
        gen_conf "${LABEL[$mi]},${LABEL[$mj]},${LABEL[$FIXED]}" "$conf"

        for caus in "${CAUSALITY_VALUES[@]}"; do
            id=$(
                export DOMAINS="$DOMAINS" REPLICATES="$REPLICATES" SEED="$SEED" SAME_MODEL="$same" \
                       REASONER="$mi,$mj" COUNTER="$mi,$mj" FIXED="$FIXED" MODELS_CONF="$conf" \
                       CAUSALITY_ABLATIONS="$caus"
                [ -n "$SEED_CACHE" ] && export WARM_CACHE="$SEED_CACHE"
                sbatch --parsable --export=ALL run.slurm
            )
            printf 'slice {%s, %s} same=%-3s causality=%-3s conf=%s -> job %s\n' \
                "$mi" "$mj" "$same" "$caus" "$(basename "$conf")" "$id"
        done
    done
done

echo "Done. Watch:  squeue -u \$USER    |    results: runs/job_<id>/results/"
