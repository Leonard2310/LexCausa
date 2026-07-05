#!/usr/bin/env bash
#SBATCH --job-name=lexcausa
#SBATCH --partition=gpus
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh && conda activate cerberus
export HF_TOKEN=hf_...
DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"     # SLURM esegue una copia dello script: usa questa
cd "$DIR"
mkdir -p logs

rm -f endpoints.json
cerberus up &                          # avvia i server in background
UP=$!
trap 'kill -INT "$UP" 2>/dev/null || true; wait "$UP" 2>/dev/null || true' EXIT

echo "attendo endpoints.json..."
for _ in $(seq 1 400); do [ -f endpoints.json ] && break; sleep 3; done
[ -f endpoints.json ] || { echo "server non partiti (vedi .cerberus/*/logs)"; exit 1; }

#python experiment.py                   # il tuo esperimento
# all'uscita, il trap ferma i server
