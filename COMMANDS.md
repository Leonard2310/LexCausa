# LexCausa Runbook (Commands Only)

Runtime commands for this repository: database, backend/frontend, pipeline/API calls,
cache warmup, tuning scripts, and demo startup.

Notes:
- Run from project root (`LexCausa/`) unless stated otherwise.
- Python commands use `poetry`.
- Frontend lives in `src/frontend/`.

## 1. Neo4j (Database)

### Start Neo4j (Docker)
```bash
docker compose up -d
```

### Stop Neo4j
```bash
docker compose down
```

### View Neo4j logs
```bash
docker compose logs -f
```

### Initialize / load the knowledge base
```bash
poetry run python src/db/db_orchestrator.py
```

### Full clean reload of the knowledge base
```bash
poetry run python src/db/db_orchestrator.py --clean
```

### Check database status
```bash
poetry run python src/db/db_orchestrator.py --check
```

## 2. App Runtime (Backend + Frontend)

### One-command local startup (recommended)
```bash
make dev
```

### Stop local dev stack started by Makefile
```bash
make dev-stop
```

### Start Flask API backend
```bash
poetry run python src/api_server.py
```

### Start backend with custom host/port
```bash
API_HOST=127.0.0.1 API_PORT=8001 DEBUG=true poetry run python src/api_server.py
```

### Start frontend (Vite dev)
```bash
cd src/frontend && npm run dev
```

### Frontend lint
```bash
cd src/frontend && npm run lint
```

### Frontend build (production bundle)
```bash
cd src/frontend && npm run build
```

### Frontend preview (after build)
```bash
cd src/frontend && npm run preview
```

### Stop leftover backend processes
```bash
pkill -f "python.*src/api_server.py" || true
```

### Stop leftover frontend dev server (Vite)
```bash
pkill -f "node .*vite" || true
```

## 3. Public Demo (Cloudflare Tunnel)

### Start quick public demo (backend + frontend + tunnel)
```bash
bash scripts/start_public_demo.sh
```

### Start demo with custom ports / instance name
```bash
API_PORT=8001 FRONTEND_PORT=3001 INSTANCE_NAME=colleague bash scripts/start_public_demo.sh
```

### Start demo with explicit script args
```bash
bash scripts/start_public_demo.sh --instance colleague --api-port 8001 --frontend-port 3001 --host 127.0.0.1
```

### Kill leftover tunnel processes
```bash
pkill -f "cloudflared tunnel --url" || true
```

## 4. API Runtime (Manual Calls)

### Health check
```bash
curl http://127.0.0.1:8000/health
```

### Get frontend/backend runtime defaults
```bash
curl http://127.0.0.1:8000/api/settings
```

### `/api/chat` (search)
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Put your claim here",
    "top_k": 100,
    "include_precedents": true,
    "max_precedents": 5
  }'
```

### `/api/chat` with claim-context memory enabled (SQLite pre-retrieval cache)
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Put your claim here",
    "top_k": 100,
    "include_precedents": true,
    "max_precedents": 5,
    "claim_context_memory_enabled": true
  }'
```

### `/api/reason`
```bash
curl -X POST http://127.0.0.1:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "include_precedents": true,
    "max_statutes": 100,
    "max_precedents": 5
  }'
```

### `/api/counter_reason` (requires `reasoner_conclusion`)
```bash
curl -X POST http://127.0.0.1:8000/api/counter_reason \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "reasoner_conclusion": "Put the reasoner conclusion here",
    "include_precedents": true,
    "max_statutes": 100,
    "max_precedents": 5
  }'
```

### `/api/counter_reason/stream` (SSE)
```bash
curl -N -X POST http://127.0.0.1:8000/api/counter_reason/stream \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "reasoner_conclusion": "Put the reasoner conclusion here",
    "include_precedents": true,
    "max_statutes": 100,
    "max_precedents": 5,
    "settings": {
      "counter_enable_causality": true,
      "counter_pass_taxonomy_attacks": true,
      "counter_pass_norms": true
    }
  }'
```

### `/api/pipeline` (JSON)
```bash
curl -X POST http://127.0.0.1:8000/api/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "include_precedents": true,
    "max_statutes": 100,
    "max_precedents": 5
  }'
```

### `/api/pipeline` with claim-context memory enabled
```bash
curl -X POST http://127.0.0.1:8000/api/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "include_precedents": true,
    "max_statutes": 100,
    "max_precedents": 5,
    "claim_context_memory_enabled": true,
    "claim_context_memory_overwrite": false
  }'
```

### `/api/pipeline/stream` (SSE, terminal-friendly)
```bash
curl -N -X POST http://127.0.0.1:8000/api/pipeline/stream \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "include_precedents": true,
    "max_statutes": 100,
    "max_precedents": 5
  }'
```

### `/api/pipeline/stop` (stop active SSE run)
```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/stop \
  -H "Content-Type: application/json" \
  -d '{}'
```

### `/api/evaluate`
```bash
curl -X POST http://127.0.0.1:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "domain": "CIVILE",
    "reasoner_output": {},
    "counter_output": {},
    "settings": {
      "aqa_alpha": 0.3,
      "aqa_beta": 0.4,
      "aqa_gamma": 0.3
    }
  }'
```

### `/api/evaluate/stream` (SSE)
```bash
curl -N -X POST http://127.0.0.1:8000/api/evaluate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "domain": "CIVILE",
    "reasoner_output": {},
    "counter_output": {},
    "settings": {
      "aqa_alpha": 0.3,
      "aqa_beta": 0.4,
      "aqa_gamma": 0.3
    }
  }'
```

### `/api/doe/log` (persist one consolidated DoE log/report)
```bash
curl -X POST http://127.0.0.1:8000/api/doe/log \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "mode": "automatic_ab",
    "reasoner_shared": {},
    "baseline": {
      "label": "A (Baseline)",
      "description": "Counter taxonomy ON (norms)",
      "settings": {},
      "status": "done",
      "duration_ms": 1000,
      "metrics": {},
      "view": {
        "counter_reasoner": {},
        "evaluation": {}
      }
    },
    "treatment": {
      "label": "B (Treatment)",
      "description": "Counter taxonomy ON (identity + attacks + norms)",
      "settings": {},
      "status": "done",
      "duration_ms": 900,
      "metrics": {},
      "view": {
        "counter_reasoner": {},
        "evaluation": {}
      }
    },
    "delta": {
      "duration_ms": -100
    }
  }'
```

### `/api/pdf/export` (persist one exported PDF artifact)
```bash
curl -X POST http://127.0.0.1:8000/api/pdf/export \
  -F "pdf=@/absolute/path/to/report.pdf" \
  -F "claim=Put your claim here" \
  -F "prefix=doe_automatic" \
  -F "export_context=doe" \
  -F "client_filename=lexcausa_doe_report.pdf"
```

## 5. Pipeline Configuration via API / Frontend-equivalent payload

### Example `/api/pipeline` with selected runtime overrides
```bash
curl -X POST http://127.0.0.1:8000/api/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Put your claim here",
    "include_precedents": true,
    "max_statutes": 100,
    "max_precedents": 5,
    "claim_context_memory_enabled": true,
    "settings": {
      "reasoner_model": "gpt_oss_120b",
      "counter_model": "gpt_oss_120b",
      "reasoner_temperature": 0.0,
      "counter_temperature": 0.3,
      "llm_max_tokens": 7168,
      "search_min_kept_statutes": 8,
      "search_use_top_n_libri": 3,
      "chain_min_steps": 3,
      "chain_max_steps": 10,
      "reasoner_enable_causality": true,
      "counter_enable_causality": true,
      "counter_pass_taxonomy_attacks": true,
      "counter_pass_norms": true
    }
  }'
```

## 6. Claim Context Memory (SQLite pre-retrieval cache)

SQLite file:
```bash
ls -lh cache/claim_context_cache.sqlite
```

### Count cached claims (requires `sqlite3`)
```bash
sqlite3 cache/claim_context_cache.sqlite "SELECT count(*) FROM claim_context_cache;"
```

### Show latest cache entries (requires `sqlite3`)
```bash
sqlite3 cache/claim_context_cache.sqlite "SELECT substr(cache_key,1,10), updated_at, hit_count FROM claim_context_cache ORDER BY updated_at DESC LIMIT 10;"
```

## 7. Batch Warmup / Retrieval Capture Scripts (claims.md)

Script used to call `/api/chat` for all covered claims in `claims.md`.
It can:
- save one JSON file per claim in `logs/api_chat_memory/`
- warm the SQLite claim-context memory

### Help
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py --help
```

### Run on all covered claims (save per-claim JSON only)
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py
```

### Run on all covered claims + warm claim-context memory (SQLite) + save JSON
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py --claim-context-memory
```

### Warm claim-context memory only (no per-claim JSON files)
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py --claim-context-memory --cache-only
```

### Force overwrite cached claim-context memory entries
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py --claim-context-memory --overwrite-claim-context-memory
```

### Resume and skip claims already exported to JSON
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py --claim-context-memory --skip-existing
```

### Restrict to one domain
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py --claim-context-memory --domains penale
```

### Adjust capture settings (top-k / precedents / delay / timeout)
```bash
poetry run python scripts/capture_api_chat_retrieval_memory.py \
  --claim-context-memory \
  --top-k 100 \
  --max-precedents 5 \
  --delay 0.25 \
  --timeout none
```

## 8. Retrieval Tuning Scripts

### Help
```bash
poetry run python scripts/tune_retrieval_claims.py --help
```

### Supervised retrieval tuning (default if `claims_gold_labels.json` exists)
```bash
poetry run python scripts/tune_retrieval_claims.py --top-k 30
```

### Unsupervised retrieval tuning
```bash
poetry run python scripts/tune_retrieval_claims.py --unsupervised
```

### Retrieval tuning with explicit LLM query terms
```bash
poetry run python scripts/tune_retrieval_claims.py --query-terms-mode llm
```

### Retrieval tuning without progress bars (CI/log-friendly)
```bash
poetry run python scripts/tune_retrieval_claims.py --query-terms-mode llm --no-progress
```

## 9. AQA Tuning Scripts

### AQA tuning with gold dataset
```bash
poetry run python scripts/tune_aqa_with_gold_dataset.py
```

### AQA tuning with real + synthetic dataset
```bash
poetry run python scripts/tune_aqa_real_plus_synth.py
```

## 10. DoE / Experiment Scripts

### Generate DoE run plan
```bash
poetry run python experiments/doe/scripts/generate_run_plan.py
```

### Run DoE batch
```bash
poetry run python experiments/doe/scripts/run_doe.py
```

### Extract metrics from DoE outputs
```bash
poetry run python experiments/doe/scripts/extract_metrics.py
```

### Analyze DoE results
```bash
poetry run python experiments/doe/scripts/analyze_doe.py
```

### DoE scripts help (recommended before first run)
```bash
poetry run python experiments/doe/scripts/run_doe.py --help
poetry run python experiments/doe/scripts/extract_metrics.py --help
poetry run python experiments/doe/scripts/analyze_doe.py --help
```

## 11. Logs (Runtime)

### Latest pipeline logs
```bash
ls -lt logs | head
```

### Latest consolidated DoE logs
```bash
ls -lt logs/*_doe.log | head
```

### Latest DoE JSON reports
```bash
ls -lt logs/doe_reports | head
```

### Latest exported PDFs persisted by the backend
```bash
ls -lt logs/pdf_exports/pipeline logs/pdf_exports/doe
```

### Latest retrieval capture outputs
```bash
ls -lt logs/api_chat_memory | head
```

### Tail a specific pipeline log
```bash
tail -f logs/<timestamp>_<claim_slug>.log
```

### Tail a specific consolidated DoE log
```bash
tail -f logs/<timestamp>_<claim_slug>_doe.log
```

## 12. Multi-DoE full-factorial Reasoner×Counter (Ibisco / vLLM offline) — THESIS RUN

This is the thesis experiment
Scripts:
`scripts/run_multi_doe_ibisco.py` (run) + `scripts/analyze_multi_doe.py` (analysis).
Design: **4 Reasoner × 4 Counter × 2 planning(on/off) × causality(ON) × 22 claims × 10 replicas = 7,040 runs.**

### Pre-flight (once)
```bash
docker compose up -d
poetry run python src/db/db_orchestrator.py --check     # KB loaded (statutes + precedents)
huggingface-cli scan-cache                              # confirm the 5 models are cached
```

### Environment (models already downloaded → offline, no HF token)
```bash
export VLLM_HF_CACHE_DIR=/path/to/hf_cache     # passed to vLLM download_dir
# or, if it is the standard HF cache:           export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# GPU (adapt to the node):
export VLLM_TENSOR_PARALLEL_SIZE=<num_gpus>     # e.g. 2 or 4
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_MAX_MODEL_LEN=30000                 # ~30k tokens (thesis budget); do NOT set VLLM_QUANTIZATION
# Neo4j (if not already in a .env):
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=<password>
# Notes: the script forces LLM_BACKEND=local. HF_TOKEN is only needed to *download* the
# gated meta-llama/* models (Llama-3.3-70B, Llama-4-Maverick); skip it if they are cached.
```

### Run the DoE (7,040 runs)
```bash
python scripts/run_multi_doe_ibisco.py \
  --fixed-model llama_4_maverick_17b \
  --evaluator-model llama_4_maverick_17b \
  --reasoner-models deepseek_r1,gpt_oss_120b,groq_llama_3_3_70b_versatile,qwen_25_72b \
  --counter-models  deepseek_r1,gpt_oss_120b,groq_llama_3_3_70b_versatile,qwen_25_72b \
  --planning-ablations on,off \
  --causality-ablations on \
  --replicates 10 --seed 42 \
  --out experiments/multi_doe/runs/ibisco_$(date +%Y%m%d_%H%M%S)
```
At startup the script prints `Run matrix: N runs` — it **must be 7040** (14080 means you wrongly
passed `--causality-ablations on,off`). `metrics.csv` is written incrementally in `--out/`.

### Analyze (manual; works on a partial metrics.csv too)
```bash
poetry run python scripts/analyze_multi_doe.py \
  --run-dir experiments/multi_doe/runs/ibisco_<ts> \
  --output  experiments/multi_doe/analysis/ibisco_<ts>
poetry add matplotlib    # optional: enables the win/tie/loss heatmaps in analysis/.../heatmaps/
```
Produces `doe_analysis.json`: factorial ANOVA (+eta^2), Friedman (blocked by the 22 claims) +
Dunn/Holm with win/tie/loss matrices, Wilcoxon signed-rank (reasoning-vs-instruction and planning),
bootstrap CI; the 3 AQA dimensions analyzed separately.

### Critical notes
- **`--fixed-model llama_4_maverick_17b` is required** (else retrieval/AQA/evaluator use the first
  loaded reasoner model, not Maverick → wrong design).
- **`--causality-ablations on` (NOT `on,off`)**: the thesis holds causality enabled (1 level). The
  example in the script docstring showing `on,off` is misleading.
- `total_tokens` = completion (output) tokens. Reasoning aliases = `{deepseek_r1, gpt_oss_120b}`
  (CoT stripped before scoring). Hyperparameters are pinned in the payload (Reasoner temp 0.0,
  Counter 0.3, `llm_max_tokens` 7168, AQA alpha/beta/gamma = 0.3/0.4/0.3, verdict thresholds ±0.2).

---

## 13. Multi-DoE on OpenRouter (cloud, paid) — Qwen3-30B R×C

Cloud alternative to section 12: runs `LLM_BACKEND=openrouter` against the live Flask API
(`scripts/run_multi_doe.py`). Provider is pinned to Alibaba (fallback DeepInfra for models Alibaba
does not serve, e.g. Scout aux). Models: `qwen3_30b_instruct` + `qwen3_30b_thinking` (R/C),
`llama_4_scout` (aux/evaluator, on DeepInfra).

### Clean restart with the new code (do this after any code/.env change)
```bash
# 0. one-time: persist the OpenAI SDK dependency (OpenRouter backend needs it)
poetry add openai

# 1. stop everything (backend, frontend, Neo4j)
make dev-stop
pkill -f "python.*src/api_server.py" 2>/dev/null || true   # belt-and-suspenders

# 2. start Neo4j and wait until healthy (citation verification needs it UP)
docker compose up -d neo4j
until docker ps --format '{{.Names}} {{.Status}}' | grep -q 'lexcausa-neo4j.*healthy'; do sleep 2; done
poetry run python src/db/db_orchestrator.py --check         # KB loaded

# 3. start the backend on the OpenRouter backend, logging to a file
LLM_BACKEND=openrouter poetry run python src/api_server.py > /tmp/lexcausa_or.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 1; done
echo "backend up (LLM_BACKEND=openrouter)"
```
`.env` already sets `OPENROUTER_API_KEY`, `OPENROUTER_AUX_MODEL=llama_4_scout`,
`OPENROUTER_REASONING_MAX_TOKENS=6000` (caps thinking tokens so planning-off does not exceed the
60-min client timeout), `CHAIN_MIN_STEPS=2`.

### Smoke test (1 claim, thinking×thinking, planning on+off) — verify before the full run
```bash
poetry run python scripts/run_multi_doe.py --claims-file claims_calib.md \
  --reasoner-models qwen3_30b_thinking --counter-models qwen3_30b_thinking \
  --pairing cross --planning-ablations on,off --replicates 1 \
  --min-kept 8 --max-statutes 100 --max-precedents 5 \
  --out experiments/multi_doe/runs/or_smoke_$(date +%Y%m%d_%H%M%S)
```
Check `metrics.csv`: `status=completed` on both, `counter_abstained=False`, `counter_steps>0`,
`aqa_contra>0`, and `fidelity` populated. C=`qwen3_30b_instruct` legitimately abstains (weak in the
Counter role) — that is a recorded outcome (`counter_abstained=True`), not a bug.

### Full DoE (2 R × 2 C × 2 planning × causality(ON) × 12 claims × 5 reps = 480 runs)
```bash
poetry run python scripts/run_multi_doe.py --claims-file claims_calib.md \
  --reasoner-models qwen3_30b_instruct,qwen3_30b_thinking \
  --counter-models  qwen3_30b_instruct,qwen3_30b_thinking \
  --pairing cross --planning-ablations on,off --replicates 5 \
  --min-kept 8 --max-statutes 100 --max-precedents 5 \
  --out experiments/multi_doe/runs/or_$(date +%Y%m%d_%H%M%S)
```
Budget ~$45-60. Thinking cells are slow (client timeout is 60 min); a full serial run spans hours.
Monitor with `tail -f /tmp/lexcausa_or.log`. Analyze with `scripts/analyze_multi_doe.py` (section 12).

### Reasoning-token control (thinking models only)
```bash
OPENROUTER_REASONING_MAX_TOKENS=6000   # explicit cap (preferred; portable across providers)
OPENROUTER_REASONING_EFFORT=low        # alt.: low|medium|high (ignored if MAX_TOKENS>0)
```
Cutting reasoning too hard makes the thinking Counter regress toward instruct-style abstention;
6000 is a safe middle ground. Both are mutually exclusive (the token cap wins).

---

## 14. Multi-DoE extension — the 10 claims the thesis did not cover (600 runs, 2 machines)

The thesis campaign (`or_doe480_4w_s*` + `or_sc240_s*` = **720 runs**) covered only 12 of the 22
claims: `C1-C3, P1-P3, M1-M3, A1-A3`. This section runs the **same design on the remaining 10**
(`C4-C6, P4-P6, M4, A4-A6`) so the two can be merged into a single **1320-run** dataset spanning
all 22 claims.

Design (identical to the thesis, do not change — the merge depends on it):

| parameter | value |
|---|---|
| claims | `claims_doe10_remaining.md` (10, extracted from `claims.md`) |
| models R x C | `gpt_oss_120b`, `llama_3_3_70b` (cross 2x2) |
| aux / evaluator | `llama_4_scout` (fixed) |
| paradigms | `on,off,single` |
| replicates | 5 |
| seed | 42 |
| retrieval | `--min-kept 8 --max-statutes 100 --max-precedents 5` |
| total | 10 x 2 x 2 x 3 x 5 = **600 runs** |

`claims_doe10_remaining.md` was extracted from `claims.md` preserving the `## CLAIM ...` domain
headers; the claim texts are byte-identical to the originals (verified).

### Running it — split over two machines
Global shard-count 8; each machine runs 4 of those shards locally (one backend per shard, ports
8001-8004). The partition is `df.iloc[i::8]`, disjoint by construction.

```bash
# PC A (macOS) — global shards 0-3, 300 runs
caffeinate -i bash scripts/run_doe600_pcA.sh

# PC B (Linux) — global shards 4-7, 300 runs (no caffeinate on Linux)
systemd-inhibit --what=idle:sleep --why="LexCausa DoE" bash scripts/run_doe600_pcB.sh
```

Both scripts run a **preflight** that aborts unless the matrix is exactly 600 runs, and print a
`CONFIG FINGERPRINT` plus the claims-file sha256 — **these must be identical on both machines**,
otherwise the shard indices no longer line up and you silently get gaps and duplicates. Copy
`claims_doe10_remaining.md` to PC B first (it is a new file).

Expected: **~$11**, **~13 h** wall-clock with both machines (83.8 h of serial work; measured
parallelism is 3.24x per 4-worker machine). Extrapolated from the real per-run token counts and
durations of `or_doe720_merged`.

### Merging everything into the 1320-run dataset
```bash
# after copying PC B's *_g4..g7 directories next to PC A's
poetry run python scripts/merge_doe_shards.py \
    --shards "experiments/multi_doe/runs/or_doe600_pc*_g*" \
             "experiments/multi_doe/runs/or_doe480_4w_s*" \
             "experiments/multi_doe/runs/or_sc240_s*" \
    --out experiments/multi_doe/runs/or_doe1320_all22 --expect 1320
```
The merge aborts on duplicate `run_id` — the tripwire for "the machines built different matrices".

### KNOWN BUG: the `domain` column is wrong for the last claim of every section
`_parse_claims_file` updates `current_domain` as soon as it sees a `## CLAIM ...` header, but the
previous claim is still buffered and is only flushed at the *next* `### ` line — by which time the
domain has already advanced. **The last claim of each section inherits the next section's domain.**

Confirmed in the thesis data: `C3 -> PENALE`, `P3 -> MISTO`, `M3 -> AMMINISTRATIVO` (180 of 720
rows, 25%). The same applies to the new campaign (`C6`, `P6`, `M4`).

`domain` never reaches the pipeline (it is not in the API payload), so **the runs themselves are
valid** — only the per-domain aggregation is affected. Fix it post-hoc before any per-domain
analysis:

```python
df["domain"] = df.claim_id.str[0].map(
    {"C": "CIVILE", "P": "PENALE", "M": "MISTO", "A": "AMMINISTRATIVO"}
)
```
Corrected, every domain has exactly 180 runs in the thesis campaign (against 240/120 as recorded),
which is the balance the design intends — an independent confirmation that the remap is right.
