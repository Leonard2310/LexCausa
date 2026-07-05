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

## 12. Multi-DoE full-factorial Reasoner×Counter (Ibisco / Cerberus llama.cpp) — THESIS RUN

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

### Serve the models with Cerberus (once, from the project dir)
Models run under Cerberus (llama.cpp), not in-process. The repo-root `models.conf`
serves three labels — `oss120` (gpt-oss-120b), `llama33` (Llama-3.3-70B-Instruct),
`assistant` (Llama-4-Scout-17B) — which the LexCausa aliases (`gpt_oss_120b`,
`llama_3_3_70b_instruct`, `llama_4_scout_17b`) resolve to via `CERBERUS_ALIAS_MAP`
in `src/config.py`. Then:
```bash
cerberus validate            # schema + how many nodes to allocate
cerberus download            # fetch the exact GGUFs (login node; HF_TOKEN for gated repos)
# allocate GPUs (salloc ...) then, from the project dir:
cerberus up                  # serves all models; writes endpoints.json
cerberus status              # UP/DOWN
```
See `Cerberus/docs/progetto.md` for the full workflow (models.conf fields, sizing,
batch `run.sbatch`). If the DoE runs from a different cwd than `models.conf`, point
the client at the map: `export CERBERUS_ENDPOINTS=/path/project/endpoints.json`.

### Environment
```bash
# Neo4j (if not already in a .env):
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=<password>
# Notes: the script forces LLM_BACKEND=local (→ Cerberus). All inference goes through
# the served endpoints; no HF token or GPU env vars are needed at run time (models are
# already up). HF_TOKEN is only needed earlier, by `cerberus download`, for gated repos.
```

### Run the reduced DoE (2 models vary as Reasoner × Counter)
The script defaults already encode the reduced, resource-constrained design, so the
run below is equivalent to calling it with no model flags:
```bash
python scripts/run_multi_doe_ibisco.py \
  --fixed-model llama_4_scout_17b \
  --evaluator-model llama_4_scout_17b \
  --reasoner-models gpt_oss_120b,llama_3_3_70b_instruct \
  --counter-models  gpt_oss_120b,llama_3_3_70b_instruct \
  --planning-ablations on \
  --causality-ablations on \
  --replicates 1 --seed 42 \
  --out experiments/multi_doe/runs/ibisco_$(date +%Y%m%d_%H%M%S)
```
`assistant` (Llama-4-Scout) is the fixed support/evaluator model and does **not** vary
in the DoE — only `oss120` and `llama33` cross as Reasoner × Counter (2×2 cells).
`metrics.csv` is written incrementally in `--out/`.

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
- **`--fixed-model llama_4_scout_17b`** pins retrieval/AQA/evaluator to Llama-4-Scout
  (`assistant`); it is not a DoE factor. It is the default, so it can be omitted.
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
