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
      "counter_pass_causal_identity": true,
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
      "description": "Counter tassonomia ON (norms)",
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
      "description": "Counter tassonomia ON (identity + attacks + norms)",
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
  -F "prefix=doe_automatico" \
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
      "llm_max_tokens": 8192,
      "search_min_kept_statutes": 8,
      "search_use_top_n_libri": 3,
      "chain_min_steps": 3,
      "chain_max_steps": 10,
      "reasoner_enable_causality": true,
      "counter_enable_causality": true,
      "counter_pass_causal_identity": true,
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
