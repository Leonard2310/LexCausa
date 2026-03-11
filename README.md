# LexCausa: Framework for Causal-Aware Structured Multi-Step Reasoning in Legal Argument Generation

<p align="center">
   <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
   <img src="https://img.shields.io/badge/Neo4j-Knowledge_Graph-4581C3?logo=neo4j&logoColor=white" alt="Neo4j">
   <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React">
   <img src="https://img.shields.io/badge/LLM-Groq_Cloud-F55036?logo=lightning&logoColor=white" alt="Groq">
   <img src="https://img.shields.io/badge/Framework-LangChain-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
   <img src="https://img.shields.io/badge/ASPIC+-Argumentation-8B5CF6" alt="ASPIC+">
   <img src="https://img.shields.io/badge/License-AGPL--3.0-yellow" alt="License">
   <img src="https://img.shields.io/badge/Version-0.10.0-brightgreen" alt="Version">
</p>

> ⚠️ **Work in Progress** - This project is under active development as part of a Master's thesis in Computer Engineering.

**LexCausa** is an AI-powered legal reasoning system for Italian law. It combines Knowledge Graphs (Neo4j), Large Language Models (Groq Cloud), and structured causal reasoning to analyze legal claims, find relevant statutes/precedents, and build logical argumentation chains.



## 🎯 Features

- **Legal Claim Classification**: Automatic claim classification and routing for Civil, Penal, and Administrative law via LLM
- **Domain Router**: Lightweight pre-routing agent that classifies claims as CIVILE, PENALE, AMMINISTRATIVO, or ENTRAMBI
- **Hybrid Statute Retrieval**: Hybrid search on 4200+ Normattiva statutes using Legal-BERT embeddings + Neo4j fulltext (rank fusion), with strict source/libro pre-filtering
- **Citation Graph Expansion (Neo4j CITES)**: Retrieval expands seed statutes with cited statutes via `(:Statute)-[:CITES]->(:Statute)` before LLM relevance/applicability filters
- **Progressive Search**: Adaptive retrieval that progressively expands results when post-filtering yields too few statutes, with configurable expansion steps and max rounds
- **Pre-Retrieval LLM Filtering**: Soft LLM-based relevance filtering for statutes and precedents before they enter the reasoning pipeline (default-YES policy: discards only clearly irrelevant items)
- **Unified Pipeline**: Search, Reasoning, Full Pipeline, and DoE all share the same singleton `LegalSearchPipeline`, ensuring consistency and thread safety
- **Shared Retrieved Context**: Both Reasoner and CounterReasoner receive the same retrieved statutes/precedents and build opposing arguments from the same evidence base
- **Planned Iterative Chain Generation**: Reasoner and Counter-Reasoner first create an execution plan (3-10 steps), then generate one LLM step per planned objective with anti-repetition and consistency checks
- **Reasoner Agent**: Builds structured argumentative chains (Premise → Statute → Precedent → Causal Link → Conclusion) only on the provided knowledge base, with causality classification, precise statute and precedent citations, and a provisional causality bootstrap (plan → taxonomy anchors → enriched KB) before expensive step generation
- **Counter-Reasoner Agent**: Generates counter-arguments using the causality taxonomy, selects multiple attacks from the attack pool, assigns attacks per planned step, applies attack blacklisting/feasibility filtering to stabilize generation, enforces claim-fact lock (no inversion of explicit facts), supports abstention flow when opposition is insufficient, and includes second-pass targeted retrieval and step expansion for multi-attack coverage
- **Repetition Detection**: Jaccard similarity-based detection (threshold 0.70) prevents duplicate reasoning steps across the chain
- **Polisher-Evaluator Agent**: Modular mixin architecture (ConsistencyMixin + ScoringMixin + NLPUtilsMixin + AQAEngineMixin) evaluating the dialectical exchange with consistency checking against Neo4j KB, citation repair, AQA scoring, counter-gate abstention classification, reasoner plausibility locking, and verdict generation
- **Consistency Checker**: Verifies statute and precedent citations against Neo4j KB, classifies articles as core/peripheral, repairs mismatches via LLM-constrained rewriting (with verbatim quote validation), and drops unreliable citations
- **AQA (Argument Quality Assessment)**: Three-dimensional scoring — Cogency (α), NormSupport (β), Semantics (γ) — with configurable weights, active cross-attacks with domain-aware rules, attack-type classification (6 types with per-type damage multipliers), and precedent influence scoring
- **Cross-Attack Computation**: Active domain-aware cross-attack engine with severity categorization, NLI contradiction detection via LLM, attack-type classification (contradiction, exception, derogation, extinction, factual_impediment, general_opposition), and configurable damage multipliers
- **Precedent Influence Scoring**: ASPIC+ links receive precedent delta based on recency, bindingness (cassazione/appello/tribunale), stance confidence, and semantic similarity
- **ASPIC+ Metagraph Visualization**: Interactive SVG frontend component displaying the dialectical meta-graph with PRO/CONTRA columns, curved attack arrows with damage values, chain flow arrows, and detail panel for selected links
- **Attack Text Details**: Expandable frontend panel showing full attacker/target text for each active cross-attack with type, multiplier, NLI label, overlap, and damage
- **Centralized Prompt Registry**: All 100+ LLM prompts managed in a single `prompt_registry.py` with typed `PromptKey` enum — covering classification, routing, filtering, reasoning, counter-reasoning, AQA, NLI, consistency repair, and abstention gate
- **Attack Coverage Scoring**: AQA bonus for counter-argumentation breadth — clusters weak reasoner links into axes, measures how many distinct axes are attacked, and applies diminishing-return weights (1st hit 100%, 2nd 30%, 3rd 10%)
- **Counter-Reasoner Abstention Gate**: Polisher-Evaluator gate (`POLISHER_COUNTER_GATE`) that classifies counter-arguments as `OPPOSING_STRONG`, `OPPOSING_LIMITATIVE`, `AGREEING`, or `UNCLEAR`; weak/agreeing counters trigger abstention, skipping AQA evaluation
- **Reasoner Plausibility Locking**: Optional lock (`aqa_lock_reasoner_plausibility`) that freezes reasoner-side plausibility in A/B DoE tests to isolate counter-reasoner effects
- **Per-Role Model Fallback**: Separate fallback chains for Reasoner and Counter-Reasoner (`reasoner_model_fallback_aliases` / `counter_model_fallback_aliases`) with independent default temperatures
- **Attack Precondition Evaluation**: LLM-based verification of taxonomy preconditions for each attack, with SATISFIED/UNSATISFIED/UNCLEAR status and intra-run caching
- **Counter-Reasoner Second-Pass Retrieval**: Targeted additional retrieval triggered when statutes opposing the claim are insufficient, with configurable thresholds and limits
- **Counter Step Expansion**: Optional multi-attack step expansion that spawns satellite steps when a single counter-step targets multiple attacks
- **Retrieval Fail-Fast Scope**: Thread-local context manager (`retrieval_llm_fail_fast_scope`) for fast fallback in retrieval filters without full retry
- **Resilient Groq Client**: Automatic retry with exponential backoff, dynamic API key discovery (V1..V99), model fallback, model-down cache with configurable TTL; smart error classification (model-down vs. rate-limit vs. transient vs. request-too-large)
- **Caching & Filtering Efficiency**: Intra-run caching for legal-context extraction, statute applicability decisions, attack preconditions, fact-lock checks, and plan target alignment, plus claim-context SQLite cache for reusing pre-retrieval outputs across repeated runs
- **Cancellation & Interruptibility**: Pipeline stop endpoint and cooperative cancellation propagation across API, agents, and long-running generation/retrieval loops
- **DoE A/B Workflow**: Dedicated DoE tab with one shared Reasoner run and two Counter/Evaluator setups (baseline vs. treatment), live A/B switching in the UI, consolidated DoE log/report persistence, and automatic delta analysis
- **DoE Batch Runner**: Automated multi-claim batch execution (`run_doe_batch.py`) with resume capability (`--resume`), checkpoint recovery (`--start-from`), selective runs (`--only`), dry-run mode, abstention classification, and stability metrics
- **DoE Statistical Analysis**: Post-hoc analysis script (`scripts/analyze_doe_results.py`) computing paired t-tests, sign tests, Cohen's d, domain breakdowns, verdict flips, error analysis, and intra-claim consistency from batch CSV results
- **Causality Taxonomy**: Structured causality taxonomy (Material, Legal, Concurrent) used by Reasoner and Counter-Reasoner for arguments and attacks
- **Knowledge Graph**: Neo4j database with statutes, precedents, and causal relationships
- **Centralized Configuration**: All parameters (150+ settings: models, retries, AQA weights, search, truncation, attack params, coverage, abstention, per-role fallback, domain-specific weights, etc.) managed by `src/config.py` (Pydantic Settings) and environment variables
- **Frontend Settings Panel**: Collapsible panel to configure per-step LLM model, temperature, max tokens, search parameters, AQA weights, chain min/max steps, and attack parameters — without touching code
- **Per-Claim Pipeline Logging**: Every pipeline run is logged to `logs/<timestamp>_<slug>.log` for full auditability
- **DoE and PDF Artifacts**: DoE runs can persist a consolidated A/B log + JSON report, and exported PDFs are saved automatically under `logs/pdf_exports/<pipeline|doe>/` in addition to browser download
- **Pre-Retrieval Claim Context Memory (SQLite)**: Optional cache of final applicable statutes and precedents per claim (reusable across Search/Reasoner/Counter/Pipeline runs and warmable via script)
- **React Frontend**: Modern four-tab interface (Search, Reasoning, Full Pipeline, DoE) with ASPIC+ Metagraph visualization, attack details, PDF export, and A/B comparison controls on Vite + React 18
- **Live Pipeline Streaming**: Real-time phase progress, token streaming for chain generation (including retry attempts), refinement phase control events (`reasoner_refinement_started/completed`) with live chain reset/replace behavior in the frontend, plus SSE endpoints for full pipeline, standalone Counter-Reasoner, and standalone Evaluator



## 🏗️ Agent and Pipeline Architecture

<p align="center">
   <img src="assets/LexCausa_architecture.svg" alt="LexCausa Architecture" width="100%">
</p>

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Frontend (React + Vite)                           │
│ Search │ Reasoning │ Full Pipeline │ DoE (A/B) │ ⚙️ Settings Panel      │
│      + ASPIC+ Metagraph SVG + Attack Details + PDF Export               │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Flask API Server (:8000)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  GET  /health              → Health check with API version              │
│  GET  /api/settings        → Defaults & available models                │
│  POST /api/chat            → LegalSearchPipeline (unified retrieval)    │
│  POST /api/reason          → Reasoner (iterative chain generation)      │
│  POST /api/counter_reason  → Counter-Reasoner (iterative counter-chain) │
│  POST /api/counter_reason/stream → Counter-Reasoner SSE                 │
│  POST /api/pipeline        → Full Pipeline (Router→Reasoner→Counter→AQA)│
│  POST /api/pipeline/stream → Full Pipeline SSE token streaming          │
│  POST /api/pipeline/stop   → Stop active SSE pipeline run               │
│  POST /api/evaluate        → Polisher-Evaluator (standalone evaluation) │
│  POST /api/evaluate/stream → Polisher-Evaluator SSE                     │
│  POST /api/doe/log         → Consolidated DoE log/report persistence    │
│  POST /api/pdf/export      → Persist exported PDF artifacts             │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────┐
│   Resilient Groq Client      │  │  LegalSearchPipeline                 │
│   (groq_client.py)           │  │  (Singleton, thread-safe)            │
├──────────────────────────────┤  ├──────────────────────────────────────┤
│  Dynamic key discovery (V1…N)│  │  ClaimClassifier → book routing      │
│  Model fallback + down cache │  │  Hybrid retrieval (vector + fulltext)│
│  Smart error classification  │  │  Progressive + CITES expansion       │
│  Exponential backoff         │  │  Pre-retrieval LLM filtering         │
│                              │  │  Shared Retrieved Context            │
└──────────────────────────────┘  └────────────────┬─────────────────────┘
                    │                              │
                    │         ┌────────────────────┘
                    ▼         ▼ statutes + precedents
┌─────────────────────────────────────────────────────────────────────────┐
│              Reasoner / Counter-Reasoner / Polisher-Evaluator           │
├─────────────────────────────────────────────────────────────────────────┤
│  Reasoner: iterative primary chain on shared retrieved context (ASPIC+) │
│  Counter-Reasoner: iterative attack chain on shared retrieved context   │
│  Polisher-Evaluator (4 Mixins):                                         │
│    ├─ ConsistencyChecker: KB verification → citation repair/drop        │
│    ├─ ScoringMixin: readability, coherence, argument quality            │
│    ├─ NLPUtilsMixin: Flesch/FOG/SMOG, NLI via LLM                       │
│    └─ AQAEngine: Cogency(α)+NormSupport(β)+Semantics(γ)→verdict         │
│         └─ Cross-attacks (6 types) + precedent influence → net_plaus.   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Neo4j Knowledge Base + Taxonomy                       │
├─────────────────────────────────────────────────────────────────────────┤
│  📚 Italian statutes KB (Normattiva): Civil + Penal + Administrative    │
│     (L. 7 agosto 1990, n. 241)                                          │
│  ⚖️  9112 precedent chunks from 792 rulings (ITA-CaseHold)              │
│  📊 768-dim Vector Index (Legal-BERT) on statutes                       │
│  🔗 Statute citation edges: `CITES` from normalized reference fields    │
│  🔗 Causality taxonomy (Material, Legal, Concurrent)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.11.x or 3.12.x
- **Docker**: For running Neo4j
- **Node.js**: 18+ (for frontend)
- **Poetry**: Python dependency management
- **Groq API Key**: Free tier available at [console.groq.com](https://console.groq.com)

### Supported Platforms
- ✅ macOS (Apple Silicon M1/M2/M3)
- ✅ macOS (Intel)
- ✅ Windows 10/11 (x64)
- ✅ Linux (x64)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LexCausa.git
cd LexCausa
```

### 2. Install Python Dependencies

**macOS/Linux:**
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --no-root
```

**Windows (PowerShell):**
```powershell
# Install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install dependencies
poetry install --no-root
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Groq Cloud API (up to 3 keys for rotation)
GROQ_API_KEY_V1=your_groq_api_key_here
GROQ_API_KEY_V2=your_second_key_here
...
GROQ_API_KEY_VN=your_third_key_here

# API Server
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

### 4. Start Neo4j Database

```bash
docker compose up -d
```

Wait for Neo4j to start (check at http://localhost:7474).

### 5. Initialize the Knowledge Base

```bash
poetry run python src/db/db_orchestrator.py
```

This will:
- Create schema (indexes, constraints, graph structure)
- Load Civil Code, Penal Code, and Administrative Law (L. 241/1990) from `*_normattiva.csv` with embeddings
- Build statute citation edges `(:Statute)-[:CITES]->(:Statute)` from normalized internal references
- Load ITA-CaseHold precedents metadata (no embeddings)
- Wait for indexes to come online (vector for statutes, fulltext for precedents)

Use `--clean` for a full wipe and reload, or `--check` to inspect database status.

### 6. Install Frontend Dependencies

```bash
cd src/frontend
npm install
```

### 7. Run the Application

```bash
# Terminal 1: Start the API server
poetry run python src/api_server.py

# Terminal 2: Start the frontend dev server
cd src/frontend && npm run dev
```

The frontend will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## 📁 Project Structure

```
LexCausa/
├── src/
│   ├── config.py                  # Centralized configuration (150+ Pydantic Settings)
│   ├── api_server.py              # Flask API server (DoE, SSE, PDF export endpoints)
│   ├── agents/                    # LLM agents with explicit orchestration
│   │   ├── base.py               # Base agent class + progressive search + filters
│   │   ├── router.py             # Domain router (CIVILE/PENALE/ENTRAMBI)
│   │   ├── reasoner.py           # Iterative reasoning agent (ASPIC+)
│   │   ├── counter_reasoner.py   # Iterative counter-argumentation agent (ASPIC+)
│   │   ├── polisher_evaluator.py # Mixin compositor (Consistency+Scoring+NLP+AQA)
│   │   ├── aspic_formatter.py    # ASPIC+ IR formatting
│   │   ├── evaluation/           # Evaluation sub-package (modular mixins)
│   │   │   ├── models.py         # Dataclasses (CitationCheck, ConsistencyReport, etc.)
│   │   │   ├── consistency_checker.py  # KB verification, citation repair, chain regen
│   │   │   ├── aqa_engine.py     # AQA pipeline, cross-attacks, precedent influence
│   │   │   ├── scoring.py        # Readability, coherence, argument quality scoring
│   │   │   └── nlp_utils.py      # Flesch/FOG/SMOG, NLI via LLM, text utilities
│   │   └── tools/                # Agent tools
│   │       ├── neo4j_tools.py    # Neo4j hybrid search pipeline
│   │       ├── prompt_registry.py # Centralized prompt registry (100+ PromptKey enum)
│   │       ├── taxonomy_tools.py # Causality taxonomy
│   │       ├── config_loader.py  # Taxonomy config loader
│   │       └── config_taxonomy.json
│   ├── services/                  # Core services
│   │   ├── groq_client.py        # Resilient Groq client (dynamic key discovery, rotation)
│   │   ├── claim_classifier.py   # LLM claim classification
│   │   ├── claim_context_memory.py # SQLite pre-retrieval claim context cache
│   │   ├── pipeline_control.py   # Cooperative cancellation primitives/exceptions
│   │   └── legal_search.py       # Hybrid legal search pipeline (vector + fulltext fusion)
│   ├── db/                        # Database management
│   │   ├── db_orchestrator.py    # Full DB lifecycle (clean/schema/load/verify)
│   │   └── data_loader.py        # Centralized data loading (CSV/parquet + statute embeddings)
│   ├── data/                      # Data files
│   │   ├── embeddings/           # Pre-computed embeddings (.npy)
│   │   ├── precedents/           # ITA-CaseHold precedents (parquet)
│   │   └── statutes/              # Civil + Penal + Administrative CSVs (`*_normattiva.csv`)
│   └── frontend/                  # React frontend (Vite + React 18)
│       └── src/
│           ├── App.jsx            # Main app with Search/Reasoning/Pipeline/DoE tabs
│           ├── AspicMetagraph.jsx # ASPIC+ meta-graph SVG visualization
│           └── AttackTextDetails.jsx # Cross-attack detail panel
├── scripts/                       # Utility scripts
│   ├── analyze_doe_results.py    # Statistical analysis of DoE A/B batch results
│   ├── capture_api_chat_retrieval_memory.py  # Claim context memory warmup
│   ├── start_public_demo.sh      # Cloudflare tunnel public demo launcher
│   ├── tune_aqa_real_plus_synth.py  # AQA tuning (real + synthetic)
│   ├── tune_aqa_with_gold_dataset.py  # AQA tuning (gold dataset)
│   └── tune_retrieval_claims.py  # Supervised retrieval tuning
├── experiments/                   # DoE experiments
│   └── doe/
│       ├── doe_settings.json     # DoE configuration
│       └── scripts/
│           └── run_doe_batch.py   # Automated DoE batch runner (resume, checkpoint, dry-run)
├── notebooks/                     # Normattiva extractors + embeddings notebooks
├── logs/                          # Pipeline/DoE logs, reports, AQA artifacts, exported PDFs
├── compose.yml                    # Docker Compose for Neo4j
├── pyproject.toml                 # Poetry configuration
└── README.md
```

## 🔧 Configuration

All configuration is managed through environment variables and the `src/config.py` Settings class (150+ parameters total).
Runtime-tunable settings (model, temperature, max tokens, search parameters, AQA weights, chain steps, attack parameters) can also be adjusted from the **frontend Settings panel** without restarting the server.
To keep this README stable across tuning changes, defaults are intentionally **not duplicated** here.

### Required (`.env`)

These variables **must** be set in the `.env` file:

| Variable | Description |
|----------|-------------|
| `NEO4J_URI` | Neo4j connection URI |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `GROQ_API_KEY_V1` | Primary Groq API key |
| `GROQ_API_KEY_V2…VN` | Additional Groq API keys (dynamic discovery V1…V99) |

### Optional — LLM & Server

These can be overridden in `.env` or via the frontend Settings panel:

- Groq client behavior (retry/backoff, key rotation, model-down cache TTL)
- LLM generation knobs (temperature, max tokens)
- API runtime settings (host/port/debug)

Model catalog and aliases (`groq_llama_scout_17b`, `groq_llama_maverick_17b`, `gpt_oss_120b`, `gpt_oss_20b`, `qwen_qwen3_32b`, `groq_llama_3_3_70b_versatile`) are code-defined in `src/config.py`.

### Optional — Embedding & Search

Configurable areas:

- Embedding model and tokenization limits
- Search breadth (`top_k`, top classified libri, precedent limits)
- Hybrid retrieval tuning (vector/fulltext weights, candidate pool sizes, priority decay, keyword bonus)
- Domain-specific hybrid search weights (`search_vector_weight_civile`, `search_fulltext_weight_penale`, etc.) and keyword bonus thresholds per domain
- Citation expansion tuning (`SEARCH_CITES_ENABLED`, per-seed limit, max added, score decay, multi-seed bonus)
- Progressive search thresholds/steps for expansion rounds

Pre-retrieval claim context memory (SQLite):

- Optional per-claim cache for the **final applicable** statutes and precedents produced by `prepare_claim_context(...)`
- Reused by `/api/chat`, `/api/reason`, `/api/counter_reason`, and full pipeline endpoints when enabled
- Can be toggled from the frontend Pipeline input (Use memory / Overwrite memory)
- Can be pre-populated in batch using `scripts/capture_api_chat_retrieval_memory.py --claim-context-memory`

Debug support:

- Existing backend/pipeline logs print, for each retrieved statute:
  `vector_rank_score`, `fulltext_rank_score`, `fusion_score`, `keyword_bonus`, `priority_multiplier`.

### Supervised Retrieval Tuning (Claims Gold Labels)

The script `scripts/tune_retrieval_claims.py` supports supervised tuning using
gold labels in `claims_gold_labels.json`.

Query-term extraction for the fulltext branch is LLM-only (no salient fallback).

Metrics used:
- `Hit@5`, `Hit@10`
- `MRR`
- `nDCG@10`

Run examples:

```bash
# supervised (default if claims_gold_labels.json is present)
poetry run python scripts/tune_retrieval_claims.py --top-k 30

# force proxy-only fallback (unsupervised)
poetry run python scripts/tune_retrieval_claims.py --unsupervised

# fulltext query terms are LLM-extracted (LLM-only mode)
poetry run python scripts/tune_retrieval_claims.py --query-terms-mode llm

# disable progress bars (CI/log-friendly)
poetry run python scripts/tune_retrieval_claims.py --query-terms-mode llm --no-progress
```

Output report:
- `logs/tuning/retrieval_tuning_<timestamp>.json`
- `logs/tuning/retrieval_tuning_latest.json`

### Optional — AQA (Argument Quality Assessment)

Configurable areas:

- AQA enable/disable and weight triplet (α, β, γ)
- Verdict thresholds and top-K attack retention
- Norm support scoring strategy
- Attack gating/tuning (semantic overlap, strength ratio, damage factor, cross-codice flags)
- Attack coverage scoring (enabled flag, similarity/overlap thresholds, min attack value, bonus weight/max/diminishing weights)
- Reasoner plausibility locking for DoE A/B isolation (`aqa_lock_reasoner_plausibility`)
- Precedent recency window and dominant-attack reporting limits

### Optional — Chain Generation

Configurable areas:

- Retry and robustness controls for chain generation
- Planned step bounds (min/max)
- Counter-Reasoner second-pass retrieval (enabled flag, thresholds, limits)
- Counter step expansion (enabled flag, min attacks, satellite limits)
- Model-down cache behavior used by the resilient Groq client

For the authoritative, always-updated list of settings, env aliases, and descriptions, see `src/config.py`.


## 🌐 Public Demo (Cloudflare Tunnel)

For a quick public URL without deploying to a cloud server, use `scripts/start_public_demo.sh`.

### Prerequisites

Install `cloudflared` once:

```bash
brew install cloudflared
```

### Run one demo instance

```bash
bash scripts/start_public_demo.sh
```

The script starts:
- Flask API (`src/api_server.py`) on `127.0.0.1:8000`
- Vite frontend on `127.0.0.1:3000`
- Cloudflare Quick Tunnel

It prints a temporary public URL like `https://...trycloudflare.com`.

### Run multiple isolated instances

```bash
# Instance A
API_PORT=8000 FRONTEND_PORT=3000 INSTANCE_NAME=you bash scripts/start_public_demo.sh

# Instance B (different terminal)
API_PORT=8001 FRONTEND_PORT=3001 INSTANCE_NAME=colleague bash scripts/start_public_demo.sh
```

Each terminal gets its own `trycloudflare.com` URL.

### Script arguments

```bash
bash scripts/start_public_demo.sh --instance colleague --api-port 8001 --frontend-port 3001 --host 127.0.0.1
```

### Stop

Press `Ctrl+C` in the tunnel terminal.  
The script stops backend + frontend processes for that instance.

### Troubleshooting

`Port XXXX is already in use`:

```bash
pkill -f "python.*src/api_server.py" || true
pkill -f "node .*vite" || true
pkill -f "cloudflared tunnel --url" || true
```

`Blocked request. This host (...) is not allowed`:
- The script already handles this by setting `--http-host-header`.
- If you run components manually, use:

```bash
cloudflared tunnel --url http://127.0.0.1:3000 --http-host-header 127.0.0.1:3000
```


## 🧪 Agent & Pipeline Development Status

### ✅ Completed
- [x] Neo4j Knowledge Base with Civil/Penal/Administrative statutes
- [x] Normattiva-based statute pipeline (`*_normattiva.csv` + `*_normattiva_embeddings.npy`)
- [x] Hybrid statute retrieval: exact cosine search (source/libro pre-filtered) + fulltext rank fusion
- [x] Neo4j `CITES` graph: citation edges built from statute references and used in pre-filter retrieval expansion
- [x] Claim classification and book routing via LLM
- [x] Domain Router Agent (CIVILE / PENALE / ENTRAMBI)
- [x] Unified, thread-safe, configurable LegalSearchPipeline (allow-list, soft-filtering)
- [x] Progressive search with adaptive expansion when post-filtering yields too few results
- [x] Pre-retrieval LLM filtering for statutes and precedents (default-YES soft filter)
- [x] Shared context retrieval for both Reasoner and CounterReasoner
- [x] Reasoner Agent: plan-then-execute generation (ASPIC+) with 3-10 planned steps and anti-repetition validation
- [x] Counter-Reasoner Agent: plan-then-execute counter-generation (ASPIC+) with multi-attack selection, per-step attack assignment, attack blacklist, and feasibility filtering for more stable counter plans
- [x] CounterReasoner second-pass targeted retrieval with improved filtering discipline and attack-coverage-based activation
- [x] Taxonomy anchor filtering with claim-aware applicability (soft core / hard accessory) for Reasoner and CounterReasoner
- [x] BaseAgent caching for legal context extraction and statute applicability decisions (reduces duplicate LLM calls and improves consistency)
- [x] Explicit precedent citation by full title in Reasoner and Counter-Reasoner prompts
- [x] Repetition detection in reasoning steps (Jaccard similarity, threshold 0.70)
- [x] Polisher-Evaluator Agent: modular mixin architecture (ConsistencyMixin + ScoringMixin + NLPUtilsMixin + AQAEngineMixin)
- [x] Consistency Checker: KB verification for statutes and precedents, core/peripheral classification, LLM-constrained repair with verbatim quote validation, citation drop
- [x] AQA (Argument Quality Assessment): Cogency + NormSupport + Semantics scoring with configurable weights
- [x] Cross-attack computation: domain-aware rules, severity categorization, NLI contradiction detection via LLM, 6 attack types (contradiction, exception, derogation, extinction, factual_impediment, general_opposition) with per-type damage multipliers
- [x] Precedent influence scoring: recency, bindingness (cassazione/appello/tribunale), stance confidence, semantic similarity
- [x] Verdict generation (plausible / implausible / uncertain) with winning_side and confidence mapping
- [x] ASPIC+ Metagraph SVG visualization: interactive PRO/CONTRA graph with attack arrows, chain flow, and detail panel
- [x] Attack Text Details: expandable panel with full attacker/target text, type, multiplier, NLI label, overlap, damage
- [x] Resilient Groq Client: retry + dynamic API key discovery (V1…V99) + model fallback + model-down cache with TTL
- [x] Smart error classification: model-down (503) vs. rate-limit (429) vs. transient errors
- [x] Cooperative cancellation across API, agents, and pipeline services + `/api/pipeline/stop` endpoint for interrupting active SSE runs
- [x] NLI contradiction detection via LLM (replaces local DeBERTa model)
- [x] ASPIC+ IR formatting for Reasoner and Counter-Reasoner
- [x] Prescriptive prompts and structured output for all agents
- [x] Centralized configuration via Pydantic Settings (150+ parameters including truncation, attack, chain, coverage, abstention, per-role fallback, domain-specific weights)
- [x] Frontend Settings Panel: per-step model selection, temperature, max tokens, search params, AQA weights, chain min/max steps, attack parameters
- [x] Per-claim pipeline logging (`logs/` directory)
- [x] Optional pre-retrieval claim context memory (SQLite) with frontend toggles and script-based warmup
- [x] Reasoner provisional causality bootstrap (plan draft → provisional classification → taxonomy anchors → enriched KB before step generation)
- [x] Reasoner post-hoc anchor refinement fallback with streaming control events and frontend live reset/replace of the support chain
- [x] React frontend (Vite) with four tabs + ASPIC+ Metagraph + Attack Details + settings panel
- [x] DoE A/B workflow with shared Reasoner, baseline/treatment Counter setups, consolidated log/report persistence, and comparative frontend analysis
- [x] SSE endpoints for standalone Counter-Reasoner and standalone Evaluator
- [x] Full PDF export for Pipeline/DoE results, including automatic artifact persistence under `logs/pdf_exports/`
- [x] Precedent ingestion and fulltext search (ITA-CaseHold)
- [x] Centralized data loading module (`data_loader.py`) with path resolution via Settings
- [x] Improving precedent utilization in reasoning chains (better citation coverage, richer contextual integration)
- [x] Supervised Tuning Domain-specific retrieval parameters (metodology weights, overlap multipliers and counts)
- [x] Supervised Tuning attack evaluation parameters (damage multipliers, strength ratios, severity thresholds) for more balanced verdicts
- [x] Centralized Prompt Registry (`prompt_registry.py`) with 100+ typed `PromptKey` enum entries covering all agent prompts
- [x] Attack coverage scoring (AQA bonus for counter-argumentation breadth across weak-point axes)
- [x] Counter-Reasoner abstention gate with label classification (OPPOSING_STRONG, OPPOSING_LIMITATIVE, AGREEING, UNCLEAR)
- [x] Reasoner plausibility locking for DoE A/B isolation
- [x] Per-role model fallback chains (separate Reasoner / Counter-Reasoner fallback aliases and temperatures)
- [x] Attack precondition evaluation with taxonomy-aware LLM verification and caching
- [x] Counter-Reasoner second-pass targeted retrieval for insufficient opposing statutes
- [x] Counter step expansion for multi-attack steps
- [x] Retrieval fail-fast scope (thread-local context manager for fast filter fallback)
- [x] DoE batch runner with resume, checkpoint recovery, selective runs, dry-run, abstention tracking, and stability metrics
- [x] DoE statistical analysis script (paired t-test, sign test, Cohen's d, domain breakdowns, verdict flips, error analysis)
- [x] Domain-specific hybrid search weights (per-domain vector/fulltext weight pairs and keyword bonus thresholds)
- [x] ASPIC+ repair and AQA report artifact persistence under `logs/aspic_repairs/` and `logs/aqa_reports/`


### 🚧 In Progress
- [ ] Export reasoning chains to structured formats (JSON-LD, RDF)
- [ ] Extended claim-level caching: persist additional pipeline artifacts (e.g., classification/stance/evaluation outputs) beyond pre-retrieval statutes/precedents
- [ ] Explainability layer: generate a final, user-facing explanation of the reasoning and verdict (with traceable links to retrieved statutes/precedents and attack outcomes)

### 📋 Planned
- [ ] Extend statutory/procedural coverage with additional corpora: c.p.a., c.p.p., c.p.c., labour law corpus, health liability corpus, consumer law corpus, military disciplinary corpus, and industrial property corpus
- [ ] LLM memory layer: maintain conversational context across pipeline steps to improve coherence and reduce redundant reasoning
- [ ] Full argumentation framework (Dung-style grounded semantics visualization)
- [ ] Multi-turn dialogue with context retention
- [ ] Multi-jurisdiction support: extend statutory coverage to additional countries and decouple the framework from Italy-specific statutes (pluggable statute loaders + jurisdiction configs)


## 📄 License

**Open Source**: This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).  
See the [LICENSE](LICENSE) file for details.

### Commercial licensing

A commercial license is available for organizations that want to use LexCausa in proprietary and/or commercial products **without** AGPL-3.0 obligations.  
See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for details and contacts.


## 📚 References

- **Normattiva (statutes source)**: All statutes used in LexCausa (Civil Code, Penal Code, Administrative Law 241/1990) are extracted from Normattiva and stored in `*_normattiva.csv`. Official portal: [https://www.normattiva.it](https://www.normattiva.it).
- **ITA-CaseHold (precedents source)**: The ITA-CaseHold dataset, used for legal precedent extraction and summarization in this project, was introduced by Licari et al. at ICAIL 2023. Publication: [https://doi.org/10.1145/3594536.3595177](https://doi.org/10.1145/3594536.3595177).

## 👤 Authors

**Leonardo Catello**  
Master's Thesis in Computer Engineering  
Email: leonardo.catello@hotmail.com

**Salvatore Maione**  
Master's Thesis in Computer Engineering  
Email: salvatore22maione@gmail.com


## 🧾 Citation

If you use LexCausa in academic work, please cite the repository.  
See [CITATION.cff](CITATION.cff).
---

*This project is part of a Master's thesis and is not intended for production legal use.*
