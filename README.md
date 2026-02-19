# LexCausa: Framework for Causal-Aware Structured Multi-Step Reasoning in Legal Argument Generation

<p align="center">
   <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
   <img src="https://img.shields.io/badge/Neo4j-Knowledge_Graph-4581C3?logo=neo4j&logoColor=white" alt="Neo4j">
   <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React">
   <img src="https://img.shields.io/badge/LLM-Groq_Cloud-F55036?logo=lightning&logoColor=white" alt="Groq">
   <img src="https://img.shields.io/badge/Framework-LangChain-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
   <img src="https://img.shields.io/badge/ASPIC+-Argumentation-8B5CF6" alt="ASPIC+">
   <img src="https://img.shields.io/badge/License-AGPL--3.0-yellow" alt="License">
   <img src="https://img.shields.io/badge/Version-0.9.0-brightgreen" alt="Version">
</p>

> ⚠️ **Work in Progress** - This project is under active development as part of a Master's thesis in Computer Engineering.

**LexCausa** is an AI-powered legal reasoning system for Italian law. It combines Knowledge Graphs (Neo4j), Large Language Models (Groq Cloud), and structured causal reasoning to analyze legal claims, find relevant statutes/precedents, and build logical argumentation chains.



## 🎯 Features

- **Legal Claim Classification**: Automatic claim classification and routing for Civil, Penal, and Administrative law via LLM
- **Domain Router**: Lightweight pre-routing agent that classifies claims as CIVILE, PENALE, AMMINISTRATIVO, or ENTRAMBI
- **Semantic Search**: Vector search on 3900+ articles using Legal-BERT, with unified and configurable pipeline
- **Progressive Search**: Adaptive retrieval that progressively expands results when post-filtering yields too few statutes, with configurable expansion steps and max rounds
- **Pre-Retrieval LLM Filtering**: Soft LLM-based relevance filtering for statutes and precedents before they enter the reasoning pipeline (default-YES policy: discards only clearly irrelevant items)
- **Unified Pipeline**: All functionalities (Search, Reasoning, Full Pipeline) share the same singleton `LegalSearchPipeline`, ensuring consistency and thread safety
- **Stance Classifier (NLI)**: Classifies statutes and precedents as SUPPORT, AGAINST, or NEUTRAL relative to the claim using NLI-style prompting
- **Iterative Reasoning Chain Generation**: Reasoner and Counter-Reasoner build chains step-by-step with dedicated LLM calls; the LLM autonomously decides when to conclude, bounded by configurable min/max step limits and a separate EVALUATION phase
- **Reasoner Agent**: Builds structured argumentative chains (Premise → Statute → Precedent → Causal Link → Conclusion) only on the provided knowledge base, with causality classification, precise statute and precedent citations
- **Counter-Reasoner Agent**: Generates independent counter-arguments using the causality taxonomy, identifying attacking causalities and building attack reasoning chains with explicit precedent citation
- **Repetition Detection**: Jaccard similarity-based detection (threshold 0.70) prevents duplicate reasoning steps across the chain
- **Polisher-Evaluator Agent**: Modular mixin architecture (ConsistencyMixin + ScoringMixin + NLPUtilsMixin + AQAEngineMixin) evaluating the dialectical exchange with consistency checking against Neo4j KB, citation repair, AQA scoring, and verdict generation
- **Consistency Checker**: Verifies statute and precedent citations against Neo4j KB, classifies articles as core/peripheral, repairs mismatches via LLM-constrained rewriting (with verbatim quote validation), and drops unreliable citations
- **AQA (Argument Quality Assessment)**: Three-dimensional scoring — Cogency (α), NormSupport (β), Semantics (γ) — with configurable weights, active cross-attacks with domain-aware rules, attack-type classification (6 types with per-type damage multipliers), and precedent influence scoring
- **Cross-Attack Computation**: Active domain-aware cross-attack engine with severity categorization, NLI contradiction detection via LLM, attack-type classification (contradiction, exception, derogation, extinction, factual_impediment, general_opposition), and configurable damage multipliers
- **Precedent Influence Scoring**: ASPIC+ links receive precedent delta based on recency, bindingness (cassazione/appello/tribunale), stance confidence, and semantic similarity
- **ASPIC+ Metagraph Visualization**: Interactive SVG frontend component displaying the dialectical meta-graph with PRO/CONTRA columns, curved attack arrows with damage values, chain flow arrows, and detail panel for selected links
- **Attack Text Details**: Expandable frontend panel showing full attacker/target text for each active cross-attack with type, multiplier, NLI label, overlap, and damage
- **Resilient Groq Client**: Automatic retry with exponential backoff, dynamic API key discovery (V1..V99), model fallback, model-down cache with configurable TTL; smart error classification (model-down vs. rate-limit vs. transient)
- **Causality Taxonomy**: Structured causality taxonomy (Material, Legal, Concurrent) used by Reasoner and Counter-Reasoner for arguments and attacks
- **Knowledge Graph**: Neo4j database with statutes, precedents, and causal relationships
- **Centralized Configuration**: All parameters (90+ settings: models, retries, AQA weights, search, truncation, attack params, etc.) managed by `src/config.py` (Pydantic Settings) and environment variables
- **Frontend Settings Panel**: Collapsible panel to configure per-step LLM model, temperature, max tokens, search parameters, AQA weights, chain min/max steps, and attack parameters — without touching code
- **Per-Claim Pipeline Logging**: Every pipeline run is logged to `logs/<timestamp>_<slug>.log` for full auditability
- **React Frontend**: Modern three-tab interface (Search, Reasoning, Full Pipeline) with ASPIC+ Metagraph visualization on Vite + React 18
- **Live Pipeline Streaming**: Real-time phase progress, token streaming for chain generation, and live evaluation/AQA status updates



## 🏗️ Agent and Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Frontend (React + Vite)                           │
│   Search Tab │ Reasoning Tab │ Full Pipeline Tab │ ⚙️ Settings Panel    │
│                    + ASPIC+ Metagraph SVG + Attack Details              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Flask API Server (:8000)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  GET  /api/settings  → defaults & available models                      │
│  POST /api/chat      → LegalSearchPipeline (unified retrieval)          │
│  POST /api/reason    → Reasoner (iterative chain generation)            │
│  POST /api/counter   → Counter-Reasoner (iterative counter-chain)       │
│  POST /api/pipeline  → Full Pipeline (Router→Reasoner→Counter→AQA)      │
│  POST /api/evaluate  → Polisher-Evaluator (standalone evaluation)       │
│  GET  /api/health    → Health check with API version                    │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────┐
│   Resilient Groq Client      │  │  LegalSearchPipeline                 │
│   (groq_client.py)           │  │  (Singleton, thread-safe)            │
├──────────────────────────────┤  ├──────────────────────────────────────┤
│  Dynamic key discovery (V1…N)│  │  ClaimClassifier → book routing      │
│  Model fallback + down cache │  │  Legal-BERT → 768-dim vectors        │
│  Smart error classification  │  │  Progressive search + expansion      │
│  Exponential backoff         │  │  Pre-retrieval LLM filtering         │
│                              │  │  StanceClassifier (NLI)              │
└──────────────────────────────┘  └────────────────┬─────────────────────┘
                    │                              │
                    │         ┌────────────────────┘
                    ▼         ▼ statutes + precedents
┌─────────────────────────────────────────────────────────────────────────┐
│              Reasoner / Counter-Reasoner / Polisher-Evaluator           │
├─────────────────────────────────────────────────────────────────────────┤
│  Reasoner: iterative chain on SUPPORT articles (ASPIC+)                 │
│  Counter-Reasoner: iterative attack chain on AGAINST articles (ASPIC+)  │
│  Polisher-Evaluator (4 Mixins):                                         │
│    ├─ ConsistencyChecker: KB verification → citation repair/drop        │
│    ├─ ScoringMixin: readability, coherence, argument quality            │
│    ├─ NLPUtilsMixin: Flesch/FOG/SMOG, NLI via LLM                      │
│    └─ AQAEngine: Cogency(α)+NormSupport(β)+Semantics(γ)→verdict        │
│         └─ Cross-attacks (6 types) + precedent influence → net_plaus.   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Neo4j Knowledge Base + Taxonomy                       │
├─────────────────────────────────────────────────────────────────────────┤
│  📚 Italian statutes KB: Civil Code + Penal Code + Administrative Law    │
│     (L. 7 agosto 1990, n. 241)                                           │
│  ⚖️  9112 precedent chunks from 792 rulings (ITA-CaseHold)              │
│  📊 768-dim Vector Index (Legal-BERT) on statutes                       │
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
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_FALLBACK_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct

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
- Load Civil Code, Penal Code, and Administrative Law (L. 241/1990) articles with embeddings
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
│   ├── config.py                  # Centralized configuration (90+ Pydantic Settings)
│   ├── api_server.py              # Flask API server (8 endpoints)
│   ├── agents/                    # LangChain/LangGraph agents
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
│   │       ├── neo4j_tools.py    # Neo4j search pipeline
│   │       ├── taxonomy_tools.py # Causality taxonomy
│   │       ├── config_loader.py  # Taxonomy config loader
│   │       └── config_taxonomy.json
│   ├── services/                  # Core services
│   │   ├── groq_client.py        # Resilient Groq client (dynamic key discovery, rotation)
│   │   ├── claim_classifier.py   # LLM claim classification
│   │   ├── stance_classifier.py  # NLI stance classification
│   │   └── legal_search.py       # Legal search pipeline
│   ├── db/                        # Database management
│   │   ├── db_orchestrator.py    # Full DB lifecycle (clean/schema/load/verify)
│   │   └── data_loader.py        # Centralized data loading (CSV/parquet + statute embeddings)
│   ├── data/                      # Data files
│   │   ├── embeddings/           # Pre-computed embeddings (.npy)
│   │   ├── precedents/           # ITA-CaseHold precedents (parquet)
│   │   └── statutes/              # Civil + Penal + Administrative law CSVs
│   └── frontend/                  # React frontend (Vite + React 18)
│       └── src/
│           ├── App.jsx            # Main app with three tabs + settings
│           ├── AspicMetagraph.jsx # ASPIC+ meta-graph SVG visualization
│           └── AttackTextDetails.jsx # Cross-attack detail panel
├── logs/                          # Per-claim pipeline logs (auto-generated)
├── compose.yml                    # Docker Compose for Neo4j
├── pyproject.toml                 # Poetry configuration
└── README.md
```

## 🔧 Configuration

All configuration is managed through environment variables and the `src/config.py` Settings class (90+ parameters total).
Runtime-tunable settings (model, temperature, max tokens, search parameters, AQA weights, chain steps, attack parameters) can also be adjusted from the **frontend Settings panel** without restarting the server.

### Required (`.env`)

These variables **must** be set in the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |
| `GROQ_API_KEY_V1` | - | Primary Groq API key |
| `GROQ_API_KEY_V2…VN` | - | Additional Groq API keys (dynamic discovery V1…V99) |

### Optional — LLM & Server

These have sensible defaults and can be overridden in `.env` or via the frontend Settings panel:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_MODEL` | `llama-4-scout-17b-16e-instruct` | Primary LLM model |
| `GROQ_FALLBACK_MODEL` | `llama-4-maverick-17b-128e-instruct` | Fallback LLM model |
| `GROQ_MAX_RETRIES` | `3` | Max retries per API call |
| `GROQ_RETRY_BASE_DELAY` | `1.0` | Base delay (s) for exponential backoff |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature |
| `LLM_MAX_TOKENS` | `8192` | LLM max output tokens |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `DEBUG` | `true` | Enable debug mode |

### Optional — Embedding & Search

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `nlpaueb/legal-bert-base-uncased` | Embedding model |
| `EMBEDDING_DIM` | `768` | Embedding vector dimension |
| `EMBEDDING_MAX_LENGTH` | `512` | Max token length for embeddings |
| `SEARCH_TOP_K_DEFAULT` | `100` | Default statute results to retrieve |
| `SEARCH_USE_TOP_N_LIBRI` | `3` | Top classified books to query |
| `PRECEDENTS_LIMIT_DEFAULT` | `5` | Default number of precedents |
| `SEARCH_MIN_KEPT_STATUTES` | `10` | Min statutes after filtering to trigger progressive expansion |
| `SEARCH_EXPANSION_STEP` | `10` | Additional statutes per expansion round |
| `SEARCH_MAX_EXPANSIONS` | `5` | Max expansion rounds |

### Optional — AQA (Argument Quality Assessment)

| Variable | Default | Description |
|----------|---------|-------------|
| `AQA_ENABLED` | `true` | Enable AQA scoring |
| `AQA_ALPHA` | `0.3` | AQA weight for Cogency |
| `AQA_BETA` | `0.4` | AQA weight for NormSupport |
| `AQA_GAMMA` | `0.3` | AQA weight for Semantics |
| `AQA_ATTACK_TOP_K` | `3` | Top-K cross-attacks to retain |
| `AQA_VERDICT_POS_THRESHOLD` | `0.2` | Threshold for plausible verdict |
| `AQA_VERDICT_NEG_THRESHOLD` | `-0.2` | Threshold for implausible verdict |
| `AQA_EMBEDDING_MODEL` | `all-mpnet-base-v2` | Sentence embedding model for AQA |
| `AQA_TFIDF_MAX_FEATURES` | `5000` | Max TF-IDF features |
| `AQA_NORMSUPPORT_MAX_CITATIONS` | `3` | Max citations for NormSupport |
| `AQA_NORMSUPPORT_CITATION_WEIGHT` | `0.7` | Citation weight in NormSupport |
| `AQA_NORMSUPPORT_RETRIEVED_WEIGHT` | `0.3` | Retrieved weight in NormSupport |
| `AQA_NORMSUPPORT_RETRIEVED_AGG` | `avg` | Aggregation for retrieved scores |
| `AQA_MIN_SEMANTIC_OVERLAP` | `0.3` | Min semantic overlap to validate an attack |
| `AQA_MIN_STRENGTH_RATIO` | `1.2` | Min base_score ratio for attacks |
| `AQA_DAMAGE_FACTOR` | `0.5` | Excess damage scaling factor |
| `AQA_ALLOW_FACTUAL_ATTACKS` | `true` | Allow factual attacks on normative links |
| `AQA_ALLOW_CROSS_CODICE` | `true` | Allow cross-codice attacks |
| `AQA_MAX_AGE` | `50.0` | Max precedent age for recency scoring |
| `AQA_DOMINANT_ATTACKS_LIMIT` | `10` | Max dominant attacks in report |

### Optional — Chain Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAIN_MAX_RETRIES` | `5` | Max retries for chain generation |
| `CHAIN_MAX_STEPS` | `10` | Safety cap: max steps per iterative chain |
| `CHAIN_MIN_STEPS` | `3` | Min steps before the LLM may conclude |
| `MODEL_DOWN_TTL` | `300.0` | Seconds before retrying a down model |


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
- [x] Neo4j Knowledge Base with Civil/Penal Code
- [x] Legal-BERT embeddings and vector search
- [x] Claim classification and book routing via LLM
- [x] Domain Router Agent (CIVILE / PENALE / ENTRAMBI)
- [x] Unified, thread-safe, configurable LegalSearchPipeline (allow-list, soft-filtering, stance)
- [x] Progressive search with adaptive expansion when post-filtering yields too few results
- [x] Pre-retrieval LLM filtering for statutes and precedents (default-YES soft filter)
- [x] Stance Classifier (NLI): SUPPORT / AGAINST / NEUTRAL classification for statutes and precedents
- [x] Reasoner Agent: iterative step-by-step chain generation (ASPIC+), with LLM evaluation phase and min/max step bounds
- [x] Counter-Reasoner Agent: iterative counter-argumentation (ASPIC+), same iterative architecture as Reasoner
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
- [x] NLI contradiction detection via LLM (replaces local DeBERTa model)
- [x] ASPIC+ IR formatting for Reasoner and Counter-Reasoner
- [x] Prescriptive prompts and structured output for all agents
- [x] Centralized configuration via Pydantic Settings (90+ parameters including truncation, attack, chain settings)
- [x] Frontend Settings Panel: per-step model selection, temperature, max tokens, search params, AQA weights, chain min/max steps, attack parameters
- [x] Per-claim pipeline logging (`logs/` directory)
- [x] React frontend (Vite) with three tabs + ASPIC+ Metagraph + Attack Details + settings panel
- [x] Precedent ingestion and fulltext search (ITA-CaseHold)
- [x] Centralized data loading module (`data_loader.py`) with path resolution via Settings
- [x] Improving precedent utilization in reasoning chains (better citation coverage, richer contextual integration)

### 🚧 In Progress
- [ ] Tuning attack evaluation parameters (damage multipliers, strength ratios, severity thresholds) for more balanced verdicts

### 📋 Planned
- [ ] Claim-level caching: persist retrieval results (statutes, precedents, classification, stance) per claim to skip redundant searches on re-execution
- [ ] LLM memory layer: maintain conversational context across pipeline steps to improve coherence and reduce redundant reasoning
- [ ] Full argumentation framework (Dung-style grounded semantics visualization)
- [ ] Export reasoning chains to structured formats (JSON-LD, RDF)
- [ ] Multi-turn dialogue with context retention


## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.


## 📚 References

The ITA-CaseHold dataset, used for legal precedent extraction and summarization in this project, was introduced by Licari et al. at ICAIL 2023. Their work presents a method for extracting legal holdings from Italian case documents using Italian-LEGAL-BERT, and provides a valuable benchmark for research in Italian legal NLP. For more details, see their publication at the International Conference on Artificial Intelligence and Law: [https://doi.org/10.1145/3594536.3595177](https://doi.org/10.1145/3594536.3595177).

The Italian Civil Code dataset leveraged in LexCausa is based on the unsupervised law article mining approach described by Tagarelli and Simeri (2022). Their research applies deep pre-trained language models to the Italian civil code, enabling advanced legal text mining and representation. The full article is available in Artificial Intelligence and Law: [https://doi.org/10.1007/s10506-021-09301-8](https://doi.org/10.1007/s10506-021-09301-8).

## 👤 Authors

**Leonardo Catello**  
Master's Thesis in Computer Engineering  
Email: leonardo.catello@hotmail.com

**Salvatore Maione**  
Master's Thesis in Computer Engineering  
Email: salvatore22maione@gmail.com

---

*This project is part of a Master's thesis and is not intended for production legal use.*
