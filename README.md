# LexCausa

> ⚠️ **Work in Progress** - This project is under active development as part of a Master's thesis in Computer Engineering.

**LexCausa** is an AI-powered legal reasoning system for Italian law. It combines Knowledge Graphs (Neo4j), Large Language Models (Groq Cloud), and structured causal reasoning to analyze legal claims, find relevant statutes/precedents, and build logical argumentation chains.



## 🎯 Features

- **Legal Claim Classification**: Automatic claim classification and routing to the correct book of the Civil/Penal Code via LLM
- **Domain Router**: Lightweight pre-routing agent that classifies claims as CIVILE, PENALE, or ENTRAMBI
- **Semantic Search**: Vector search on 3900+ articles using Legal-BERT, with unified and configurable pipeline
- **Unified Pipeline**: All functionalities (Search, Reasoning, Full Pipeline) share the same singleton `LegalSearchPipeline`, ensuring consistency and thread safety
- **Stance Classifier (NLI)**: Classifies statutes and precedents as SUPPORT, AGAINST, or NEUTRAL relative to the claim using NLI-style prompting
- **Reasoner Agent**: Builds structured argumentative chains (Premise → Statute → Precedent → Causal Link → Conclusion) only on the provided knowledge base, with causality classification and precise citations
- **Counter-Reasoner Agent**: Generates independent counter-arguments using the causality taxonomy, identifying attacking causalities and building attack reasoning chains
- **Polisher-Evaluator Agent**: Evaluates the dialectical exchange Reasoner/Counter-Reasoner with consistency checking against Neo4j KB, citation repair, AQA scoring, and verdict generation
- **AQA (Argument Quality Assessment)**: Three-dimensional scoring — Cogency (α), NormSupport (β), Semantics (γ) — with configurable weights, cross-attacks, and precedent influence
- **Resilient Groq Client**: Automatic retry with exponential backoff, API key rotation (up to 3 keys), and model fallback; smart error classification (model-down vs. rate-limit vs. transient)
- **Causality Taxonomy**: Structured causality taxonomy (Material, Legal, Concurrent) used by Reasoner and Counter-Reasoner for arguments and attacks
- **Knowledge Graph**: Neo4j database with statutes, precedents, and causal relationships
- **Centralized Configuration**: All parameters (models, retries, AQA weights, search settings, etc.) managed by `src/config.py` and environment variables
- **Frontend Settings Panel**: Collapsible panel to configure per-step LLM model, temperature, max tokens, search parameters, and AQA weights — without touching code
- **Per-Claim Pipeline Logging**: Every pipeline run is logged to `logs/<timestamp>_<slug>.log` for full auditability
- **React Frontend**: Modern three-tab interface (Search, Reasoning, Full Pipeline) on Vite + React 18



## 🏗️ Agent and Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Frontend (React + Vite)                           │
│   Search Tab │ Reasoning Tab │ Full Pipeline Tab │ ⚙️ Settings Panel    │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Flask API Server (:8000)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  GET  /api/settings  → defaults & available models                      │
│  POST /api/chat      → LegalSearchPipeline (unified retrieval)          │
│  POST /api/reason    → Reasoner (structured reasoning)                  │
│  POST /api/counter   → Counter-Reasoner (counter-argumentation)         │
│  POST /api/pipeline  → Full Pipeline (Router→Reasoner→Counter→AQA)      │
│  POST /api/evaluate  → Polisher-Evaluator (standalone evaluation)       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────┐
│   Resilient Groq Client      │  │  LegalSearchPipeline                 │
│   (groq_client.py)           │  │  (Singleton, thread-safe)            │
├──────────────────────────────┤  ├──────────────────────────────────────┤
│  Key rotation (3 keys)       │  │  ClaimClassifier → book routing      │
│  Model fallback              │  │  Legal-BERT → 768-dim vectors        │
│  Smart error classification  │  │  Vector Search → filtered results    │
│  Exponential backoff         │  │  StanceClassifier (NLI)              │
└──────────────────────────────┘  └────────────────┬─────────────────────┘
                    │                              │
                    │         ┌────────────────────┘
                    ▼         ▼ statutes + precedents
┌─────────────────────────────────────────────────────────────────────────┐
│              Reasoner / Counter-Reasoner / Polisher-Evaluator           │
├─────────────────────────────────────────────────────────────────────────┤
│  Reasoner: argumentative chain on SUPPORT articles (ASPIC+)             │
│  Counter-Reasoner: attack chain on AGAINST articles (ASPIC+)            │
│  Polisher-Evaluator: KB consistency → citation repair → AQA scoring     │
│    └─ AQA: Cogency (α) + NormSupport (β) + Semantics (γ) → verdict     │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Neo4j Knowledge Base + Taxonomy                       │
├─────────────────────────────────────────────────────────────────────────┤
│  📚 3964 articles (925 Penal Code + 3039 Civil Code)                    │
│  ⚖️  9112 precedent chunks from 792 rulings (ITA-CaseHold)              │
│  📊 768-dim Vector Indexes (Legal-BERT)                                 │
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
GROQ_API_KEY_V3=your_third_key_here
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
poetry run python -c "from src.db.db_orchestrator import DBOrchestrator; DBOrchestrator().run_full_ingestion()"
```

This will:
- Load Civil Code and Penal Code articles
- Generate Legal-BERT embeddings
- Create vector indexes in Neo4j

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
│   ├── config.py                  # Centralized configuration (Pydantic Settings)
│   ├── api_server.py              # Flask API server (7 endpoints)
│   ├── agents/                    # LangChain/LangGraph agents
│   │   ├── base.py               # Base agent class + AgentConfig
│   │   ├── router.py             # Domain router (CIVILE/PENALE/ENTRAMBI)
│   │   ├── reasoner.py           # Main reasoning agent (ASPIC+)
│   │   ├── counter_reasoner.py   # Counter-argumentation agent (ASPIC+)
│   │   ├── polisher_evaluator.py # Consistency check + AQA scoring
│   │   ├── aspic_formatter.py    # ASPIC+ IR formatting
│   │   └── tools/                # Agent tools
│   │       ├── neo4j_tools.py    # Neo4j search pipeline
│   │       ├── taxonomy_tools.py # Causality taxonomy
│   │       ├── config_loader.py  # Taxonomy config loader
│   │       └── config_taxonomy.json
│   ├── services/                  # Core services
│   │   ├── groq_client.py        # Resilient Groq client (retry + rotation)
│   │   ├── claim_classifier.py   # LLM claim classification
│   │   ├── stance_classifier.py  # NLI stance classification
│   │   └── legal_search.py       # Legal search pipeline
│   ├── db/                        # Database management
│   │   ├── db_orchestrator.py
│   │   ├── neo4j_ingestion.py
│   │   └── neo4j_schema.py
│   ├── data/                      # Data files
│   │   ├── embeddings/           # Pre-computed embeddings (.npy)
│   │   ├── precedenti/           # ITA-CaseHold precedents
│   │   └── statuti/              # Civil + Penal Code CSVs
│   └── frontend/                  # React frontend (Vite + React 18)
├── logs/                          # Per-claim pipeline logs (auto-generated)
├── compose.yml                    # Docker Compose for Neo4j
├── pyproject.toml                 # Poetry configuration
└── README.md
```

## 🔧 Configuration

All configuration is managed through environment variables and the `src/config.py` Settings class (37 parameters total).
Runtime-tunable settings (model, temperature, max tokens, search parameters, AQA weights) can also be adjusted from the **frontend Settings panel** without restarting the server.

### Required (`.env`)

These variables **must** be set in the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |
| `GROQ_API_KEY_V1` | - | Primary Groq API key |
| `GROQ_API_KEY_V2` | - | Secondary Groq API key (rotation) |
| `GROQ_API_KEY_V3` | - | Tertiary Groq API key (rotation) |

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
| `AQA_NLI_MODEL` | `DeBERTa-v3-base-mnli` | NLI model for AQA |
| `AQA_TFIDF_MAX_FEATURES` | `5000` | Max TF-IDF features |
| `AQA_NORMSUPPORT_MAX_CITATIONS` | `3` | Max citations for NormSupport |
| `AQA_NORMSUPPORT_CITATION_WEIGHT` | `0.7` | Citation weight in NormSupport |
| `AQA_NORMSUPPORT_RETRIEVED_WEIGHT` | `0.3` | Retrieved weight in NormSupport |
| `AQA_NORMSUPPORT_RETRIEVED_AGG` | `avg` | Aggregation for retrieved scores |



## 🧪 Agent & Pipeline Development Status

### ✅ Completed
- [x] Neo4j Knowledge Base with Civil/Penal Code
- [x] Legal-BERT embeddings and vector search
- [x] Claim classification and book routing via LLM
- [x] Domain Router Agent (CIVILE / PENALE / ENTRAMBI)
- [x] Unified, thread-safe, configurable LegalSearchPipeline (allow-list, soft-filtering, stance)
- [x] Stance Classifier (NLI): SUPPORT / AGAINST / NEUTRAL classification for statutes and precedents
- [x] Reasoner Agent: structured argumentative chain (ASPIC+), only on SUPPORT articles
- [x] Counter-Reasoner Agent: independent counter-argumentation (ASPIC+), only on AGAINST articles
- [x] Polisher-Evaluator Agent: consistency verification against Neo4j KB, citation repair, chain regeneration
- [x] AQA (Argument Quality Assessment): Cogency + NormSupport + Semantics scoring with configurable weights
- [x] Verdict generation (plausible / implausible / uncertain) with winning_side and confidence mapping
- [x] Resilient Groq Client: retry + API key rotation (3 keys) + model fallback + model-down cache
- [x] Smart error classification: model-down (503) vs. rate-limit (429) vs. transient errors
- [x] ASPIC+ IR formatting for Reasoner and Counter-Reasoner
- [x] Prescriptive prompts and structured output for all agents
- [x] Centralized configuration via Pydantic Settings (50+ parameters)
- [x] Frontend Settings Panel: per-step model selection, temperature, max tokens, search params, AQA weights
- [x] Per-claim pipeline logging (`logs/` directory)
- [x] React frontend (Vite) with three tabs + settings panel
- [x] Precedent ingestion and vector search (ITA-CaseHold)

### 🚧 In Progress
- [ ] Cross-attack computation in AQA (logic implemented, disabled at runtime — needs activation and testing)
- [ ] Attack graph visualization (dialectical meta-graph)
- [ ] Explainability of reasoning (detailed trace UI)
- [ ] Official Gazette ingestion

### 📋 Planned
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