# LexCausa

> ⚠️ **Work in Progress** - This project is under active development as part of a Master's thesis in Computer Engineering.

**LexCausa** is an AI-powered legal reasoning system for Italian law. It combines Knowledge Graphs (Neo4j), Large Language Models (Groq Cloud), and structured causal reasoning to analyze legal claims, find relevant statutes/precedents, and build logical argumentation chains.



## 🎯 Features

- **Legal Claim Classification**: Automatic claim classification and routing to the correct book of the Civil/Penal Code via LLM
- **Semantic Search**: Vector search on 3900+ articles using Legal-BERT, with unified and configurable pipeline
- **Unified Pipeline**: All functionalities (Search, Reasoning, Full Pipeline) share the same singleton `LegalSearchPipeline`, ensuring consistency and thread safety
- **Reasoner Agent**: Builds structured argumentative chains (Premise → Statute → Precedent → Causal Link → Conclusion) only on the provided knowledge base, with causality classification and precise citations
- **Counter-Reasoner Agent**: Generates counter-arguments using the causality taxonomy, identifying attacking causalities and building attack reasoning chains
- **Polisher-Evaluator Agent**: (In development) Evaluates the dialectical exchange Reasoner/Counter-Reasoner, determines the prevailing side and polishes the final response
- **Causality Taxonomy**: Structured causality taxonomy (Material, Legal, Concurrent) used by Reasoner and Counter-Reasoner for arguments and attacks
- **Knowledge Graph**: Neo4j database with statutes, precedents, and causal relationships
- **Centralized Configuration**: All parameters (models, allow-list, top_k, etc.) managed by `src/config.py` and environment variables
- **React Frontend**: Modern three-tab interface (Search, Reasoning, Full Pipeline) on Vite + React 18



## 🏗️ Agent and Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                              │
│         Search Tab │ Reasoning Tab │ Full Pipeline Tab                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Flask API Server (8000)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  /api/chat      → LegalSearchPipeline (unified retrieval)                   │
│  /api/reason    → Reasoner Agent (structured reasoning)                     │
│  /api/counter   → Counter-Reasoner Agent (counter-argumentation)            │
│  /api/evaluate  → Polisher-Evaluator Agent (dialectical evaluation, WIP)    │
└──────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              LegalSearchPipeline (Singleton, thread-safe)                   │
│    Vector retrieval, allow-list, stance classification, soft-filtering      │
├──────────────────────────────────────────────────────────────────────────────┤
│  ClaimClassifier (Groq LLM) → book routing                                  │
│  Legal-BERT Embeddings → 768-dim vectors                                    │
│  Vector Search → Filtered by books and allow-list                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Reasoner / Counter-Reasoner / Polisher-Evaluator         │
├──────────────────────────────────────────────────────────────────────────────┤
│  Reasoner: structured argumentative chain, only on provided knowledge base   │
│  Counter-Reasoner: dialectical attack via causality taxonomy                │
│  Polisher-Evaluator: evaluation and final synthesis (in development)        │
└──────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Neo4j Knowledge Base + Taxonomy                          │
├──────────────────────────────────────────────────────────────────────────────┤
│  📚 3964 articles (Civil + Penal Code)                                      │
│  📊 768-dim Vector Index (Legal-BERT)                                       │
│  🔗 Causality taxonomy (Material, Legal, Concurrent)                        │
│  ⚖️  Precedents and causal relationships                                    │
└──────────────────────────────────────────────────────────────────────────────┘
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

# Groq Cloud API
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

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
cd ../..
```

## 🎮 Usage

### Start the Backend

```bash
poetry run python src/api_server.py
```

The API will be available at http://localhost:8000

### Start the Frontend

```bash
cd src/frontend
npm start
```

Open http://localhost:3000 in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/chat` | POST | Legal search with classification |
| `/api/reason` | POST | Causal chain reasoning |
| `/api/counter-reason` | POST | Counter-arguments (WIP) |
| `/api/evaluate` | POST | Final evaluation (WIP) |

### Example Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Il venditore non ha consegnato la merce", "top_k": 5}'
```

## 📁 Project Structure

```
LexCausa/
├── src/
│   ├── config.py              # Centralized configuration
│   ├── api_server.py          # Flask API server
│   ├── agents/                # LangChain/LangGraph agents
│   │   ├── base.py           # Base agent class
│   │   ├── reasoner.py       # Main reasoning agent
│   │   ├── counter_reasoner.py  # (WIP)
│   │   ├── polisher_evaluator.py  # (WIP)
│   │   └── tools/            # Agent tools
│   │       ├── neo4j_tools.py
│   │       └── taxonomy_tools.py
│   ├── services/              # Core services
│   │   ├── claim_classifier.py
│   │   └── legal_search.py
│   ├── db/                    # Database management
│   │   ├── db_orchestrator.py
│   │   ├── neo4j_ingestion.py
│   │   └── neo4j_schema.py
│   ├── data/                  # Data files
│   │   ├── tassonomia_causale.json
│   │   └── statuti/
│   └── frontend/              # React frontend
├── compose.yml                # Docker Compose for Neo4j
├── pyproject.toml             # Poetry configuration
└── README.md
```

## 🔧 Configuration

All configuration is managed through environment variables and the `src/config.py` Settings class:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |
| `GROQ_API_KEY` | - | Groq Cloud API key |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | LLM model |
| `EMBEDDING_MODEL` | `nlpaueb/legal-bert-base-uncased` | Embedding model |
| `API_PORT` | `8000` | API server port |



## 🧪 Agent & Pipeline Development Status

### ✅ Completed
- [x] Neo4j Knowledge Base with Civil/Penal Code
- [x] Legal-BERT embeddings and vector search
- [x] Claim classification and book routing via LLM
- [x] Unified, thread-safe, configurable LegalSearchPipeline (allow-list, soft-filtering, stance)
- [x] Reasoner Agent: structured argumentative chain, serialized output, only on provided knowledge base
- [x] Counter-Reasoner Agent: dialectical attack via causality taxonomy, structured counter-arguments
- [x] Prescriptive prompts and structured output for all agents
- [x] Centralized configuration via Pydantic Settings
- [x] React frontend (Vite) with three tabs

### 🚧 In Progress
- [ ] Polisher-Evaluator Agent: dialectical evaluation, scoring, final synthesis
- [ ] Formatter/Unification of PRO/COUNTER output
- [ ] Precedent ingestion and search (ITA-CASEHOLD)
- [ ] Attack graph visualization (dialectical meta-graph)
- [ ] Final logic chain & structured scoring
- [ ] Explainability of reasoning
- [ ] Official Gazette ingestion

### 📋 Planned
- [ ] Full argumentation framework (Dung-style)
- [ ] Export reasoning chains to structured formats
- [ ] Structured PRO/COUNTER output
- [ ] Multi-turn dialogue with context retention

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📚 References


### ITA-CaseHold Dataset

```bibtex
@inproceedings{10.1145/3594536.3595177,
  author = {Licari, Daniele and Bushipaka, Praveen and Marino, Gabriele and Comand\'e, Giovanni and Cucinotta, Tommaso},
  title = {Legal Holding Extraction from Italian Case Documents using Italian-LEGAL-BERT Text Summarization},
  year = {2023},
  isbn = {9798400701979},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3594536.3595177},
  doi = {10.1145/3594536.3595177},
  abstract = {Legal holdings are used in Italy as a critical component of the legal system, serving to establish legal precedents, provide guidance for future legal decisions, and ensure consistency and predictability in the interpretation and application of the law. They are written by domain experts who describe in a clear and concise manner the principle of law applied in the judgments.We introduce a legal holding extraction method based on Italian-LEGAL-BERT to automatically extract legal holdings from Italian cases. In addition, we present ITA-CaseHold, a benchmark dataset for Italian legal summarization. We conducted several experiments using this dataset, as a valuable baseline for future research on this topic.},
  booktitle = {Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law},
  pages = {148--156},
  numpages = {9},
  keywords = {Italian-LEGAL-BERT, Holding Extraction, Extractive Text Summarization, Benchmark Dataset},
  location = {Braga, Portugal},
  series = {ICAIL '23'}
}
```

### Italian Civil Code Dataset

```bibtex
@article{Lamberta,
  author    = {Andrea Tagarelli and Andrea Simeri},
  title     = {{Unsupervised law article mining based on deep pre-trained language representation models with application to the Italian civil code}},
  journal   = {Artif. Intell. Law},
  volume    = {30(3)},
  pages     = {417--473. Published: 15 September 2021},
  year      = {2022},
  doi       = {10.1007/s10506-021-09301-8}
}
```

## 👤 Authors

**Leonardo Catello**  
Master's Thesis in Computer Engineering  
Email: leonardo.catello@hotmail.com

**Salvatore Maione**  
Master's Thesis in Computer Engineering  
Email: salvatore22maione@gmail.com

---

*This project is part of a Master's thesis and is not intended for production legal use.*