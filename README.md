# LexCausa

> ⚠️ **Work in Progress** - This project is under active development as part of a Master's thesis in Computer Engineering.

**LexCausa** is an AI-powered legal reasoning system for Italian law. It combines Knowledge Graphs (Neo4j), Large Language Models (Groq Cloud), and structured causal reasoning to analyze legal claims, find relevant statutes/precedents, and build logical argumentation chains.

## 🎯 Features

- **Legal Claim Classification**: Routes claims to the appropriate book of Italian Civil/Penal Code using LLM
- **Semantic Search**: Vector similarity search on 3900+ statute articles using Legal-BERT embeddings
- **Causal Chain Reasoning**: Classifies causality type (Material, Legal, Concurrent) and builds structured arguments
- **Knowledge Graph**: Neo4j-based storage of statutes, precedents, and their relationships
- **Multi-Agent Architecture**: Reasoner, Counter-Reasoner, and Polisher-Evaluator agents (WIP)
- **React Frontend**: Interactive chat interface for legal queries

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask API Server (8000)                       │
├─────────────────────────────────────────────────────────────────┤
│  /api/chat      → Legal Search Pipeline                         │
│  /api/reason    → Reasoner Agent                                │
│  /api/counter   → Counter-Reasoner Agent (stub)                 │
│  /api/evaluate  → Polisher-Evaluator Agent (stub)               │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claim        │     │   Legal Search  │     │   Reasoner      │
│  Classifier   │     │   Pipeline      │     │   Agent         │
│  (Groq LLM)   │     │  (Legal-BERT)   │     │  (LangGraph)    │
└───────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Neo4j Knowledge Base                          │
├─────────────────────────────────────────────────────────────────┤
│  📚 3964 Statute Articles (Civil Code + Penal Code)             │
│  📊 768-dim Vector Index (Legal-BERT embeddings)                │
│  🔗 Causality Taxonomy (Material, Legal, Concurrent)            │
└─────────────────────────────────────────────────────────────────┘
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

## 🧪 Development Status

### ✅ Completed
- [x] Neo4j Knowledge Base with Civil/Penal Code
- [x] Legal-BERT embeddings and vector search
- [x] Claim classification (book routing)
- [x] Legal Search Pipeline
- [x] Reasoner Agent with LangGraph
- [x] Causality classification via LLM
- [x] React frontend with chat interface
- [x] Centralized configuration

### 🚧 In Progress
- [ ] Counter-Reasoner Agent
- [ ] Polisher-Evaluator Agent
- [ ] Precedent ingestion and search
- [ ] Attack graph visualization

### 📋 Planned
- [ ] Full argumentation framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Authors

**Leonardo Catello**  
Master's Thesis in Computer Engineering  
Email: leonardo.catello@hotmail.com

**Salvatore Maione**  
Master's Thesis in Computer Engineering  
Email: salvatore22maione@gmail.com

---

*This project is part of a Master's thesis and is not intended for production legal use.*