# LexCausa Multi-DoE Framework

Advanced Design of Experiments (DoE) for multi-dimensional ablation studies of the LexCausa legal reasoning system.

## Overview

This framework supports comprehensive evaluation of LexCausa across multiple dimensions:

### Research Questions

**RQ1: Model-Class Efficacy**
- Compare native-reasoning vs instruction-tuned models (e.g., deepseek_r1/gpt_oss_120b vs groq_llama_3_3_70b_versatile/qwen_25_72b)
- Metrics: AQA net plausibility, reasoning chain length
- Analysis: paired t-test by claim, Cohen's d effect size

**RQ2: Dialectical Pairing**
- Measure cross-model Reasoner×Counter interaction effects beyond each role's marginal contribution
- Metrics: AQA net plausibility per pairing
- Analysis: Reasoner×Counter interaction term of the factorial ANOVA (eta-squared)

**RQ3: Architectural Efficacy and Stability**
- Impact of Plan-then-Execute vs single-call (planning ablation), citation faithfulness, and token cost
- Compare: planning ON (both Reasoner + Counter) vs planning OFF
- Metrics: citation accuracy %, citation repair rate, token/quality deltas, inter-replica variance
- Analysis: planning main effect (ANOVA), bootstrap confidence intervals, percentage deltas

**Auxiliary: Token Cost Analysis**
- Token efficiency per model and condition
- Token cost per verdict (completion + reasoning tokens)
- Efficiency frontier comparison

## Architecture

```
Containerized Backend (Docker)
├── Neo4j (7687)
└── Flask API (8000)
     └── Reasoner Agent (enable_planning_reasoner toggle)
     └── Counter-Reasoner Agent (enable_planning_counter toggle)
     └── Polisher-Evaluator Agent
     └── AQA (Argument Quality Assessment)

DoE Orchestration
├── run_multi_doe.py: Experiment matrix + run execution
├── analyze_multi_doe.py: Statistical analysis per RQ
└── start_backend_doe_mode.sh: Docker startup helper
```

## Quick Start

### 1. Prerequisites

```bash
# Python dependencies already in pyproject.toml
# Docker and Docker Compose installed

# Verify setup
docker --version
docker compose version   # modern plugin
python --version         # 3.11+
```

### 2. Start Backend (Docker)

```bash
bash scripts/start_backend_doe_mode.sh

# Or manually
docker compose up -d
curl http://localhost:8000/health  # Verify
```

### 3. Configure Environment

Create `.env` file with API keys:

```bash
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpassword
GROQ_API_KEY_V1=<your-groq-key>
GROQ_API_KEY_V2=<optional-secondary-key>
# ... more keys if available
```

### 4. Run DoE (Example: Pilot Study)

```bash
# Quick pilot: 2 models, 2 planning conditions, 2 replicates per condition
python scripts/run_multi_doe.py \
  --claims-file claims.md \
  --models gpt_oss_120b,groq_llama_scout_17b \
  --domains CIVILE,PENALE \
  --planning-ablations on,off \
  --replicates 2 \
  --out experiments/multi_doe/runs/pilot_20260225

# Full study: more replicates and domains
python scripts/run_multi_doe.py \
  --claims-file claims.md \
  --models gpt_oss_120b,groq_llama_scout_17b \
  --domains CIVILE,PENALE,AMMINISTRATIVO,MISTO \
  --planning-ablations on,off \
  --replicates 10 \
  --seed 42 \
  --out experiments/multi_doe/runs/full_$(date +%Y%m%d_%H%M%S)
```

### 5. Analyze Results

```bash
python scripts/analyze_multi_doe.py \
  --run-dir experiments/multi_doe/runs/pilot_20260225 \
  --output experiments/multi_doe/analysis/pilot_20260225
```

## Output Structure

```
experiments/multi_doe/runs/20260225_101530/
├── run_matrix.csv              # Experimental design matrix
├── metrics.csv                 # All metrics per run
├── metrics.parquet             # Parquet format (optional)
├── runs/
│   ├── run_001abc.json        # Raw pipeline response
│   ├── run_002def.json
│   └── ...
└── logs/
    └── run_errors.log         # Error log

experiments/multi_doe/analysis/20260225/
├── doe_analysis.json           # Complete statistical analysis
└── [future: plots, tables]
```

## Metrics Captured Per Run

### AQA (RQ1)
- `aqa_verdict`: plausible/implausible/uncertain
- `aqa_plausibility`: final score [0,1]
- `aqa_pro`, `aqa_contra`: component scores

### Citation Faithfulness (RQ3)
- `citation_total`: total citations in counter argument
- `citation_repaired`: citations fixed by consistency checker
- `citation_valid`: citations verified against Neo4j KB
- `citation_dropped`: citations removed
- `citation_accuracy`: valid_citations / total_citations

### Planning Impact (RQ3)
- `reasoning_tokens`: tokens used in Reasoner phase
- `counter_tokens`: tokens used in Counter-Reasoner phase
- `total_tokens`: sum of all completion tokens
- `reasoning_steps`: number of steps in reasoning chain
- `counter_steps`: number of steps in counter argument
- `counter_attacks_count`: number of attacks used

### Process Metrics
- `duration_sec`: wall-clock time for run
- `status`: completed/failed
- `error`: error message if failed

## Configuration

### DoE Parameters

```python
# In run_multi_doe.py or CLI args
models = ["gpt_oss_120b", "groq_llama_scout_17b"]  # Models to test
domains = ["CIVILE", "PENALE", "AMMINISTRATIVO", "MISTO"]  # Domains to include
planning_ablations = [(True, True), (False, False)]  # (reasoner, counter)
replicates = 10  # Per condition
```

### Backend Configuration

Edit `.env` or `compose.yml`:

```yaml
# compose.yml API service
environment:
  NEO4J_URI: "bolt://neo4j:7687"
  GROQ_API_KEY_V1: "${GROQ_API_KEY_V1}"
  API_PORT: "8000"
  ENABLE_PLANNING_REASONER: "true"      # Toggle via ENV
  ENABLE_PLANNING_COUNTER: "true"        # Toggle via ENV
```

Pipeline settings can also be overridden per-run in `run_multi_doe.py`:

```python
payload = {
    "claim": "...",
    "settings": {
        "reasoner_model": model,
        "counter_model": model,
        "enable_planning_reasoner": plan_r,
        "enable_planning_counter": plan_c,
        "reasoner_temperature": 0.0,
        "counter_temperature": 0.3,
        "llm_max_tokens": 7168,
        # ... other settings
    }
}
```

## Analysis Output Format

### Model-Class Efficacy (RQ1)

```json
{
  "model_class_efficacy": {
    "aqa_plausibility": {
      "gpt_oss_120b_mean": 0.72,
      "gpt_oss_120b_std": 0.15,
      "groq_llama_scout_17b_mean": 0.58,
      "groq_llama_scout_17b_std": 0.21,
      "t_statistic": 2.34,
      "p_value": 0.042,
      "cohens_d": 0.65,
      "significant": true
    },
    "verdict_distribution": {
      "gpt_oss_120b": {"plausible": 18, "implausible": 2},
      "groq_llama_scout_17b": {"plausible": 12, "implausible": 8}
    }
  }
}
```

### Citation Faithfulness (RQ3)

```json
{
  "citation_faithfulness": {
    "gpt_oss_120b": {
      "accuracy_mean": 0.87,
      "accuracy_std": 0.08,
      "accuracy_ci_95": [0.78, 0.95],
      "citation_repaired_rate": 0.23
    },
    "groq_llama_scout_17b": {
      "accuracy_mean": 0.71,
      "accuracy_std": 0.12,
      "accuracy_ci_95": [0.58, 0.84],
      "citation_repaired_rate": 0.35
    }
  }
}
```

### Planning Ablation (RQ3)

```json
{
  "planning_ablation": {
    "token_delta": {
      "absolute": 470,
      "percentage": 23.7,
      "planning_on_preferred": false
    },
    "reasoning_steps_delta": 1.4,
    "quality_delta": 0.08,
    "planning_quality_impact": {
      "improvements": 16,
      "deteriorations": 4,
      "improvement_rate": 0.8
    }
  }
}
```

## Troubleshooting

### Docker backend not starting

```bash
# Check status
docker compose ps

# View logs
docker compose logs api neo4j

# Restart
docker compose restart

# Full rebuild
docker compose down
docker compose up -d --build
```

### API connection errors

```bash
# Test endpoint
curl -v http://localhost:8000/health

# Check if API is running
docker compose ps api

# Restart API only
docker compose restart api
```

### Out of memory

Increase Docker memory limits in `compose.yml`:

```yaml
api:
  deploy:
    resources:
      limits:
        memory: 4G
      reservations:
        memory: 2G
```

## Performance Notes

- **Typical run duration**: 60-300 seconds per claim/model/condition
- **Full study**: 2 models × 2 planning × 10 replicates × 20 claims ≈ 800 runs ≈ 5-40 hours
- **Token usage**: 1500-4000 tokens per run average
- **Disk usage**: ~200MB per 100 runs (raw JSON responses)

## Citation

If using this DoE framework, cite:

```bibtex
@software{lexcausa2026,
  title={LexCausa: Legal Reasoning System with Multi-DoE Framework},
  author={Catello, Leonardo and Maione, Salvatore},
  year={2026},
  url={https://github.com/lexcausa}
}
```

## Support

For issues:
1. Check Docker logs: `docker compose logs`
2. Verify API health: `curl http://localhost:8000/health`
3. Check run logs in `experiments/multi_doe/runs/<timestamp>/logs/`
