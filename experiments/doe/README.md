# DoE LexCausa (A/B tassonomia causale)

Questa cartella contiene tutto il necessario per eseguire un Design of Experiments
paired e bloccato per dominio:

- `A`: `enable_causality=false`
- `B`: `enable_causality=true`

## Struttura

- `doe_settings.template.json`: template di configurazione run (endpoint, timeout, settings fissi).
- `scripts/generate_run_plan.py`: genera la matrice sperimentale (`run_plan.csv`) da `claims.md`.
- `scripts/run_doe.py`: esegue i run contro `POST /api/pipeline` e salva i JSON raw.
- `scripts/extract_metrics.py`: estrae metriche e delta paired.
- `scripts/analyze_doe.py`: analisi statistica base (sign test, McNemar exact, CI bootstrap).

## Prerequisiti

1. Backend avviato (`python src/api_server.py`) su `http://127.0.0.1:8000`.
2. Variabili API key valide nel backend.
3. Ambiente Python con dipendenze progetto installate.

## Workflow rapido

### 1) Copia e personalizza la config

```powershell
Copy-Item experiments/doe/doe_settings.template.json experiments/doe/doe_settings.json
```

Modifica `experiments/doe/doe_settings.json` fissando:

- modello/i
- temperatura
- max_statutes / max_precedents
- chain_min/max_steps
- pesi AQA

Nota: durante il DoE cambierai solo `enable_causality` (gestito automaticamente da `run_plan.csv`).

### 2) Genera il piano sperimentale

Esempio pilot robusto:

```powershell
python experiments/doe/scripts/generate_run_plan.py `
  --claims-file claims.md `
  --out experiments/doe/run_plan.csv `
  --replicates 2 `
  --domains CIVILE,PENALE,AMMINISTRATIVO `
  --seed 42
```

Output:

- `experiments/doe/run_plan.csv`
- `experiments/doe/run_plan_summary.json`

### 3) Esegui i run

```powershell
python experiments/doe/scripts/run_doe.py `
  --plan experiments/doe/run_plan.csv `
  --config experiments/doe/doe_settings.json `
  --out experiments/doe/runs
```

Viene creata una run folder timestampata, ad esempio:

- `experiments/doe/runs/20260223_101530/`
- `.../raw/<run_id>.json`
- `.../requests/<run_id>.json`
- `.../run_status.csv`

### 4) Estrai metriche

```powershell
python experiments/doe/scripts/extract_metrics.py `
  --plan experiments/doe/run_plan.csv `
  --run-dir experiments/doe/runs/20260223_101530 `
  --out experiments/doe/outputs/20260223_101530
```

Output principali:

- `metrics.csv` / `metrics.parquet`
- `paired_deltas.csv`

### 5) Analizza risultati

```powershell
python experiments/doe/scripts/analyze_doe.py `
  --metrics experiments/doe/outputs/20260223_101530/metrics.csv `
  --paired experiments/doe/outputs/20260223_101530/paired_deltas.csv `
  --out experiments/doe/outputs/20260223_101530
```

Output:

- `analysis_summary.json`
- `analysis_summary.md`

## Metriche principali estratte

- `aqa_net_final`, `aqa_verdict`
- `reasoner_consistency_score`, `counter_consistency_score`
- `reasoner_repair_fail_rate`, `counter_repair_fail_rate`
- `reasoner_dropped_citations`, `counter_dropped_citations`
- `counter_gate_label`, `counter_gate_abstain`
- `counter_planning_mode`, `counter_reasoner_plan_hints_available`

## Note operative

- L'endpoint `/api/pipeline` accetta una sola esecuzione per volta (lock server-side).
- `run_doe.py` esegue in modo sequenziale; in caso di errore HTTP o timeout applica retry.
- Il piano e' paired: per ogni claim/replica esegue entrambe le condizioni con ordine AB/BA randomizzato.
