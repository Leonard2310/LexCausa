# Neo4j on IBiSCo (Cerberus-style deploy)

Deploys the LexCausa knowledge base (Neo4j) on **node 0** of a SLURM allocation,
on a **free port**, publishing an endpoint file the rest of the project reads —
the same pattern Cerberus uses for models (`endpoints.json`).

## Why this shape
- **Read-only at runtime**: only `src/db/db_orchestrator.py` writes (population).
  The pipeline (`neo4j_tools`, `legal_search`, evaluation) only reads. So the DB
  is a **single project-internal store** (`<repo>/neo4j`), populated **once**, and
  read concurrently by all runs.
- **Node 0, free port, endpoint file**: no hardcoded host/port. `neo4j_up.sh`
  picks a free port on the compute node and writes `neo4j_endpoint.json`.
- **Decoupled**: the deployer writes the endpoint; the app reads it via
  `settings` (see `src/config.py::_load_neo4j_endpoint`). Credentials come from a
  dedicated config, not the code.

## Files
| File | Role |
|------|------|
| `neo4j_up.sh` | Start/stop Neo4j on node 0; pick free ports; write `neo4j_endpoint.json`. |
| `populate_kb.sh` | One-time, idempotent KB population via `db_orchestrator.py` (marker `<data_dir>/.populated`). |
| `neo4j.config.json` | **gitignored** — real credentials + deploy params (copy from the example). |
| `neo4j.config.example.json` | Committed template. |
| `reqs_populate.txt` | Minimal deps for population (subset of the `lexcausa` env). |

Generated at repo root (both gitignored): `neo4j_endpoint.json` (coordinates) and
`neo4j/` (data store + `.populated` marker).

## One-time setup
```bash
cp deploy/neo4j/neo4j.config.example.json deploy/neo4j/neo4j.config.json
# edit: set "sandbox" to your Neo4j Singularity sandbox, and "password".
# secrets never committed: neo4j.config.json is gitignored; NEO4J_PASSWORD env overrides.
```

## Sequence (done automatically by run.slurm, on node 0)
```bash
export NEO4J_ENDPOINTS="$PWD/neo4j_endpoint.json"
deploy/neo4j/neo4j_up.sh        # 1) Neo4j up on node 0 → neo4j_endpoint.json
cerberus up &                   # 2) models
deploy/neo4j/populate_kb.sh     # 3) populate ONCE (skipped if marker present)
python scripts/run_multi_doe_ibisco.py ...   # 4) use (reads endpoint via settings)
deploy/neo4j/neo4j_up.sh stop   # teardown (run.slurm's trap does this)
```

Inspect / repopulate:
```bash
deploy/neo4j/populate_kb.sh --check     # counts only
deploy/neo4j/populate_kb.sh --clean     # wipe + repopulate (rewrites marker)
FORCE=1 deploy/neo4j/populate_kb.sh     # ignore marker
```

## Parallel runs
A Neo4j store is opened by **one** process (store-lock). Since the DB is read-only
at runtime:
- **Same allocation, multiple experiment processes** → they all read the one Neo4j
  on node 0. Nothing to do.
- **Separate concurrent allocations** → each has its own node 0 and would clash on
  the shared store. Give each its own store + endpoint:
  ```bash
  # in that run: point data_dir (config) and the endpoint elsewhere
  export NEO4J_ENDPOINTS="$RUN_DIR/neo4j_endpoint.json"
  # set "data_dir" in a per-run neo4j.config.json to a run-local path
  ```

## Credentials
- Real secrets live in `deploy/neo4j/neo4j.config.json` (gitignored).
- `NEO4J_PASSWORD` / `NEO4J_USER` env vars override the config.
- `neo4j_endpoint.json` carries **coordinates only** (host/ports/uri/user), never
  the password.
- The `system` db is recreated on each boot so the configured password is always
  authoritative (the application db `neo4j` is untouched).
