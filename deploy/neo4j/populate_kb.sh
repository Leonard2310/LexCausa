#!/bin/bash
# =============================================================================
#  populate_kb.sh — populate the LexCausa Neo4j KB (schema + statutes +
#  precedents) via src/db/db_orchestrator.py. Idempotent: runs ONCE.
#
#  Reads connection coordinates from neo4j_endpoint.json (written by
#  neo4j_up.sh) and credentials from neo4j.config.json (env NEO4J_PASSWORD
#  wins), then exports NEO4J_URI/USER/PASSWORD for db_orchestrator.
#
#  A marker file <data_dir>/.populated makes a default run a no-op after the
#  first success, so re-running the orchestrator (or restarting jobs) never
#  re-populates. --clean or FORCE=1 repopulate; --check only inspects.
#
#  Python: uses $PY311 if set, else `python3` (the active env — the main
#  `lexcausa` env already has the needed deps; see reqs_populate.txt for a
#  dedicated light env).
#
#  Usage:
#    deploy/neo4j/populate_kb.sh            # populate once (skip if marker)
#    deploy/neo4j/populate_kb.sh --check    # inspect DB status only
#    deploy/neo4j/populate_kb.sh --clean    # wipe + repopulate (rewrites marker)
#    FORCE=1 deploy/neo4j/populate_kb.sh    # repopulate ignoring the marker
# =============================================================================
set -uo pipefail
ulimit -u 8192 2>/dev/null || ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

CONFIG_FILE="$HERE/neo4j.config.json"
[ -f "$CONFIG_FILE" ] || CONFIG_FILE="$HERE/neo4j.config.example.json"

eval "$(python3 - "$CONFIG_FILE" <<'PY'
import json, shlex, sys
c = json.load(open(sys.argv[1]))
def g(k, d=""): return c.get(k, d)
print(f'CFG_USER={shlex.quote(str(g("user","neo4j")))}')
print(f'CFG_PASSWORD={shlex.quote(str(g("password","neo4jpassword")))}')
print(f'CFG_DATA_DIR={shlex.quote(str(g("data_dir","")))}')
PY
)"

DATA_DIR="${CFG_DATA_DIR:-}"; [ -n "$DATA_DIR" ] || DATA_DIR="$REPO/neo4j"
ENDPOINT_FILE="${NEO4J_ENDPOINTS:-$REPO/neo4j_endpoint.json}"
MARKER="$DATA_DIR/.populated"
PY311="${PY311:-python3}"

# ── Endpoint must exist (neo4j_up.sh runs first) ─────────────────────────────
[ -f "$ENDPOINT_FILE" ] || { echo "!! $ENDPOINT_FILE not found — run neo4j_up.sh first"; exit 1; }
eval "$(python3 - "$ENDPOINT_FILE" <<'PY'
import json, shlex, sys
e = json.load(open(sys.argv[1]))
print(f'EP_HOST={shlex.quote(str(e["host"]))}')
print(f'EP_BOLT={int(e["bolt_port"])}')
print(f'EP_HTTP={int(e["http_port"])}')
print(f'EP_USER={shlex.quote(str(e.get("user","neo4j")))}')
PY
)"

export NEO4J_URI="bolt://${EP_HOST}:${EP_BOLT}"
export NEO4J_USER="${NEO4J_USER:-$EP_USER}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-$CFG_PASSWORD}"

# ── Idempotency: default populate is a no-op once the marker exists ───────────
if [ $# -eq 0 ] && [ -f "$MARKER" ] && [ "${FORCE:-0}" != "1" ]; then
    echo ">> KB already populated (marker: $MARKER). FORCE=1 or --clean to repopulate. Skipping."
    exit 0
fi

# ── Neo4j must be reachable ───────────────────────────────────────────────────
if ! curl -sf "http://${EP_HOST}:${EP_HTTP}" >/dev/null 2>&1; then
    echo "!! Neo4j not reachable at http://${EP_HOST}:${EP_HTTP} — is neo4j_up.sh running?"; exit 1
fi

echo "== Populating KB =="
echo "   URI:    $NEO4J_URI"
echo "   Python: $PY311"
echo "   Repo:   $REPO"
PYTHONPATH="$REPO/src" "$PY311" "$REPO/src/db/db_orchestrator.py" "$@"
rc=$?

# ── Write marker only after a successful FULL (re)populate ────────────────────
if [ "$rc" -eq 0 ] && { [ $# -eq 0 ] || printf '%s\n' "$@" | grep -q -- '--clean'; }; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
    echo ">> Marker written: $MARKER"
fi
exit "$rc"
