#!/bin/bash
# =============================================================================
#  neo4j_up.sh — deploy Neo4j for LexCausa on IBiSCo, Cerberus-style.
#
#  Runs on NODE 0 of the allocation (the batch node where this script executes,
#  NOT the login node). Picks a FREE Bolt/HTTP port, starts Neo4j from a
#  Singularity sandbox bound to a project-internal data dir, waits until ready,
#  and writes `neo4j_endpoint.json` — the single source of truth the rest of the
#  project reads to connect (exactly like Cerberus' endpoints.json).
#
#  Neo4j is READ-ONLY at pipeline runtime (only db_orchestrator writes, once),
#  so the data dir is a single project-internal store shared across runs:
#  default <repo_root>/neo4j . One Neo4j process owns the store (store-lock);
#  parallel runs read it concurrently. For fully separate concurrent
#  ALLOCATIONS, override `data_dir` (config) and NEO4J_ENDPOINTS per run.
#
#  Config: deploy/neo4j/neo4j.config.json (falls back to *.example.json).
#  Env overrides: NEO4J_PASSWORD, NEO4J_ENDPOINTS, CONSOLE_LOG_DIR.
#
#  Usage:
#    deploy/neo4j/neo4j_up.sh            # start (idempotent-ish: reuses a live server)
#    deploy/neo4j/neo4j_up.sh stop       # stop the server started here
# =============================================================================
set -uo pipefail
ulimit -u 8192 2>/dev/null || ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

CONFIG_FILE="$HERE/neo4j.config.json"
[ -f "$CONFIG_FILE" ] || CONFIG_FILE="$HERE/neo4j.config.example.json"
[ -f "$CONFIG_FILE" ] || { echo "!! No Neo4j config in $HERE"; exit 1; }

# ── Load config (flat JSON → shell vars) ─────────────────────────────────────
eval "$(python3 - "$CONFIG_FILE" <<'PY'
import json, shlex, sys
c = json.load(open(sys.argv[1]))
def g(k, d=""): return c.get(k, d)
br = g("bolt_port_range", [47600, 47699]); hr = g("http_port_range", [47700, 47799])
print(f'CFG_USER={shlex.quote(str(g("user","neo4j")))}')
print(f'CFG_PASSWORD={shlex.quote(str(g("password","neo4jpassword")))}')
print(f'CFG_SANDBOX={shlex.quote(str(g("sandbox","")))}')
print(f'CFG_DATA_DIR={shlex.quote(str(g("data_dir","")))}')
print(f'CFG_BOLT_START={int(br[0])}'); print(f'CFG_BOLT_END={int(br[1])}')
print(f'CFG_HTTP_START={int(hr[0])}'); print(f'CFG_HTTP_END={int(hr[1])}')
print(f'CFG_HEAP={shlex.quote(str(g("heap_max","2G")))}')
print(f'CFG_PAGECACHE={shlex.quote(str(g("pagecache","1G")))}')
print(f'CFG_TIMEOUT={int(g("ready_timeout_sec",300))}')
print(f'CFG_APC={int(g("active_processor_count",4))}')
print(f'CFG_BOLT_POOL={int(g("bolt_thread_pool_max",40))}')
PY
)"

NEO4J_USER_="${NEO4J_USER:-$CFG_USER}"
NEO4J_PASS_="${NEO4J_PASSWORD:-$CFG_PASSWORD}"
SANDBOX="$CFG_SANDBOX"
DATA_DIR="${CFG_DATA_DIR:-}"
[ -n "$DATA_DIR" ] || DATA_DIR="$REPO/neo4j"
ENDPOINT_FILE="${NEO4J_ENDPOINTS:-$REPO/neo4j_endpoint.json}"
PID_FILE="$DATA_DIR/neo4j.pid"
CONSOLE_LOG_DIR="${CONSOLE_LOG_DIR:-$DATA_DIR/logs}"

# ── stop ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "stop" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE")"
        echo ">> Stopping Neo4j (pid $pid)…"
        kill "$pid" 2>/dev/null || true; sleep 5; kill -9 "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    # The JVM (org.neo4j) can outlive the singularity wrapper and hold the port.
    pkill -9 -f 'org.neo4j' 2>/dev/null || true
    pkill -9 -f 'docker-entrypoint.sh neo4j' 2>/dev/null || true
    rm -f "$ENDPOINT_FILE"
    echo ">> Done."
    exit 0
fi

[ -d "$SANDBOX" ] || { echo "!! Neo4j sandbox not found: $SANDBOX (set it in neo4j.config.json)"; exit 1; }

HOST="$(hostname -f 2>/dev/null || hostname)"
case "$HOST" in
  *ui*|*login*) echo "!! WARNING: this looks like a login node — Neo4j must run on node 0 (compute)." ;;
esac

# ── free-port picker (first free in [start,end] on this node) ────────────────
find_free_port() {
    python3 - "$1" "$2" <<'PY'
import socket, sys
start, end = int(sys.argv[1]), int(sys.argv[2])
for p in range(start, end + 1):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", p)); s.close(); print(p); break
    except OSError:
        s.close()
else:
    sys.exit("no free port in range")
PY
}
BOLT_PORT="$(find_free_port "$CFG_BOLT_START" "$CFG_BOLT_END")" || { echo "!! no free Bolt port"; exit 1; }
HTTP_PORT="$(find_free_port "$CFG_HTTP_START" "$CFG_HTTP_END")" || { echo "!! no free HTTP port"; exit 1; }

# ── data dir layout ──────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"/{data,logs,run,plugins,import,conf} "$CONSOLE_LOG_DIR"

# The sandbox is read-only on Lustre; docker-entrypoint.sh must write conf/,
# so bind a writable conf/ pre-seeded once with the sandbox defaults.
if [ -z "$(ls -A "$DATA_DIR/conf" 2>/dev/null)" ]; then
    cp -a "$SANDBOX/var/lib/neo4j/conf/." "$DATA_DIR/conf/" 2>/dev/null \
        || echo ">> No default conf in sandbox — entrypoint will create it."
fi

# Inject our dynamic settings into conf/neo4j.conf (idempotent: strip+re-append).
CONF="$DATA_DIR/conf/neo4j.conf"; touch "$CONF"
grep -vE '^(server\.(bolt|http)\.(listen|advertised)_address|server\.default_listen_address|server\.bolt\.thread_pool_max_size|server\.memory\.(heap\.max_size|pagecache\.size))=' \
    "$CONF" > "$CONF.tmp" 2>/dev/null || true
mv "$CONF.tmp" "$CONF"
{
  echo "server.default_listen_address=0.0.0.0"
  echo "server.bolt.listen_address=:${BOLT_PORT}"
  echo "server.http.listen_address=:${HTTP_PORT}"
  echo "server.bolt.advertised_address=${HOST}:${BOLT_PORT}"
  echo "server.http.advertised_address=${HOST}:${HTTP_PORT}"
  echo "server.memory.heap.max_size=${CFG_HEAP}"
  echo "server.memory.pagecache.size=${CFG_PAGECACHE}"
  echo "server.bolt.thread_pool_max_size=${CFG_BOLT_POOL}"
} >> "$CONF"

# Credentials live in the system db; recreate it each boot so the password is
# always the configured one (does NOT touch the application db 'neo4j').
rm -rf "$DATA_DIR/data/databases/system" "$DATA_DIR/data/transactions/system"

echo ">> Starting Neo4j on node 0 ($HOST)"
echo "   Sandbox:  $SANDBOX"
echo "   Data:     $DATA_DIR"
echo "   Bolt:     bolt://${HOST}:${BOLT_PORT}   HTTP: http://${HOST}:${HTTP_PORT}"
echo "   Console:  $CONSOLE_LOG_DIR/neo4j_console.log"

SINGULARITYENV_NEO4J_AUTH="${NEO4J_USER_}/${NEO4J_PASS_}" \
SINGULARITYENV_JAVA_OPTS="-XX:ActiveProcessorCount=${CFG_APC} -XX:ParallelGCThreads=2 -XX:ConcGCThreads=1 -XX:CICompilerCount=2" \
singularity exec --cleanenv --bind /lustre:/lustre \
    --bind "$DATA_DIR/data:/var/lib/neo4j/data" \
    --bind "$DATA_DIR/logs:/var/lib/neo4j/logs" \
    --bind "$DATA_DIR/run:/var/lib/neo4j/run" \
    --bind "$DATA_DIR/plugins:/var/lib/neo4j/plugins" \
    --bind "$DATA_DIR/import:/var/lib/neo4j/import" \
    --bind "$DATA_DIR/conf:/var/lib/neo4j/conf" \
    "$SANDBOX" /startup/docker-entrypoint.sh neo4j \
    > "$CONSOLE_LOG_DIR/neo4j_console.log" 2>&1 &

echo "$!" > "$PID_FILE"
echo ">> Neo4j PID: $(cat "$PID_FILE")"

echo -n ">> Waiting for Neo4j on :${HTTP_PORT} "
waited=0
while true; do
    if curl -sf "http://localhost:${HTTP_PORT}" >/dev/null 2>&1; then
        echo " OK (${waited}s)"
        break
    fi
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo " FAILED (process died — see $CONSOLE_LOG_DIR/neo4j_console.log)"; exit 1
    fi
    sleep 5; waited=$((waited + 5)); echo -n "."
    if [ "$waited" -ge "$CFG_TIMEOUT" ]; then
        echo " TIMEOUT (${waited}s — see $CONSOLE_LOG_DIR/neo4j_console.log)"; exit 1
    fi
done

# ── write the endpoint file (coordinates only; NO password) ──────────────────
python3 - "$ENDPOINT_FILE" "$HOST" "$BOLT_PORT" "$HTTP_PORT" "$NEO4J_USER_" <<'PY'
import json, sys
f, host, bp, hp, user = sys.argv[1:6]
json.dump({
    "host": host,
    "bolt_port": int(bp), "http_port": int(hp),
    "bolt_uri": f"bolt://{host}:{bp}", "http_uri": f"http://{host}:{hp}",
    "user": user,
}, open(f, "w"), indent=2)
print(f">> Wrote endpoint: {f}")
PY

echo "✅ Neo4j ready (bolt://${HOST}:${BOLT_PORT})."
