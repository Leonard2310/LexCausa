#!/bin/bash

# Helper script to start LexCausa backend in Docker for DoE mode

set -e

COMPOSE_FILE="${1:-compose.yml}"
HEALTHCHECK_TIMEOUT="${HEALTHCHECK_TIMEOUT:-60}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jpassword}"

echo "🐳 Starting LexCausa backend in Docker..."
echo "   Compose file: $COMPOSE_FILE"

# Resolve docker compose command (modern plugin vs legacy standalone)
if docker compose version &>/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose is not installed or not in PATH"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

echo "   Using compose command: $COMPOSE_CMD"

# Start services
echo "📦 Starting services..."
$COMPOSE_CMD -f "$COMPOSE_FILE" up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy (max ${HEALTHCHECK_TIMEOUT}s)..."
for i in $(seq 1 "$HEALTHCHECK_TIMEOUT"); do
    # Check Neo4j health
    if $COMPOSE_CMD -f "$COMPOSE_FILE" exec -T neo4j \
            cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1" &>/dev/null; then
        echo "✅ Neo4j is healthy"

        # Check API health
        if curl -sf http://localhost:8000/health | grep -q '"status"'; then
            echo "✅ Flask API is healthy"
            echo ""
            echo "✅ All services are running!"
            echo ""
            echo "📍 Service endpoints:"
            echo "   - Neo4j Browser: http://localhost:7474"
            echo "   - Neo4j Bolt:    bolt://localhost:7687"
            echo "   - Flask API:     http://localhost:8000"
            echo ""
            echo "To run DoE:"
            echo "  python scripts/run_multi_doe.py \\"
            echo "    --claims-file claims.md \\"
            echo "    --models gpt_oss_120b,groq_llama_scout_17b \\"
            echo "    --replicates 10 \\"
            echo "    --out experiments/multi_doe/runs/\$(date +%Y%m%d_%H%M%S)"
            echo ""
            echo "To stop services:"
            echo "  $COMPOSE_CMD -f $COMPOSE_FILE down"
            exit 0
        fi
    fi

    echo "   Attempt $i/$HEALTHCHECK_TIMEOUT..."
    sleep 1
done

echo "❌ Timeout waiting for services to be healthy"
echo ""
echo "Checking container status:"
$COMPOSE_CMD -f "$COMPOSE_FILE" ps

echo ""
echo "Checking logs:"
$COMPOSE_CMD -f "$COMPOSE_FILE" logs --tail=20

exit 1
