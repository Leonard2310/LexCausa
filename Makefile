SHELL := /bin/bash

.PHONY: help install install-frontend setup up down logs backend frontend run dev dev-stop test format lint

help:
	@echo "Targets disponibili:"
	@echo "  make setup           # installa dipendenze backend + frontend"
	@echo "  make up              # avvia Neo4j (docker compose up -d)"
	@echo "  make down            # ferma Neo4j"
	@echo "  make backend         # avvia backend Flask"
	@echo "  make frontend        # avvia frontend Vite"
	@echo "  make dev             # avvia tutto con un solo comando"
	@echo "  make dev-stop        # stop processi locali backend/frontend + Neo4j"
	@echo "  make test            # test backend"
	@echo "  make lint            # lint backend"
	@echo "  make format          # format backend"

install:
	poetry install --no-root

install-frontend:
	cd src/frontend && npm install

setup: install install-frontend

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

backend:
	poetry run python src/api_server.py

frontend:
	cd src/frontend && npm run dev

run: backend

dev:
	@pkill -f "python.*src/api_server.py" >/dev/null 2>&1 || true
	@pkill -f "node .*vite" >/dev/null 2>&1 || true
	@docker compose up -d neo4j
	@echo "Avvio backend + frontend (Ctrl+C ferma i processi locali; Neo4j resta attivo)..."
	@trap 'kill 0' INT TERM EXIT; \
	poetry run python src/api_server.py & \
	for i in {1..60}; do \
		if curl -sf http://127.0.0.1:8000/health >/dev/null; then \
			break; \
		fi; \
		sleep 0.5; \
	done; \
	if ! curl -sf http://127.0.0.1:8000/health >/dev/null; then \
		echo "Backend non raggiungibile su http://127.0.0.1:8000/health"; \
		exit 1; \
	fi; \
	(cd src/frontend && npm run dev) & \
	wait

dev-stop:
	-pkill -f "python.*src/api_server.py" || true
	-pkill -f "node .*vite" || true
	@docker compose stop neo4j

test:
	poetry run pytest -v

format:
	poetry run black src
	poetry run isort src

lint:
	poetry run ruff check src
	poetry run mypy src

