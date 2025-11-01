install:
	poetry install

run:
	poetry run uvicorn src.main:app --reload

test:
	poetry run pytest -v

format:
	poetry run black src tests
	poetry run isort src tests

lint:
	poetry run flake8 src tests
	poetry run mypy src

