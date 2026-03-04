.PHONY: install dev lint format test test-unit test-integration coverage docker-build docker-up docker-down run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest -x

test-unit:
	pytest tests/unit -x

test-integration:
	pytest tests/integration -x

coverage:
	pytest --cov=src --cov-report=html --cov-fail-under=80
	@echo "Open htmlcov/index.html for the coverage report"

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

run:
	python scripts/run_pipeline.py $(CONFIG)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage coverage.xml dist build *.egg-info
