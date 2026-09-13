# ai-ccas -- see CLAUDE.md for the rules these targets enforce.
.DEFAULT_GOAL := help
UV := uv

.PHONY: help install check fmt lint types test test-fast latency security voice evals workbench clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## sync all extras + dev group
	$(UV) sync --all-extras --group dev

fmt:  ## format in place
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

lint:  ## lint without fixing
	$(UV) run ruff format --check src tests scripts
	$(UV) run ruff check src tests scripts

types:  ## mypy --strict
	$(UV) run mypy

test:  ## full suite
	$(UV) run pytest

test-fast:  ## unit only
	$(UV) run pytest -m "not slow and not integration"

latency:  ## latency budget gate (CLAUDE.md Rule 3)
	$(UV) run pytest tests/latency

security:  ## zero-leakage + domain-agnosticism gates (Rules 1 & 2)
	$(UV) run pytest tests/security

workbench:  ## interactive console at http://127.0.0.1:8000 (loopback only)
	$(UV) run uvicorn ccas.api.main:create_app --factory --host 127.0.0.1 --port 8000 --reload

voice:  ## the frozen voice channel (ADR-0018). Must stay green before resuming voice
	$(UV) run pytest -m voice -q

evals:  ## provider parity (Rule 6). Ragas/DeepEval suites land with Module 6
	$(UV) run pytest tests/evals -q -rs

check: lint types test voice  ## must be green before any commit

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
