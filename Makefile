# omni-ai-ccas -- see CLAUDE.md for the rules these targets enforce.
.DEFAULT_GOAL := help
# `--frozen` honours uv.lock without re-resolving it. Without it every `uv run`
# re-checks the en-core-web-sm direct URL against GitHub, and a slow or blocked fetch
# fails the command *before* it executes anything -- which presents as a hang with no
# child process and 0% CPU, not as a network error. See docs/future-scoped-work.md 9.14.
UV := uv run --frozen
UVX := uv

.PHONY: help install check fmt lint types test test-fast latency security voice evals workbench clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## sync all extras + dev group
	$(UVX) sync --all-extras --group dev

fmt:  ## format in place
	$(UV) ruff format src tests scripts
	$(UV) ruff check --fix src tests scripts

lint:  ## lint without fixing
	$(UV) ruff format --check src tests scripts
	$(UV) ruff check src tests scripts

types:  ## mypy --strict
	$(UV) mypy

test:  ## full suite
	$(UV) pytest

test-fast:  ## unit only
	$(UV) pytest -m "not slow and not integration"

latency:  ## voice latency budget gate (Rule 3). Frozen with voice -- ADR-0018
	$(UV) pytest tests/latency -m 'voice or latency'

security:  ## zero-leakage + domain-agnosticism gates (Rules 1 & 2)
	$(UV) pytest tests/security

workbench:  ## interactive console at http://127.0.0.1:8000 (loopback only)
	$(UV) uvicorn ccas.api.main:create_app --factory --host 127.0.0.1 --port 8000 --reload

voice:  ## the frozen voice channel (ADR-0018). Must stay green before resuming voice
	$(UV) pytest -m voice -q

evals:  ## provider parity (Rule 6). Ragas/DeepEval suites land with Module 6
	$(UV) pytest tests/evals -q -rs

check: lint types test voice  ## must be green before any commit

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
