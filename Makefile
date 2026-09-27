# omni-ai-ccas -- see CLAUDE.md for the rules these targets enforce.
.DEFAULT_GOAL := help
# `--frozen` honours uv.lock without re-resolving it. Without it every `uv run`
# re-checks the en-core-web-sm direct URL against GitHub, and a slow or blocked fetch
# fails the command *before* it executes anything -- which presents as a hang with no
# child process and 0% CPU, not as a network error. See docs/future-scoped-work.md 9.14.
UV := uv run --frozen
UVX := uv

.PHONY: help install check fmt lint types test test-fast latency security voice evals eval-router workbench browser clean

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

browser:  ## browser tests for the console (ADR-0022, dev-only; needs Node + npx)
	@command -v npx >/dev/null 2>&1 || { echo "npx not found -- install Node.js to run the browser tests (dev-only, ADR-0022)"; exit 2; }
	npm install --silent --no-fund --no-audit
	@npx --yes playwright install chromium >/dev/null 2>&1 || true
	@OPENROUTER_API_KEY= TYPESAFE_API_KEY= ANTHROPIC_API_KEY= \
	  $(UV) uvicorn ccas.api.main:create_app --factory --host 127.0.0.1 --port 8765 --no-access-log & \
	  server_pid=$$!; \
	  trap 'kill $$server_pid 2>/dev/null' EXIT; \
	  for i in $$(seq 1 60); do curl -sf http://127.0.0.1:8765/health >/dev/null && break; sleep 0.5; done; \
	  npx --yes playwright test

voice:  ## the frozen voice channel (ADR-0018). Must stay green before resuming voice
	$(UV) pytest -m voice -q

eval-router:  ## measure router accuracy on held-out rows (9.18). Costs 1 LLM call/row
	$(UV) python scripts/eval_router.py --domain retail --limit 40

evals:  ## provider parity (Rule 6). Ragas/DeepEval suites land with Module 6
	$(UV) pytest tests/evals -q -rs

check: lint types test voice  ## must be green before any commit

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
