"""Drive the whole system end to end, once, and print what happened.

A chat caller (the ADR-0018 proving channel) runs through the real API surface: session
created, turns spoken, the graph routing and dispatching tools, and -- when the exchange
escalates -- the handoff retained, fetched the way an agent desktop would fetch it, and
streamed the way a copilot would watch it. A judge verdict is then recorded for the
exchange, because that is the other half of Module 6.

    uv run python scripts/e2e_run.py                  # scripted caller, live provider
    uv run python scripts/e2e_run.py --utterances "..." "..."
    uv run python scripts/e2e_run.py --judge-only     # verdict on the finished exchange

Calls are paced and capped, so a free-tier quota cannot be silently burnt: every LLM
call is printed as it happens and ``--max-calls`` bounds the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from ccas.api.main import build_app
from ccas.api.sessions import SessionManager
from ccas.config.settings import Settings
from ccas.evals.judge import SessionJudge
from ccas.llm.bindings import load_bindings
from ccas.observability.logging import configure_logging, get_logger

LOG = get_logger("scripts.e2e_run")

RULE = "─" * 76

DEFAULT_UTTERANCES = (
    "Hi, I never got my order and I want to know where it is",
    "It was order 4417-2290, placed last week",
    "No, that is everything, thank you",
)


class CallBudget:
    """Counts LLM-backed steps so a run cannot quietly outrun the daily cap.

    A caller turn costs one or two calls (router + responder); the judge costs one.
    ``max_calls`` bounds the *steps*, so the true call ceiling is roughly 2n+1.
    """

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0

    def spend(self, what: str) -> None:
        self.used += 1
        print(f"  [step {self.used}/{self.max_calls}] {what}")
        if self.used > self.max_calls:
            raise SystemExit(
                f"call budget exhausted ({self.max_calls}); stopping rather than burning quota"
            )


def build_manager(settings: Settings) -> SessionManager:
    from ccas.api.sessions import SessionManager as _Manager

    return _Manager(
        settings=settings, bindings=load_bindings(settings.models_config), provider=None
    )


async def judge_exchange_state(
    manager: SessionManager, session_id: str, budget: CallBudget
) -> dict[str, Any] | None:
    """The M6b verdict on the exchange, through the same manager the session ran on."""
    handle = manager.get(session_id)
    judge = SessionJudge(manager.bindings, manager.settings)
    state = handle.snapshot()

    async def once() -> Any:
        budget.spend(f"judge on {session_id} ({len(state.turns)} turns)")
        try:
            return await judge.judge_session(E2E_RULES, state, sampled=False)
        finally:
            await judge.aclose()

    verdict = await once()
    if verdict is None:
        return None
    return {
        "passed": verdict.passed,
        "judge": f"{verdict.judge_provider}:{verdict.judge_model}",
        "scores": {s.dimension.value: round(s.score, 3) for s in verdict.scores},
    }


E2E_RULES = (
    "Be concise. Answer only from the tool results; escalate when the data is missing. "
    "Never state a value the record does not contain."
)


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    budget = CallBudget(args.max_calls)
    manager = build_manager(settings)
    client = TestClient(build_app(manager))

    utterances = tuple(args.utterances) if args.utterances else DEFAULT_UTTERANCES
    print(
        f"{RULE}\nomni-ai-ccas end to end :: domain={args.domain}"
        f" :: {len(utterances)} turns\n{RULE}"
    )

    created = client.post("/v1/sessions", json={"domain": args.domain})
    if created.status_code != 200:
        print(f"error: session creation failed: {created.text}", file=sys.stderr)
        return 2
    session_id = created.json()["session_id"]
    print(f"session       {session_id}")

    final: dict[str, Any] = {}
    for text in utterances:
        budget.spend(f"turn: {text[:48]!r}")
        started = time.perf_counter()
        response = client.post(f"/v1/sessions/{session_id}/turns", json={"text": text})
        if response.status_code != 200:
            print(f"error: turn failed: {response.text}", file=sys.stderr)
            return 2
        final = response.json()
        intent = final.get("current_intent")
        print(
            f"turn {final['turn_index']:>2}      {time.perf_counter() - started:>5.1f}s  "
            f"intent={intent and intent['intent_id']} "
            f"conf={intent and round(intent['confidence'], 2)} "
            f"tools={len(final['tool_records'])} terminal={final['terminal']}"
        )
        if final["terminal"]:
            break

    print(f"\nsnapshot      {json.dumps(_summary(final), indent=2)}")

    # --- the agent-desktop surface -------------------------------------------------
    escalated = final.get("escalation") is not None
    print(f"\nescalated     {escalated}")
    if escalated:
        fetched = client.get(f"/v1/sessions/{session_id}/handoff")
        if fetched.status_code == 200:
            body = fetched.json()
            print(
                f"handoff       {body['handoff_id']}  queue={body['target_queue']}  "
                f"reason={body['reason']}"
            )
            print(f"cti           {body['cti_attributes']}")
        else:
            print(f"handoff       MISSING ({fetched.status_code}) -- a leak in the surface")

    # --- the copilot stream --------------------------------------------------------
    with client.websocket_connect(f"/v1/ws/copilot/{session_id}") as ws:
        seen = 0
        while True:
            message = json.loads(ws.receive_text())
            seen += 1
            if "handoff" in message:
                print(f"copilot       {seen} snapshot(s) then handoff, stream closed")
                break
            if seen > 16:
                print("copilot       (no handoff pushed; session not escalated)")
                break

    # --- the judge ------------------------------------------------------------------
    if not args.skip_judge:
        verdict = await judge_exchange_state(manager, session_id, budget)
        if verdict is None:
            print("judge         unmeasured (quota or provider unavailable)")
        else:
            print(f"judge         {verdict['judge']} -> passed={verdict['passed']}")
            for dim, score in verdict["scores"].items():
                print(f"  {dim:<18} {score:>5.2f}")

    print(
        f"\n{RULE}\nLLM-backed steps this run: {budget.used} "
        "(budget "
        f"{budget.max_calls}; a turn costs 1-2 calls, the judge 1)"
    )
    LOG.info(
        "e2e.run",
        correlation_id=session_id,
        domain=args.domain,
        turns=len(utterances),
        escalated=escalated,
        llm_calls=budget.used,
    )
    return 0


def _summary(final: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_index": final.get("turn_index"),
        "terminal": final.get("terminal"),
        "intent": final.get("current_intent") and final["current_intent"]["intent_id"],
        "slots": final.get("slots"),
        "tool_records": [
            {"tool": r["tool"], "status": r["status"]} for r in final.get("tool_records", [])
        ],
        "escalation": final.get("escalation") and final["escalation"]["reason"],
        "latency_total_ms": final.get("latency", {}).get("total_rtt_ms"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="e2e_run", description=__doc__)
    p.add_argument("--domain", default=None)
    p.add_argument(
        "--utterances", nargs="*", default=None, help="caller turns; default is a scripted caller"
    )
    p.add_argument("--max-calls", type=int, default=12, help="hard cap on LLM calls this run")
    p.add_argument("--skip-judge", action="store_true", help="stop after the desktop surface")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.domain is None:
        args.domain = Settings().default_domain
    configure_logging(Path("logs/execution.log"), console=False)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
