#!/usr/bin/env bash
# Wait for the OpenRouter daily-cap reset (00:00 UTC), then spend the approved
# envelope in priority order: the 30-case judge suite under rubric v2 (the open
# 6.12 question), then the Rule 6 parity gate (6.7) if the cap still has room.
# Start with:  nohup bash scripts/run_after_reset.sh &   (from the repo root)
# Kill with:   pkill -f run_after_reset.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs data/interim

# Target is 00:10 UTC (10-minute buffer past the 00:00 reset), computed rather
# than hardcoded, and portable: BSD date has no -d, so the epoch math is Python.
target_epoch=$("$PY" - <<'EOF'
import datetime
now = datetime.datetime.now(datetime.timezone.utc)
t = now.replace(hour=0, minute=10, second=0, microsecond=0)
if t <= now:
    t += datetime.timedelta(days=1)
print(int(t.timestamp()))
EOF
)
sleep_seconds=$(( target_epoch - $(date -u +%s) ))
echo "[run_after_reset] sleeping ${sleep_seconds}s until 00:10 UTC" | tee logs/run_after_reset.log

sleep "$sleep_seconds"

echo "[run_after_reset] quota window open -- judge suite (rubric v2)" | tee -a logs/run_after_reset.log
"$PY" scripts/run_evals.py --out data/interim/judge_suite_rubric_v2.json \
  > logs/judge_suite_rubric_v2.log 2>&1
echo "[run_after_reset] judge suite exit=$? (log: logs/judge_suite_rubric_v2.log)" | tee -a logs/run_after_reset.log

echo "[run_after_reset] parity gate (Rule 6, ~16 calls)" | tee -a logs/run_after_reset.log
"$PY" -m pytest tests/evals/test_provider_parity.py -q \
  > logs/parity_reset_run.log 2>&1
echo "[run_after_reset] parity exit=$? (log: logs/parity_reset_run.log)" | tee -a logs/run_after_reset.log

echo "[run_after_reset] done" | tee -a logs/run_after_reset.log
