#!/usr/bin/env bash
# Runs the J++ expert fits in order; each step retried once (fits resume from
# their shard checkpoints, so a transient native crash costs little).
cd "$(dirname "$0")/../.."
PY=/c/Users/toont/Documents/GitHub/jpp_lens/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
for step in "$@"; do
  set -- $step
  echo "STEP-START $step $(date +%T)"
  $PY src/jpp/fit_experts.py "$@" || { echo "STEP-RETRY $step"; $PY src/jpp/fit_experts.py "$@" || { echo "STEP-FAILED $step"; exit 1; }; }
  echo "STEP-DONE $step $(date +%T)"
done
echo ALL-DONE
