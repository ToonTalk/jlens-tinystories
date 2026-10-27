#!/usr/bin/env bash
# After run_fits.sh: 110M skip-first check, remaining expert weights, final
# evaluations, swap batteries, report. Each python step retried once (the
# transient Windows cuBLAS crash).
cd "$(dirname "$0")/../.."
PY=/c/Users/toont/Documents/GitHub/jpp_lens/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
until grep -q -E "ALL-DONE|STEP-FAILED" out/jpp/run_fits.log; do sleep 30; done
grep -q "STEP-FAILED" out/jpp/run_fits.log && { echo "fits failed"; exit 1; }
run() { echo "RUN $* $(date +%T)"; $PY "$@" || { echo "RETRY $*"; $PY "$@" || { echo "FAILED $*"; exit 1; }; }; }
run src/jpp/skip_check.py stories110M profile
run src/jpp/skip_check.py stories110M refit
run src/jpp/fit_weights.py stories110M rlens_skip16 1000 period
run src/jpp/fit_weights.py stories110M rlens_skip16 1000 mixed
run src/jpp/fit_weights.py stories110M none_skip16 100 period
run src/jpp/evaluate_jpp.py stories110M
run src/jpp/swap_demo.py stories110M
echo FINISH-DONE
