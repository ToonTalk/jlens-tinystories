#!/usr/bin/env bash
# Regenerates every artifact from a clean checkout.
# Prereqs: python 3.10+, git, a CUDA GPU (CPU works, slower).
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv 2>/dev/null || true
if [ -f .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

$PY -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
[ -d jacobian-lens ] || git clone --depth 1 https://github.com/anthropics/jacobian-lens
$PY -m pip install --quiet -e ./jacobian-lens
$PY -m pip install --quiet "transformers>=5.5" datasets huggingface_hub numpy matplotlib

$PY src/smoke_test.py                 # step 0: walkthrough flow on a tiny model
$PY src/fluency_gate.py               # step 1: both models must be fluent
for FIT in A B C; do                  # step 2: three fits, 100 then 1000 prompts
  $PY src/fit_lens.py $FIT 100
  $PY src/fit_lens.py $FIT 1000
done
$PY src/evaluate.py A B C             # step 3: readouts, divergence, slice pages
$PY src/make_report.py                # step 4: report.md + appendix.md
echo "Done. See out/report.md"
