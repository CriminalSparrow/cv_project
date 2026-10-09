#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/model_analyzer
python3 scripts/make_perf_input.py
model-analyzer profile -f profiling/config.yaml 2>&1 | tee results/model_analyzer/run.log
