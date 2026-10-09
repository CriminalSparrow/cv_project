#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/local_analyzer
# Model Analyzer 1.36 requires a JSON file instead of input-data=zero.
python3 - <<'PY'
import json
with open("results/input-data-0.json", "w") as f:
    json.dump({"data": [{"IMAGE": [0] * (256 * 256 * 3)}]}, f)
PY
model-analyzer profile -f profiling/config.yaml 2>&1 | tee results/local_analyzer/run.log
