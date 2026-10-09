#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
perf_analyzer -m selfie_segmentation -i http -u "${1:-server:8000}" \
  --concurrency-range 1:8:1 --measurement-mode time_windows \
  --measurement-interval 3000 --stability-percentage 10 --max-trials 10 \
  --percentile 95 --input-data zero \
  -f results/local_perf.csv 2>&1 | tee results/local_perf.log
