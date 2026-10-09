"""Explicit zero tensor: compatible with Model Analyzer 1.36's input-data parser."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "results/input-data-0.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"data": [{"IMAGE": [0] * (256 * 256 * 3)}]},
                           separators=(",", ":")) + "\n")
