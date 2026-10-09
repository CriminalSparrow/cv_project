"""Observe custom metrics before, during and after concurrent real inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.request import urlopen

from infer import infer, load_image, ROOT


def snapshot(url):
    with urlopen(url, timeout=10) as response:
        raw = response.read().decode()
    values = {}
    for name in ("segmentation_processing_seconds_total", "segmentation_requests_in_flight"):
        samples = [float(line.rsplit(" ", 1)[1]) for line in raw.splitlines()
                   if line.startswith(name + "{")]
        if not samples:
            raise AssertionError(f"Metric missing: {name}")
        values[name] = sum(samples)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--metrics-url", default="http://localhost:8002/metrics")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "results/local_metrics.json")
    args = parser.parse_args()
    if args.workers < 1 or args.seconds <= 0:
        parser.error("workers and seconds must be positive")
    image = load_image(ROOT / "examples/person.jpg")
    before = snapshot(args.metrics_url)
    samples = [before]
    deadline = time.monotonic() + args.seconds

    def worker():
        count = 0
        while time.monotonic() < deadline:
            infer(image, args.url, "grpc")
            count += 1
        return count

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker) for _ in range(args.workers)]
        while not all(f.done() for f in futures):
            samples.append(snapshot(args.metrics_url))
            time.sleep(0.02)
        completed = sum(f.result() for f in futures)
    after = snapshot(args.metrics_url)
    samples.append(after)
    counter = "segmentation_processing_seconds_total"
    gauge = "segmentation_requests_in_flight"
    assert before[gauge] == after[gauge] == 0, "Run this check without other clients"
    assert after[counter] > before[counter]
    assert all(b[counter] >= a[counter] for a, b in zip(samples, samples[1:]))
    assert all(s[gauge] >= 0 for s in samples)
    peak = max(s[gauge] for s in samples)
    assert peak > 0, "No active request observed; increase --seconds and retry"
    report = {"completed": completed, "workers": args.workers, "seconds": args.seconds,
              "before": before, "after": after, "peak_in_flight": peak, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
