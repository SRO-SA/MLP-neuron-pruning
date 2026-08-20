#!/usr/bin/env python3
"""Run a command while recording wall time and sampled NVIDIA memory usage."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time


def _gpu_used_bytes() -> list[int]:
    try:
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
        ], text=True, stderr=subprocess.DEVNULL)
        return [int(line.strip()) * 1024 * 1024 for line in text.splitlines()
                if line.strip()]
    except (OSError, subprocess.CalledProcessError, ValueError):
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("command is required after --")
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stop = threading.Event(); samples = []
    baseline_values = _gpu_used_bytes()
    baseline_total = sum(baseline_values)

    def monitor() -> None:
        while not stop.is_set():
            values = _gpu_used_bytes()
            samples.append({"elapsed_seconds": time.perf_counter() - started,
                            "used_bytes_by_gpu": values,
                            "used_bytes_total": sum(values)})
            stop.wait(args.poll_seconds)

    started = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    process = subprocess.run(command, check=False)
    stop.set(); thread.join(timeout=max(2.0, args.poll_seconds * 2))
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1, "command": command, "exit_code": process.returncode,
        "wall_clock_seconds": elapsed, "poll_seconds": args.poll_seconds,
        "peak_gpu_memory_used_bytes_total": max(
            (row["used_bytes_total"] for row in samples), default=0
        ),
        "baseline_gpu_memory_used_bytes_total": baseline_total,
        "baseline_gpu_memory_used_bytes_by_gpu": baseline_values,
        "peak_incremental_gpu_memory_used_bytes_total": max(
            (row["used_bytes_total"] - baseline_total for row in samples),
            default=0,
        ),
        "peak_gpu_memory_used_bytes_by_gpu": [
            max((row["used_bytes_by_gpu"][i] for row in samples
                 if len(row["used_bytes_by_gpu"]) > i), default=0)
            for i in range(max((len(row["used_bytes_by_gpu"]) for row in samples),
                               default=0))
        ],
        "sample_count": len(samples), "samples": samples,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
