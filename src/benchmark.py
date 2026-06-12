"""
benchmark.py
============
Real inference latency / throughput / memory benchmark.

Measures
--------
- prefill_latency_ms  : time from input to first token (batch=1, no KV cache)
- decode_latency_ms   : time per step averaged over max_new_tokens decode steps
- total_latency_ms    : prefill + full decode
- tokens_per_sec      : max_new_tokens / (total_latency_s)
- peak_gpu_memory_MB  : torch.cuda.max_memory_allocated() after the run

Benchmark protocol
------------------
- batch_size = 1 (latency benchmark, not throughput)
- greedy decoding (do_sample=False, temperature=1.0)
- prompt padded/truncated to exactly `prompt_len` tokens
- generate max_new_tokens new tokens
- warmup: 2 runs before timing (amortises JIT / CUDA graph effects)
- timing: median of `n_repeats` runs

Usage
-----
    from src.benchmark import run_inference_benchmark
    results = run_inference_benchmark(
        model, tokenizer, device=device,
        prompt_lens=[128, 512, 1024],
        max_new_tokens=128,
        n_repeats=5,
    )
    # results is a list of dicts, one per prompt_len

    from src.benchmark import run_benchmark_mode
    run_benchmark_mode(cfg, device, output_dir, model_configs=[
        ("baseline",  baseline_model),
        ("pruned_6%", pruned_model),
    ])
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Fixed prompts — same across all benchmark runs for reproducibility
BENCHMARK_PROMPTS = [
    "The transformer architecture was introduced in the paper Attention Is All You Need. "
    "This seminal work by Vaswani et al. replaced recurrent and convolutional networks "
    "with a purely attention-based model, achieving state-of-the-art results on machine "
    "translation and other sequence modelling tasks.",

    "In physics, quantum mechanics is a fundamental theory that provides a description "
    "of the physical properties of nature at the scale of atoms and subatomic particles. "
    "It is the foundation of all quantum physics, including quantum chemistry, quantum "
    "field theory, quantum technology, and quantum information science. Classical physics, "
    "the collection of theories that existed before the advent of modern physics, does not "
    "adequately describe these systems at small scales, and quantum mechanics was developed "
    "to explain their behavior.",

    "Python is a high-level, general-purpose programming language that emphasises code "
    "readability. It supports multiple programming paradigms including structured, object-"
    "oriented, and functional programming. Python is dynamically typed and garbage-collected. "
    "It was created by Guido van Rossum and first released in 1991. Python's comprehensive "
    "standard library contributes to its popularity across a wide range of applications, "
    "from web development to data science. The language's design philosophy values explicit "
    "code and has led to a large and active community of developers who contribute to its "
    "extensive ecosystem of libraries and tools.",
]

BENCHMARK_CSV_KEYS = [
    "label",
    "prompt_len_tokens",
    "max_new_tokens",
    "n_repeats",
    "warmup_runs",
    "prefill_median_ms",
    "prefill_p5_ms",
    "prefill_p95_ms",
    "decode_per_step_median_ms",
    "total_median_ms",
    "tokens_per_sec",
    "peak_gpu_memory_MB",
    "device",
    "dtype",
    "notes",
]


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def _sync():
    """GPU barrier — ensures CUDA ops finish before timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_single_config(
    model,
    tokenizer,
    device:         str,
    prompt_len:     int   = 512,
    max_new_tokens: int   = 128,
    n_repeats:      int   = 5,
    warmup_runs:    int   = 2,
    prompt_text:    str   = "",
) -> Dict:
    """
    Benchmark prefill + decode for a single (prompt_len, max_new_tokens) config.

    Returns a dict with timing and memory statistics.
    """
    if not prompt_text:
        # Use a concatenated version of BENCHMARK_PROMPTS
        prompt_text = " ".join(BENCHMARK_PROMPTS)

    # Build a prompt of exactly prompt_len tokens by truncating/padding
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_len,
        padding="max_length",
    )
    input_ids      = enc["input_ids"].to(device)       # [1, prompt_len]
    attention_mask = enc["attention_mask"].to(device)  # [1, prompt_len]

    model.eval()

    prefill_times: List[float] = []
    total_times:   List[float] = []
    peak_mbs:      List[float] = []

    total_runs = warmup_runs + n_repeats

    for run_i in range(total_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        _sync()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                max_new_tokens  = max_new_tokens,
                do_sample       = False,
                pad_token_id    = tokenizer.eos_token_id,
            )

        _sync()
        t1 = time.perf_counter()

        total_ms   = (t1 - t0) * 1000.0
        peak_mb    = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if torch.cuda.is_available() else 0.0
        )

        if run_i >= warmup_runs:
            total_times.append(total_ms)
            peak_mbs.append(peak_mb)

    # Prefill-only timing: one additional forward pass with no generation
    for run_i in range(warmup_runs + n_repeats):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        _sync()
        t1 = time.perf_counter()
        if run_i >= warmup_runs:
            prefill_times.append((t1 - t0) * 1000.0)

    import statistics
    pref_sorted = sorted(prefill_times)
    tot_sorted  = sorted(total_times)

    pref_med = statistics.median(prefill_times)
    pref_p5  = pref_sorted[max(0, int(0.05 * len(pref_sorted)))]
    pref_p95 = pref_sorted[min(len(pref_sorted)-1, int(0.95 * len(pref_sorted)))]
    tot_med  = statistics.median(total_times)

    # Decode latency: total - prefill, averaged over max_new_tokens steps
    decode_per_step_ms = (
        (tot_med - pref_med) / max_new_tokens
        if max_new_tokens > 0 else 0.0
    )
    tokens_per_sec = (
        max_new_tokens / (tot_med / 1000.0)
        if tot_med > 0 else 0.0
    )
    peak_mb_med = statistics.median(peak_mbs) if peak_mbs else 0.0

    result = {
        "prompt_len_tokens":          prompt_len,
        "max_new_tokens":             max_new_tokens,
        "n_repeats":                  n_repeats,
        "warmup_runs":                warmup_runs,
        "prefill_median_ms":          round(pref_med, 2),
        "prefill_p5_ms":              round(pref_p5,  2),
        "prefill_p95_ms":             round(pref_p95, 2),
        "decode_per_step_median_ms":  round(decode_per_step_ms, 3),
        "total_median_ms":            round(tot_med,  2),
        "tokens_per_sec":             round(tokens_per_sec, 1),
        "peak_gpu_memory_MB":         round(peak_mb_med, 1),
    }
    return result


def run_inference_benchmark(
    model,
    tokenizer,
    device:         str,
    label:          str         = "model",
    prompt_lens:    List[int]   = None,
    max_new_tokens: int         = 128,
    n_repeats:      int         = 5,
    warmup_runs:    int         = 2,
    dtype:          str         = "unknown",
) -> List[Dict]:
    """
    Run the benchmark across multiple prompt lengths.

    Returns a list of result dicts (one per prompt_len), each augmented with
    'label', 'device', 'dtype'.
    """
    if prompt_lens is None:
        prompt_lens = [128, 512, 1024]

    rows = []
    for plen in prompt_lens:
        logger.info(
            "[benchmark] label=%s  prompt_len=%d  max_new_tokens=%d",
            label, plen, max_new_tokens,
        )
        print(
            f"  Benchmarking {label}  prompt_len={plen}  "
            f"max_new_tokens={max_new_tokens} ... ",
            end="", flush=True,
        )
        try:
            r = benchmark_single_config(
                model, tokenizer, device,
                prompt_len=plen,
                max_new_tokens=max_new_tokens,
                n_repeats=n_repeats,
                warmup_runs=warmup_runs,
            )
            r["label"]  = label
            r["device"] = device
            r["dtype"]  = dtype
            r["notes"]  = ""
            print(
                f"total={r['total_median_ms']:.0f}ms  "
                f"prefill={r['prefill_median_ms']:.0f}ms  "
                f"tps={r['tokens_per_sec']:.0f}  "
                f"mem={r['peak_gpu_memory_MB']:.0f}MB"
            )
        except Exception as exc:
            logger.error("benchmark failed label=%s plen=%d: %s", label, plen, exc)
            r = {k: "" for k in BENCHMARK_CSV_KEYS}
            r.update({"label": label, "prompt_len_tokens": plen,
                       "device": device, "dtype": dtype, "notes": f"ERROR: {exc}"})
            print(f"FAILED: {exc}")
        rows.append(r)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


# ---------------------------------------------------------------------------
# Convenience: benchmark mode run from main experiment loop
# ---------------------------------------------------------------------------

def run_benchmark_mode(
    cfg:            dict,
    device:         str,
    output_dir:     str,
    model_configs:  List[Tuple[str, object, object]],
    prompt_lens:    Optional[List[int]] = None,
    max_new_tokens: int  = 128,
    n_repeats:      int  = 5,
) -> None:
    """
    Run inference benchmark for a list of (label, model, tokenizer) triples.

    Parameters
    ----------
    model_configs : list of (label, model, tokenizer)
        The baseline should be first, then pruned variants.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"benchmark_{ts}.csv")
    json_path = os.path.join(output_dir, f"benchmark_{ts}.json")

    if prompt_lens is None:
        prompt_lens = cfg.get("benchmark_prompt_lens", [128, 512, 1024])
    if isinstance(prompt_lens, str):
        prompt_lens = [int(x) for x in prompt_lens.split(",")]

    max_new_tokens = int(cfg.get("benchmark_max_new_tokens", max_new_tokens))
    n_repeats      = int(cfg.get("benchmark_n_repeats", n_repeats))

    W = 110
    print(f"\n{'=' * W}")
    print("INFERENCE BENCHMARK")
    print(f"  prompt_lens    : {prompt_lens}")
    print(f"  max_new_tokens : {max_new_tokens}")
    print(f"  n_repeats      : {n_repeats}")
    print(f"{'=' * W}\n")

    all_rows: List[Dict] = []

    for label, model, tokenizer in model_configs:
        print(f"\n--- {label} ---")
        rows = run_inference_benchmark(
            model, tokenizer, device,
            label=label,
            prompt_lens=prompt_lens,
            max_new_tokens=max_new_tokens,
            n_repeats=n_repeats,
            warmup_runs=2,
            dtype=str(next(model.parameters()).dtype),
        )
        all_rows.extend(rows)

    # Write CSV
    write_hdr = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=BENCHMARK_CSV_KEYS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Write JSON
    with open(json_path, "w") as fh:
        json.dump(all_rows, fh, indent=2, default=str)

    # Summary table
    print(f"\n{'=' * W}")
    print("BENCHMARK SUMMARY")
    print(f"{'─' * W}")
    print(f"  {'label':>30}  {'plen':>6}  {'prefill_ms':>10}  {'decode_ms/tok':>14}  "
          f"{'tps':>8}  {'mem_MB':>8}")
    print(f"{'─' * W}")
    for r in all_rows:
        print(
            f"  {str(r.get('label',''))[:30]:>30}  "
            f"{str(r.get('prompt_len_tokens','')):>6}  "
            f"{str(r.get('prefill_median_ms','')):>10}  "
            f"{str(r.get('decode_per_step_median_ms','')):>14}  "
            f"{str(r.get('tokens_per_sec','')):>8}  "
            f"{str(r.get('peak_gpu_memory_MB','')):>8}"
        )
    print(f"{'=' * W}")
    print(f"\nBenchmark CSV  : {csv_path}")
    print(f"Benchmark JSON : {json_path}\n")
