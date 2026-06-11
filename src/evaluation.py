"""
evaluation.py
=============
Perplexity evaluation and generation quality tests.

Perplexity
----------
We compute token-level negative log-likelihood on WikiText-2 (test split).
If the datasets library or network is unavailable, a small built-in fallback
corpus is used so the script can still run in offline environments.

    PPL = exp( -1/N * Σ_t log p(x_t | x_<t) )

where N is the total number of predicted tokens.

Generation tests
----------------
We run a small set of fixed prompts with greedy decoding and record the
generated text.  These serve as a qualitative sanity check.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in fallback corpus (used if WikiText-2 cannot be loaded)
# ---------------------------------------------------------------------------
FALLBACK_TEXTS = [
    "The transformer architecture was introduced in the paper "
    "Attention Is All You Need by Vaswani et al. in 2017. "
    "It relies entirely on attention mechanisms, dispensing with recurrence and convolutions.",

    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasises code readability, notably using significant indentation. "
    "Python is dynamically typed and garbage-collected.",

    "The human brain is the central organ of the human nervous system, "
    "and with the spinal cord makes up the central nervous system. "
    "The brain consists of the cerebrum, the brainstem and the cerebellum.",

    "In physics, the speed of light in vacuum, commonly denoted c, "
    "is a universal physical constant equal to 299,792,458 metres per second. "
    "According to the special theory of relativity, c is the upper limit for the speed "
    "at which conventional matter or energy can travel through space.",

    "Machine learning is a method of data analysis that automates analytical model building. "
    "It is a branch of artificial intelligence based on the idea that systems can learn from data, "
    "identify patterns and make decisions with minimal human intervention.",

    "The Great Wall of China is a series of walls and fortifications that were built "
    "across the historical northern borders of ancient Chinese states and Imperial China "
    "as protection against various nomadic groups from the Eurasian Steppe.",

    "Quantum mechanics is a fundamental theory in physics that provides a description of "
    "the physical properties of nature at the scale of atoms and subatomic particles. "
    "It is the foundation of all quantum physics.",

    "The Internet is a global system of interconnected computer networks that uses the "
    "Internet protocol suite to communicate between networks and devices. "
    "It carries a vast range of information resources and services.",

    "Climate change refers to long-term shifts in temperatures and weather patterns. "
    "These shifts may be natural, such as through variations in the solar cycle. "
    "But since the 1800s, human activities have been the main driver of climate change.",

    "The periodic table is a tabular arrangement of the chemical elements, "
    "ordered by their atomic number, electron configuration, and recurring chemical properties. "
    "The rows are called periods and the columns are called groups.",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(max_samples: int = 512) -> List[str]:
    """
    Try to load WikiText-2 test split.  Fall back to FALLBACK_TEXTS if
    datasets / network is unavailable.
    """
    try:
        from datasets import load_dataset  # type: ignore
        logger.info("Loading WikiText-2 (test split) …")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in ds if len(row["text"].strip()) > 50]
        texts = texts[:max_samples]
        logger.info("WikiText-2 loaded: %d samples", len(texts))
        return texts
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load WikiText-2 (%s). Using built-in fallback corpus.", exc
        )
        return FALLBACK_TEXTS


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def evaluate_perplexity(
    model,
    tokenizer,
    texts: Optional[List[str]] = None,
    max_samples: int = 512,
    max_seq_len: int = 512,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate per-token cross-entropy perplexity.

    Returns dict with keys: 'perplexity', 'nll_mean', 'n_tokens'.
    """
    if texts is None:
        texts = load_eval_dataset(max_samples)

    if device is None:
        device = str(next(model.parameters()).device)

    model.eval()
    total_nll   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating perplexity"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize with padding
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids      = enc["input_ids"].to(device)       # [B, T]
            attention_mask = enc["attention_mask"].to(device)  # [B, T]

            # Forward pass; labels = input_ids shifted inside the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            # outputs.loss is mean NLL over non-padding tokens
            loss = outputs.loss  # scalar

            # Count non-padding tokens (exclude the first token which has
            # no prediction target after the left-shift)
            n_tokens = attention_mask[:, 1:].sum().item()

            total_nll    += loss.item() * n_tokens
            total_tokens += n_tokens

            # Free CUDA memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if total_tokens == 0:
        logger.warning("No tokens evaluated — returning PPL = inf")
        return {"perplexity": float("inf"), "nll_mean": float("inf"), "n_tokens": 0}

    nll_mean   = total_nll / total_tokens
    perplexity = math.exp(nll_mean)

    logger.info("Perplexity: %.4f  (NLL=%.4f, tokens=%d)", perplexity, nll_mean, total_tokens)
    return {"perplexity": perplexity, "nll_mean": nll_mean, "n_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Generation sanity tests
# ---------------------------------------------------------------------------

GENERATION_PROMPTS = [
    "The capital of France is",
    "Explain what a GPU is in simple terms.",
    "Write a short Python function to add two numbers.",
    "The key idea behind transformer models is",
    "Once upon a time, in a land far away,",
]


def run_generation_tests(
    model,
    tokenizer,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 80,
    device: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Run greedy-decoding generation on a fixed set of prompts.

    Returns a list of dicts with keys 'prompt' and 'generated'.
    """
    if prompts is None:
        prompts = GENERATION_PROMPTS
    if device is None:
        device = str(next(model.parameters()).device)

    model.eval()
    results = []

    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # greedy
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            generated_ids = out[0][enc["input_ids"].shape[-1]:]
            generated     = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append({"prompt": prompt, "generated": generated})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results
