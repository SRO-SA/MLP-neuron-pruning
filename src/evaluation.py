"""
evaluation.py
=============
Perplexity evaluation and generation quality tests.

──────────────────────────────────────────────────────────────────────────
BUG FIX (v2)
──────────────────────────────────────────────────────────────────────────
The original code passed `labels=input_ids` directly to the model.
When the batch contains padded sequences, padding token IDs are included
in the loss computation.  HuggingFace CausalLM shifts labels by 1 and
computes cross-entropy with ignore_index=-100; if we do not set
`labels[padding_positions] = -100`, padding tokens contribute to the mean
loss, inflating PPL and making it incomparable across models with different
effective sequence lengths.

Fix: always set `labels[attention_mask == 0] = -100` before passing to
the model.  Also count n_tokens as the number of non-padded *shifted*
positions, which matches HF's internal denominator exactly.

──────────────────────────────────────────────────────────────────────────
PPL formula
──────────────────────────────────────────────────────────────────────────
    PPL = exp( -1/N * Σ_t log p(x_t | x_{<t}) )

HF reports loss = mean NLL per token (over non-(-100) positions).
We reconstruct total NLL by multiplying by n_tokens, then divide by the
running total to get the corpus-level mean NLL.
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
    "It relies entirely on attention mechanisms, dispensing with recurrence and convolutions "
    "entirely. The model achieved state-of-the-art results on machine translation tasks.",

    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasises code readability, notably using significant indentation. "
    "Python is dynamically typed and garbage-collected. "
    "It supports multiple programming paradigms including structured, object-oriented and functional.",

    "The human brain is the central organ of the human nervous system, "
    "and with the spinal cord makes up the central nervous system. "
    "The brain consists of the cerebrum, the brainstem and the cerebellum. "
    "It controls most of the activities of the body.",

    "In physics, the speed of light in vacuum, commonly denoted c, "
    "is a universal physical constant equal to 299,792,458 metres per second. "
    "According to the special theory of relativity, c is the upper limit for the speed "
    "at which conventional matter or energy can travel through space.",

    "Machine learning is a method of data analysis that automates analytical model building. "
    "It is a branch of artificial intelligence based on the idea that systems can learn from data, "
    "identify patterns and make decisions with minimal human intervention. "
    "Deep learning uses neural networks with many layers.",

    "The Great Wall of China is a series of walls and fortifications that were built "
    "across the historical northern borders of ancient Chinese states and Imperial China "
    "as protection against various nomadic groups from the Eurasian Steppe. "
    "Construction started as early as the 7th century BC.",

    "Quantum mechanics is a fundamental theory in physics that provides a description of "
    "the physical properties of nature at the scale of atoms and subatomic particles. "
    "It is the foundation of all quantum physics including quantum chemistry, "
    "quantum field theory, quantum technology, and quantum information science.",

    "The Internet is a global system of interconnected computer networks that uses the "
    "Internet protocol suite to communicate between networks and devices. "
    "It carries a vast range of information resources and services, "
    "such as the interlinked hypertext documents and applications of the World Wide Web.",

    "Climate change refers to long-term shifts in temperatures and weather patterns. "
    "These shifts may be natural, such as through variations in the solar cycle. "
    "But since the 1800s, human activities have been the main driver of climate change, "
    "primarily due to the burning of fossil fuels.",

    "The periodic table is a tabular arrangement of the chemical elements, "
    "ordered by their atomic number, electron configuration, and recurring chemical properties. "
    "The rows are called periods and the columns are called groups. "
    "Elements in the same group share similar chemical properties.",

    "Neural networks are computing systems inspired by biological neural networks that "
    "constitute animal brains. An artificial neural network consists of layers of nodes "
    "that process information using connectionist approaches to computation. "
    "Modern deep networks have billions of parameters.",

    "The solar system consists of the Sun and the objects that orbit it. "
    "It formed approximately 4.6 billion years ago from the gravitational collapse "
    "of a giant molecular cloud. The vast majority of the system's mass is in the Sun, "
    "with most of the remaining mass contained in the planet Jupiter.",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Canonical names for supported eval datasets
SUPPORTED_EVAL_DATASETS = ("wikitext2", "c4", "wikitext103", "lambada")


def load_eval_dataset(
    max_samples: int = 512,
    use_fallback_corpus: bool = True,
    dataset_name: str = "wikitext2",
) -> List[str]:
    """
    Load an evaluation text corpus.

    Parameters
    ----------
    max_samples : int
        Maximum number of text samples to return.
    use_fallback_corpus : bool
        If True, fall back to FALLBACK_TEXTS when the dataset cannot be loaded.
        Set to False in the config when you need trustworthy perplexity numbers.
    dataset_name : str
        One of: "wikitext2" (default), "c4", "wikitext103", "lambada".
        Unrecognised names fall through to wikitext2 with a warning.
    """
    ds_key = dataset_name.lower().strip()

    loaders = {
        "wikitext2":   _load_wikitext2,
        "wikitext103": _load_wikitext103,
        "c4":          _load_c4,
        "lambada":     _load_lambada,
    }
    loader = loaders.get(ds_key)
    if loader is None:
        logger.warning(
            "Unknown dataset_name '%s'; falling back to wikitext2.", dataset_name
        )
        loader = _load_wikitext2

    try:
        texts = loader(max_samples)
        logger.info("Dataset '%s' loaded: %d samples", ds_key, len(texts))
        return texts
    except Exception as exc:  # noqa: BLE001
        if not use_fallback_corpus:
            logger.error(
                "Could not load '%s': %s\n"
                "  Re-run with use_fallback_corpus: true to use the built-in "
                "fallback corpus (PPL will be indicative only).",
                dataset_name, exc,
            )
            raise
        logger.warning(
            "Could not load '%s' (%s). Using built-in fallback corpus "
            "(note: PPL computed on %d short samples — indicative only).",
            dataset_name, exc, len(FALLBACK_TEXTS),
        )
        return FALLBACK_TEXTS


def load_all_eval_datasets(
    dataset_names: List[str],
    max_samples: int = 512,
    use_fallback_corpus: bool = True,
) -> "Dict[str, List[str]]":
    """
    Load multiple evaluation datasets at once.

    Returns
    -------
    Dict mapping dataset_name → list of text samples.
    If a dataset fails and use_fallback_corpus=True, it maps to FALLBACK_TEXTS.
    """
    out: Dict[str, List[str]] = {}
    for name in dataset_names:
        out[name] = load_eval_dataset(
            max_samples=max_samples,
            use_fallback_corpus=use_fallback_corpus,
            dataset_name=name,
        )
    return out


# ---------------------------------------------------------------------------
# Per-dataset loader helpers
# ---------------------------------------------------------------------------

def _load_wikitext2(max_samples: int) -> List[str]:
    from datasets import load_dataset  # type: ignore
    ds    = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"] for row in ds if len(row["text"].strip()) > 100]
    return texts[:max_samples]


def _load_wikitext103(max_samples: int) -> List[str]:
    from datasets import load_dataset  # type: ignore
    ds    = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    texts = [row["text"] for row in ds if len(row["text"].strip()) > 100]
    return texts[:max_samples]


def _load_c4(max_samples: int) -> List[str]:
    from datasets import load_dataset  # type: ignore
    # C4 validation split; streaming avoids downloading the full ~300 GB
    ds    = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for row in ds:
        t = row.get("text", "").strip()
        if len(t) > 100:
            texts.append(t)
        if len(texts) >= max_samples:
            break
    return texts


def _load_lambada(max_samples: int) -> List[str]:
    from datasets import load_dataset  # type: ignore
    ds    = load_dataset("EleutherAI/lambada_openai", split="test")
    texts = [row["text"] for row in ds if len(row["text"].strip()) > 20]
    return texts[:max_samples]


# ---------------------------------------------------------------------------
# Perplexity (main)
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

    Correctly masks padding tokens so they are excluded from the loss.

    Returns dict with keys: 'perplexity', 'nll_mean', 'n_tokens'.
    """
    if texts is None:
        texts = load_eval_dataset(max_samples)

    if device is None:
        device = str(next(model.parameters()).device)

    model.eval()
    total_nll    = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating perplexity"):
            batch_texts = texts[i : i + batch_size]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids      = enc["input_ids"].to(device)       # [B, T]
            attention_mask = enc["attention_mask"].to(device)  # [B, T]

            # ── KEY FIX: mask padding tokens so they don't contribute to loss ──
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100   # ignore_index for HF cross-entropy

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # outputs.loss = mean NLL over the non-(-100) shifted positions
            loss = outputs.loss

            # Count non-padding prediction targets (positions 1..T, where label != -100)
            # HF shifts by 1 internally: labels[..., 1:] are the targets.
            # After our masking, padding positions in labels[..., 1:] are -100.
            n_tokens = (labels[:, 1:] != -100).sum().item()

            if n_tokens == 0:
                continue

            total_nll    += loss.item() * n_tokens
            total_tokens += n_tokens

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if total_tokens == 0:
        logger.warning("No tokens evaluated — returning PPL = inf")
        return {"perplexity": float("inf"), "nll_mean": float("inf"), "n_tokens": 0}

    nll_mean   = total_nll / total_tokens
    perplexity = math.exp(min(nll_mean, 20.0))  # cap at exp(20) ≈ 5e8 to avoid overflow

    logger.info(
        "Perplexity: %.4f  (NLL=%.4f, tokens=%d)", perplexity, nll_mean, total_tokens
    )
    return {"perplexity": perplexity, "nll_mean": nll_mean, "n_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Perplexity sanity check on a tiny, fixed text (no padding)
# ---------------------------------------------------------------------------

SANITY_TEXT = (
    "The transformer architecture was introduced in the paper Attention Is All You Need "
    "by Vaswani et al. in 2017. It relies entirely on self-attention mechanisms, dispensing "
    "with recurrence and convolutions. The model encodes each token by attending to all "
    "other tokens in the sequence, weighted by learned key-query dot products."
)


def evaluate_on_fixed_text(
    model,
    tokenizer,
    device: Optional[str] = None,
    text: str = SANITY_TEXT,
    label: str = "model",
) -> Dict:
    """
    Evaluate perplexity on a single, fixed text with NO padding.
    This removes all ambiguity about padding handling and gives a clean PPL number.

    Interpretation
    --------------
    - For Qwen2.5-0.5B on this text, expected PPL ≈ 10-30 (if eval is correct).
    - If PPL > 1000, the evaluation or model is broken.
    - If PPL for the pruned model is > 5× baseline on this same text, pruning is aggressive.
    """
    if device is None:
        device = str(next(model.parameters()).device)

    model.eval()
    enc      = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]               # [1, T]  (no padding)
    n_tokens  = input_ids.shape[1] - 1        # first token has no prediction target

    with torch.no_grad():
        out  = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss.item()                 # mean NLL, no padding, so exact

    ppl = math.exp(min(loss, 20.0))

    print(f"\n  PPL sanity check [{label}]:")
    print(f"    Text (first 80 chars): '{text[:80]}…'")
    print(f"    Tokens : {n_tokens}")
    print(f"    Loss   : {loss:.4f}")
    print(f"    PPL    : {ppl:.4f}")

    if ppl > 500:
        print(f"    ⚠  PPL is very high — model or evaluation may be broken!")
    elif ppl > 50:
        print(f"    ⚠  PPL is elevated — check model or dataset match.")
    else:
        print(f"    ✓  PPL looks reasonable for a language model.")

    return {"label": label, "loss": loss, "n_tokens": n_tokens, "perplexity": ppl}


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
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = out[0][enc["input_ids"].shape[-1]:]
            generated_ids = out[0][enc["input_ids"].shape[-1]:]
            generated     = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append({"prompt": prompt, "generated": generated})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results
