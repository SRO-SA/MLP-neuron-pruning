"""CPU test for per-document PPL sufficient-statistic collection."""
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from src.evaluation import evaluate_perplexity


class _ToyTokenizer:
    def __call__(
        self, texts, *, return_tensors, padding, truncation, max_length
    ):
        del return_tensors, padding, truncation
        sequences = []
        for text in texts:
            length = min(max(len(text.split()) + 1, 2), max_length)
            sequences.append(torch.arange(1, length + 1) % 5)
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = sequence
            attention_mask[row, :len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _UniformCausalLM(nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, labels=None):
        del attention_mask, labels
        logits = torch.zeros(
            (*input_ids.shape, self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        ) + self.anchor
        return SimpleNamespace(logits=logits)


class PairedNLLEvaluationTests(unittest.TestCase):
    def test_collects_one_aligned_record_per_document(self):
        result = evaluate_perplexity(
            _UniformCausalLM(),
            _ToyTokenizer(),
            texts=["one two", "one two three four"],
            max_seq_len=16,
            batch_size=2,
            device="cpu",
            collect_per_example=True,
        )
        self.assertEqual(result["n_tokens"], 6)
        self.assertEqual(len(result["per_example"]), 2)
        self.assertEqual(
            [row["n_tokens"] for row in result["per_example"]], [2, 4]
        )
        self.assertAlmostEqual(result["perplexity"], 5.0, places=5)
        for index, row in enumerate(result["per_example"]):
            self.assertEqual(row["sample_index"], index)
            self.assertAlmostEqual(row["nll_mean"], torch.log(torch.tensor(5.0)).item())


if __name__ == "__main__":
    unittest.main()
