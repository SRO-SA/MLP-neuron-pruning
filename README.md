# qwen_swiglu_pruning

> **Research prototype — not a production optimisation.**
> This project tests a static RMSNorm-bounded SwiGLU neuron pruning score for
> structured MLP width reduction.  No novelty is claimed.

---

## Project goal

Measure how much of a Qwen2.5-0.5B language model's MLP capacity can be
removed statically (without training or fine-tuning) while minimising the
impact on perplexity.

The target reductions are:
- **MLP parameter count** — fewer stored weights, smaller checkpoint
- **Theoretical MLP FLOPs** — proportional to the surviving intermediate width
- **Generation quality** — tracked via perplexity and greedy-decode samples

---

## Theory

### SwiGLU MLP

Each transformer layer contains a three-projection MLP:

```
g  = r · W_gate.T            # [seq, d_ff]
u  = r · W_up.T              # [seq, d_ff]
a  = SiLU(g) ⊙ u             # SwiGLU gating
m  = a · W_down.T            # [seq, d_model]
```

Neuron `i`'s contribution (for a single token vector `r ∈ ℝ^d_model`) is:

```
c_i(r) = SiLU(r · w_gate_i) × (r · w_up_i) × w_down_i
```

where:
- `w_gate_i = W_gate[i, :]`  (row i of gate_proj, shape `[d_model]`)
- `w_up_i   = W_up[i,   :]`  (row i of up_proj,   shape `[d_model]`)
- `w_down_i = W_down[:, i]`  (col i of down_proj, shape `[d_model]`)

### RMSNorm input bound

The MLP input `r` passes through a RMSNorm with learnable scale `γ`.  For any
input `x`:

```
r_k = x_k / RMS(x) × γ_k   →   ||r||_2 ≤ R = √d_model × ||γ||_∞
```

### Proposed score (method `rmsnorm_bound_angle`)

Using `|SiLU(x)| ≤ |x|` and Cauchy–Schwarz:

```
||c_i(r)|| ≤ R² × ( ||w_gate_i|| × ||w_up_i|| + |w_gate_i · w_up_i| ) / 2
                 × ||w_down_i||
```

The `(norm_product + dot_product) / 2` term is tighter than the pure
norm-product bound: it penalises neurons whose gate and up vectors are nearly
**orthogonal** (small dot product → small output regardless of input norm).

---

## Pruning methods

| ID | Name | Score formula |
|----|------|---------------|
| A  | `random`              | Uniform random (baseline) |
| B  | `down_norm`           | `‖w_down_i‖` |
| C  | `product_norm`        | `‖w_gate_i‖ × ‖w_up_i‖ × ‖w_down_i‖` |
| D  | `rmsnorm_bound_angle` | `R² × (‖w_gate_i‖ × ‖w_up_i‖ + |w_gate_i · w_up_i|) / 2 × ‖w_down_i‖` |

Neurons with the **lowest score** are pruned first.

---

## File layout

```
qwen_swiglu_pruning/
├── run_experiment.py      ← single entry point
├── requirements.txt
├── configs/
│   └── default.yaml       ← all hyperparameters live here
├── src/
│   ├── model_utils.py     ← load model, get_mlp_weights, count_parameters
│   ├── scoring.py         ← compute_scores, get_keep_indices
│   ├── pruning.py         ← prune_layer_mlp, prune_model
│   ├── evaluation.py      ← evaluate_perplexity, run_generation_tests
│   ├── flops.py           ← estimate_mlp_flops
│   └── diagnostics.py     ← per-layer MLP norm logging (no pruning)
├── results/               ← CSV + JSON output written here
└── scripts/
    └── run_qwen05b.sh     ← convenience shell wrapper
```

---

## Fresh pod setup — reproducible locked environment (recommended)

The paper benchmarks were produced in an exactly pinned environment.
`setup_repro_env.sh` creates an isolated venv with those exact versions,
verifies them with `--strict`, and writes `env_fingerprint.json`.

```bash
cd /root/workspace/MLP-neuron-pruning/qwen_swiglu_pruning
RESET_ENV=1 bash setup_repro_env.sh
source /workspace/venvs/qwen-pruning/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
SKIP_DOWNLOAD=1 bash scripts/ready_check.sh
```

**Locked environment (from `env_expected.yaml`):**

| Package | Version |
|---------|---------|
| Python | 3.10.12 |
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| torchaudio | 2.7.1+cu128 |
| transformers | 4.57.6 |
| datasets | 3.6.0 |
| accelerate | 1.14.0 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| pyyaml | 6.0.3 |

**Expected Qwen/Qwen3-30B-A3B model layout on the locked pod:**

```
expert_layout      : unpacked (nn.ModuleList of 128 independent experts)
expert[0].gate_proj: [768, 2048]
expert[0].up_proj  : [768, 2048]
expert[0].down_proj: [2048, 768]
```

If the model loads with a different layout (e.g. packed fused tensors),
`check_env.py --strict` will fail with: *"Model layout differs from the
locked paper environment."*

### Environment guard in `run_experiment.py`

`run_experiment.py` checks the environment at startup and prints a warning
if the locked venv is not active or package versions differ.  To bypass:

```bash
ALLOW_UNLOCKED_ENV=1 python run_experiment.py --config configs/default.yaml
```

For strict paper-benchmark mode (fails hard if env is wrong):

```bash
PAPER_BENCHMARK_MODE=1 python run_experiment.py --config configs/default.yaml
```

### Manual install (without the reproducible setup script)

```bash
# Step 1 — PyTorch stack (exact CUDA 12.8 wheels):
pip install -r requirements-torch-cu128.txt

# Step 2 — all other deps:
pip install -r requirements-locked.txt

# Step 3 — verify:
python scripts/check_env.py --strict --skip-model-layout
```

> **torchaudio note:** Install torch, torchvision, and torchaudio together
> from the same wheel index.  A version mismatch causes `undefined symbol`
> errors when Transformers loads model weights.  `requirements-torch-cu128.txt`
> pins all three to the same compatible release.

---

## Install (original / minimal)

```bash
# Python 3.10+, CUDA optional
pip install -r requirements.txt
```

The script downloads `Qwen/Qwen2.5-0.5B` from the HuggingFace Hub on first
run (~1 GB).  Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` if you
have already cached the model locally.

---

## Run

### Full experiment (all methods × all ratios)

```bash
python run_experiment.py --config configs/default.yaml
```

Or using the shell script:

```bash
bash scripts/run_qwen05b.sh
```

### Quick smoke test (one method, two ratios)

```bash
python run_experiment.py --config configs/default.yaml \
    --methods rmsnorm_bound_angle \
    --pruning-ratios 0.0 0.2
```

### Diagnostic mode (no pruning — logs per-layer MLP output norms)

```bash
python run_experiment.py --config configs/default.yaml --diagnostics-only
```

### Bound analysis mode (threshold-based pruning, not fixed ratios)

```bash
# Full pipeline: distributions + calibration + PPL experiments
python run_experiment.py --config configs/default.yaml --bound-analysis

# Distributions and count tables only (no PPL)
python run_experiment.py --config configs/default.yaml --bound-analysis --no-ppl

# PPL without activation verification
python run_experiment.py --config configs/default.yaml --bound-analysis --no-activation-verification

# Quick PPL: only cumul_score_sum at alpha=1e-4/1e-3/1e-2
python run_experiment.py --config configs/default.yaml --bound-ppl-only

# Compare static bound scores to calibration-data activation scores
python run_experiment.py --config configs/default.yaml --activation-verification-only
```

### Override device / dtype inline

Edit `configs/default.yaml` or pass a modified config.

---

## Expected outputs

After a run, the `results/` directory will contain:

```
results/
├── results_YYYYMMDD_HHMMSS.csv   ← summary table (one row per run)
├── results_YYYYMMDD_HHMMSS.json  ← full results incl. generation examples
└── generations_YYYYMMDD_HHMMSS.json  ← per-prompt generated text
```

The CSV columns are:

```
model_name, pruning_method, pruning_ratio,
total_params_before, total_params_after,
mlp_params_before, mlp_params_after, mlp_params_reduction_pct,
mlp_flops_before, mlp_flops_after, mlp_flops_reduction_pct,
perplexity, perplexity_delta, forward_pass_ok, notes
```

---

## What to expect (rough ballpark)

### Fixed-ratio pruning (--main experiment)

| Ratio | PPL increase (typical) | MLP FLOP reduction |
|-------|------------------------|--------------------|
| 5%    | ~0.1–0.5               | ~5%                |
| 10%   | ~0.5–2.0               | ~10%               |
| 20%   | ~2.0–10                | ~20%               |
| 30%   | may diverge            | ~30%               |

### Bound analysis mode (--bound-analysis)

**Finding: the RMSNorm worst-case bound is highly conservative for Qwen2.5-0.5B.**

When running `--bound-analysis` on Qwen2.5-0.5B:

- **Zero neurons** fall below any tested absolute threshold (up to 1e-2).
- **Zero neurons** fall below any tested relative threshold (score/median < 1e-2).
- Only the **cumulative budget** criterion (cumul_score_sum) selects any neurons:
  - α = 1e-4 → ~25 neurons (≈ 0.021% of all MLP neurons)
  - α = 1e-3 → ~315 neurons (≈ 0.270%)
  - α = 1e-2 → ~2452 neurons (≈ 2.100%)
- The **calibrated budget** (cumul_mlp_norm) selects zero neurons at all α
  because each neuron's worst-case bound score is small relative to the actual
  MLP output norm — confirming how conservative the bound is.

The α = 1e-4 result (25 neurons removed) produced a PPL change of approximately
+0.21 on a preliminary 12-sample corpus. **This must be reproduced on real
WikiText-2 before drawing conclusions** (set `use_fallback_corpus: false` in the
config to ensure the real dataset is used).

**Interpretation:** The theory is not wrong — it is simply very conservative.
The static weight-based worst-case bound cannot certify most neurons as
near-zero because their theoretical maximum contribution is non-negligible.
Data-driven calibration (activation scores, Wanda-style) is likely necessary
for practical structured pruning at useful sparsity levels.

Use `--activation-verification-only` to measure how well the static bound
correlates with actual activation magnitudes on calibration data.

---

## Limitations / caveats

- **No fine-tuning.** Accuracy degrades monotonically with pruning ratio.
  Recovery requires at least a few steps of post-pruning training.
- **Static scores only.** No activation statistics are used.  Dynamic or
  calibration-based scores (e.g. Wanda, SparseGPT) typically perform better.
- **Uniform per-layer pruning.** The same ratio is applied to every layer.
  Layer-adaptive allocation (e.g. sensitivity-based) is left for future work.
- **Small model.** Qwen2.5-0.5B is relatively small; larger models tend to be
  more compressible.
- This is a **research prototype**.  No stability guarantees.  Do not use in
  production.
