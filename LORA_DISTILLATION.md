# LoRA Finetuning with Logit Distillation — Design & Implementation Guide

This document explains the LoRA finetuning and logit distillation features added
to the training pipeline. It starts from first principles — no prior knowledge of
recurrence retrofitting, LoRA, or distillation is assumed.

---

## Table of Contents

1. [Background: What problem are we solving?](#1-background-what-problem-are-we-solving)
2. [Retrofitting recurrence into a pretrained LLM](#2-retrofitting-recurrence-into-a-pretrained-llm)
3. [The problem with full finetuning](#3-the-problem-with-full-finetuning)
4. [LoRA: Low-Rank Adaptation](#4-lora-low-rank-adaptation)
5. [Knowledge distillation via logits](#5-knowledge-distillation-via-logits)
6. [Putting it all together: LoRA + Distillation for looped models](#6-putting-it-all-together-lora--distillation-for-looped-models)
7. [Implementation walkthrough](#7-implementation-walkthrough)
8. [CLI reference and examples](#8-cli-reference-and-examples)
9. [Design decisions and trade-offs](#9-design-decisions-and-trade-offs)

---

## 1. Background: What problem are we solving?

Large language models (LLMs) like Llama are **feedforward**: each input token
passes through every layer exactly once. A 32-layer model always does 32 layers
of compute regardless of how hard the question is.

**Recurrent models** can loop over the same block of layers multiple times. Easy
questions get one pass; hard questions get many. This is more efficient — you get
more compute from fewer unique parameters — and more flexible.

But training a recurrent LLM from scratch is expensive. This project
**retrofits** recurrence into an *already-trained* feedforward model, reusing
its existing knowledge. The result is a model that can think in loops while
starting from a strong pretrained baseline.

---

## 2. Retrofitting recurrence into a pretrained LLM

A standard Llama model has layers numbered 0 through N-1 arranged in sequence:

```
Input -> [Layer 0] -> [Layer 1] -> ... -> [Layer N-1] -> Output
```

Retrofitting splits these layers into three groups:

```
Input -> [Prelude: layers 0..P] -> [Recurrent Block: layers P+1..P+B] x R -> [Coda: layers P+B+1..N-1] -> Output
```

- **Prelude** (non-recurrent): runs once, processes raw input into a good
  internal representation.
- **Recurrent block**: runs R times in a loop. Each pass refines the hidden
  state — like iterating on a thought. R is variable at inference time.
- **Coda** (non-recurrent): runs once, converts the refined hidden state into
  output logits.

The conversion scripts in `convert_pretrained_model/` handle the restructuring.
Layer selection is guided by **Block Influence** and **Angular Distance** metrics
(see `dev/layer_analysis.py`) which identify which layers are redundant enough
to be collapsed into a loop.

After conversion, the model architecture is correct but its performance has
degraded — it wasn't trained to loop. The training pipeline (`train.py`) then
finetunes the model to learn how to use recurrence effectively. That is where
LoRA and distillation come in.

---

## 3. The problem with full finetuning

After converting a pretrained model to a looped architecture, we need to
finetune it so it learns to produce good outputs through recurrence. The naive
approach is to update **all** parameters. This has two problems:

1. **Memory**: a 1B-parameter model needs ~4 GB just for the weights in
   bfloat16, plus another ~8 GB for optimizer states (Adam stores two extra
   buffers per parameter). Doubling the model size for the teacher model (see
   below) makes this worse.

2. **Catastrophic forgetting**: updating every weight can erase the pretrained
   knowledge we are trying to preserve. We want the model to learn *how to
   loop*, not to relearn language from scratch.

LoRA addresses both problems.

---

## 4. LoRA: Low-Rank Adaptation

### The core idea

LoRA (Low-Rank Adaptation) freezes the original model weights and injects small
trainable matrices alongside them. For a weight matrix W of shape (d_out, d_in):

```
Original:   output = W @ input
With LoRA:  output = W @ input + (B @ A) @ input
```

Where:
- **W** is frozen (not updated).
- **A** has shape (r, d_in) and **B** has shape (d_out, r).
- **r** (the "rank") is much smaller than d_in or d_out (typically 8-64).

The number of new trainable parameters is r * (d_in + d_out) instead of
d_in * d_out — a massive reduction. For a 2048x2048 matrix with rank 16, that
is 65,536 new parameters instead of 4,194,304 (a 64x reduction).

### Why it works

The key insight from the LoRA paper is that the weight *changes* during
finetuning tend to live in a low-dimensional subspace. You don't need to update
the full matrix — a low-rank perturbation captures most of the useful adaptation.

### LoRA hyperparameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `r` | Rank of the low-rank matrices. Higher = more capacity, more parameters. | 16 |
| `lora_alpha` | Scaling factor. The LoRA output is multiplied by `lora_alpha / r`. Higher alpha = stronger LoRA effect. | 32 |
| `target_modules` | Which weight matrices get LoRA adapters. `"all-linear"` targets every linear layer. | `"all-linear"` |
| `lora_dropout` | Dropout applied to the LoRA path during training. | 0.0 |

---

## 5. Knowledge distillation via logits

### The core idea

The original pretrained model (before looping was added) was good at its job. We
want the looped model to **match** its outputs — specifically, its probability
distribution over the next token.

This is **knowledge distillation**: a *teacher* model (the original) guides a
*student* model (the looped version) by providing soft targets.

### Why soft targets beat hard labels

Suppose the teacher model sees "The capital of France is" and outputs:

| Token   | Probability |
|---------|------------|
| Paris   | 0.85       |
| Lyon    | 0.05       |
| Marseille | 0.03     |
| ...     | ...        |

A hard label just says "Paris". But the soft distribution tells the student that
Lyon and Marseille are more plausible than, say, "banana" — this is richer
supervision. The dark knowledge in these secondary probabilities helps the
student generalize better.

### Temperature scaling

To make the soft distribution even more informative, we divide the logits by a
**temperature** T before computing softmax:

```
softmax(logits / T)
```

- T = 1: normal distribution (peaked).
- T > 1: smoother distribution (spreads probability mass, reveals more about
  relative preferences).
- T = 2 (our default): a moderate smoothing that exposes the teacher's
  ranking of alternatives without losing too much signal.

### The distillation loss

We use **KL divergence** to measure how different the student and teacher
distributions are:

```
KL_loss = KL_div( log_softmax(student_logits / T), softmax(teacher_logits / T) ) * T^2
```

The T^2 factor compensates for the gradients being scaled down by the
temperature division, keeping gradient magnitudes consistent regardless of T.

### Combined loss

The final training loss blends the standard language modeling loss (cross-entropy
against the ground truth labels) with the distillation loss:

```
total_loss = alpha * CE_loss + (1 - alpha) * KL_loss
```

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `alpha` | Weight of the CE loss. At 0.5, both losses contribute equally. | 0.5 |
| `temperature` | Softmax temperature for distillation (see above). | 2.0 |
| `teacher_model_name` | HuggingFace model ID of the original (non-looped) model. | (required) |

---

## 6. Putting it all together: LoRA + Distillation for looped models

The full pipeline works like this:

```
                        +-----------------------+
                        | Original Pretrained   |   <-- Teacher (frozen)
                        | Model (e.g. Llama-1B) |
                        +----------+------------+
                                   |
                            teacher logits
                                   |
                                   v
+-----------+    +------+    +----------+    +-----------+
| Input IDs |--->|Looped|--->| Student  |--->| KL Loss   |--+
+-----------+    |Model |    | Logits   |    +-----------+  |
                 |+LoRA |    +----+-----+                   |
                 +------+         |                         |  total_loss = alpha * CE + (1-alpha) * KL
                                  v                         |
                             +----------+                   |
                             | CE Loss  |-------------------+
                             | (vs true |
                             |  labels) |
                             +----------+
```

1. The **student** is the looped/recurrent model with LoRA adapters. Only the
   LoRA parameters are trained — everything else is frozen.
2. The **teacher** is the original pretrained model loaded separately. It never
   updates.
3. On each training step, both models process the same input tokens.
4. The CE loss keeps the student grounded in the actual task (next-token
   prediction on the training data).
5. The KL loss keeps the student close to the teacher's behavior, preserving
   pretrained knowledge while the student learns to use recurrence.

The features are **composable** — you can use LoRA without distillation, distillation
without LoRA, both together, or neither:

| LoRA | Distillation | Use case |
|------|-------------|----------|
| Off  | Off         | Full-parameter recurrence training (original behavior) |
| On   | Off         | Memory-efficient finetuning of the looped model |
| Off  | On          | Full-parameter training guided by the teacher |
| On   | On          | Memory-efficient finetuning guided by the teacher (recommended) |

---

## 7. Implementation walkthrough

All changes are in `train.py`. Here is how each piece maps to code.

### 7.1 Configuration (CLISettings dataclass)

Two new config dicts were added (`train.py:107-112`):

```python
lora: dict[str, Any] = field(
    default_factory=lambda: dict(
        enabled=False, r=16, lora_alpha=32,
        target_modules="all-linear", lora_dropout=0.0
    )
)
distillation: dict[str, Any] = field(
    default_factory=lambda: dict(
        enabled=False, teacher_model_name=None,
        temperature=2.0, alpha=0.5
    )
)
```

These are dicts (not separate dataclasses) to match the existing codebase
convention used by `optim_config`, `muon`, `use_ellis_adam`, etc. This also
means they work naturally with jsonargparse:
`--lora.enabled=true --lora.r=32`.

Validation in `__post_init__` (`train.py:147-153`) ensures incompatible options
are caught early:

```python
if self.lora["enabled"]:
    assert not self.muon["use_muon"]        # Muon has custom param grouping
    assert not self.use_ellis_adam[...]      # ELLISAdam likewise
    assert not self.throttle                 # Throttle splits params by recur/non-recur
if self.distillation["enabled"]:
    assert self.distillation["teacher_model_name"] is not None
```

### 7.2 Applying LoRA adapters (startup function)

LoRA is applied **after** model loading but **before** DDP wrapping
(`train.py:373-386`):

```python
if cfg.lora["enabled"]:
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=cfg.lora["r"],
        lora_alpha=cfg.lora["lora_alpha"],
        target_modules=cfg.lora["target_modules"],
        lora_dropout=cfg.lora["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
```

**Why this ordering matters:**
- LoRA must be applied *before* DDP so that DDP wraps the PeftModel. DDP then
  correctly tracks only the trainable (LoRA) parameters for gradient
  synchronization.
- LoRA must be applied *after* model loading so the pretrained weights are
  already in place when the adapters are injected.

PEFT is imported lazily (inside the `if` block) so it is not required when
LoRA is disabled.

### 7.3 Optimizer: only trainable parameters

With LoRA, most parameters are frozen (`requires_grad=False`). We filter to
pass only trainable parameters to the optimizer (`train.py:496-498`):

```python
if cfg.lora["enabled"]:
    params = [p for p in model.parameters() if p.requires_grad]
    optim_config = cfg.optim_config.copy()
```

This avoids wasting optimizer state memory on frozen parameters. For a rank-16
LoRA on a 1B model, this typically reduces optimizer memory from ~8 GB to
~50 MB.

### 7.4 Loading the teacher model

The teacher is loaded right after the optimizer setup (`train.py:518-533`):

```python
teacher_model = None
if cfg.distillation["enabled"]:
    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.distillation["teacher_model_name"],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=local_device,
        torch_dtype=weight_dtype,
        attn_implementation="sdpa",
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
```

Key decisions:
- **Same device and dtype** as the student for efficient computation.
- **eval mode + requires_grad_(False)**: disables dropout, disables gradient
  tracking. The teacher never updates.
- **Not wrapped with DDP**: the teacher only does inference, no gradient
  synchronization needed. Each GPU runs it independently on its local data.
- Added to the `state` dict (`train.py:688`) so it flows through to the
  training loop.

### 7.5 Checkpoint save/load with LoRA

**Saving** (`train.py:208-214`): When LoRA is enabled, we save only the adapter
weights using PEFT's `get_peft_model_state_dict`. This produces a checkpoint
containing just the small LoRA matrices instead of the entire model:

```python
if cfg.lora["enabled"]:
    from peft import get_peft_model_state_dict
    model_state = get_peft_model_state_dict(unwrap)
else:
    model_state = unwrap.state_dict()
```

**Loading** (`train.py:236-241`): On resume, the base model is recreated from
scratch (via `startup()`), LoRA is re-applied, and then only the adapter weights
are loaded:

```python
if cfg.lora["enabled"]:
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(unwrap, ckpt["model"])
else:
    unwrap.load_state_dict(ckpt["model"], strict=True)
```

The model-only save (`save_model_only`) calls `model.save_pretrained()` which,
for a PeftModel, automatically saves just the adapter config and weights. These
can later be loaded with `PeftModel.from_pretrained(base_model, adapter_path)`.

### 7.6 The distillation forward/backward pass

The training loop defines separate forward/backward functions for each mode.
This follows the existing pattern where `tightly_scoped_fwd_bwd` (recurrent)
and `non_rec_fwd_bwd` (non-recurrent) are defined inside the inner loop to
capture loop variables like `num_steps` and `is_accumulating`.

**Shared KL computation** (`train.py:964-977`):

```python
def _compute_kl_distill_loss(student_logits, teacher_logits, labels):
    valid_mask = labels != -100
    if valid_mask.any():
        s_logits = student_logits[valid_mask] / distill_temperature
        t_logits = teacher_logits[valid_mask] / distill_temperature
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(s_logits, dim=-1),
            torch.nn.functional.softmax(t_logits, dim=-1),
            reduction="batchmean",
        ) * (distill_temperature ** 2)
    else:
        kl_loss = student_logits.new_tensor(0.0)
    return kl_loss
```

**Key details:**
- `labels != -100` masks out padding tokens. KL divergence is only computed
  on positions where we have actual supervision. This prevents the model from
  wasting capacity matching the teacher's predictions on padding.
- `reduction="batchmean"` divides by the number of valid positions, giving
  a per-token average KL.
- The `T^2` scaling is the standard distillation correction from the Hinton
  et al. (2015) distillation paper.

**Recurrent model distillation** (`train.py:979-1001`):

```python
def distill_fwd_bwd(model, input_ids, labels):
    with ...:  # DDP sync context
        with torch.autocast(...):
            outputs = model(input_ids, labels=labels, num_steps=num_steps,
                            output_details=distill_output_details)
            student_logits = outputs["logits"]
            ce_loss = outputs["loss"]

            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits

            kl_loss = _compute_kl_distill_loss(student_logits, teacher_logits, labels)
            loss = distill_alpha * ce_loss + (1 - distill_alpha) * kl_loss

        (loss / accumulation_steps).backward()
```

Note that `distill_output_details` sets `return_logits=True` so the Huginn
model returns logits alongside its internal loss computation. The teacher runs
inside `torch.no_grad()` to avoid building a computation graph for it.

**Function selection** (`train.py:1027-1034`):

```python
if cfg.distillation["enabled"]:
    fwd_bwd_func = non_rec_distill_fwd_bwd if cfg.non_recurrent_model else distill_fwd_bwd
    loss, log_ppl, num_steps_no_grad, num_steps_with_grad, kl_loss_val = fwd_bwd_func(...)
else:
    fwd_bwd_func = non_rec_fwd_bwd if cfg.non_recurrent_model else tightly_scoped_fwd_bwd
    loss, log_ppl, num_steps_no_grad, num_steps_with_grad = fwd_bwd_func(...)
```

This gives four concrete code paths (recurrent/non-recurrent x distill/no-distill),
keeping each path simple and readable.

### 7.7 Metrics and logging

KL loss is tracked in `metrics_to_agg_data_step["kl_loss"]` (only allocated
when distillation is enabled). The existing aggregation function
`distributed_and_agg_metrics` averages it across ranks (added to `keys_to_mean`
at `train.py:740`).

The KL loss flows into wandb automatically through the existing pattern:
```python
**{f"train/{k}": v for k, v in agg_metrics.items()}
```
This logs it as `train/kl_loss` without any special-casing in the wandb block.

---

## 8. CLI reference and examples

### LoRA-only finetuning

```bash
uv run python train.py \
  --model_name="smcleish/Recurrent-Llama-3.2-2-4-2" \
  --lora.enabled=true \
  --lora.r=16 \
  --lora.lora_alpha=32 \
  --max_steps=5000
```

### Distillation-only (full parameter training with teacher guidance)

```bash
uv run python train.py \
  --model_name="smcleish/Recurrent-Llama-3.2-2-4-2" \
  --distillation.enabled=true \
  --distillation.teacher_model_name="meta-llama/Llama-3.2-1B" \
  --distillation.temperature=2.0 \
  --distillation.alpha=0.5 \
  --max_steps=10000
```

### LoRA + Distillation (recommended for memory-efficient retrofitting)

```bash
uv run python train.py \
  --model_name="smcleish/Recurrent-Llama-3.2-2-4-2" \
  --lora.enabled=true \
  --lora.r=16 \
  --lora.lora_alpha=32 \
  --distillation.enabled=true \
  --distillation.teacher_model_name="meta-llama/Llama-3.2-1B" \
  --distillation.temperature=2.0 \
  --distillation.alpha=0.5 \
  --mean_recurrence_schedule.turn_on=true \
  --mean_recurrence_schedule.warmup=0.125 \
  --mean_recurrence_schedule.max_mean_rec=32 \
  --max_steps=25000
```

### LoRA parameters reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lora.enabled` | bool | `false` | Enable LoRA adapters |
| `--lora.r` | int | `16` | Rank of the low-rank decomposition |
| `--lora.lora_alpha` | int | `32` | Scaling factor (effective scale = alpha/r) |
| `--lora.target_modules` | str | `"all-linear"` | Which modules to adapt. Can also be a list like `"q_proj,v_proj"` |
| `--lora.lora_dropout` | float | `0.0` | Dropout probability on the LoRA path |

### Distillation parameters reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--distillation.enabled` | bool | `false` | Enable logit distillation |
| `--distillation.teacher_model_name` | str | `null` | HuggingFace model ID for the teacher. **Required** when enabled. |
| `--distillation.temperature` | float | `2.0` | Softmax temperature for computing KL divergence |
| `--distillation.alpha` | float | `0.5` | Weight of CE loss. KL loss weight is `(1 - alpha)`. |

---

## 9. Design decisions and trade-offs

### Why LoRA instead of full finetuning or other PEFT methods?

- **LoRA** is the most mature and widely-tested parameter-efficient method. It
  works with any linear layer, composes well with DDP/FSDP, and has stable PEFT
  library support.
- **Adapters** (bottleneck modules inserted between layers) would conflict with
  the existing adapter layers used in the looped architecture for dimension
  matching.
- **Prefix tuning / prompt tuning** would interfere with the recurrence
  mechanism's position encoding.
- **Full finetuning** works but uses 50-100x more memory for optimizer states.

### Why `target_modules="all-linear"` by default?

Targeting all linear layers (attention projections + MLP) gives the adapter
maximum expressiveness. For the recurrent block specifically, this means the LoRA
adapters can modify how information flows during each recurrence step. If memory
is tight, restricting to attention projections only (`"q_proj,k_proj,v_proj,o_proj"`)
is a reasonable alternative.

For Huginn-style models (custom modeling code), the linear layer names differ
(e.g. `Wqkv`, `out_proj`, `fc`, `down_proj`). The `"all-linear"` default
auto-discovers linear layers by type rather than name, so it works regardless of
naming convention.

### Why KL divergence and not MSE on logits?

KL divergence operates on probability distributions and naturally handles the
fact that logit scales can vary between models. MSE would penalize logit
magnitude differences that don't matter for the final distribution. KL is also
the standard choice from Hinton et al. (2015) and is well-studied.

### Why apply LoRA before DDP, not after?

DDP wraps the model and tracks which parameters need gradient synchronization.
If LoRA were applied after DDP, the adapter parameters would not be registered
with DDP's reducer, causing incorrect gradient synchronization. The correct
wrapping order is: BaseModel -> PeftModel -> DDP.

### Why not wrap the teacher with DDP?

The teacher model only does forward inference (no gradients). DDP is only needed
for synchronizing gradients during backward passes. Wrapping the teacher in DDP
would waste communication bandwidth for no benefit.

### Why save only adapter weights in checkpoints?

The base model weights never change during LoRA training. Saving them in every
checkpoint would waste storage proportional to the full model size. By saving
only the adapter state dict (via `get_peft_model_state_dict`), checkpoints are
tiny (~10-50 MB for rank-16 LoRA vs ~2-4 GB for the full model).

On resume, the full pipeline reruns: load base model -> apply LoRA -> load
adapter weights. This is slightly slower than loading a single file but saves
substantial disk space, especially with frequent checkpointing.

### Why restrict LoRA to AdamW only (no Muon/ELLISAdam)?

Muon separates parameters into "body" and "non-body" groups using heuristics
based on parameter names (checking for "norm", "embed_tokens", etc.). LoRA
parameters don't fit these patterns and would be misclassified. ELLISAdam has
similar tensor-aware logic. Rather than adding fragile special-casing, we
restrict LoRA to AdamW which treats all parameters uniformly. This covers the
primary use case; support for other optimizers can be added if needed.

### Why define distillation functions inside the inner loop?

The existing `tightly_scoped_fwd_bwd` and `non_rec_fwd_bwd` are defined inside
the data iteration loop to capture `num_steps` and `is_accumulating` as
closures. The distillation variants follow the same convention for consistency.
Python does not create closures eagerly (variables are looked up at call time),
so defining the distillation functions even when distillation is disabled has no
correctness cost — they simply never get called.

### Why temperature=2.0 and alpha=0.5 as defaults?

- **T=2.0** is a common starting point in the distillation literature. It
  smooths the teacher's distribution enough to reveal the ranking of
  alternatives without losing the signal of the top prediction. Values of 1-4
  are typical; 2 is a balanced choice.
- **alpha=0.5** weights CE and KL equally. In practice, the best ratio depends
  on the data and model. Equal weighting is a safe starting point — the CE loss
  grounds the model in the actual task, while the KL loss transfers teacher
  knowledge.

### Distillation only computes KL on valid (non-padded) positions

The training data may contain padding tokens (marked with `label=-100`). These
positions have no meaningful target — neither the ground truth labels nor the
teacher's predictions are informative there. Computing KL on padding would waste
gradient signal on matching the teacher's arbitrary behavior on non-content
tokens. The `valid_mask = labels != -100` filter ensures we only distill on
real content.
