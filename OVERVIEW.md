# Retrofitting Recurrence — Codebase Overview

Research implementation of [arXiv:2511.07384](https://arxiv.org/abs/2511.07384): adding recurrent mechanisms to pretrained transformer LMs to improve reasoning.

Pretrained models: [HuggingFace collection](https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence)

---

## What can I do?

### Convert a pretrained model to recurrent architecture

```bash
uv run python convert_pretrained_model/convert_llama.py  # for Llama
uv run python convert_pretrained_model/convert_olmo.py   # for OLMo
```

These scripts load a pretrained HF model, restructure its layers into a looped/recurrent block, add linear adapter layers where dimensions change, and save a converted model ready for recurrent fine-tuning.

**Supporting files in `convert_pretrained_model/`:**
| File | Purpose |
|------|---------|
| `looped_llama.py` / `looped_olmo.py` | Looped model class definitions (`LoopedLlamaForCausalLM`, etc.) |
| `raven_modeling_minimal_llama.py` / `*_olmo.py` | Minimal forward-pass implementations for validation |
| `raven_modeling_minimal_compare_*.py` | Comparison variants used during conversion validation |
| `raven_modeling_minimal_with_qk_norm.py` | QK-norm variant |

---

### Train a model

```bash
uv run python train.py [args]
```

The main training script. Key capabilities:
- **Recurrent + non-recurrent** model support
- **Curriculum learning** for recurrence: gradually ramp up `mean_recurrence_steps` using linear or `1-sqrt` schedules
- **Backward depth scheduling**: optionally reduce backprop depth to save memory
- **Multiple optimizers**: AdamW, Muon, ELLISAdam
- **Distributed training** via NCCL (multi-GPU, multi-node)
- **WandB** experiment tracking
- **Checkpointing**: full state saved/restored (model, optimizer, scheduler, RNG)
- **Parquet streaming** datasets with stateful, rank-aware data loading

See `shells/` for example invocations per figure in the paper:
| Script | Models / scenarios |
|--------|-------------------|
| `shells/llama.sh` | Llama-3.2 (1B–4B): from-scratch vs pretrained init, max recurrence |
| `shells/tinyllama.sh` | TinyLlama 1.1B: optimizer ablations, dataset mixing, two-phase training |
| `shells/olmo.sh` | OLMo-2 1B: Nemotron + Math, Muon optimizer |

Configure machine-specific paths by copying `shells/machine_config.sh.example` → `shells/machine_config.sh`.

**Custom optimizer:** `ellisadam.py` — ELLISAdam with tensor-wise gradient normalization, atan modifications, and update clipping.

---

### Prepare training data

**Download datasets:**
```bash
uv run python utils/download_ds.py          # FineWeb-Edu, Nemotron Math, Nemotron SFT
```

**Convert to parquet:**
```bash
uv run python utils/to_parquet.py
```

**Pack sequences (tokenize + fill fixed-length contexts):**
```bash
uv run python preprocess_data_packing.py [args]
```
Applies chat template, tokenizes, packs sequences, optionally masks non-assistant tokens in loss.

**Mix datasets into shards:**
```bash
uv run python mix_datasets.py
```
Combines FineWeb-Edu + Nemotron datasets, shuffles, and writes 516 equally-sized parquet shards.

**Stateful dataloader** (used internally by `train.py`): `stateful_parquet_dataset.py` — rank-aware, resumable parquet streaming with shuffle and epoch management.

---

### Generate text from a trained model

```bash
uv run python generate.py
# or via shell example:
bash shells/generate.sh
```

Loads a model from HF, runs generation with a configurable number of recurrence steps (`num_steps`), and prints output. Default model: `smcleish/Recurrent-Llama-3.2-train-recurrence-4`.

---

### Evaluate a model

**Benchmark evaluation** (lm_eval harness):
```bash
bash shells/eval.sh
```
Tasks: LAMBADA, HellaSwag, ARC, MMLU, OpenBookQA, PIQA, Social IQA, WinoGrande, ASDiv, GSM8K (CoT), Minerva Math.

Custom GSM8K config (8-shot, "Let's think step by step."): `eval_yamls/gsm8k-cot-sean.yaml`

**Validation loss across recurrence depths:**
```bash
uv run python multi_recurence_eval.py
```
Computes cross-entropy loss at recurrence depths `[1, 2, 4, 8, 16, 32, 64]` for a set of checkpoints. Saves results to JSON for plotting.

---

### Plot and analyse results

**Evaluation result plots:**
```bash
uv run python plot_evals.py
```
Aggregates JSON eval results into DataFrames, computes FLOPs accounting for recurrence curriculum, and generates figures.

**Paper figures:**
```bash
uv run python paper_plots/plot.py
```
Full figure generation for the paper. WandB data can be pulled with `paper_plots/pull_from_wandb.py`. Pre-computed data lives in `paper_plots/data/`.

**Scheduler analysis:**
```bash
uv run python paper_plots/scheduling_options.py
```

---

### Analyse layer structure for recurrence retrofitting

```bash
uv run python dev/layer_analysis.py --model_name meta-llama/Llama-3.2-1B
```

Before converting a pretrained model to a recurrent architecture you need to decide which layers become the prelude, recurrent block, and coda. The paper (arXiv:2511.07384) used manual grid search for this; `dev/layer_analysis.py` automates the decision with two complementary metrics from the literature:

**1. Block Influence (BI)** — from ShortGPT ([arXiv:2403.03853](https://arxiv.org/abs/2403.03853))

Measures how much each layer changes the hidden state: `BI_i = 1 - E[cos_sim(H_i, H_{i+1})]`. A low BI means the layer's output is nearly identical to its input — it is redundant and safe to drop. High-BI layers do the most "work" and are the best candidates for the recurrent block.

**2. Angular Distance** — from Encode-Think-Decode ([arXiv:2510.07358](https://arxiv.org/abs/2510.07358))

A normalised variant: `d(i, i+1) = (1/pi) * arccos(cos_sim(H_i, H_{i+1}))`. The shape of the angular distance curve across layers reveals three functional zones: early layers with high distance (encoding), middle layers where distance plateaus (thinking/reasoning), and final layers where distance rises again (decoding). The script applies the **Kneedle algorithm** to this curve to automatically locate the Encoder/Thinking and Thinking/Decoder boundaries.

The two metrics are complementary: angular distance identifies *where* to split (functional zones), while BI identifies *what* to drop within those zones (redundant layers).

**Calibration data.** Both metrics are computed from hidden states collected during a single forward pass over a calibration corpus — no gradients needed. The key requirement is that the calibration data be *generic and diverse* so the measured layer behaviour reflects the model's general structure rather than quirks of a specific domain. The original papers chose corpora that satisfy this:

- **C4 validation** (ETD's choice, our default): a large cleaned web crawl. Web text is close to what most LLMs were pretrained on, so layer activations are "typical" rather than out-of-distribution. The validation split is fixed and reproducible.
- **PG19** (ShortGPT's choice): long-form book text from Project Gutenberg. The long sequences help average out positional effects and give a stable per-layer signal.

Both work well because they are broad-domain, in-distribution text that the model processes naturally. The metrics are robust to the exact corpus choice — what matters is having enough tokens to average over. The script defaults to C4 (256 samples) and falls back to a small set of hardcoded prompts if the dataset can't be downloaded (`--use_builtin_prompts true`).

**Output:**
- Per-layer table with BI, angular distance, and assigned zone (Prelude / Recurrent / Coda / DROPPED)
- Suggested `LoopConfig` args (`prelude_size`, `start_index`, `block_size`, `coda_size`) ready for the conversion scripts
- Two-panel PNG plot saved to `dev/outputs/` (BI bar chart + angular distance curve with zone shading)

**Key flags:**

| Flag | Description |
|------|-------------|
| `--model_name` | Any HuggingFace model name or local path |
| `--num_samples` | Number of C4 validation texts to use (default 256; ETD used 10K) |
| `--use_builtin_prompts` | Skip C4 download, use hardcoded prompts |
| `--target_recurrent_size` | Force the recurrent block to exactly N layers (drops lowest-BI layers) |
| `--save_plot` | Custom output path for the plot (default: `dev/outputs/<model>_layers.png`) |
| `--device` | `auto` (default), `cuda`, or `cpu` |

---

### Count parameters / FLOPs

```bash
uv run python param_counter.py
```

Breaks down params by component (embeddings, prelude layers, recurrent block, coda) and computes effective FLOPs under a given recurrence curriculum (`6D·N₁ + 2D·N₂` formula).

---

### Prepare non-recurrent baselines

```bash
uv run python utils/untie_embeds_hf.py
```

Unties the shared embedding / lm_head weights in a Llama model so input and output projections are independent — needed for fair non-recurrent baseline comparisons.

---

## Project structure at a glance

```
train.py                       Main training loop
generate.py                    Text generation / inference
ellisadam.py                   ELLISAdam optimizer
multi_recurence_eval.py        Offline validation at multiple recurrence depths
plot_evals.py                  Aggregate + plot eval results
param_counter.py               Parameter and FLOP counting
preprocess_data_packing.py     Tokenize and pack datasets
mix_datasets.py                Mix dataset shards
stateful_parquet_dataset.py    Distributed stateful parquet dataloader

convert_pretrained_model/      Model conversion pipeline (Llama & OLMo)
dev/                           Analysis & development scripts (outputs in dev/outputs/)
utils/                         Download, convert, and prep utilities
shells/                        Shell script examples for training and eval
eval_yamls/                    lm_eval task configs
paper_plots/                   Paper figure generation + pre-computed data
```

## Key dependencies

| Package | Role |
|---------|------|
| `torch 2.9` | Core ML framework |
| `transformers 4.51` | Model loading (pinned — KV-cache API) |
| `flash-attn` | Efficient attention |
| `lm-eval` | Evaluation harness |
| `muon-optimizer` | Muon optimizer |
| `wandb` | Experiment tracking |
| `jsonargparse` | CLI config parsing |
