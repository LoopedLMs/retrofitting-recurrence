"""Compute layer selection metrics for retrofitting recurrence into a pretrained HF model.

Implements two complementary analysis methods:
  1. Block Influence (BI) from ShortGPT (arXiv:2403.03853):
     BI_i = 1 - E[cos_sim(H_i, H_{i+1})]
     Low BI = redundant layer (safe to drop).
     Original paper calibrated on PG19.

  2. Angular Distance from Encode-Think-Decode (arXiv:2510.07358):
     d(i, i+1) = (1/π) * arccos(cos_sim(H_i, H_{i+1}))
     Used with the Kneedle algorithm to find Encoder/Thinking/Decoder boundaries.
     Original paper calibrated on C4 validation (10K examples).

By default uses C4 validation split as calibration data (matching ETD).
Pass --use_builtin_prompts=true to skip dataset download and use hardcoded prompts.

Usage:
    uv run python dev/layer_analysis.py --model_name meta-llama/Llama-3.2-1B
    uv run python dev/layer_analysis.py --model_name ByteDance/Ouro-1.4B-Thinking
    uv run python dev/layer_analysis.py --model_name allenai/OLMo-2-0425-1B --num_samples 1024
    uv run python dev/layer_analysis.py --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --use_builtin_prompts true
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fallback calibration prompts (used when --use_builtin_prompts or dataset loading fails)
BUILTIN_PROMPTS = [
    "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
    "To solve the equation 3x + 7 = 22, first subtract 7 from both sides to get 3x = 15, then divide by 3.",
    "In Python, a list comprehension provides a concise way to create lists based on existing iterables.",
    "The mitochondria is the powerhouse of the cell, responsible for producing ATP through oxidative phosphorylation.",
    "Once upon a time, in a kingdom far beyond the mountains, there lived a young dragon who could not breathe fire.",
    "The Federal Reserve adjusts interest rates to control inflation and maintain economic stability.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight.",
    "The quick brown fox jumps over the lazy dog, demonstrating every letter in the English alphabet.",
    "Machine learning models learn patterns from data without being explicitly programmed for each specific task.",
    "The French Revolution of 1789 fundamentally transformed European politics and the concept of individual rights.",
    "Consider a recursive function that computes the nth Fibonacci number: f(n) = f(n-1) + f(n-2) with base cases f(0)=0, f(1)=1.",
    "Water reaches its maximum density at approximately 4 degrees Celsius, which is why ice floats on liquid water.",
    "The categorical imperative states that one should act only according to maxims that could become universal laws.",
    "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.name;",
    "In a parallel circuit, the total resistance is the reciprocal of the sum of the reciprocals of individual resistances.",
    "The economy showed signs of recovery as unemployment fell to 3.8 percent and consumer spending increased by 2.1 percent.",
]


def load_calibration_texts(num_samples: int = 256) -> list[str]:
    """Load calibration texts from C4 validation split (matching ETD paper).

    Falls back to built-in prompts if the dataset can't be loaded.
    """
    try:
        from datasets import load_dataset

        print(f"Loading C4 validation split ({num_samples} samples)...")
        ds = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
        )
        texts = []
        for ex in ds:
            if ex["text"].strip():
                texts.append(ex["text"])
            if len(texts) >= num_samples:
                break
        if len(texts) >= 16:
            print(f"  Loaded {len(texts)} calibration texts from C4")
            return texts
    except Exception as e:
        print(f"  Could not load C4 dataset: {e}")

    print("  Falling back to built-in calibration prompts")
    return BUILTIN_PROMPTS


def kneedle(values: list[float], direction: str = "decreasing") -> int:
    """Find the knee/elbow point in a curve using the Kneedle algorithm.

    Normalizes the curve to [0,1]x[0,1], then finds the point of maximum
    distance from the diagonal connecting the first and last points.

    Args:
        values: The y-values of the curve (x assumed to be equally spaced).
        direction: 'decreasing' for a knee in a falling curve,
                   'increasing' for a knee in a rising curve.

    Returns:
        Index of the knee point.
    """
    n = len(values)
    if n < 3:
        return 0

    x = np.linspace(0, 1, n)
    y = np.array(values, dtype=np.float64)

    # Normalize y to [0, 1]
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-12:
        return n // 2
    y_norm = (y - y_min) / (y_max - y_min)

    if direction == "decreasing":
        y_norm = 1.0 - y_norm

    # Distance from the diagonal (line from first to last point)
    # Line: from (x[0], y_norm[0]) to (x[-1], y_norm[-1])
    dx = x[-1] - x[0]
    dy = y_norm[-1] - y_norm[0]
    line_len = math.sqrt(dx**2 + dy**2)

    if line_len < 1e-12:
        return n // 2

    # Signed distance from each point to the line
    distances = np.abs(dy * x - dx * y_norm + x[-1] * y_norm[0] - y_norm[-1] * x[0]) / line_len

    return int(np.argmax(distances))


def compute_layer_metrics(
    hidden_states: list[torch.Tensor],
    mask: torch.Tensor | None = None,
) -> tuple[list[float], list[float]]:
    """Compute Block Influence and Angular Distance between consecutive layers.

    Args:
        hidden_states: List of (batch, seq_len, hidden_dim) tensors, one per layer
                       (including embedding layer at index 0).
        mask: Optional (batch, seq_len) boolean mask. True = real token, False = padding.

    Returns:
        Tuple of (block_influence, angular_distance) lists, each of length num_layers.
        Index i corresponds to the transition from layer i to layer i+1.
    """
    block_influence = []
    angular_distance = []

    for i in range(len(hidden_states) - 1):
        h_in = hidden_states[i].float()
        h_out = hidden_states[i + 1].float()

        # Cosine similarity: (batch, seq_len)
        cos_sim = F.cosine_similarity(h_in, h_out, dim=-1)

        if mask is not None:
            # Only average over real (non-padding) positions
            cos_sim = cos_sim.masked_fill(~mask, 0.0)
            num_valid = mask.sum()
            bi = 1.0 - (cos_sim.sum() / num_valid).item()

            cos_sim_clamped = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            ang_dist = (1.0 / math.pi) * torch.arccos(cos_sim_clamped)
            ang_dist = ang_dist.masked_fill(~mask, 0.0)
            angular_distance.append((ang_dist.sum() / num_valid).item())
        else:
            bi = 1.0 - cos_sim.mean().item()

            cos_sim_clamped = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            ang_dist = (1.0 / math.pi) * torch.arccos(cos_sim_clamped)
            angular_distance.append(ang_dist.mean().item())

        block_influence.append(bi)

    return block_influence, angular_distance


def suggest_config(
    angular_distance: list[float],
    block_influence: list[float],
    num_layers: int,
    target_recurrent_size: int | None = None,
) -> dict:
    """Suggest a (prelude, recurrent, coda) config based on the metrics.

    Uses angular distance + Kneedle to find encode/think/decode boundaries,
    then uses BI to identify which dropped layers are least impactful.

    Args:
        angular_distance: Per-layer angular distances.
        block_influence: Per-layer block influence scores.
        num_layers: Total number of transformer layers.
        target_recurrent_size: If set, force the recurrent block to this size.

    Returns:
        Dict with suggested configuration.
    """
    n = len(angular_distance)

    # Find E/T boundary: knee in the decreasing part of angular distance
    # Look at the first ~70% of layers for the first knee
    first_portion = max(3, int(0.7 * n))
    et_boundary = kneedle(angular_distance[:first_portion], direction="decreasing")

    # Find T/D boundary: knee in the increasing part (search from the end)
    # Reverse the tail portion and find the knee
    reversed_tail = list(reversed(angular_distance[et_boundary:]))
    td_offset = kneedle(reversed_tail, direction="decreasing")
    td_boundary = n - td_offset

    # Ensure valid boundaries
    et_boundary = max(1, et_boundary)
    td_boundary = min(n - 1, td_boundary)
    if td_boundary <= et_boundary + 1:
        # Fallback: split roughly into thirds
        et_boundary = n // 3
        td_boundary = 2 * n // 3

    # Natural split from angular distance
    prelude_end = et_boundary  # layers [0, prelude_end)
    think_start = et_boundary  # layers [think_start, think_end)
    think_end = td_boundary
    coda_start = td_boundary  # layers [coda_start, num_layers)

    prelude_size = prelude_end
    think_size = think_end - think_start
    coda_size = num_layers - coda_start

    # For the retrofitting architecture: we may want to drop some layers.
    # Identify which layers in the "think" zone have lowest BI (most redundant)
    think_bi = [(i, block_influence[i]) for i in range(think_start, min(think_end, len(block_influence)))]
    think_bi.sort(key=lambda x: x[1])

    if target_recurrent_size is not None and target_recurrent_size < think_size:
        # Keep the highest-BI layers in the think zone
        keep_count = target_recurrent_size
        drop_count = think_size - keep_count
        dropped = [idx for idx, _ in think_bi[:drop_count]]
        kept = sorted([idx for idx, _ in think_bi[drop_count:]])
    else:
        dropped = []
        kept = list(range(think_start, think_end))
        keep_count = think_size

    return {
        "prelude_layers": list(range(prelude_size)),
        "recurrent_layers": kept,
        "coda_layers": list(range(coda_start, num_layers)),
        "dropped_layers": sorted(dropped),
        "prelude_size": prelude_size,
        "recurrent_size": len(kept),
        "coda_size": coda_size,
        "start_index": kept[0] if kept else think_start,
        "et_boundary": et_boundary,
        "td_boundary": td_boundary,
    }


def plot_metrics(
    block_influence: list[float],
    angular_distance: list[float],
    config: dict,
    model_name: str,
    save_path: str,
) -> None:
    """Plot BI and angular distance with suggested boundaries."""
    n = len(block_influence)
    layers = list(range(n))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Layer Analysis: {model_name}", fontsize=14)

    # --- Block Influence ---
    colors = []
    for i in layers:
        if i < config["et_boundary"]:
            colors.append("#2196F3")  # blue = prelude
        elif i >= config["td_boundary"]:
            colors.append("#4CAF50")  # green = coda
        elif i in config["dropped_layers"]:
            colors.append("#BDBDBD")  # grey = dropped
        else:
            colors.append("#FF9800")  # orange = recurrent

    ax1.bar(layers, block_influence, color=colors, edgecolor="black", linewidth=0.3)
    ax1.set_ylabel("Block Influence (BI)")
    ax1.set_title("Block Influence (higher = more important)")
    ax1.axvline(x=config["et_boundary"] - 0.5, color="red", linestyle="--", alpha=0.7, label="E/T boundary")
    ax1.axvline(x=config["td_boundary"] - 0.5, color="red", linestyle="--", alpha=0.7, label="T/D boundary")
    ax1.legend()

    # --- Angular Distance ---
    ax2.plot(layers, angular_distance, "o-", color="#333333", markersize=4, linewidth=1.5)
    ax2.fill_between(
        range(config["et_boundary"]),
        0,
        [angular_distance[i] for i in range(config["et_boundary"])],
        alpha=0.2,
        color="#2196F3",
        label="Prelude (Encoder)",
    )
    ax2.fill_between(
        range(config["et_boundary"], config["td_boundary"]),
        0,
        [angular_distance[i] for i in range(config["et_boundary"], config["td_boundary"])],
        alpha=0.2,
        color="#FF9800",
        label="Recurrent (Thinking)",
    )
    ax2.fill_between(
        range(config["td_boundary"], n),
        0,
        [angular_distance[i] for i in range(config["td_boundary"], n)],
        alpha=0.2,
        color="#4CAF50",
        label="Coda (Decoder)",
    )
    ax2.axvline(x=config["et_boundary"] - 0.5, color="red", linestyle="--", alpha=0.7)
    ax2.axvline(x=config["td_boundary"] - 0.5, color="red", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Angular Distance")
    ax2.set_xlabel("Layer Index")
    ax2.set_title("Angular Distance (ETD method)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")


def layer_analysis(
    model_name: str = "allenai/OLMo-2-0425-1B",
    device: str = "auto",
    dtype: str = "bfloat16",
    max_tokens: int = 4096,
    num_samples: int = 10000,
    batch_size: int = 8,
    use_builtin_prompts: bool = False,
    target_recurrent_size: int | None = 4,
    save_plot: str | None = None,
) -> None:
    """Analyze layer redundancy and functional zones for a pretrained HF model.

    Computes Block Influence (ShortGPT) and Angular Distance (ETD) metrics,
    then suggests a (prelude, recurrent, coda) configuration for retrofitting
    recurrence.

    Args:
        model_name: HuggingFace model name or local path.
        device: Device to run on ('auto', 'cuda', 'cpu').
        dtype: Model dtype ('bfloat16', 'float16', 'float32').
        max_tokens: Max tokens per calibration prompt.
        num_samples: Number of C4 validation samples for calibration (ETD used 10K).
        use_builtin_prompts: Skip dataset download, use hardcoded prompts instead.
        target_recurrent_size: Force recurrent block to this many layers.
        save_plot: Path to save the analysis plot. Defaults to dev/{model_short_name}_layers.png.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # --- Load calibration data ---
    calibration_texts = BUILTIN_PROMPTS if use_builtin_prompts else load_calibration_texts(num_samples=num_samples)

    # --- Load model and tokenizer ---
    print(f"Loading {model_name} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} transformer layers")

    # --- Collect hidden states over calibration texts ---
    num_batches = (len(calibration_texts) + batch_size - 1) // batch_size
    print(f"Running {len(calibration_texts)} calibration texts in {num_batches} batches (bs={batch_size})...")
    all_bi = [[] for _ in range(num_layers)]
    all_ang = [[] for _ in range(num_layers)]

    with torch.no_grad(), tqdm(total=len(calibration_texts), desc="Samples", unit="sample") as pbar:
        for batch_start in range(0, len(calibration_texts), batch_size):
            batch_texts = calibration_texts[batch_start : batch_start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
                padding=True,
            ).to(device)

            outputs = model(**inputs)
            # hidden_states: tuple of (num_layers + 1) tensors, each (B, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states
            pad_mask = inputs["attention_mask"].bool()  # (B, seq_len)

            bi_vals, ang_vals = compute_layer_metrics(hidden_states, mask=pad_mask)
            for i in range(num_layers):
                all_bi[i].append(bi_vals[i])
                all_ang[i].append(ang_vals[i])
            pbar.update(len(batch_texts))

    # Average over calibration prompts
    block_influence = [np.mean(scores) for scores in all_bi]
    angular_distance = [np.mean(scores) for scores in all_ang]

    # --- Suggest configuration ---
    config = suggest_config(angular_distance, block_influence, num_layers, target_recurrent_size)

    # --- Print results ---
    print(f"\n{'=' * 72}")
    print(f"  Layer Analysis Results: {model_name}")
    print(f"{'=' * 72}")
    print(f"\n{'Layer':>5}  {'BI':>8}  {'AngDist':>8}  {'Zone':<12}")
    print(f"{'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 12}")

    for i in range(num_layers):
        if i < config["et_boundary"]:
            zone = "Prelude"
        elif i >= config["td_boundary"]:
            zone = "Coda"
        elif i in config["dropped_layers"]:
            zone = "DROPPED"
        else:
            zone = "Recurrent"

        print(f"{i:>5}  {block_influence[i]:>8.4f}  {angular_distance[i]:>8.4f}  {zone:<12}")

    print(f"\n{'=' * 72}")
    print("  Suggested Configuration")
    print(f"{'=' * 72}")
    print(f"  Prelude layers:    {config['prelude_layers']}")
    print(f"  Recurrent layers:  {config['recurrent_layers']}")
    print(f"  Coda layers:       {config['coda_layers']}")
    if config["dropped_layers"]:
        print(f"  Dropped layers:    {config['dropped_layers']}")
    print(f"\n  (P, R, C) = ({config['prelude_size']}, {config['recurrent_size']}, {config['coda_size']})")
    print(f"  start_index = {config['start_index']}")
    print(f"  Layers retained: {config['prelude_size'] + config['recurrent_size'] + config['coda_size']}/{num_layers}")

    print("\n  LoopConfig args:")
    print(f"    prelude_size: {config['prelude_size']}")
    print(f"    start_index:  {config['start_index']}")
    print(f"    block_size:   {config['recurrent_size']}")
    print(f"    coda_size:    {config['coda_size']}")

    # --- Plot ---
    if save_plot is None:
        import os

        os.makedirs("dev/outputs", exist_ok=True)
        short_name = model_name.replace("/", "_").replace("-", "_")
        save_plot = f"dev/outputs/{short_name}_layers.png"

    plot_metrics(block_influence, angular_distance, config, model_name, save_plot)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for layer analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze layer redundancy and functional zones for a pretrained HF model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/OLMo-2-0425-1B",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens per calibration prompt.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of C4 validation samples for calibration.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model forward passes.",
    )
    parser.add_argument(
        "--use_builtin_prompts",
        action="store_true",
        help="Skip dataset download and use hardcoded prompts instead.",
    )
    parser.add_argument(
        "--target_recurrent_size",
        type=int,
        default=4,
        help="Force recurrent block to this many layers.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save the analysis plot. Defaults to dev/{model_short_name}_layers.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_analysis(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        use_builtin_prompts=args.use_builtin_prompts,
        target_recurrent_size=args.target_recurrent_size,
        save_plot=args.save_plot,
    )


if __name__ == "__main__":
    main()
