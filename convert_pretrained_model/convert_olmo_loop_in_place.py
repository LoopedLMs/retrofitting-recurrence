"""Convert OLMo-2-0425-1B to Raven (loop-in-place) format with no layer dropping.

All original layers are preserved and assigned to prelude / core_block / coda.
The adapter is newly initialized (random). Everything else copies from pretrained weights.

Usage:
    uv run python convert_pretrained_model/convert_olmo_loop_in_place.py
    uv run python convert_pretrained_model/convert_olmo_loop_in_place.py --prelude 7 --core 4 --coda 5
    uv run python convert_pretrained_model/convert_olmo_loop_in_place.py --source allenai/OLMo-2-0425-1B --prelude 3 --core 10 --coda 3
"""

import torch
from jsonargparse import CLI
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Template model that has the Raven config + modeling files with QK-norm
RAVEN_TEMPLATE = "smcleish/Recurrent-OLMo-2-0425-untrained"


def weight_mapping(
    src_state_dict: dict[str, torch.Tensor],
    dst_state_dict: dict[str, torch.Tensor],
    mapping_cfg: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    """Copy weights from a standard OLMo-2 state dict into a Raven state dict.

    Reuses the mapping logic from convert_olmo.py. The adapter weight is left
    at its random initialization.
    """
    # Embeddings + final norm + LM head
    dst_state_dict["transformer.wte.weight"] = src_state_dict["model.embed_tokens.weight"]
    dst_state_dict["lm_head.weight"] = src_state_dict["lm_head.weight"]
    dst_state_dict["transformer.ln_f.weight"] = src_state_dict["model.norm.weight"]

    # Initialize adapter to [0 | I] so it passes through the prelude output
    # and ignores the random initial state: adapter(cat([x, prelude_out])) = prelude_out
    n_embd = dst_state_dict["transformer.adapter.weight"].shape[0]
    adapter_weight = torch.zeros(n_embd, 2 * n_embd, dtype=dst_state_dict["transformer.adapter.weight"].dtype)
    adapter_weight[:, n_embd:] = torch.eye(n_embd)
    dst_state_dict["transformer.adapter.weight"] = adapter_weight

    def copy_layer(src_i: int, tgt_prefix: str) -> None:
        # Attention: fuse Q/K/V into a single Wqkv
        q_w = src_state_dict[f"model.layers.{src_i}.self_attn.q_proj.weight"]
        k_w = src_state_dict[f"model.layers.{src_i}.self_attn.k_proj.weight"]
        v_w = src_state_dict[f"model.layers.{src_i}.self_attn.v_proj.weight"]
        dst_state_dict[f"{tgt_prefix}.attn.Wqkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        dst_state_dict[f"{tgt_prefix}.attn.proj.weight"] = src_state_dict[f"model.layers.{src_i}.self_attn.o_proj.weight"]

        # QK-norm (OLMo-2 specific)
        if f"model.layers.{src_i}.self_attn.q_norm.weight" in src_state_dict:
            dst_state_dict[f"{tgt_prefix}.attn.q_norm.weight"] = src_state_dict[
                f"model.layers.{src_i}.self_attn.q_norm.weight"
            ]
            dst_state_dict[f"{tgt_prefix}.attn.k_norm.weight"] = src_state_dict[
                f"model.layers.{src_i}.self_attn.k_norm.weight"
            ]

        # MLP: fuse gate + up into fc
        gate_proj = src_state_dict[f"model.layers.{src_i}.mlp.gate_proj.weight"]
        up_proj = src_state_dict[f"model.layers.{src_i}.mlp.up_proj.weight"]
        dst_state_dict[f"{tgt_prefix}.mlp.fc.weight"] = torch.cat([gate_proj, up_proj], dim=0)
        dst_state_dict[f"{tgt_prefix}.mlp.proj.weight"] = src_state_dict[f"model.layers.{src_i}.mlp.down_proj.weight"]

        # Layer norms (OLMo-2 uses post-attention and post-feedforward norms)
        dst_state_dict[f"{tgt_prefix}.norm_1.weight"] = src_state_dict[f"model.layers.{src_i}.post_attention_layernorm.weight"]
        dst_state_dict[f"{tgt_prefix}.norm_2.weight"] = src_state_dict[
            f"model.layers.{src_i}.post_feedforward_layernorm.weight"
        ]

    for j, src_i in enumerate(mapping_cfg["prelude_idx"]):
        copy_layer(src_i, f"transformer.prelude.{j}")

    for j, src_i in enumerate(mapping_cfg["core_idx"]):
        copy_layer(src_i, f"transformer.core_block.{j}")

    for j, src_i in enumerate(mapping_cfg["coda_idx"]):
        copy_layer(src_i, f"transformer.coda.{j}")

    return dst_state_dict


def check_same(
    src_model: torch.nn.Module,
    raven_model: torch.nn.Module,
    tokenizer,
    device: str = "cpu",
) -> None:
    """Verify that a single forward pass (1 recurrence) produces matching logits."""
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    src_model.eval().to(device=device, dtype=torch.float32)
    raven_model.eval().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        src_out = src_model(**inputs)
        logits_src = src_out.logits

        raven_out = raven_model(
            **inputs,
            output_details={"return_logits": True, "return_latents": True, "return_head": True, "return_stats": False},
            num_steps=1,
        )
        logits_raven = raven_out.logits

    same_shape = logits_src.shape == logits_raven.shape
    mse = torch.nn.functional.mse_loss(logits_src, logits_raven).item()
    close = torch.allclose(logits_src, logits_raven, atol=1e-3, rtol=1e-3)

    print("\nSanity check (1 recurrence):")
    print(f"  Same shape: {same_shape} ({logits_src.shape})")
    print(f"  Values close (atol=1e-3): {close}")
    print(f"  MSE: {mse:.6f}")

    if not close:
        max_diff = (logits_src - logits_raven).abs().max().item()
        print(f"  Max absolute diff: {max_diff:.6f}")
        print("  WARNING: Logits don't match closely. Check weight mapping.")


def convert(
    source: str = "allenai/OLMo-2-0425-1B",
    prelude: int = 7,
    core: int = 4,
    coda: int = 5,
    save_dir: str | None = None,
    skip_check: bool = False,
) -> None:
    """Convert a pretrained OLMo-2 model to Raven loop-in-place format.

    Args:
        source: HuggingFace model name or local path for the source OLMo-2 model.
        prelude: Number of prelude (encoder) layers.
        core: Number of core (recurrent) layers.
        coda: Number of coda (decoder) layers.
        save_dir: Where to save the converted model. Defaults to models/OLMo-2-...-loop-in-place-{P}-{R}-{C}/
        skip_check: Skip the logit-matching sanity check.
    """
    if save_dir is None:
        short_name = source.split("/")[-1]
        save_dir = f"models/{short_name}-loop-in-place-{prelude}-{core}-{coda}"

    # Validate layer split
    src_config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    num_layers = src_config.num_hidden_layers
    assert prelude + core + coda == num_layers, (
        f"Layer split ({prelude}+{core}+{coda}={prelude + core + coda}) must equal total layers ({num_layers})"
    )

    print(f"Converting {source} → Raven loop-in-place")
    print(f"  Split: prelude={prelude}, core={core}, coda={coda}")
    print(f"  All {num_layers} layers preserved (no dropping)")

    # Build layer mapping
    mapping_cfg = {
        "prelude_idx": list(range(prelude)),
        "core_idx": list(range(prelude, prelude + core)),
        "coda_idx": list(range(prelude + core, num_layers)),
    }
    print(f"  Prelude layers: {mapping_cfg['prelude_idx']}")
    print(f"  Core layers:    {mapping_cfg['core_idx']}")
    print(f"  Coda layers:    {mapping_cfg['coda_idx']}")

    # Load Raven config from template and override layer counts
    raven_config = AutoConfig.from_pretrained(RAVEN_TEMPLATE, trust_remote_code=True)
    raven_config.n_layers = num_layers
    raven_config.n_layers_in_prelude = prelude
    raven_config.n_layers_in_recurrent_block = core
    raven_config.n_layers_in_coda = coda
    # Match source model properties
    raven_config.n_embd = src_config.hidden_size
    raven_config.n_heads = src_config.num_attention_heads
    raven_config.num_key_value_heads = src_config.num_key_value_heads
    raven_config.head_dim = src_config.hidden_size // src_config.num_attention_heads
    raven_config.intermediate_size = src_config.intermediate_size
    raven_config.vocab_size = src_config.vocab_size
    raven_config.padded_vocab_size = src_config.vocab_size
    raven_config.norm_eps = src_config.rms_norm_eps
    raven_config.rope_base = getattr(src_config, "rope_theta", 500000.0)
    raven_config.rope_theta = raven_config.rope_base
    raven_config.tie_embeddings = src_config.tie_word_embeddings
    raven_config.max_position_embeddings = src_config.max_position_embeddings
    raven_config.torch_dtype = str(src_config.torch_dtype).removeprefix("torch.")
    raven_config.qk_bias = False
    raven_config.init_values["embed_scale"] = 1.0

    # Create empty Raven model
    print("\nCreating Raven model from config...")
    raven_model = AutoModelForCausalLM.from_config(raven_config, trust_remote_code=True)

    # Load source model weights
    print("Loading source model weights...")
    src_model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(source)

    # Copy weights
    print("Copying weights...")
    raven_state = weight_mapping(src_model.state_dict(), raven_model.state_dict(), mapping_cfg)
    raven_model.load_state_dict(raven_state)

    # Report adapter params
    adapter_params = sum(p.numel() for n, p in raven_model.named_parameters() if "adapter" in n)
    total_params = sum(p.numel() for p in raven_model.parameters())
    print(f"\nAdapter params: {adapter_params:,} ({adapter_params / total_params * 100:.2f}%)")
    print(f"Total params:   {total_params:,}")

    # Sanity check
    if not skip_check:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        check_same(src_model, raven_model, tokenizer, device=device)

    # Save
    print(f"\nSaving to {save_dir}/")
    raven_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done!")


if __name__ == "__main__":
    CLI(convert)
