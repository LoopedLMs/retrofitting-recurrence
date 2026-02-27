import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerBase

from convert_pretrained_model.common import force_attn_impl, get_looped_model


def get_olmo_huginn_config(
    olmo_config_name: str,
    huginn_recurrent_base: str,
    *,
    prelude: int,
    core: int,
    coda: int,
) -> AutoConfig:
    """
    Construct a Huginn-style OLMo configuration ("raven_config") by cloning a base
    Huginn config and updating it with hyperparameters from a source OLMo config.

    Args:
        olmo_config_name: Hugging Face identifier or path for the source OLMo config.
        huginn_recurrent_base: Hugging Face identifier or path for the base
            Huginn/Recurrent OLMo config to clone.
        prelude: Number of prelude layers.
        core: Number of layers in the recurrent block.
        coda: Number of coda layers.

    Returns:
        `AutoConfig` for the Huginn-style recurrent OLMo model (the `raven_config`).
    """
    raven_config = AutoConfig.from_pretrained(huginn_recurrent_base, trust_remote_code=True)
    olmo_config = AutoConfig.from_pretrained(olmo_config_name, trust_remote_code=True)
    
    if olmo_config.tie_word_embeddings:
        print('olmo model has tied embeddings but this models won\'t have ("tie_embeddings": False), check you mean this')
        
    update_dict = {
        "head_dim": int(olmo_config.hidden_size / olmo_config.num_attention_heads),
        "intermediate_size": olmo_config.intermediate_size,
        "n_embd": olmo_config.hidden_size,
        "n_heads": olmo_config.num_attention_heads,
        "num_key_value_heads": olmo_config.num_key_value_heads,
        "n_layers": prelude + core + coda,
        "n_layers_in_coda": coda,
        "n_layers_in_prelude": prelude,
        "n_layers_in_recurrent_block": core,
        "norm_eps": olmo_config.rms_norm_eps,
        "vocab_size": olmo_config.vocab_size,
        "padded_vocab_size": olmo_config.vocab_size,
        "rope_base": olmo_config.rope_theta,
        "tie_embeddings": False,
        "torch_dtype": olmo_config.torch_dtype,
        "qk_bias": False,
        "max_position_embeddings": olmo_config.max_position_embeddings,
    }

    for key, value in update_dict.items():
        setattr(raven_config, key, value)

    # From scratch Huginn scales embedding weights.
    # Force no extra embedding scaling in the converted model
    raven_config.init_values["embed_scale"] = 1.0

    return raven_config


def weight_mapping(olmo_state_dict, huginn_state_dict, mapping_cfg):
    # 0. transfer token embeddings & lm head (shape-compatible)
    huginn_state_dict["transformer.wte.weight"] = olmo_state_dict["model.embed_tokens.weight"]
    huginn_state_dict["lm_head.weight"] = olmo_state_dict["lm_head.weight"]
    huginn_state_dict["transformer.ln_f.weight"] = olmo_state_dict["model.norm.weight"]

    # Initialize adapter to [0 | I] so it passes through the prelude output
    # and ignores the random initial state: adapter(cat([x, prelude_out])) = prelude_out
    n_embd = huginn_state_dict["transformer.adapter.weight"].shape[0]
    adapter_weight = torch.zeros(n_embd, 2 * n_embd, dtype=huginn_state_dict["transformer.adapter.weight"].dtype)
    adapter_weight[:, n_embd:] = torch.eye(n_embd)
    huginn_state_dict["transformer.adapter.weight"] = adapter_weight

    def copy_layer(src_i, tgt_prefix):
        """
        helper to copy a single layer
        """
        # attn
        q_w = olmo_state_dict[f"model.layers.{src_i}.self_attn.q_proj.weight"]
        k_w = olmo_state_dict[f"model.layers.{src_i}.self_attn.k_proj.weight"]
        v_w = olmo_state_dict[f"model.layers.{src_i}.self_attn.v_proj.weight"]

        # cat along out-features → (n_embd + 2*n_kv*hdim, n_embd)
        huginn_state_dict[f"{tgt_prefix}.attn.Wqkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        huginn_state_dict[f"{tgt_prefix}.attn.proj.weight"] = olmo_state_dict[f"model.layers.{src_i}.self_attn.o_proj.weight"]

        # MLP
        gate_proj = olmo_state_dict[f"model.layers.{src_i}.mlp.gate_proj.weight"]
        up_proj = olmo_state_dict[f"model.layers.{src_i}.mlp.up_proj.weight"]
        huginn_state_dict[f"{tgt_prefix}.mlp.fc.weight"] = torch.cat([gate_proj, up_proj], dim=0)
        huginn_state_dict[f"{tgt_prefix}.mlp.proj.weight"] = olmo_state_dict[f"model.layers.{src_i}.mlp.down_proj.weight"]

        if f"model.layers.{src_i}.self_attn.q_norm.weight" in olmo_state_dict:
            huginn_state_dict[f"{tgt_prefix}.attn.q_norm.weight"] = olmo_state_dict[
                f"model.layers.{src_i}.self_attn.q_norm.weight"
            ]
            huginn_state_dict[f"{tgt_prefix}.attn.k_norm.weight"] = olmo_state_dict[
                f"model.layers.{src_i}.self_attn.k_norm.weight"
            ]

        # LayerNorms
        # OLMo-2 uses post-norm, this is implemented in raven_modeling_minimal_olmo.py
        huginn_state_dict[f"{tgt_prefix}.norm_1.weight"] = olmo_state_dict[
            f"model.layers.{src_i}.post_attention_layernorm.weight"
        ]
        huginn_state_dict[f"{tgt_prefix}.norm_2.weight"] = olmo_state_dict[
            f"model.layers.{src_i}.post_feedforward_layernorm.weight"
        ]

    # 2. prelude → core → coda
    for j, src_i in enumerate(mapping_cfg["prelude_idx"]):
        copy_layer(src_i, f"transformer.prelude.{j}")

    for j, src_i in enumerate(mapping_cfg["core_idx"]):
        copy_layer(src_i, f"transformer.core_block.{j}")

    for j, src_i in enumerate(mapping_cfg["coda_idx"]):
        copy_layer(src_i, f"transformer.coda.{j}")

    return huginn_state_dict


def get_olmo_huginn(
    looped_olmo_model: torch.nn.Module,
    olmo_config_name: str,
    save_name: str | None,
    mapping_cfg: dict[str, list[int]],
    huginn_recurrent_base: str,
    *,
    prelude: int,
    core: int,
    coda: int,
) -> AutoModelForCausalLM:
    """
    Build a Huginn-style recurrent OLMo model from a looped OLMo checkpoint and optionally cache it on disk.

    Args:
        looped_olmo_model: Looped OLMo model whose weights will be remapped into the Huginn architecture.
        olmo_config_name: Hugging Face identifier or path for the *original*
            (non-recurrent) OLMo config to mirror.
        save_name: Optional path under which to load/save the converted Huginn model.
        mapping_cfg: Layer index mapping configuration for prelude/core/coda segments.
        huginn_recurrent_base: Hugging Face identifier or path for the base
            Huginn/Recurrent OLMo config to clone.
        prelude: Number of prelude layers.
        core: Number of layers in the recurrent block.
        coda: Number of coda layers.

    Returns:
        The instantiated Huginn-style `AutoModelForCausalLM` with remapped weights.
    """
    if save_name is not None:
        # Return cached (already converted) model if it exists
        if os.path.exists(save_name):
            return AutoModelForCausalLM.from_pretrained(save_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Get Raven config and overwrite with OLMo hyperparameters
    raven_olmo_config = get_olmo_huginn_config(
        olmo_config_name,
        huginn_recurrent_base,
        prelude=prelude,
        core=core,
        coda=coda,
    )
    model = AutoModelForCausalLM.from_config(raven_olmo_config, trust_remote_code=True)

    huginn_state_dict = weight_mapping(
        olmo_state_dict=looped_olmo_model.state_dict(), huginn_state_dict=model.state_dict(), mapping_cfg=mapping_cfg
    )
    model.load_state_dict(huginn_state_dict)
    if save_name is not None:
        model.save_pretrained(save_name)
        # Remove tie_word_embeddings from saved config.json to avoid conflict
        # with RavenConfig.__init__ which passes it explicitly via tie_embeddings
        config_path = os.path.join(save_name, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        cfg.pop("tie_word_embeddings", None)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return model


def check_same(
    looped_olmo: torch.nn.Module,
    olmo_huginn: AutoModelForCausalLM,
    olmo_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """
    Compare a looped OLMo model against its Huginn-style recurrent counterpart
    by running them on the same input and printing diagnostic statistics.

    Args:
        looped_olmo: The looped OLMo model to use as the reference.
        olmo_huginn: The Huginn-style recurrent OLMo model to compare against.
        olmo_tokenizer: Tokenizer used to encode the test input string.
    """
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = olmo_tokenizer(input_text, return_tensors="pt").to(olmo_huginn.device)
    looped_inputs = {k: v.clone() for k, v in inputs.items()}
    huginn_inputs = {k: v.clone() for k, v in inputs.items()}

    with torch.no_grad():
        logits_looped = looped_olmo(**looped_inputs).logits

        huginn_out = olmo_huginn(
            **huginn_inputs,
            output_details={"return_logits": True, "return_latents": False, "return_head": False, "return_stats": False},
            num_steps=1,
        )
        logits_huginn = huginn_out.logits

    # Compare logits
    same_shape = logits_looped.shape == logits_huginn.shape
    print(f"Same logits shape: {same_shape}")
    print(f"Logits dtypes: looped={logits_looped.dtype}, huginn={logits_huginn.dtype}")
    logits_close = torch.allclose(logits_looped, logits_huginn, atol=1e-4, rtol=1e-4)
    logits_mse = torch.nn.functional.mse_loss(logits_looped, logits_huginn).item()
    print(f"Logits allclose: {logits_close}")
    print(f"Logits MSE: {logits_mse:.6f}")


def convert(
    model_name: str = "allenai/OLMo-2-0425-1B",
    save_root: str = "models",
    prelude: int = 7,
    core: int = 4,
    coda: int = 5,
    start_index: int = 7,
    huginn_recurrent_base: str = "smcleish/Recurrent-OLMo-2-0425-untrained",
) -> None:
    """
    Convert a pretrained OLMo-2 model into a recurrent Huginn-style model and run a sanity check.
    """

    # Treat save_root as a root output folder and construct a subdirectory name
    # based on the source model name and the prelude/core/coda split.
    model_base = os.path.basename(model_name.rstrip("/"))
    subdir_name = f"{model_base}_pre{prelude}_core{core}_coda{coda}"
    save_name = str(Path(save_root) / subdir_name)

    looped_args = {
        "prelude_size": prelude,
        "start_index": start_index,
        "block_size": core,
        "coda_size": coda,
        "num_rec": 1,
    }
    mapping_cfg = {
        "prelude_idx": list(range(prelude)),
        "core_idx": list(range(start_index, start_index + core)),
        "coda_idx": list(range(start_index + core, start_index + core + coda)),
    }

    # Get vanilla looped OLMo-2 model for sanity-checking activations against Huginn OLMo
    # This is also used as an intermediate model to create the Huginn OLMo model
    looped_olmo_model, olmo_tokenizer = get_looped_model(model_name, looped_args)

    # Create the Huginn OLMo model
    olmo_huginn = get_olmo_huginn(
        looped_olmo_model,
        model_name,
        save_name,
        mapping_cfg,
        huginn_recurrent_base,
        prelude=prelude,
        core=core,
        coda=coda,
    )
    total_params = sum(p.numel() for p in olmo_huginn.parameters())
    print(f"Total params: {total_params:,}")

    # Check that the Huginn OLMo model matches the vanilla looped OLMo-2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    looped_olmo_model.eval().to(device=device, dtype=torch.float32)
    olmo_huginn.eval().to(device=device, dtype=torch.float32)

    check_same(looped_olmo_model, olmo_huginn, olmo_tokenizer)


def main() -> None:
    """
    CLI entrypoint to convert a OLMo-2 model into a recurrent Huginn-style
    model using a prelude / recurrent core / coda decomposition.
    """

    parser = argparse.ArgumentParser(description="Convert a OLMo-2 model into a recurrent Huginn-style model.")
    parser.add_argument(
        "--source",
        "--model-name",
        dest="model_name",
        default="allenai/OLMo-2-0425-1B",
        help="Path or HF id of the source OLMo-2 model.",
    )
    parser.add_argument(
        "--save-name",
        "--output",
        dest="save_name",
        default="models",
        help="Output folder to store the converted recurrent model subdirectory.",
    )
    parser.add_argument(
        "--prelude",
        type=int,
        default=7,
        help="Number of prelude layers.",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=4,
        help="Number of layers in the recurrent block.",
    )
    parser.add_argument(
        "--coda",
        type=int,
        default=5,
        help="Number of coda layers.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=7,
        help="Index of the first layer in the recurrent block.",
    )
    parser.add_argument(
        "--huginn-recurrent-base",
        type=str,
        default="smcleish/Recurrent-OLMo-2-0425-untrained",
        help=(
            "Hugging Face id or local path for the base Huginn/Recurrent OLMo config "
            "to clone before copying OLMo weights."
        ),
    )
    args = parser.parse_args()

    convert(
        model_name=args.model_name,
        save_root=args.save_name,
        prelude=args.prelude,
        core=args.core,
        coda=args.coda,
        start_index=args.start_index,
        huginn_recurrent_base=args.huginn_recurrent_base,
    )


if __name__ == "__main__":
    main()
