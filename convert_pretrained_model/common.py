import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_edited_model(model_name: str, extra_args: dict | None = None):
    """
    Load a pretrained model and swap in the looped/recurrent architecture for LLaMA or OLMo-2.
    """
    if extra_args is None:
        extra_args = {}

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    lower_name = model_name.lower()
    if "llama" in lower_name:
        from convert_pretrained_model.looped_llama import LoopedLlamaForCausalLM

        model_cls = LoopedLlamaForCausalLM
    elif "olmo-2" in lower_name:
        from convert_pretrained_model.looped_olmo import LoopedOlmo2ForCausalLM

        model_cls = LoopedOlmo2ForCausalLM
    else:
        raise ValueError(f"Unsupported model_name for looped conversion: {model_name}")

    # Use the standard eager attention implementation to avoid depending on
    # `scaled_dot_product_attention` backends, which can be unavailable or
    # misconfigured on some systems.
    model = model_cls.from_pretrained(
        model_name,
        config=config,
        attn_implementation="eager",
        torch_dtype="bfloat16",
    )
    model.rec_post_init(extra_args, {})
    return model


def force_attn_impl(name: str) -> None:
    """
    Force a specific scaled dot-product attention implementation for reproducibility.
    """
    if name == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif name == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        raise ValueError(f"attn impl not found: {name}")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def get_looped_model(model_name: str, looped_args: dict):
    """
    Load a looped/recurrent pretrained model (LLaMA or OLMo-2) plus its tokenizer.
    """
    model = get_edited_model(model_name, looped_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


