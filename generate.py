import torch
from jsonargparse import CLI
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "smcleish/Recurrent-Llama-3.2-train-recurrence-4"


def load_model_and_tokenizer(model_name: str, device: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(
    prompt: str = "The key to solving hard math problems is",
    model_name: str = DEFAULT_MODEL,
    num_recurrence_steps: int = 32,
    max_new_tokens: int = 200,
    device: str = "cuda",
    temperature: float = 0.0,
) -> None:
    print(f"Loading {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print(f"Prompt: {prompt!r}  |  recurrence steps: {num_recurrence_steps}")
    print("-" * 60)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_steps=torch.tensor([num_recurrence_steps, 0]),
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    CLI(generate)
