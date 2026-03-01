"""Tokenize and sequence-pack a HuggingFace dataset into fixed-length parquet shards.

Two dataset modes are supported, selected by --q_col:

  Pretraining text  (q_col="text")
    Each example is a raw document. The text is appended with an EOS token,
    tokenized without padding, then packed into seq_length = max_length + 1
    chunks. With wrapped_packing=True, documents flow across chunk boundaries
    so no tokens are wasted on padding.

  Instruction / chat  (q_col != "text")
    Each example is a question/answer pair. A system prompt is prepended and
    the conversation is formatted with the model's chat template, then padded
    or truncated to max_length + 1.

The output is written as a HuggingFace dataset on disk (parquet shards) and
can be consumed by train.py via --is_parquet_dataset=true.
"""

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from jsonargparse import CLI
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl import pack_dataset

from train import DEFAULT_SYS_PROMPT, Message


def format_and_tokenize_examples(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    q_col: str,
    a_col: str,
    max_length: int,
    take_loss_over_all_tokens: bool,
    return_type: str | None = "pt",
) -> dict:
    """Tokenize a batch of examples, formatting them as chat or plain text.

    For plain-text datasets (q_col="text") documents are simply appended with
    an EOS token and tokenized without padding so they can be packed later.

    For chat datasets the system prompt, user question, and assistant answer
    are formatted with the model's chat template and padded/truncated to
    max_length + 1.

    Args:
        examples: Batch dict from dataset.map().
        tokenizer: Tokenizer for the target model.
        q_col: Column name for the input text or question.
        a_col: Column name for the answer (only used when q_col != "text").
        max_length: Maximum sequence length (chat mode pads/truncates to this).
        take_loss_over_all_tokens: If True, compute loss over all tokens rather
            than only assistant tokens (chat mode only).
        return_type: Tensor return type passed to the tokenizer ("pt" or None).

    Returns:
        Dict with "input_ids" and "attention_mask" keys.
    """
    conversations: list = []
    for idx in range(len(examples[q_col])):
        if q_col != "text":
            conversations.append(
                [
                    Message(role="system", content=DEFAULT_SYS_PROMPT),
                    Message(role="user", content=examples[q_col][idx].strip()),
                    Message(role="Huginn", content=examples[a_col][idx].strip()),
                ]
            )
        else:
            conversations.append(examples[q_col][idx].strip() + tokenizer.eos_token)

    if q_col != "text":
        chat_encoding = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            padding="max_length",
            max_length=max_length + 1,
            return_tensors=return_type,
            return_dict=True,
            truncation=True,
        )
        if take_loss_over_all_tokens:
            chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]
    else:
        chat_encoding = tokenizer(
            conversations,
            padding=False,
            return_tensors=return_type,
            truncation=False,
            add_special_tokens=True,
        )
        chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]

    return {
        "input_ids": chat_encoding["input_ids"],
        "attention_mask": chat_encoding["attention_mask"],
    }


def pad_or_truncate(example: dict, tokenizer_pad_id: int, max_len: int) -> dict:
    """Pad or truncate a single example's tensors to exactly max_len tokens.

    Applied after packing so that every sequence is exactly the expected
    length, which avoids variable-length batches in the dataloader.

    Args:
        example: Single example with "input_ids" and "attention_mask" tensors.
        tokenizer_pad_id: Token ID used to pad input_ids.
        max_len: Target sequence length.

    Returns:
        The same example dict with tensors resized to max_len.
    """
    for key, pad_id in [("input_ids", tokenizer_pad_id), ("attention_mask", 0)]:
        tensor: torch.Tensor = example[key]
        length = tensor.shape[0]
        if length < max_len:
            tensor = torch.cat([tensor, torch.full((max_len - length,), pad_id, dtype=tensor.dtype)])
        elif length > max_len:
            tensor = tensor[:max_len]
        example[key] = tensor
    return example


def process_data(
    tokenizer_name: str = "smcleish/Recurrent-Llama-3.2-untrained",
    out_path: str = "del",
    dataset_location: str = "/path/to/dataset",
    q_col: str = "text",
    a_col: str = "answer",
    max_length: int = 1024,
    max_samples: int | None = None,
    take_loss_over_all_tokens: bool = False,
    num_proc: int = 96,
    batch_size: int = 1024,
    pack: bool = True,
    wrapped_packing: bool = True,
    cache_path: str = "/tmp",
    save_path: str = "/tmp",
) -> None:
    """Tokenize and pack a dataset into fixed-length parquet shards.

    Output is written to:
        {save_path}/preprocessed_data{packing_suffix}/{tokenizer_name}/{out_path}/dataset

    Args:
        tokenizer_name: HuggingFace tokenizer identifier or local path.
        out_path: Subdirectory name for this specific dataset variant.
        dataset_location: Path to the raw dataset saved with save_to_disk().
        q_col: Column name for text/questions. Use "text" for pretraining data.
        a_col: Column name for answers (chat mode only, ignored when q_col="text").
        max_length: Sequence length for packing/padding (actual tensors are
            max_length + 1 to accommodate the shifted label).
        max_samples: If set, truncate the dataset to this many examples before
            tokenization. Useful for quick experiments.
        take_loss_over_all_tokens: For chat data, compute loss over all tokens
            instead of only assistant tokens.
        num_proc: Number of parallel workers for dataset.map().
        batch_size: Batch size used in dataset.map() and for Arrow writer shards.
        pack: Whether to pack tokenized sequences into fixed-length chunks.
        wrapped_packing: If True, use wrapped packing (documents flow across
            chunk boundaries). Requires pack=True.
        cache_path: Directory for intermediate Arrow cache files.
        save_path: Root directory under which the final dataset is saved.
    """
    assert not (wrapped_packing and not pack), "wrapped_packing=True requires pack=True."

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if wrapped_packing:
        tokenizer.model_max_length = int(1e30)

    packing_suffix = "_packed_wrapped" if wrapped_packing else ("_packed" if pack else "")
    dataset_save_dir = f"{save_path}/preprocessed_data{packing_suffix}/{tokenizer_name}/{out_path}/dataset"

    dataset = load_from_disk(dataset_location)
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        assert len(splits) == 1, f"Expected a single-split DatasetDict, got splits: {splits}"
        dataset = dataset[splits[0]]
    assert isinstance(dataset, Dataset)

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    tokenized_dataset: Dataset = dataset.map(
        format_and_tokenize_examples,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "q_col": q_col,
            "a_col": a_col,
            "max_length": max_length,
            "take_loss_over_all_tokens": take_loss_over_all_tokens,
            "return_type": None if pack else "pt",
        },
        cache_file_name=f"{cache_path}/processing_cache/tmp_cache_{out_path}.arrow",
    )

    if pack:
        # ffd and bfd are the same algorithm, just renamed across trl versions:
        # https://github.com/huggingface/trl/commit/0353d6766144981040ce47eb16925bb7f5e6ecf7
        tokenized_dataset = pack_dataset(
            tokenized_dataset,
            seq_length=max_length + 1,
            strategy="wrapped" if wrapped_packing else "ffd",
            map_kwargs={
                "num_proc": num_proc,
                "desc": "packing",
                "batch_size": batch_size,
                "writer_batch_size": batch_size,
                "cache_file_name": f"{cache_path}/processing_cache/tmp_cache_packing{'_wrapped' if wrapped_packing else ''}_{out_path}.arrow",
            },
        )

        tokenized_dataset.set_format("pt")
        if "position_ids" in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.remove_columns(["position_ids"])

        tokenized_dataset = tokenized_dataset.map(
            pad_or_truncate,
            fn_kwargs={"tokenizer_pad_id": tokenizer.pad_token_id, "max_len": max_length + 1},
            num_proc=num_proc,
            cache_file_name=f"{cache_path}/processing_cache/tmp_cache_padding_{out_path}.arrow",
        )

    print(f"Saving dataset to {dataset_save_dir} ...")
    tokenized_dataset.save_to_disk(dataset_save_dir, num_proc=num_proc, max_shard_size="2GB")
    tokenized_dataset.cleanup_cache_files()
    dataset.cleanup_cache_files()
    print(f"Done. {len(tokenized_dataset):,} packed sequences saved.")


if __name__ == "__main__":
    CLI(process_data)
