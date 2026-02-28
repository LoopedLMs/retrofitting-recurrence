#!/usr/bin/env bash
# Tokenizes and sequence-packs Nemotron-CC-Math-v1-4plus into parquet files
# ready for train.py (--is_parquet_dataset=true).
#
# Run AFTER download_nemotron_cpu.sh.

cd ~/retrofitting-recurrence
uv sync
source .venv/bin/activate
source shells/_machine_config.sh
validate_config || exit 1

TOKENIZER="allenai/OLMo-2-0425-1B"
DATASET_LOCATION="$HF_DATASETS_CACHE/Nemotron-CC-Math-v1-4plus"
MAX_LENGTH=1024
NUM_PROC="${NUM_PROC:-32}"
SAVE_PATH="$PROCESSED_DATA_PATH/OLMo-2-0425-1B"

python preprocess_data_packing.py \
    --tokenizer_name "$TOKENIZER" \
    --out_path "olmo_2_0425_1b_nemotron_cc_math_v1_4plus" \
    --dataset_location "$DATASET_LOCATION" \
    --q_col "text" \
    --max_length "$MAX_LENGTH" \
    --num_proc "$NUM_PROC" \
    --pack true \
    --wrapped_packing true \
    --save_path "$SAVE_PATH" \
    --cache_path "/tmp/nemotron_preprocess_cache"

