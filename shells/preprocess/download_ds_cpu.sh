#!/usr/bin/env bash
# Downloads HuggingFace datasets to disk via utils/download_ds.py.
#
# Usage:
#   bash shells/preprocess/download_nemotron_cpu.sh --datasets nemotron-math
#   bash shells/preprocess/download_nemotron_cpu.sh --datasets nemotron-math sft-general

cd ~/retrofitting-recurrence
uv sync
source .venv/bin/activate
source shells/_machine_config.sh
validate_config || exit 1

python utils/download_ds.py --dataset-path "$HF_DATASETS_CACHE" --datasets nemotron-math
