#!/usr/bin/env bash
cd ~/retrofitting-recurrence
uv sync
source .venv/bin/activate
source shells/_machine_config.sh
validate_config || exit 1

python -m convert_pretrained_model.convert_olmo \
    --source "allenai/OLMo-2-0425-1B" \
    --save-name "models" \
    --prelude 7 \
    --core 4 \
    --coda 5 \
    --start-index 7
