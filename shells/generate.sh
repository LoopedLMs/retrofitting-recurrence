cd "$(dirname "$0")/.."
uv sync
source .venv/bin/activate
source shells/machine_config.sh
validate_config || exit 1

MODEL_NAME="smcleish/Recurrent-Llama-3.2-train-recurrence-4"
PROMPT="The key to solving hard math problems is"
NUM_RECURRENCE_STEPS=4
MAX_NEW_TOKENS=200

python generate.py \
    --model_name="$MODEL_NAME" \
    --prompt="$PROMPT" \
    --num_recurrence_steps=$NUM_RECURRENCE_STEPS \
    --max_new_tokens=$MAX_NEW_TOKENS
