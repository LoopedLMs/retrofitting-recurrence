cd "$(dirname "$0")/.."
uv sync
source .venv/bin/activate
source shells/_machine_config.sh
validate_config || exit 1

# ============================================================================
# Experiment 1a: Adapter-only (frozen backbone)
# ============================================================================
# OLMo-2-0425-1B with (7, 4, 5) split — all 16 layers preserved
# Only the adapter (~4M params) is trained
# Higher LR since adapter is the only trainable component

python train.py \
    --epochs=1 \
    --max_length=1024 \
    --out_path=outputs \
    --run_name="olmo_adapter_only" \
    --optim_config.lr=1e-3 \
    --optim_config.weight_decay=1e-4 \
    --model_name="models/OLMo-2-0425-1B_pre7_core4_coda5" \
    --tokenizer_name="allenai/OLMo-2-0425-1B" \
    --dataset_location="allenai/c4" \
    --dataset_args.q_col="text" \
    --dataset_config="en" \
    --max_samples=10000 \
    --scheduler_args.cooldown=0.6 \
    --scheduler_args.warmup=0.005 \
    --max_grad_norm=1.0 \
    --micro_batch_size=8 \
    --batch_size=64 \
    --no_amp=false \
    --max_steps=500 \
    --compile=false \
    --save_interval=1000 \
    --freeze_backbone=true \
    --mean_recurrence_schedule.turn_on=true \
    --mean_recurrence_schedule.warmup=0.75 \
    --mean_recurrence_schedule.start=2 \
    --mean_recurrence_schedule.max_mean_rec=4 \
    --mean_recurrence_schedule.warmup_type="1-sqrt" \
    --mean_backprop_depth_schedule.turn_on=false
