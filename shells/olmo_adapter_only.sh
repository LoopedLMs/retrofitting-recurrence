#!/usr/bin/env bash
cd ~/retrofitting-recurrence
uv sync
source .venv/bin/activate
source shells/_machine_config.sh
validate_config || exit 1

# ============================================================================
# Experiment 1a: Adapter-only (frozen backbone), skip input injection at step 0
# ============================================================================
# OLMo-2-0425-1B with (7, 4, 5) split — all 16 layers preserved
# Only the adapter (~4M params) is trained
# Higher LR since adapter is the only trainable component
# initial_state_mode="skip-adapter": step 0 passes input_embeds directly to
# the core block without noise or adapter injection

torchrun --nproc_per_node="${NUM_GPUS:-1}" train.py \
    --epochs=1 \
    --max_length=1024 \
    --out_path=outputs \
    --run_name="olmo_adapter_only_skip_init" \
    --optim_config.lr=1e-3 \
    --optim_config.weight_decay=1e-4 \
    --model_name="models/OLMo-2-0425-1B_pre7_core4_coda5" \
    --preprocessed_data_path="$PROCESSED_DATA_PATH/OLMo-2-0425-1B/preprocessed_data_packed_wrapped/allenai/OLMo-2-0425-1B/olmo_2_0425_1b_nemotron_cc_math_v1_4plus_wrapped_packing/dataset" \
    --is_parquet_dataset=true \
    --scheduler_args.cooldown=0.6 \
    --scheduler_args.warmup=0.005 \
    --max_grad_norm=1.0 \
    --micro_batch_size=8 \
    --batch_size=64 \
    --no_amp=false \
    --max_steps=500 \
    --compile=true \
    --save_interval=1000 \
    --freeze_backbone=true \
    --mean_recurrence_schedule.turn_on=true \
    --mean_recurrence_schedule.warmup=0.75 \
    --mean_recurrence_schedule.start=2 \
    --mean_recurrence_schedule.max_mean_rec=4 \
    --mean_recurrence_schedule.warmup_type="1-sqrt" \
    --mean_backprop_depth_schedule.turn_on=false \
    --initial_state_mode="skip-adapter" \
    --fix_num_steps=true
