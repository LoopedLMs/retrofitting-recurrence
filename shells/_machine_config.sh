# Machine-specific configuration template
# Copy this file to machine_config.sh and fill in your paths
# machine_config.sh is gitignored — never commit it

# ============================================================================
# Directory Configuration
# ============================================================================
BASE_DIR="~/retrofitting-recurrence"

# HuggingFace cache (models, tokenizers, datasets)
export HF_HOME="/srv/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Local model storage (corresponds to models/ in the repo, gitignored)
export MODELS_DIR="$BASE_DIR/models"

# Preprocessed/packed parquet datasets (used in shells/*.sh via $PROCESSED_DATA_PATH)
export PROCESSED_DATA_PATH="$BASE_DIR/processed_datasets"

# ============================================================================
# SLURM Configuration
# ============================================================================

# GPU jobs (training, evaluation)
export SLURM_PARTITION_GPU="your-gpu-partition"
export SLURM_QOS_GPU="your-gpu-qos"
export NUM_GPUS="${NUM_GPUS:-2}"               # Number of GPUs per job (overridable via env)
export SLURM_MEM_GPU="64G"
export SLURM_MAIL_USER="your.email@example.com"

# CPU jobs (data preprocessing, downloads)
export SLURM_PARTITION_CPU="your-cpu-partition"
export SLURM_QOS_CPU="your-cpu-qos"
export SLURM_MEM_CPU="32G"

# ============================================================================
# Validation
# ============================================================================

validate_config() {
    local required_vars=(HF_HOME HF_DATASETS_CACHE MODELS_DIR PROCESSED_DATA_PATH)
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            echo "ERROR: $var not set in machine_config.sh" >&2
            return 1
        fi
    done
    return 0
}
