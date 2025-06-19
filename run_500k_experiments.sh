#!/bin/bash

# Spurious Rewards Long-term Experiments (500k iterations)
# Testing Any Valid and Diversity methods for extended training

echo "=========================================="
echo "Starting 500k iterations experiments"
echo "Date: $(date)"
echo "=========================================="

# Create output directory for logs
LOG_DIR="logs/500k_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# GPU selection (modify if needed)
GPU_ID=0

# Common parameters
COMMON_PARAMS="--dataset simple_graph \
               --n_layer 1 \
               --n_head 1 \
               --n_embd 120 \
               --max_iters 500000 \
               --num_nodes 100 \
               --num_of_paths 20 \
               --test_interval 5000 \
               --device cuda:$GPU_ID \
               --temperature 1.0 \
               --num_eval_batches 10 \
               --seed 42 \
               --learning_rate 0.0005 \
               --checkpoint_interval 50000"

# Function to run experiment
run_experiment() {
    local reward_type=$1
    local extra_params=$2
    local log_file="$LOG_DIR/${reward_type}_500k.log"
    
    echo ""
    echo "=========================================="
    echo "Starting $reward_type experiment"
    echo "Log file: $log_file"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Run the experiment
    python train_spurious_rewards.py \
        --reward_type $reward_type \
        $COMMON_PARAMS \
        $extra_params \
        2>&1 | tee $log_file
    
    echo "Completed $reward_type experiment at $(date)"
}

# Option 1: Run experiments sequentially (safer for single GPU)
echo "Running experiments sequentially..."

# Run Any Valid
run_experiment "any_valid" ""

# Run Diversity
run_experiment "diversity" "--diversity_weight 0.1"

# Option 2: Run experiments in parallel (uncomment if you have multiple GPUs)
# echo "Running experiments in parallel..."
# 
# # Run Any Valid on GPU 0
# CUDA_VISIBLE_DEVICES=0 python train_spurious_rewards.py \
#     --reward_type any_valid \
#     $COMMON_PARAMS \
#     2>&1 | tee "$LOG_DIR/any_valid_500k.log" &
# 
# # Run Diversity on GPU 1
# CUDA_VISIBLE_DEVICES=1 python train_spurious_rewards.py \
#     --reward_type diversity \
#     --diversity_weight 0.1 \
#     $COMMON_PARAMS \
#     2>&1 | tee "$LOG_DIR/diversity_500k.log" &
# 
# # Wait for both to complete
# wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Logs saved in: $LOG_DIR"
echo "Completion time: $(date)"
echo "=========================================="

# Optional: Send notification (uncomment and modify if needed)
# echo "500k experiments completed" | mail -s "Experiment Status" your_email@example.com