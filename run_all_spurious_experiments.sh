#!/bin/bash

# 运行所有Spurious Rewards实验的脚本

# 实验配置
NUM_NODES=100
NUM_PATHS=20
MAX_ITERS=200000
TEST_INTERVAL=1000

echo "Starting Spurious Rewards Experiments..."
echo "======================================="

# 1. Baseline - 标准训练
echo "Running Baseline (Standard Training)..."
python train_spurious_rewards.py \
    --reward_type standard \
    --num_nodes $NUM_NODES \
    --num_of_paths $NUM_PATHS \
    --max_iters $MAX_ITERS \
    --test_interval $TEST_INTERVAL \
    --seed 42

# 2. Any Valid Reward
echo "Running Any Valid Reward..."
python train_spurious_rewards.py \
    --reward_type any_valid \
    --num_nodes $NUM_NODES \
    --num_of_paths $NUM_PATHS \
    --max_iters $MAX_ITERS \
    --test_interval $TEST_INTERVAL \
    --seed 42

# 3. Mixed Reward (不同比例)
for alpha in 0.3 0.5 0.7; do
    echo "Running Mixed Reward (alpha=$alpha)..."
    python train_spurious_rewards.py \
        --reward_type mixed \
        --mixed_alpha $alpha \
        --num_nodes $NUM_NODES \
        --num_of_paths $NUM_PATHS \
        --max_iters $MAX_ITERS \
        --test_interval $TEST_INTERVAL \
        --seed 42
done

# 4. Diversity Reward
echo "Running Diversity Reward..."
python train_spurious_rewards.py \
    --reward_type diversity \
    --diversity_weight 0.1 \
    --num_nodes $NUM_NODES \
    --num_of_paths $NUM_PATHS \
    --max_iters $MAX_ITERS \
    --test_interval $TEST_INTERVAL \
    --seed 42

# 5. Phase-Aware Training
echo "Running Phase-Aware Training..."
python train_spurious_rewards.py \
    --reward_type phase_aware \
    --phase_aware_transition 120000 \
    --num_nodes $NUM_NODES \
    --num_of_paths $NUM_PATHS \
    --max_iters $MAX_ITERS \
    --test_interval $TEST_INTERVAL \
    --seed 42

echo "======================================="
echo "All experiments completed!"
echo "Results saved in out/spurious_rewards/"