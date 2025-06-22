"""
test_gradient_competition.py
测试梯度竞争分析是否能正常运行
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试import
try:
    from model import GPTConfig, GPT
    print("✓ Model imports successful")
except ImportError as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

# 测试数据路径
data_dir = 'data/simple_graph/100'
if os.path.exists(data_dir):
    print(f"✓ Data directory exists: {data_dir}")
else:
    print(f"✗ Data directory not found: {data_dir}")

# 测试checkpoint
checkpoint_dir = "out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430"
if os.path.exists(checkpoint_dir):
    print(f"✓ Checkpoint directory exists")
    
    # 列出一些checkpoint
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
    print(f"  Found {len(ckpts)} checkpoints")
    if ckpts:
        print(f"  Example: {ckpts[0]}")
else:
    print(f"✗ Checkpoint directory not found: {checkpoint_dir}")

print("\nIf all checks pass, you can run the gradient competition analysis.")