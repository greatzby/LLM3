# 检查脚本 - check_config.py
import pickle
import numpy as np
import torch

# 1. 检查200节点数据的配置
print("=== Checking 200-node data configuration ===")
with open('data/simple_graph/200/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
print(f"Data vocab_size: {meta['vocab_size']}")
print(f"Data block_size: {meta['block_size']}")

# 2. 检查实际数据中的token范围
val_data = np.memmap('data/simple_graph/200/val.bin', dtype=np.uint16, mode='r')
sample = val_data[:10000]
print(f"Actual token range in data: {np.min(sample)} to {np.max(sample)}")

# 3. 检查训练时保存的checkpoint
checkpoint_path = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250625_173711/ckpt_100000.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"\n=== Checking checkpoint configuration ===")
print(f"Model vocab_size: {checkpoint['model_args']['vocab_size']}")
print(f"Model block_size: {checkpoint['model_args']['block_size']}")

# 4. 检查训练时的iter_num确认这是200节点训练的
print(f"Checkpoint iter_num: {checkpoint.get('iter_num', 'N/A')}")