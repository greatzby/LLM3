import os
import torch
import numpy as np
import pickle
from model import GPTConfig, GPT
from tqdm import tqdm

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    model = GPT(GPTConfig(**checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('cuda:0')
    return model, checkpoint['model_args']

def get_batch_from_val(val_data, batch_size, block_size):
    """从val.bin获取批次 - 与训练代码一致"""
    data_size = block_size + 1
    ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
    
    x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to('cuda:0'), y.to('cuda:0')
    return x, y

def test_model_like_training(model, val_data, block_size, num_eval_batches=10, val_batch_size=64):
    """完全复现训练时的test_model函数"""
    total_correct = 0
    total_count = 0
    
    for batch_idx in range(num_eval_batches):
        X, Y = get_batch_from_val(val_data, val_batch_size, block_size)
        
        # 使用dummy targets获取完整logits
        dummy_targets = torch.zeros_like(X)
        with torch.no_grad():
            logits, _ = model(X, targets=dummy_targets)
        
        preds = torch.argmax(logits, dim=-1)
        
        # Token级别准确率
        batch_correct = (preds == Y).float().sum().item()
        batch_total = Y.numel()
        
        total_correct += batch_correct
        total_count += batch_total
    
    return total_correct / total_count

# 主分析
def main():
    data_dir = 'data/simple_graph/100'
    
    # 加载meta信息
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    block_size = meta['block_size']
    
    # 加载val.bin - 与训练一致！
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    print(f"Loaded val.bin with {len(val_data)} tokens")
    
    # 测试关键checkpoints
    checkpoints = [100000, 120000, 140000, 160000]
    
    print("\nTesting with val.bin (same as training):")
    print("="*50)
    
    for ckpt in checkpoints:
        ckpt_path = f'out/simple_graph_1_1_120_100_original_seed42/{ckpt}_ckpt_20.pt'
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found")
            continue
            
        model, model_args = load_model(ckpt_path)
        
        # 使用与训练相同的参数
        accuracy = test_model_like_training(
            model, 
            val_data, 
            block_size,
            num_eval_batches=10,  # 与训练一致
            val_batch_size=64     # 与训练一致
        )
        
        print(f"Checkpoint {ckpt}: TF Accuracy = {accuracy:.3f}")
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()