"""
精确复现训练代码的TF评估，找出差异原因
"""
import os
import torch
import numpy as np
import pickle
import networkx as nx
from model import GPT, GPTConfig
import torch.nn.functional as F
from contextlib import nullcontext
import matplotlib.pyplot as plt

def load_checkpoint_and_model(checkpoint_path, device='cuda'):
    """加载checkpoint和模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def test_model_exact(model, val_data, block_size, device, num_eval_batches=10, train_batch_size=1024, val_batch_size=64):
    """完全复制训练代码的test_model函数"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # 完全复制get_batch逻辑
    def get_batch(split):
        data = val_data  # 只用val
        bs = val_batch_size
        data_size = block_size + 1
        
        ix = torch.randint((len(data) - data_size) // data_size, (bs,)) * data_size
        
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        x, y = x.to(device), y.to(device)
        return x, y
    
    # 完全复制test_model逻辑
    total_correct = 0
    total_count = 0
    batch_accuracies = []
    
    # 收集详细信息用于分析
    position_correct = np.zeros(block_size)
    position_total = np.zeros(block_size)
    
    for batch_idx in range(num_eval_batches):
        X, Y = get_batch('val')
        with ctx:
            logits, _ = model(X, Y)
        preds = torch.argmax(logits, dim=-1)
        
        # 批次准确率（包括所有token）
        batch_correct = (preds == Y).float().sum().item()
        batch_total = Y.numel()
        batch_accuracy = batch_correct / batch_total
        batch_accuracies.append(batch_accuracy)
        
        total_correct += batch_correct
        total_count += batch_total
        
        # 分析每个位置
        for pos in range(block_size):
            mask = Y[:, pos] != -100  # 检查是否是有效位置
            if mask.any():
                correct = ((preds[:, pos] == Y[:, pos]) & mask).sum().item()
                total = mask.sum().item()
                position_correct[pos] += correct
                position_total[pos] += total
    
    overall_accuracy = total_correct / total_count
    accuracy_std = np.std(batch_accuracies)
    
    # 计算每个位置的准确率
    position_accuracy = np.zeros(block_size)
    for pos in range(block_size):
        if position_total[pos] > 0:
            position_accuracy[pos] = position_correct[pos] / position_total[pos]
    
    return overall_accuracy, accuracy_std, position_accuracy, position_total

def analyze_sequence_structure(val_data, block_size, itos, num_samples=1000):
    """分析val数据的序列结构"""
    data_size = block_size + 1
    num_sequences = (len(val_data) - data_size) // data_size
    
    # 统计信息
    seq_lengths = []
    padding_ratios = []
    
    for i in range(min(num_samples, num_sequences)):
        start_idx = i * data_size
        seq = val_data[start_idx:start_idx + data_size]
        
        # 找到实际序列长度（第一个padding之前）
        actual_length = 0
        for j, token in enumerate(seq):
            if token == 0:  # padding
                actual_length = j
                break
            elif j == len(seq) - 1:
                actual_length = len(seq)
        
        seq_lengths.append(actual_length)
        padding_ratio = (data_size - actual_length) / data_size
        padding_ratios.append(padding_ratio)
    
    return seq_lengths, padding_ratios

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430'
    data_dir = 'data/simple_graph/100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载元数据
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    
    print(f"Block size: {block_size}")
    print(f"Vocab size: {len(itos)}")
    
    # 加载验证数据
    val_data_path = os.path.join(data_dir, 'val.bin')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # 分析数据结构
    print("\nAnalyzing data structure...")
    seq_lengths, padding_ratios = analyze_sequence_structure(val_data, block_size, itos)
    print(f"Average sequence length: {np.mean(seq_lengths):.1f}")
    print(f"Average padding ratio: {np.mean(padding_ratios):.3f}")
    
    # 测试不同的checkpoint
    checkpoints = {
        '50k': 50000,
        '100k': 100000,
        '190k': 190000,
        '200k': 200000
    }
    
    results = {}
    
    for name, iteration in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Testing checkpoint {name} (iteration {iteration})...")
        
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
        
        # 使用完全相同的参数
        accuracy, std, pos_acc, pos_total = test_model_exact(
            model, val_data, block_size, device, 
            num_eval_batches=10, val_batch_size=64
        )
        
        print(f"TF Accuracy: {accuracy:.4f} (±{std:.4f})")
        
        # 检查checkpoint中保存的历史
        if 'tf_history' in checkpoint:
            history = checkpoint['tf_history']
            if len(history) > 0:
                # 找到最接近的记录
                test_interval = 2000
                closest_idx = iteration // test_interval - 1
                if 0 <= closest_idx < len(history):
                    print(f"Training log TF: {history[closest_idx]:.4f}")
        
        # 显示前几个位置的准确率
        print("\nPosition-wise accuracy (first 10):")
        for pos in range(min(10, block_size)):
            if pos_total[pos] > 0:
                print(f"  Pos {pos}: {pos_acc[pos]:.3f} ({int(pos_total[pos])} samples)")
        
        results[name] = {
            'accuracy': accuracy,
            'std': std,
            'pos_acc': pos_acc
        }
    
    # 如果仍然显示72%而不是15%，让我们深入分析
    if results['200k']['accuracy'] > 0.5:
        print("\n" + "="*60)
        print("DEBUGGING: Why is TF not showing collapse?")
        print("="*60)
        
        # 1. 检查padding的影响
        print("\n1. Padding analysis:")
        print(f"   If {np.mean(padding_ratios):.1%} of tokens are padding,")
        print(f"   and model always predicts padding correctly,")
        print(f"   minimum accuracy would be ~{np.mean(padding_ratios):.1%}")
        
        # 2. 分析非padding位置的准确率
        print("\n2. Non-padding positions:")
        for name in ['50k', '200k']:
            pos_acc = results[name]['pos_acc']
            # 假设前几个位置不是padding
            non_pad_positions = [i for i in range(3, 10) if pos_acc[i] < 0.99]
            if non_pad_positions:
                non_pad_acc = np.mean([pos_acc[i] for i in non_pad_positions])
                print(f"   {name}: {non_pad_acc:.3f} on likely non-padding positions")
        
        # 3. 尝试不同的batch size
        print("\n3. Testing with different batch sizes:")
        for bs in [32, 128, 256]:
            model_200k, _ = load_checkpoint_and_model(
                os.path.join(base_dir, 'ckpt_200000.pt'), device
            )
            acc, _, _, _ = test_model_exact(
                model_200k, val_data, block_size, device,
                num_eval_batches=5, val_batch_size=bs
            )
            print(f"   Batch size {bs}: {acc:.4f}")
    
    # 绘制位置准确率对比
    plt.figure(figsize=(12, 6))
    positions = list(range(min(20, block_size)))
    
    for name in ['50k', '200k']:
        pos_acc = results[name]['pos_acc']
        plt.plot(positions, [pos_acc[p] for p in positions], 
                marker='o', label=name)
    
    plt.xlabel('Position in Sequence')
    plt.ylabel('Accuracy')
    plt.title('Position-wise Accuracy: Stable vs Collapsed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_dir, 'position_accuracy_debug.png'))
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(base_dir, 'position_accuracy_debug.png')}")

if __name__ == "__main__":
    main()