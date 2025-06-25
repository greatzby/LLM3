"""
直接复现训练代码的评估逻辑，逐位置分析
"""
import os
import torch
import numpy as np
import pickle
from model import GPT, GPTConfig
from contextlib import nullcontext
import matplotlib.pyplot as plt

def load_checkpoint_and_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, checkpoint

def analyze_tf_breakdown(model, val_data, block_size, device):
    """完全复现训练时的评估，但保存详细信息"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    batch_size = 64
    num_batches = 10
    data_size = block_size + 1
    
    # 收集所有预测
    all_predictions = []
    all_targets = []
    all_logits = []
    
    for batch_idx in range(num_batches):
        # 完全复制训练代码的数据加载
        ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
        
        x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        x, y = x.to(device), y.to(device)
        
        with ctx:
            logits, _ = model(x, y)
        
        preds = torch.argmax(logits, dim=-1)
        
        all_predictions.append(preds.cpu())
        all_targets.append(y.cpu())
        all_logits.append(logits.cpu())
    
    # 合并所有批次
    all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples, block_size]
    all_targets = torch.cat(all_targets, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # 计算整体准确率（应该匹配训练日志）
    total_correct = (all_predictions == all_targets).sum().item()
    total_count = all_targets.numel()
    overall_accuracy = total_correct / total_count
    
    print(f"Overall TF Accuracy: {overall_accuracy:.4f}")
    print(f"Total tokens: {total_count}, Correct: {total_correct}")
    
    # 按位置分析
    position_analysis = {}
    for pos in range(block_size):
        pos_preds = all_predictions[:, pos]
        pos_targets = all_targets[:, pos]
        
        # 所有token的准确率
        pos_correct = (pos_preds == pos_targets).sum().item()
        pos_total = len(pos_targets)
        pos_accuracy = pos_correct / pos_total
        
        # 分别统计padding和非padding
        padding_mask = (pos_targets == 0)
        non_padding_mask = ~padding_mask
        
        padding_correct = ((pos_preds == pos_targets) & padding_mask).sum().item()
        padding_total = padding_mask.sum().item()
        
        non_padding_correct = ((pos_preds == pos_targets) & non_padding_mask).sum().item()
        non_padding_total = non_padding_mask.sum().item()
        
        position_analysis[pos] = {
            'overall_acc': pos_accuracy,
            'overall_correct': pos_correct,
            'overall_total': pos_total,
            'padding_acc': padding_correct / padding_total if padding_total > 0 else 0,
            'padding_total': padding_total,
            'non_padding_acc': non_padding_correct / non_padding_total if non_padding_total > 0 else 0,
            'non_padding_total': non_padding_total
        }
    
    return overall_accuracy, position_analysis, all_predictions, all_targets

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430'
    data_dir = 'data/simple_graph/100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载元数据
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    block_size = meta['block_size']
    
    # 加载验证数据
    val_data_path = os.path.join(data_dir, 'val.bin')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # 分析两个checkpoint
    results = {}
    
    for name, iteration in [('stable_100k', 100000), ('collapsed_200k', 200000)]:
        print(f"\n{'='*60}")
        print(f"Analyzing {name}...")
        
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, _ = load_checkpoint_and_model(checkpoint_path, device)
        
        overall_acc, pos_analysis, predictions, targets = analyze_tf_breakdown(model, val_data, block_size, device)
        results[name] = (overall_acc, pos_analysis)
        
        # 打印详细分解
        print(f"\nPosition-by-position breakdown:")
        print(f"{'Pos':>3} {'Overall':>8} {'Padding':>8} {'Non-Pad':>8} {'P-Total':>8} {'NP-Total':>8}")
        print("-" * 50)
        
        for pos in range(block_size):
            stats = pos_analysis[pos]
            print(f"{pos:3d} {stats['overall_acc']:8.3f} {stats['padding_acc']:8.3f} "
                  f"{stats['non_padding_acc']:8.3f} {stats['padding_total']:8d} "
                  f"{stats['non_padding_total']:8d}")
        
        # 计算加权平均
        total_padding = sum(stats['padding_total'] for stats in pos_analysis.values())
        total_non_padding = sum(stats['non_padding_total'] for stats in pos_analysis.values())
        
        if total_padding > 0:
            weighted_padding_acc = sum(stats['padding_acc'] * stats['padding_total'] 
                                      for stats in pos_analysis.values()) / total_padding
        else:
            weighted_padding_acc = 0
            
        if total_non_padding > 0:
            weighted_non_padding_acc = sum(stats['non_padding_acc'] * stats['non_padding_total'] 
                                         for stats in pos_analysis.values()) / total_non_padding
        else:
            weighted_non_padding_acc = 0
        
        print(f"\nWeighted averages:")
        print(f"  Padding accuracy: {weighted_padding_acc:.3f} ({total_padding}/{total_padding + total_non_padding} = {total_padding/(total_padding + total_non_padding):.1%})")
        print(f"  Non-padding accuracy: {weighted_non_padding_acc:.3f} ({total_non_padding}/{total_padding + total_non_padding} = {total_non_padding/(total_padding + total_non_padding):.1%})")
        print(f"  Combined: {weighted_padding_acc * total_padding/(total_padding + total_non_padding) + weighted_non_padding_acc * total_non_padding/(total_padding + total_non_padding):.3f}")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图1：整体准确率对比
    positions = list(range(block_size))
    
    for name in ['stable_100k', 'collapsed_200k']:
        _, pos_analysis = results[name]
        overall_accs = [pos_analysis[pos]['overall_acc'] for pos in positions]
        ax1.plot(positions, overall_accs, marker='o', label=name)
    
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Overall Accuracy')
    ax1.set_title('Position-wise Accuracy (All Tokens)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # 图2：padding vs non-padding
    x = np.arange(block_size)
    width = 0.35
    
    _, pos_analysis = results['collapsed_200k']
    padding_accs = [pos_analysis[pos]['padding_acc'] for pos in positions]
    non_padding_accs = [pos_analysis[pos]['non_padding_acc'] for pos in positions]
    
    ax2.bar(x - width/2, padding_accs, width, label='Padding', alpha=0.7)
    ax2.bar(x + width/2, non_padding_accs, width, label='Non-padding', alpha=0.7)
    
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Collapsed Model: Padding vs Non-padding Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'tf_breakdown_analysis.png'))
    
    print(f"\n\nVisualization saved to: {os.path.join(base_dir, 'tf_breakdown_analysis.png')}")

if __name__ == "__main__":
    main()