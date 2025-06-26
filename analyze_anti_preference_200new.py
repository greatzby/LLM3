"""
直接复现训练代码的评估逻辑，逐位置分析 - 全面版本
分析整个验证集而不是随机采样
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
    
    # 打印模型配置以调试
    print(f"Model config: vocab_size={model_args.get('vocab_size', 'N/A')}, block_size={model_args.get('block_size', 'N/A')}")
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, checkpoint

def analyze_tf_breakdown_comprehensive(model, val_data, block_size, device, vocab_size, max_samples=None):
    """分析整个验证集（或指定数量的样本）"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    batch_size = 64
    data_size = block_size + 1
    
    # 计算总共有多少个完整序列
    total_sequences = (len(val_data) - data_size) // data_size
    
    # 如果指定了max_samples，限制分析数量
    if max_samples is not None:
        total_sequences = min(total_sequences, max_samples)
    
    print(f"Total sequences in validation set: {total_sequences}")
    print(f"Analyzing {total_sequences} sequences...")
    
    # 首先检查数据范围
    print("Checking data range...")
    sample_data = val_data[:min(10000, len(val_data))]
    min_token = int(np.min(sample_data))
    max_token = int(np.max(sample_data))
    print(f"Token range in validation data: {min_token} to {max_token}")
    print(f"Model vocab size: {vocab_size}")
    
    if max_token >= vocab_size:
        print(f"WARNING: Data contains tokens ({max_token}) >= vocab_size ({vocab_size})")
        print("This will cause CUDA errors. Please check if the model was trained with the correct vocab_size.")
        return None, None, None, None
    
    # 收集所有预测
    all_predictions = []
    all_targets = []
    all_logits = []
    
    # 处理完整的批次
    num_full_batches = total_sequences // batch_size
    num_processed = 0
    
    for batch_idx in range(num_full_batches):
        # 顺序读取数据，不随机
        batch_start = batch_idx * batch_size
        indices = []
        for i in range(batch_size):
            seq_idx = batch_start + i
            data_start = seq_idx * data_size
            indices.append(data_start)
        
        x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in indices])
        y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in indices])
        
        # 额外的安全检查
        if torch.max(x) >= vocab_size or torch.max(y) >= vocab_size:
            print(f"ERROR: Batch {batch_idx} contains out-of-vocabulary tokens")
            print(f"x max: {torch.max(x).item()}, y max: {torch.max(y).item()}")
            return None, None, None, None
        
        x, y = x.to(device), y.to(device)
        
        with ctx:
            logits, _ = model(x, y)
        
        preds = torch.argmax(logits, dim=-1)
        
        all_predictions.append(preds.cpu())
        all_targets.append(y.cpu())
        all_logits.append(logits.cpu())
        
        num_processed += batch_size
        
        if batch_idx % 10 == 0:
            print(f"  Processed {num_processed}/{total_sequences} sequences...")
    
    # 处理剩余的不完整批次
    remaining = total_sequences % batch_size
    if remaining > 0:
        batch_start = num_full_batches * batch_size
        indices = []
        for i in range(remaining):
            seq_idx = batch_start + i
            data_start = seq_idx * data_size
            indices.append(data_start)
        
        x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in indices])
        y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in indices])
        
        x, y = x.to(device), y.to(device)
        
        with ctx:
            logits, _ = model(x, y)
        
        preds = torch.argmax(logits, dim=-1)
        
        all_predictions.append(preds.cpu())
        all_targets.append(y.cpu())
        all_logits.append(logits.cpu())
        
        num_processed += remaining
    
    print(f"  Total processed: {num_processed} sequences")
    
    # 合并所有批次
    all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples, block_size]
    all_targets = torch.cat(all_targets, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # 计算整体准确率
    total_correct = (all_predictions == all_targets).sum().item()
    total_count = all_targets.numel()
    overall_accuracy = total_correct / total_count
    
    print(f"Overall TF Accuracy: {overall_accuracy:.4f}")
    print(f"Total tokens: {total_count}, Correct: {total_correct}")
    
    # 按位置分析
    position_analysis = {}
    
    # 200节点的最大node token是201
    max_node_token = vocab_size - 1
    
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
        
        # 进一步分析非padding中的细节
        newline_mask = (pos_targets == 1)
        node_mask = (pos_targets >= 2) & (pos_targets <= max_node_token)
        
        newline_correct = ((pos_preds == pos_targets) & newline_mask).sum().item()
        newline_total = newline_mask.sum().item()
        
        node_correct = ((pos_preds == pos_targets) & node_mask).sum().item()
        node_total = node_mask.sum().item()
        
        position_analysis[pos] = {
            'overall_acc': pos_accuracy,
            'overall_correct': pos_correct,
            'overall_total': pos_total,
            'padding_acc': padding_correct / padding_total if padding_total > 0 else 0,
            'padding_total': padding_total,
            'non_padding_acc': non_padding_correct / non_padding_total if non_padding_total > 0 else 0,
            'non_padding_total': non_padding_total,
            'newline_acc': newline_correct / newline_total if newline_total > 0 else 0,
            'newline_total': newline_total,
            'node_acc': node_correct / node_total if node_total > 0 else 0,
            'node_total': node_total
        }
    
    return overall_accuracy, position_analysis, all_predictions, all_targets

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250625_173711'
    data_dir = 'data/simple_graph/200'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载元数据
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    
    print(f"Data metadata: block_size={block_size}, vocab_size={vocab_size}")
    
    # 加载验证数据
    val_data_path = os.path.join(data_dir, 'val.bin')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # 分析两个checkpoint
    results = {}
    
    for name, iteration in [('stable_100k', 100000), ('collapsed_200k', 200000)]:
        print(f"\n{'='*60}")
        print(f"Analyzing {name}...")
        
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        
        # 检查checkpoint是否存在
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            continue
            
        model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
        
        # 检查模型的vocab_size
        model_vocab_size = model.config.vocab_size
        if model_vocab_size != vocab_size:
            print(f"WARNING: Model vocab_size ({model_vocab_size}) != Data vocab_size ({vocab_size})")
            print("This model may not be compatible with 200-node data.")
            
        # 分析整个验证集（或指定最大样本数）
        overall_acc, pos_analysis, predictions, targets = analyze_tf_breakdown_comprehensive(
            model, val_data, block_size, device, model_vocab_size, max_samples=None  # None表示分析全部
        )
        
        if overall_acc is None:
            print(f"Skipping {name} due to vocabulary mismatch")
            continue
            
        results[name] = (overall_acc, pos_analysis, predictions, targets)
        
        # 打印详细分解
        print(f"\nPosition-by-position breakdown:")
        print(f"{'Pos':>3} {'Overall':>8} {'Padding':>8} {'Non-Pad':>8} {'Newline':>8} {'Node':>8} {'P-Tot':>6} {'NP-Tot':>6} {'NL-Tot':>6} {'N-Tot':>6}")
        print("-" * 80)
        
        for pos in range(block_size):
            stats = pos_analysis[pos]
            print(f"{pos:3d} {stats['overall_acc']:8.3f} {stats['padding_acc']:8.3f} "
                  f"{stats['non_padding_acc']:8.3f} {stats['newline_acc']:8.3f} "
                  f"{stats['node_acc']:8.3f} {stats['padding_total']:6d} "
                  f"{stats['non_padding_total']:6d} {stats['newline_total']:6d} "
                  f"{stats['node_total']:6d}")
        
        # 计算加权平均
        total_padding = sum(stats['padding_total'] for stats in pos_analysis.values())
        total_non_padding = sum(stats['non_padding_total'] for stats in pos_analysis.values())
        total_newline = sum(stats['newline_total'] for stats in pos_analysis.values())
        total_node = sum(stats['node_total'] for stats in pos_analysis.values())
        
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
            
        if total_newline > 0:
            weighted_newline_acc = sum(stats['newline_acc'] * stats['newline_total'] 
                                     for stats in pos_analysis.values()) / total_newline
        else:
            weighted_newline_acc = 0
            
        if total_node > 0:
            weighted_node_acc = sum(stats['node_acc'] * stats['node_total'] 
                                  for stats in pos_analysis.values()) / total_node
        else:
            weighted_node_acc = 0
        
        print(f"\nWeighted averages:")
        print(f"  Padding accuracy: {weighted_padding_acc:.3f} ({total_padding}/{total_padding + total_non_padding} = {total_padding/(total_padding + total_non_padding):.1%})")
        print(f"  Non-padding accuracy: {weighted_non_padding_acc:.3f} ({total_non_padding}/{total_padding + total_non_padding} = {total_non_padding/(total_padding + total_non_padding):.1%})")
        print(f"  - Newline accuracy: {weighted_newline_acc:.3f} ({total_newline}/{total_padding + total_non_padding} = {total_newline/(total_padding + total_non_padding):.1%})")
        print(f"  - Node accuracy: {weighted_node_acc:.3f} ({total_node}/{total_padding + total_non_padding} = {total_node/(total_padding + total_non_padding):.1%})")
        print(f"  Combined: {weighted_padding_acc * total_padding/(total_padding + total_non_padding) + weighted_non_padding_acc * total_non_padding/(total_padding + total_non_padding):.3f}")
    
    # 只有当两个结果都存在时才进行对比分析
    if len(results) < 2:
        print("\nNot enough results for comparative analysis. Please check model compatibility.")
        return
        
    # 对比分析
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    _, pos_analysis_before, _, _ = results['stable_100k']
    _, pos_analysis_after, _, _ = results['collapsed_200k']
    
    # 计算各类准确率变化
    for token_type in ['padding', 'newline', 'node']:
        print(f"\n{token_type.upper()} accuracy changes:")
        total_before = sum(stats[f'{token_type}_total'] for stats in pos_analysis_before.values())
        total_after = sum(stats[f'{token_type}_total'] for stats in pos_analysis_after.values())
        
        if total_before > 0 and total_after > 0:
            acc_before = sum(stats[f'{token_type}_acc'] * stats[f'{token_type}_total'] 
                           for stats in pos_analysis_before.values()) / total_before
            acc_after = sum(stats[f'{token_type}_acc'] * stats[f'{token_type}_total'] 
                          for stats in pos_analysis_after.values()) / total_after
            
            print(f"  Before: {acc_before:.3f}")
            print(f"  After: {acc_after:.3f}")
            print(f"  Change: {acc_after - acc_before:+.3f}")
    
    # 可视化部分保持不变...
    # [可视化代码保持原样]

if __name__ == "__main__":
    main()