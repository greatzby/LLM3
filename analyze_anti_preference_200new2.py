"""
分析整个验证集的节点预测策略
确认节点策略是否真的改变了
"""
import os
import torch
import numpy as np
import pickle
from model import GPT, GPTConfig
from contextlib import nullcontext
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def load_checkpoint_and_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, checkpoint

def analyze_node_predictions_comprehensive(model, val_data, block_size, device):
    """分析整个验证集的预测，特别关注节点预测"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    batch_size = 64
    data_size = block_size + 1
    
    # 计算总序列数
    total_sequences = (len(val_data) - data_size) // data_size
    print(f"Analyzing {total_sequences} sequences...")
    
    # 收集所有预测
    all_token_predictions = Counter()
    node_predictions = Counter()  # 只统计节点（token 2-101）
    position_node_predictions = {pos: Counter() for pos in range(block_size)}
    
    # 同时收集真实分布用于对比
    all_token_targets = Counter()
    node_targets = Counter()
    
    # 处理所有批次
    num_full_batches = total_sequences // batch_size
    num_processed = 0
    
    for batch_idx in range(num_full_batches):
        # 顺序读取
        batch_start = batch_idx * batch_size
        indices = []
        for i in range(batch_size):
            seq_idx = batch_start + i
            data_start = seq_idx * data_size
            indices.append(data_start)
        
        x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in indices])
        y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in indices])
        
        x, y = x.to(device), y.to(device)
        
        with ctx:
            logits, _ = model(x, y)
        
        preds = torch.argmax(logits, dim=-1)
        
        # 统计预测
        preds_np = preds.cpu().numpy()
        targets_np = y.cpu().numpy()
        
        for seq_idx in range(batch_size):
            for pos in range(block_size):
                pred_token = int(preds_np[seq_idx, pos])
                target_token = int(targets_np[seq_idx, pos])
                
                # 统计所有token
                all_token_predictions[pred_token] += 1
                all_token_targets[target_token] += 1
                
                # 统计节点
                if 2 <= pred_token <= 101:
                    node_id = pred_token - 2
                    node_predictions[node_id] += 1
                    position_node_predictions[pos][node_id] += 1
                
                if 2 <= target_token <= 101:
                    node_id = target_token - 2
                    node_targets[node_id] += 1
        
        num_processed += batch_size
        if batch_idx % 10 == 0:
            print(f"  Processed {num_processed}/{total_sequences} sequences...")
    
    # 处理剩余批次
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
        
        # 统计
        preds_np = preds.cpu().numpy()
        targets_np = y.cpu().numpy()
        
        for seq_idx in range(remaining):
            for pos in range(block_size):
                pred_token = int(preds_np[seq_idx, pos])
                target_token = int(targets_np[seq_idx, pos])
                
                all_token_predictions[pred_token] += 1
                all_token_targets[target_token] += 1
                
                if 2 <= pred_token <= 101:
                    node_id = pred_token - 2
                    node_predictions[node_id] += 1
                    position_node_predictions[pos][node_id] += 1
                
                if 2 <= target_token <= 101:
                    node_id = target_token - 2
                    node_targets[node_id] += 1
    
    return {
        'all_tokens': all_token_predictions,
        'nodes': node_predictions,
        'position_nodes': position_node_predictions,
        'true_tokens': all_token_targets,
        'true_nodes': node_targets
    }

def analyze_node_strategy_change(results_before, results_after):
    """分析节点策略的变化"""
    print("\n" + "="*60)
    print("NODE PREDICTION STRATEGY ANALYSIS")
    print("="*60)
    
    # 1. 总体token分布
    print("\n1. Overall Token Distribution:")
    
    total_before = sum(results_before['all_tokens'].values())
    total_after = sum(results_after['all_tokens'].values())
    
    print(f"\nBefore collapse ({total_before} total predictions):")
    for token, count in results_before['all_tokens'].most_common(5):
        pct = count / total_before * 100
        token_name = {0: '[PAD]', 1: 'newline'}.get(token, f'node_{token-2}')
        print(f"  {token_name}: {count} ({pct:.1f}%)")
    
    print(f"\nAfter collapse ({total_after} total predictions):")
    for token, count in results_after['all_tokens'].most_common(5):
        pct = count / total_after * 100
        token_name = {0: '[PAD]', 1: 'newline'}.get(token, f'node_{token-2}')
        print(f"  {token_name}: {count} ({pct:.1f}%)")
    
    # 2. 节点预测分析
    print("\n2. Node Predictions Analysis:")
    
    nodes_before = results_before['nodes']
    nodes_after = results_after['nodes']
    
    total_nodes_before = sum(nodes_before.values())
    total_nodes_after = sum(nodes_after.values())
    
    print(f"\nTotal node predictions:")
    print(f"  Before: {total_nodes_before} ({total_nodes_before/total_before*100:.1f}%)")
    print(f"  After: {total_nodes_after} ({total_nodes_after/total_after*100:.1f}%)")
    
    # 3. Top预测的节点
    print("\n3. Most Frequently Predicted Nodes:")
    
    print("\nBefore collapse - Top 15 nodes:")
    for node, count in nodes_before.most_common(15):
        pct = count / total_nodes_before * 100 if total_nodes_before > 0 else 0
        print(f"  Node {node}: {count} ({pct:.1f}%)")
    
    print("\nAfter collapse - Top 15 nodes:")
    for node, count in nodes_after.most_common(15):
        pct = count / total_nodes_after * 100 if total_nodes_after > 0 else 0
        print(f"  Node {node}: {count} ({pct:.1f}%)")
    
    # 4. 节点分组分析
    print("\n4. Node Group Analysis:")
    
    def group_nodes(nodes_dict):
        groups = {
            'early (0-19)': 0,
            'mid-early (20-39)': 0,
            'mid (40-59)': 0,
            'mid-late (60-79)': 0,
            'late (80-99)': 0
        }
        
        for node, count in nodes_dict.items():
            if node < 20:
                groups['early (0-19)'] += count
            elif node < 40:
                groups['mid-early (20-39)'] += count
            elif node < 60:
                groups['mid (40-59)'] += count
            elif node < 80:
                groups['mid-late (60-79)'] += count
            else:
                groups['late (80-99)'] += count
        
        return groups
    
    groups_before = group_nodes(nodes_before)
    groups_after = group_nodes(nodes_after)
    
    print("\nNode predictions by group:")
    print("Group            | Before         | After          | Change")
    print("-"*65)
    
    for group in groups_before.keys():
        before_pct = groups_before[group] / total_nodes_before * 100 if total_nodes_before > 0 else 0
        after_pct = groups_after[group] / total_nodes_after * 100 if total_nodes_after > 0 else 0
        print(f"{group:16} | {before_pct:5.1f}% ({groups_before[group]:5}) | "
              f"{after_pct:5.1f}% ({groups_after[group]:5}) | {after_pct-before_pct:+6.1f}%")
    
    # 5. 统计指标
    print("\n5. Statistical Measures:")
    
    if nodes_before and nodes_after:
        # 计算平均节点编号
        avg_before = sum(node * count for node, count in nodes_before.items()) / total_nodes_before
        avg_after = sum(node * count for node, count in nodes_after.items()) / total_nodes_after
        
        # 计算标准差
        var_before = sum(count * (node - avg_before)**2 for node, count in nodes_before.items()) / total_nodes_before
        var_after = sum(count * (node - avg_after)**2 for node, count in nodes_after.items()) / total_nodes_after
        
        std_before = np.sqrt(var_before)
        std_after = np.sqrt(var_after)
        
        print(f"\nAverage node number:")
        print(f"  Before: {avg_before:.2f} (std: {std_before:.2f})")
        print(f"  After: {avg_after:.2f} (std: {std_after:.2f})")
        print(f"  Shift: {avg_after - avg_before:.2f}")
    
    return nodes_before, nodes_after

def visualize_node_strategy(nodes_before, nodes_after, save_path):
    """可视化节点策略变化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top节点对比
    ax = axes[0, 0]
    
    top_before = dict(nodes_before.most_common(20))
    top_after = dict(nodes_after.most_common(20))
    
    all_top_nodes = sorted(set(top_before.keys()) | set(top_after.keys()))
    
    x = np.arange(len(all_top_nodes))
    width = 0.35
    
    before_counts = [top_before.get(n, 0) for n in all_top_nodes]
    after_counts = [top_after.get(n, 0) for n in all_top_nodes]
    
    ax.bar(x - width/2, before_counts, width, label='Before', alpha=0.8)
    ax.bar(x + width/2, after_counts, width, label='After', alpha=0.8)
    
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Prediction Count')
    ax.set_title('Top 20 Most Predicted Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(all_top_nodes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 节点分布直方图
    ax = axes[0, 1]
    
    # 创建完整的节点分布
    node_dist_before = np.zeros(100)
    node_dist_after = np.zeros(100)
    
    for node, count in nodes_before.items():
        if 0 <= node <= 99:
            node_dist_before[node] = count
    
    for node, count in nodes_after.items():
        if 0 <= node <= 99:
            node_dist_after[node] = count
    
    bins = np.arange(0, 101, 5)
    ax.hist(np.repeat(np.arange(100), node_dist_before.astype(int)), 
            bins=bins, alpha=0.5, label='Before', density=True)
    ax.hist(np.repeat(np.arange(100), node_dist_after.astype(int)), 
            bins=bins, alpha=0.5, label='After', density=True)
    
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Density')
    ax.set_title('Node Prediction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 累积分布
    ax = axes[1, 0]
    
    # 计算累积分布
    nodes_sorted_before = sorted(nodes_before.items())
    nodes_sorted_after = sorted(nodes_after.items())
    
    if nodes_sorted_before:
        cum_before = np.cumsum([c for _, c in nodes_sorted_before])
        cum_before = cum_before / cum_before[-1]
        ax.plot([n for n, _ in nodes_sorted_before], cum_before, 'b-', label='Before', linewidth=2)
    
    if nodes_sorted_after:
        cum_after = np.cumsum([c for _, c in nodes_sorted_after])
        cum_after = cum_after / cum_after[-1]
        ax.plot([n for n, _ in nodes_sorted_after], cum_after, 'r-', label='After', linewidth=2)
    
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Node Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 变化热图
    ax = axes[1, 1]
    
    # 创建变化矩阵
    change_matrix = np.zeros((10, 10))
    
    total_before = sum(nodes_before.values())
    total_after = sum(nodes_after.values())
    
    for i in range(10):
        for j in range(10):
            node = i * 10 + j
            before_pct = nodes_before.get(node, 0) / total_before * 100 if total_before > 0 else 0
            after_pct = nodes_after.get(node, 0) / total_after * 100 if total_after > 0 else 0
            change_matrix[i, j] = after_pct - before_pct
    
    im = ax.imshow(change_matrix, cmap='RdBu', vmin=-2, vmax=2)
    ax.set_xlabel('Node ID (ones)')
    ax.set_ylabel('Node ID (tens)')
    ax.set_title('Change in Prediction Frequency (%)')
    
    # 添加colorbar
    plt.colorbar(im, ax=ax)
    
    # 添加网格
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")

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
    
    # 加载验证数据
    val_data_path = os.path.join(data_dir, 'val.bin')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # 分析两个模型
    results = {}
    
    for name, iteration in [('before', 100000), ('after', 200000)]:
        print(f"\n{'='*60}")
        print(f"Analyzing {name} collapse (iter {iteration})...")
        
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, _ = load_checkpoint_and_model(checkpoint_path, device)
        
        results[name] = analyze_node_predictions_comprehensive(model, val_data, block_size, device)
    
    # 分析节点策略变化
    nodes_before, nodes_after = analyze_node_strategy_change(results['before'], results['after'])
    
    # 可视化
    save_path = os.path.join(base_dir, 'node_strategy_comprehensive.png')
    visualize_node_strategy(nodes_before, nodes_after, save_path)
    
    # 最终总结
    print("\n" + "="*60)
    print("FINAL CONCLUSION")
    print("="*60)
    
    if nodes_before and nodes_after:
        avg_before = sum(n * c for n, c in nodes_before.items()) / sum(nodes_before.values())
        avg_after = sum(n * c for n, c in nodes_after.items()) / sum(nodes_after.values())
        shift = avg_after - avg_before
        
        if abs(shift) > 5:
            print(f"\n✅ NODE STRATEGY CHANGE CONFIRMED!")
            print(f"   Average node shifted by {shift:.1f}")
            print(f"   This is a systematic reorganization of prediction strategy")
        else:
            print(f"\n❌ No significant node strategy change detected")
            print(f"   Average shift only {shift:.1f}")
    else:
        print("\n⚠️ Insufficient node predictions to determine strategy change")

if __name__ == "__main__":
    main()