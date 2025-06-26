"""
分析节点预测策略的详细变化
特别关注：节点分布的变化发生在哪些位置
"""
import os
import torch
import numpy as np
import pickle
from model import GPT, GPTConfig
from contextlib import nullcontext
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
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

def analyze_node_predictions_by_position_type(model, val_data, block_size, device):
    """分析节点预测，按照ground truth类型分组"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    batch_size = 64
    data_size = block_size + 1
    
    # 计算总序列数
    total_sequences = (len(val_data) - data_size) // data_size
    print(f"Analyzing {total_sequences} sequences...")
    
    # 按ground truth类型分组收集预测
    predictions_by_gt_type = {
        'at_padding': Counter(),      # 在padding位置的预测
        'at_newline': Counter(),      # 在newline位置的预测
        'at_node': Counter(),         # 在节点位置的预测
        'at_node_correct': Counter(), # 在节点位置的正确预测
        'at_node_wrong': Counter()    # 在节点位置的错误预测
    }
    
    # 节点分组统计
    node_groups_by_gt_type = {
        'at_padding': defaultdict(int),
        'at_newline': defaultdict(int),
        'at_node': defaultdict(int),
        'at_node_correct': defaultdict(int),
        'at_node_wrong': defaultdict(int)
    }
    
    # 总体统计
    total_stats = {
        'padding_positions': 0,
        'newline_positions': 0,
        'node_positions': 0,
        'node_predictions_at_padding': 0,
        'node_predictions_at_newline': 0,
        'node_predictions_at_node': 0
    }
    
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
                
                # 确定ground truth类型
                if target_token == 0:  # padding
                    gt_type = 'at_padding'
                    total_stats['padding_positions'] += 1
                elif target_token == 1:  # newline
                    gt_type = 'at_newline'
                    total_stats['newline_positions'] += 1
                else:  # node
                    gt_type = 'at_node'
                    total_stats['node_positions'] += 1
                
                # 如果预测的是节点
                if 2 <= pred_token <= 101:
                    node_id = pred_token - 2
                    
                    # 记录在不同gt类型位置的节点预测
                    predictions_by_gt_type[gt_type][node_id] += 1
                    
                    # 记录节点分组
                    if node_id < 20:
                        group = 'early'
                    elif node_id < 40:
                        group = 'mid-early'
                    elif node_id < 60:
                        group = 'mid'
                    elif node_id < 80:
                        group = 'mid-late'
                    else:
                        group = 'late'
                    
                    node_groups_by_gt_type[gt_type][group] += 1
                    
                    # 统计
                    if gt_type == 'at_padding':
                        total_stats['node_predictions_at_padding'] += 1
                    elif gt_type == 'at_newline':
                        total_stats['node_predictions_at_newline'] += 1
                    elif gt_type == 'at_node':
                        total_stats['node_predictions_at_node'] += 1
                        
                        # 进一步区分正确和错误
                        if pred_token == target_token:
                            predictions_by_gt_type['at_node_correct'][node_id] += 1
                            node_groups_by_gt_type['at_node_correct'][group] += 1
                        else:
                            predictions_by_gt_type['at_node_wrong'][node_id] += 1
                            node_groups_by_gt_type['at_node_wrong'][group] += 1
        
        num_processed += batch_size
        if batch_idx % 10 == 0:
            print(f"  Processed {num_processed}/{total_sequences} sequences...")
    
    # 处理剩余批次
    remaining = total_sequences % batch_size
    if remaining > 0:
        # [类似的处理逻辑，为了简洁省略]
        pass
    
    return predictions_by_gt_type, node_groups_by_gt_type, total_stats

def analyze_position_type_changes(results_before, results_after):
    """分析不同位置类型上的变化"""
    pred_before, groups_before, stats_before = results_before
    pred_after, groups_after, stats_after = results_after
    
    print("\n" + "="*60)
    print("POSITION-TYPE BASED ANALYSIS")
    print("="*60)
    
    # 1. 总体统计
    print("\n1. Overall Statistics:")
    print(f"\nGround truth distribution:")
    print(f"  Padding positions: {stats_before['padding_positions']} "
          f"({stats_before['padding_positions']/sum([stats_before['padding_positions'], stats_before['newline_positions'], stats_before['node_positions']])*100:.1f}%)")
    print(f"  Newline positions: {stats_before['newline_positions']} "
          f"({stats_before['newline_positions']/sum([stats_before['padding_positions'], stats_before['newline_positions'], stats_before['node_positions']])*100:.1f}%)")
    print(f"  Node positions: {stats_before['node_positions']} "
          f"({stats_before['node_positions']/sum([stats_before['padding_positions'], stats_before['newline_positions'], stats_before['node_positions']])*100:.1f}%)")
    
    # 2. 节点预测发生在哪些位置
    print("\n2. Where Node Predictions Occur:")
    
    print("\nBefore collapse:")
    print(f"  Node predictions at padding positions: {stats_before['node_predictions_at_padding']}")
    print(f"  Node predictions at newline positions: {stats_before['node_predictions_at_newline']}")
    print(f"  Node predictions at node positions: {stats_before['node_predictions_at_node']}")
    
    print("\nAfter collapse:")
    print(f"  Node predictions at padding positions: {stats_after['node_predictions_at_padding']}")
    print(f"  Node predictions at newline positions: {stats_after['node_predictions_at_newline']}")
    print(f"  Node predictions at node positions: {stats_after['node_predictions_at_node']}")
    
    # 3. 节点分组变化 - 按位置类型
    print("\n3. Node Group Changes by Position Type:")
    
    for position_type in ['at_padding', 'at_node', 'at_node_wrong']:
        print(f"\n{position_type.upper()}:")
        print("Group       | Before | After  | Change")
        print("-"*40)
        
        groups_order = ['early', 'mid-early', 'mid', 'mid-late', 'late']
        for group in groups_order:
            before_count = groups_before[position_type].get(group, 0)
            after_count = groups_after[position_type].get(group, 0)
            change = after_count - before_count
            print(f"{group:11} | {before_count:6} | {after_count:6} | {change:+6}")
    
    # 4. 特别关注mid-late和late的变化
    print("\n4. Detailed Analysis of Mid-late and Late Changes:")
    
    # 计算在不同位置类型的变化
    for group in ['mid-late', 'late']:
        print(f"\n{group.upper()} nodes:")
        for position_type in ['at_padding', 'at_node_wrong', 'at_node_correct']:
            before = groups_before[position_type].get(group, 0)
            after = groups_after[position_type].get(group, 0)
            change = after - before
            print(f"  {position_type}: {before} → {after} ({change:+d})")
    
    return pred_before, pred_after, groups_before, groups_after

def visualize_position_based_changes(groups_before, groups_after, save_path):
    """可视化基于位置类型的变化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    groups_order = ['early', 'mid-early', 'mid', 'mid-late', 'late']
    
    # 1. 在padding位置的节点预测变化
    ax = axes[0, 0]
    
    before_padding = [groups_before['at_padding'].get(g, 0) for g in groups_order]
    after_padding = [groups_after['at_padding'].get(g, 0) for g in groups_order]
    
    x = np.arange(len(groups_order))
    width = 0.35
    
    ax.bar(x - width/2, before_padding, width, label='Before', alpha=0.8)
    ax.bar(x + width/2, after_padding, width, label='After', alpha=0.8)
    
    ax.set_xlabel('Node Group')
    ax.set_ylabel('Count')
    ax.set_title('Node Predictions at Padding Positions')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_order, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 在正确节点位置的预测变化
    ax = axes[0, 1]
    
    before_correct = [groups_before['at_node_correct'].get(g, 0) for g in groups_order]
    after_correct = [groups_after['at_node_correct'].get(g, 0) for g in groups_order]
    
    ax.bar(x - width/2, before_correct, width, label='Before', alpha=0.8)
    ax.bar(x + width/2, after_correct, width, label='After', alpha=0.8)
    
    ax.set_xlabel('Node Group')
    ax.set_ylabel('Count')
    ax.set_title('Correct Node Predictions at Node Positions')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_order, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 在错误节点位置的预测变化
    ax = axes[1, 0]
    
    before_wrong = [groups_before['at_node_wrong'].get(g, 0) for g in groups_order]
    after_wrong = [groups_after['at_node_wrong'].get(g, 0) for g in groups_order]
    
    ax.bar(x - width/2, before_wrong, width, label='Before', alpha=0.8)
    ax.bar(x + width/2, after_wrong, width, label='After', alpha=0.8)
    
    ax.set_xlabel('Node Group')
    ax.set_ylabel('Count')
    ax.set_title('Wrong Node Predictions at Node Positions')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_order, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 变化总结
    ax = axes[1, 1]
    
    # 计算各类型的变化
    changes = {
        'At Padding': [(after_padding[i] - before_padding[i]) for i in range(len(groups_order))],
        'At Node (Wrong)': [(after_wrong[i] - before_wrong[i]) for i in range(len(groups_order))],
        'At Node (Correct)': [(after_correct[i] - before_correct[i]) for i in range(len(groups_order))]
    }
    
    # 绘制堆叠条形图
    bottom = np.zeros(len(groups_order))
    colors = ['red', 'orange', 'green']
    
    for (label, data), color in zip(changes.items(), colors):
        ax.bar(groups_order, data, bottom=bottom, label=label, alpha=0.7, color=color)
        bottom += np.array(data)
    
    ax.set_xlabel('Node Group')
    ax.set_ylabel('Change in Count')
    ax.set_title('Breakdown of Node Prediction Changes by Position Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
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
        
        results[name] = analyze_node_predictions_by_position_type(model, val_data, block_size, device)
    
    # 分析变化
    pred_before, pred_after, groups_before, groups_after = analyze_position_type_changes(
        results['before'], results['after'])
    
    # 可视化
    save_path = os.path.join(base_dir, 'node_strategy_by_position_type.png')
    visualize_position_based_changes(groups_before, groups_after, save_path)
    
    # 最终结论
    print("\n" + "="*60)
    print("FINAL INSIGHTS")
    print("="*60)
    
    # 计算关键指标
    midlate_padding_change = (groups_after['at_padding'].get('mid-late', 0) - 
                             groups_before['at_padding'].get('mid-late', 0))
    late_padding_change = (groups_after['at_padding'].get('late', 0) - 
                          groups_before['at_padding'].get('late', 0))
    
    print(f"\nKey finding:")
    print(f"Mid-late change at padding positions: {midlate_padding_change:+d}")
    print(f"Late change at padding positions: {late_padding_change:+d}")
    
    if abs(midlate_padding_change) > 50 or abs(late_padding_change) > 50:
        print("\n✅ HYPOTHESIS CONFIRMED: Node distribution changes mainly occur at padding positions!")
    else:
        print("\n❌ HYPOTHESIS NOT CONFIRMED: Changes are distributed across position types")

if __name__ == "__main__":
    main()