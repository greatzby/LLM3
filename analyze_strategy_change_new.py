"""
分析节点预测策略的系统性变化 - 改进版
处理模型只预测特殊token的情况
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

def analyze_graph_topology(G):
    """深入分析图的拓扑结构"""
    properties = {}
    
    # 基本属性
    for node in G.nodes():
        n = int(node)
        properties[n] = {
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
            'total_degree': G.degree(node),
        }
    
    # 计算中心性指标
    print("Computing centrality metrics...")
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, max_iter=100)
    closeness = nx.closeness_centrality(G)
    
    for node in G.nodes():
        n = int(node)
        properties[n]['betweenness'] = betweenness[node]
        properties[n]['pagerank'] = pagerank[node]
        properties[n]['closeness'] = closeness[node]
    
    # 计算节点分组特征
    node_groups = {
        'early': [n for n in range(20)],         # 0-19
        'mid_early': [n for n in range(20, 40)], # 20-39
        'mid': [n for n in range(40, 60)],       # 40-59
        'mid_late': [n for n in range(60, 80)],  # 60-79
        'late': [n for n in range(80, 100)]      # 80-99
    }
    
    return properties, node_groups

def analyze_all_predictions(model, val_data, meta, device, num_samples=2000):
    """分析模型的所有预测（包括特殊token）"""
    block_size = meta['block_size']
    
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # 统计
    all_token_predictions = Counter()
    node_predictions = Counter()
    position_token_dist = defaultdict(Counter)
    position_node_dist = defaultdict(Counter)
    special_token_stats = {'[PAD]': 0, 'newline': 0}
    
    # 分析真实数据
    true_token_counts = Counter()
    true_node_counts = Counter()
    
    data_size = block_size + 1
    num_sequences = (len(val_data) - data_size) // data_size
    
    for i in range(min(num_samples, num_sequences)):
        idx = i * data_size
        x = torch.from_numpy(val_data[idx:idx+block_size].astype(np.int64)).unsqueeze(0).to(device)
        y = val_data[idx+1:idx+1+block_size]
        
        # 获取预测
        with ctx:
            logits, _ = model(x)
        
        if len(logits.shape) == 3:
            preds = torch.argmax(logits[0], dim=-1).cpu().numpy()
        else:
            # 单token预测
            pred_token = torch.argmax(logits[0]).item()
            all_token_predictions[pred_token] += 1
            if pred_token == 0:
                special_token_stats['[PAD]'] += 1
            elif pred_token == 1:
                special_token_stats['newline'] += 1
            elif 2 <= pred_token <= 101:
                node_predictions[pred_token - 2] += 1
            continue
        
        # 分析每个位置
        for pos in range(min(len(preds), len(y))):
            pred_token = int(preds[pos])
            true_token = int(y[pos])
            
            # 统计所有token
            all_token_predictions[pred_token] += 1
            position_token_dist[pos][pred_token] += 1
            
            # 统计真实token
            true_token_counts[true_token] += 1
            
            # 特殊token统计
            if pred_token == 0:
                special_token_stats['[PAD]'] += 1
            elif pred_token == 1:
                special_token_stats['newline'] += 1
            elif 2 <= pred_token <= 101:
                # 节点预测
                pred_node = pred_token - 2
                node_predictions[pred_node] += 1
                position_node_dist[pos][pred_node] += 1
            
            # 统计真实节点
            if 2 <= true_token <= 101:
                true_node = true_token - 2
                true_node_counts[true_node] += 1
    
    return {
        'all_tokens': all_token_predictions,
        'nodes': node_predictions,
        'position_tokens': position_token_dist,
        'position_nodes': position_node_dist,
        'special_tokens': special_token_stats,
        'true_tokens': true_token_counts,
        'true_nodes': true_node_counts
    }

def analyze_token_distribution(results_before, results_after):
    """分析token分布的变化"""
    print("\n" + "="*60)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # 1. 整体token分布
    print("\n1. Overall Token Distribution:")
    
    print("\nBefore collapse - Top 10 tokens:")
    total_before = sum(results_before['all_tokens'].values())
    for token, count in results_before['all_tokens'].most_common(10):
        pct = count / total_before * 100 if total_before > 0 else 0
        token_name = {0: '[PAD]', 1: 'newline'}.get(token, f'node_{token-2}' if 2 <= token <= 101 else f'token_{token}')
        print(f"  {token_name}: {count} ({pct:.1f}%)")
    
    print("\nAfter collapse - Top 10 tokens:")
    total_after = sum(results_after['all_tokens'].values())
    for token, count in results_after['all_tokens'].most_common(10):
        pct = count / total_after * 100 if total_after > 0 else 0
        token_name = {0: '[PAD]', 1: 'newline'}.get(token, f'node_{token-2}' if 2 <= token <= 101 else f'token_{token}')
        print(f"  {token_name}: {count} ({pct:.1f}%)")
    
    # 2. 特殊token统计
    print("\n2. Special Token Statistics:")
    print(f"\nBefore collapse:")
    print(f"  [PAD]: {results_before['special_tokens']['[PAD]']}")
    print(f"  newline: {results_before['special_tokens']['newline']}")
    print(f"  Total nodes: {sum(results_before['nodes'].values())}")
    
    print(f"\nAfter collapse:")
    print(f"  [PAD]: {results_after['special_tokens']['[PAD]']}")
    print(f"  newline: {results_after['special_tokens']['newline']}")
    print(f"  Total nodes: {sum(results_after['nodes'].values())}")
    
    # 3. 节点预测（如果有）
    if results_before['nodes'] or results_after['nodes']:
        print("\n3. Node Predictions:")
        
        if results_before['nodes']:
            print("\nBefore collapse - Top nodes:")
            for node, count in results_before['nodes'].most_common(10):
                print(f"  Node {node}: {count}")
        else:
            print("\nBefore collapse: No node predictions!")
        
        if results_after['nodes']:
            print("\nAfter collapse - Top nodes:")
            for node, count in results_after['nodes'].most_common(10):
                print(f"  Node {node}: {count}")
        else:
            print("\nAfter collapse: No node predictions!")
    
    return total_before > 0 and total_after > 0

def visualize_token_analysis(results_before, results_after, save_path):
    """可视化token分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Token类型分布饼图
    ax = axes[0, 0]
    
    # 统计token类型
    types_before = {
        '[PAD]': results_before['special_tokens']['[PAD]'],
        'newline': results_before['special_tokens']['newline'],
        'nodes': sum(results_before['nodes'].values())
    }
    
    types_after = {
        '[PAD]': results_after['special_tokens']['[PAD]'],
        'newline': results_after['special_tokens']['newline'],
        'nodes': sum(results_after['nodes'].values())
    }
    
    # 创建子图
    ax1 = plt.subplot(221)
    if sum(types_before.values()) > 0:
        ax1.pie(types_before.values(), labels=types_before.keys(), autopct='%1.1f%%')
        ax1.set_title('Before Collapse')
    
    ax2 = plt.subplot(222)
    if sum(types_after.values()) > 0:
        ax2.pie(types_after.values(), labels=types_after.keys(), autopct='%1.1f%%')
        ax2.set_title('After Collapse')
    
    # 2. 位置级别的token分布
    ax = axes[1, 0]
    
    positions = range(min(20, len(results_before['position_tokens'])))
    pad_before = []
    newline_before = []
    node_before = []
    pad_after = []
    newline_after = []
    node_after = []
    
    for pos in positions:
        total_before = sum(results_before['position_tokens'][pos].values())
        total_after = sum(results_after['position_tokens'][pos].values())
        
        if total_before > 0:
            pad_before.append(results_before['position_tokens'][pos].get(0, 0) / total_before)
            newline_before.append(results_before['position_tokens'][pos].get(1, 0) / total_before)
            node_count = sum(count for token, count in results_before['position_tokens'][pos].items() if 2 <= token <= 101)
            node_before.append(node_count / total_before)
        else:
            pad_before.append(0)
            newline_before.append(0)
            node_before.append(0)
        
        if total_after > 0:
            pad_after.append(results_after['position_tokens'][pos].get(0, 0) / total_after)
            newline_after.append(results_after['position_tokens'][pos].get(1, 0) / total_after)
            node_count = sum(count for token, count in results_after['position_tokens'][pos].items() if 2 <= token <= 101)
            node_after.append(node_count / total_after)
        else:
            pad_after.append(0)
            newline_after.append(0)
            node_after.append(0)
    
    ax.plot(positions, pad_before, 'b-', label='[PAD] before', linewidth=2)
    ax.plot(positions, newline_before, 'g-', label='newline before', linewidth=2)
    ax.plot(positions, node_before, 'r-', label='nodes before', linewidth=2)
    ax.plot(positions, pad_after, 'b--', label='[PAD] after', linewidth=2)
    ax.plot(positions, newline_after, 'g--', label='newline after', linewidth=2)
    ax.plot(positions, node_after, 'r--', label='nodes after', linewidth=2)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Proportion')
    ax.set_title('Token Type Distribution by Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 真实vs预测的token分布对比
    ax = axes[1, 1]
    
    # 获取最常见的token
    true_dist = results_before['true_tokens']
    pred_before = results_before['all_tokens']
    pred_after = results_after['all_tokens']
    
    common_tokens = list(set(list(true_dist.keys())[:20]) | 
                        set(list(pred_before.keys())[:20]) | 
                        set(list(pred_after.keys())[:20]))[:10]
    
    if common_tokens:
        token_labels = []
        true_counts = []
        pred_before_counts = []
        pred_after_counts = []
        
        for token in common_tokens[:5]:  # 只显示前5个
            if token == 0:
                token_labels.append('[PAD]')
            elif token == 1:
                token_labels.append('\\n')
            else:
                token_labels.append(f'n{token-2}')
            
            true_counts.append(true_dist.get(token, 0))
            pred_before_counts.append(pred_before.get(token, 0))
            pred_after_counts.append(pred_after.get(token, 0))
        
        x = np.arange(len(token_labels))
        width = 0.25
        
        ax.bar(x - width, true_counts, width, label='True', alpha=0.8)
        ax.bar(x, pred_before_counts, width, label='Before', alpha=0.8)
        ax.bar(x + width, pred_after_counts, width, label='After', alpha=0.8)
        
        ax.set_xlabel('Token')
        ax.set_ylabel('Count')
        ax.set_title('True vs Predicted Token Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(token_labels)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430'
    data_dir = 'data/simple_graph/100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading data and models...")
    
    # 加载元数据
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # 加载图
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    G = nx.read_graphml(graph_path)
    
    print("Analyzing graph topology...")
    graph_props, node_groups = analyze_graph_topology(G)
    
    # 加载验证数据
    val_data_path = os.path.join(data_dir, 'val.bin')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # 加载模型
    print("\nLoading model checkpoints...")
    ckpt_before = os.path.join(base_dir, 'ckpt_100000.pt')
    ckpt_after = os.path.join(base_dir, 'ckpt_200000.pt')
    
    model_before, _ = load_checkpoint_and_model(ckpt_before, device)
    model_after, _ = load_checkpoint_and_model(ckpt_after, device)
    
    # 分析所有预测
    print("\nAnalyzing predictions before collapse...")
    results_before = analyze_all_predictions(model_before, val_data, meta, device)
    
    print("Analyzing predictions after collapse...")
    results_after = analyze_all_predictions(model_after, val_data, meta, device)
    
    # 分析token分布
    has_data = analyze_token_distribution(results_before, results_after)
    
    # 可视化
    if has_data:
        save_path = os.path.join(base_dir, 'token_distribution_analysis.png')
        visualize_token_analysis(results_before, results_after, save_path)
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # 诊断问题
    if sum(results_before['nodes'].values()) == 0 and sum(results_after['nodes'].values()) == 0:
        print("\n⚠️ CRITICAL FINDING: Model NEVER predicts any nodes!")
        print("\nBefore collapse: Only predicts [PAD]")
        print("After collapse: Only predicts newline")
        print("\nThis is even more extreme than expected:")
        print("- The model has completely given up on the actual task")
        print("- It only outputs formatting tokens, never actual path nodes")
        print("\nThis suggests a catastrophic failure mode where the model")
        print("finds it 'safer' to predict special tokens than risk being wrong on nodes.")
    else:
        print("\n✅ NODE STRATEGY ANALYSIS COMPLETE")
        if sum(results_before['nodes'].values()) > sum(results_after['nodes'].values()):
            print("Model predictions shifted dramatically after collapse")
    
    print("\n🔍 Next steps:")
    print("1. Examine training logs to see when this behavior emerged")
    print("2. Check if reward/loss signals encouraged this degenerate behavior")
    print("3. Your entropy regularization likely prevents this collapse")

if __name__ == "__main__":
    main()