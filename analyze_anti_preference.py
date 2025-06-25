"""
分析不同训练阶段模型的输出分布，验证反偏好现象
"""
import os
import torch
import numpy as np
import pickle
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
import torch.nn.functional as F

def load_checkpoint_and_model(checkpoint_path, device='cuda'):
    """加载checkpoint和模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从checkpoint获取模型配置
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def load_graph_and_metadata(data_dir):
    """加载图结构和元数据"""
    # 加载图
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    G = nx.read_graphml(graph_path)
    
    # 加载元数据
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    return G, meta

def extract_training_edges(train_file_path):
    """从训练文件中提取所有出现过的边"""
    training_edges = set()
    
    with open(train_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line != 'x':
                tokens = line.split()
                # 解析路径：source target path...
                path = tokens[2:]  # 跳过source和target
                
                # 提取路径中的所有边
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    training_edges.add(edge)
    
    return training_edges

def analyze_predictions(model, test_data, G, training_edges, stoi, itos, device='cuda', num_samples=1000):
    """分析模型在测试数据上的预测分布"""
    
    results = {
        'train_path_prob': [],
        'valid_non_train_prob': [],
        'invalid_prob': [],
        'predictions': []
    }
    
    # 读取测试数据
    with open(test_data, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    # 随机采样
    sampled_lines = np.random.choice(test_lines, min(num_samples, len(test_lines)), replace=False)
    
    for line in sampled_lines:
        tokens = line.split()
        source, target = tokens[0], tokens[1]
        path = tokens[2:]
        
        # 对路径中的每个位置进行预测（除了最后一个）
        for i in range(len(path) - 1):
            current_node = path[i]
            true_next = path[i + 1]
            
            # 构建输入序列：source target path_so_far
            input_sequence = [source, target] + path[:i+1]
            input_ids = [stoi[token] for token in input_sequence]
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # 获取模型预测
            with torch.no_grad():
                logits, _ = model(input_tensor)
                # 获取最后一个位置的logits
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
            
            # 分析预测分布
            train_prob = 0.0
            valid_non_train_prob = 0.0
            invalid_prob = 0.0
            
            # 获取当前节点的所有可能的下一跳
            current_node_neighbors = list(G.successors(current_node))
            
            # 遍历所有可能的预测
            for next_node_idx in range(2, len(itos)):  # 从2开始，跳过PAD和\n
                if next_node_idx >= len(probs):
                    break
                    
                next_node = itos[next_node_idx]
                prob = probs[next_node_idx].item()
                
                edge = (current_node, next_node)
                
                # 分类
                if edge in training_edges:
                    train_prob += prob
                elif next_node in current_node_neighbors:
                    valid_non_train_prob += prob
                else:
                    invalid_prob += prob
            
            results['train_path_prob'].append(train_prob)
            results['valid_non_train_prob'].append(valid_non_train_prob)
            results['invalid_prob'].append(invalid_prob)
            
            # 记录预测细节
            top_k = 5
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            predictions = []
            for idx, prob in zip(top_indices, top_probs):
                if idx >= 2 and idx < len(itos):
                    pred_node = itos[idx.item()]
                    edge = (current_node, pred_node)
                    edge_type = 'train' if edge in training_edges else \
                               'valid_non_train' if pred_node in current_node_neighbors else \
                               'invalid'
                    predictions.append({
                        'node': pred_node,
                        'prob': prob.item(),
                        'type': edge_type
                    })
            
            results['predictions'].append({
                'current': current_node,
                'true_next': true_next,
                'top_predictions': predictions
            })
    
    return results

def plot_distribution_comparison(results_dict, save_path):
    """绘制不同checkpoint的分布对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Output Distribution Analysis: Stable vs Collapsed States', fontsize=16)
    
    checkpoints = list(results_dict.keys())
    
    for idx, (checkpoint, results) in enumerate(results_dict.items()):
        ax = axes[idx // 2, idx % 2]
        
        # 计算平均概率
        avg_train = np.mean(results['train_path_prob'])
        avg_valid_non_train = np.mean(results['valid_non_train_prob'])
        avg_invalid = np.mean(results['invalid_prob'])
        
        # 绘制饼图
        labels = ['Training Path', 'Valid Non-Training', 'Invalid']
        sizes = [avg_train, avg_valid_non_train, avg_invalid]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        
        ax.set_title(f'Checkpoint {checkpoint}\n'
                    f'Avg probs: Train={avg_train:.3f}, Valid-NonTrain={avg_valid_non_train:.3f}, Invalid={avg_invalid:.3f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 绘制概率分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Probability Distribution Histograms', fontsize=16)
    
    for idx, (checkpoint, results) in enumerate(results_dict.items()):
        ax = axes[idx // 2, idx % 2]
        
        # 绘制直方图
        bins = np.linspace(0, 1, 21)
        ax.hist(results['train_path_prob'], bins=bins, alpha=0.6, label='Training Path', color='#2ecc71')
        ax.hist(results['valid_non_train_prob'], bins=bins, alpha=0.6, label='Valid Non-Training', color='#3498db')
        ax.hist(results['invalid_prob'], bins=bins, alpha=0.6, label='Invalid', color='#e74c3c')
        
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'Checkpoint {checkpoint}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_hist.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430'
    data_dir = 'data/simple_graph/100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 要分析的checkpoints
    checkpoints_to_analyze = {
        '50k': 50000,
        '100k': 100000,
        '190k': 190000,
        '200k': 200000
    }
    
    # 加载图和元数据
    print("Loading graph and metadata...")
    G, meta = load_graph_and_metadata(data_dir)
    stoi, itos = meta['stoi'], meta['itos']
    
    # 提取训练集中的所有边
    print("Extracting training edges...")
    train_file = os.path.join(data_dir, 'train_20.txt')
    training_edges = extract_training_edges(train_file)
    print(f"Found {len(training_edges)} unique edges in training data")
    
    # 统计图中的总边数
    total_edges = G.number_of_edges()
    print(f"Total edges in graph: {total_edges}")
    print(f"Training coverage: {len(training_edges)/total_edges*100:.1f}%")
    
    # 测试文件
    test_file = os.path.join(data_dir, 'test.txt')
    
    # 分析每个checkpoint
    results_dict = {}
    
    for name, iteration in checkpoints_to_analyze.items():
        print(f"\nAnalyzing checkpoint {name} (iteration {iteration})...")
        
        # 加载模型
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
        
        # 分析预测
        results = analyze_predictions(model, test_file, G, training_edges, stoi, itos, device)
        results_dict[name] = results
        
        # 打印统计
        avg_train = np.mean(results['train_path_prob'])
        avg_valid_non_train = np.mean(results['valid_non_train_prob'])
        avg_invalid = np.mean(results['invalid_prob'])
        
        print(f"Average probabilities:")
        print(f"  Training path nodes: {avg_train:.3f}")
        print(f"  Valid non-training nodes: {avg_valid_non_train:.3f}")
        print(f"  Invalid nodes: {avg_invalid:.3f}")
        
        # 打印一些具体的预测示例
        print(f"\nExample predictions:")
        for i in range(min(3, len(results['predictions']))):
            pred = results['predictions'][i]
            print(f"  Current: {pred['current']}, True next: {pred['true_next']}")
            for j, top_pred in enumerate(pred['top_predictions'][:3]):
                print(f"    Top-{j+1}: {top_pred['node']} (p={top_pred['prob']:.3f}, type={top_pred['type']})")
    
    # 绘制对比图
    print("\nGenerating visualization...")
    save_path = os.path.join(base_dir, 'anti_preference_analysis.png')
    plot_distribution_comparison(results_dict, save_path)
    
    # 保存详细结果
    results_save_path = os.path.join(base_dir, 'anti_preference_results.pkl')
    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  Plots: {save_path}")
    print(f"  Data: {results_save_path}")
    
    # 生成总结报告
    print("\n" + "="*60)
    print("ANTI-PREFERENCE ANALYSIS SUMMARY")
    print("="*60)
    
    stable_avg_train = np.mean([results_dict['50k']['train_path_prob'] + results_dict['100k']['train_path_prob']]) / 2
    collapsed_avg_train = np.mean([results_dict['190k']['train_path_prob'] + results_dict['200k']['train_path_prob']]) / 2
    
    stable_avg_valid = np.mean([results_dict['50k']['valid_non_train_prob'] + results_dict['100k']['valid_non_train_prob']]) / 2
    collapsed_avg_valid = np.mean([results_dict['190k']['valid_non_train_prob'] + results_dict['200k']['valid_non_train_prob']]) / 2
    
    print(f"\nStable Phase (50k-100k):")
    print(f"  Training path preference: {stable_avg_train:.1%}")
    print(f"  Valid non-training paths: {stable_avg_valid:.1%}")
    
    print(f"\nCollapsed Phase (190k-200k):")
    print(f"  Training path preference: {collapsed_avg_train:.1%}")
    print(f"  Valid non-training paths: {collapsed_avg_valid:.1%}")
    
    print(f"\nChange:")
    print(f"  Training path: {stable_avg_train:.1%} → {collapsed_avg_train:.1%} (Δ = {collapsed_avg_train - stable_avg_train:.1%})")
    print(f"  Valid non-training: {stable_avg_valid:.1%} → {collapsed_avg_valid:.1%} (Δ = {collapsed_avg_valid - stable_avg_valid:.1%})")
    
    if collapsed_avg_train < stable_avg_train and collapsed_avg_valid > stable_avg_valid:
        print("\n✓ ANTI-PREFERENCE CONFIRMED: Model actively avoids training paths!")
    
    print("="*60)

if __name__ == "__main__":
    main()