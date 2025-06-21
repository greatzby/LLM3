"""
图结构理解探测 - 探测模型是否真正理解了图结构
这是对教授"diagnostic rather than descriptive"要求的回应
"""

import os
import torch
import numpy as np
import networkx as nx
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
from model import GPTConfig, GPT

def convert_to_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def load_meta(data_path):
    """加载meta信息"""
    with open(os.path.join(data_path, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)

def load_model(checkpoint_path, device='cuda:0'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def probe_edge_prediction(model, graph, num_samples=1000):
    """测试模型是否知道哪些节点相连"""
    model.eval()
    device = next(model.parameters()).device
    
    edges = set(graph.edges())
    nodes = list(graph.nodes())
    
    predictions = []
    labels = []
    
    for _ in range(num_samples):
        # 随机选择两个节点
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        
        # 构造prompt: "node1 node2 node1"
        # 节点id转换为token id
        prompt = torch.tensor([int(node1)+2, int(node2)+2, int(node1)+2], device=device).unsqueeze(0)
        
        # 获取下一个token的logits
        with torch.no_grad():
            logits, _ = model(prompt)
            next_token_logits = logits[0, -1, :]
            
            # 检查node2的logit
            node2_logit = next_token_logits[int(node2)+2].item()
            predictions.append(float(node2_logit))
            
            # 真实标签
            is_edge = (node1, node2) in edges or (node2, node1) in edges
            labels.append(int(is_edge))
    
    # 计算AUC
    if len(set(labels)) > 1:
        auc = float(roc_auc_score(labels, predictions))
    else:
        auc = 0.5
    
    # 计算准确率（使用中位数作为阈值）
    threshold = np.median(predictions)
    binary_preds = [1 if p > threshold else 0 for p in predictions]
    accuracy = float(accuracy_score(labels, binary_preds))
    
    return {
        'edge_prediction_auc': auc,
        'edge_prediction_accuracy': accuracy
    }

def probe_path_validity_understanding(model, graph, num_samples=500):
    """测试模型是否理解路径的有效性"""
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    
    for _ in range(num_samples):
        # 生成一个部分路径
        path = generate_partial_path(graph, length=4)
        
        if len(path) < 4:
            continue
        
        # 获取所有可能的下一步
        current_node = path[-1]
        valid_next_nodes = list(graph.neighbors(current_node))
        all_nodes = list(graph.nodes())
        invalid_nodes = [n for n in all_nodes if n not in valid_next_nodes and n != current_node]
        
        if not valid_next_nodes or not invalid_nodes:
            continue
        
        # 构造prompt
        prompt_tokens = []
        for node in path:
            prompt_tokens.append(int(node) + 2)  # 节点id转token id
        prompt = torch.tensor(prompt_tokens, device=device).unsqueeze(0)
        
        # 获取logits
        with torch.no_grad():
            logits, _ = model(prompt)
            next_token_logits = logits[0, -1, :]
        
        # 比较valid vs invalid节点的平均logit
        valid_logits = [float(next_token_logits[int(n)+2].item()) for n in valid_next_nodes]
        invalid_sample = np.random.choice(invalid_nodes, min(10, len(invalid_nodes)), replace=False)
        invalid_logits = [float(next_token_logits[int(n)+2].item()) for n in invalid_sample]
        
        avg_valid = float(np.mean(valid_logits))
        avg_invalid = float(np.mean(invalid_logits))
        
        results.append({
            'valid_score': avg_valid,
            'invalid_score': avg_invalid,
            'discrimination': avg_valid - avg_invalid
        })
    
    # 统计结果
    if results:
        discriminations = [r['discrimination'] for r in results]
        
        return {
            'path_validity_discrimination': float(np.mean(discriminations)),
            'discrimination_std': float(np.std(discriminations)),
            'positive_discrimination_rate': float(sum(1 for d in discriminations if d > 0) / len(discriminations))
        }
    else:
        return {
            'path_validity_discrimination': 0.0,
            'discrimination_std': 0.0,
            'positive_discrimination_rate': 0.5
        }

def probe_shortest_path_preference(model, graph, num_samples=200):
    """测试模型是否偏好最短路径"""
    model.eval()
    device = next(model.parameters()).device
    
    preferences = []
    
    for _ in range(num_samples):
        # 随机选择起点和终点
        nodes = list(graph.nodes())
        source, target = np.random.choice(nodes, 2, replace=False)
        
        try:
            # 找到所有路径（限制长度避免计算爆炸）
            all_paths = list(nx.all_simple_paths(graph, source, target, cutoff=10))
            if len(all_paths) < 2:
                continue
            
            # 找到最短路径
            shortest_length = min(len(p) for p in all_paths)
            shortest_paths = [p for p in all_paths if len(p) == shortest_length]
            longer_paths = [p for p in all_paths if len(p) > shortest_length]
            
            if not longer_paths:
                continue
            
            # 构造prompt
            prompt = torch.tensor([int(source)+2, int(target)+2, int(source)+2], device=device).unsqueeze(0)
            
            # 获取第一步的概率分布
            with torch.no_grad():
                logits, _ = model(prompt)
                probs = torch.softmax(logits[0, -1, :], dim=-1)
            
            # 比较最短路径第一步vs较长路径第一步的概率
            shortest_first_steps = set(p[1] for p in shortest_paths if len(p) > 1)
            longer_first_steps = set(p[1] for p in longer_paths if len(p) > 1) - shortest_first_steps
            
            if shortest_first_steps and longer_first_steps:
                shortest_prob = float(sum(probs[int(n)+2].item() for n in shortest_first_steps))
                longer_prob = float(sum(probs[int(n)+2].item() for n in longer_first_steps))
                
                # 归一化
                total = shortest_prob + longer_prob
                if total > 0:
                    preferences.append(shortest_prob / total)
        
        except nx.NetworkXNoPath:
            continue
        except Exception as e:
            print(f"Error in shortest path preference: {e}")
            continue
    
    return {
        'shortest_path_preference': float(np.mean(preferences)) if preferences else 0.5,
        'preference_std': float(np.std(preferences)) if preferences else 0.0,
        'num_samples_analyzed': len(preferences)
    }

def generate_partial_path(graph, length=4):
    """生成一个部分路径用于测试"""
    nodes = list(graph.nodes())
    start = np.random.choice(nodes)
    path = [start]
    
    current = start
    for _ in range(length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        next_node = np.random.choice(neighbors)
        path.append(next_node)
        current = next_node
    
    return path

def probe_multi_hop_reasoning(model, graph, num_samples=500):
    """测试模型的多跳推理能力"""
    model.eval()
    device = next(model.parameters()).device
    
    hop_accuracies = {1: [], 2: [], 3: []}
    
    for _ in range(num_samples):
        # 随机选择起点
        nodes = list(graph.nodes())
        start = np.random.choice(nodes)
        
        # 测试不同跳数的推理
        for hops in [1, 2, 3]:
            # 找到所有k跳可达的节点
            reachable = set()
            current = {start}
            
            for _ in range(hops):
                next_nodes = set()
                for node in current:
                    next_nodes.update(graph.neighbors(node))
                current = next_nodes
                reachable.update(current)
            
            if not reachable:
                continue
            
            # 随机选择一个目标节点
            target = np.random.choice(list(reachable))
            
            # 构造prompt
            prompt = torch.tensor([int(start)+2, int(target)+2, int(start)+2], device=device).unsqueeze(0)
            
            # 获取模型预测
            with torch.no_grad():
                logits, _ = model(prompt)
                pred = torch.argmax(logits[0, -1, :]).item()
            
            # 检查预测是否是有效的第一步
            if pred >= 2 and pred <= 101:
                pred_node = pred - 2
                # 检查是否存在通过这个节点到达目标的路径
                try:
                    path = nx.shortest_path(graph, str(pred_node), str(target))
                    if len(path) <= hops:
                        hop_accuracies[hops].append(1)
                    else:
                        hop_accuracies[hops].append(0)
                except:
                    hop_accuracies[hops].append(0)
            else:
                hop_accuracies[hops].append(0)
    
    return {
        f'{k}_hop_accuracy': float(np.mean(v)) if v else 0.0
        for k, v in hop_accuracies.items()
    }

def main():
    # 配置
    checkpoints = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    base_dir = 'out/simple_graph_1_1_120_100_original_seed42'
    graph_path = 'data/simple_graph/100/path_graph.graphml'
    data_path = 'data/simple_graph/100'
    output_dir = 'analysis_results/graph_understanding'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图
    graph = nx.read_graphml(graph_path)
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    all_results = {}
    
    for ckpt in tqdm(checkpoints, desc="Probing checkpoints"):
        ckpt_path = os.path.join(base_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found, skipping...")
            continue
        
        try:
            model = load_model(ckpt_path)
            
            # 执行各种探测
            print(f"\nProbing checkpoint {ckpt}...")
            edge_results = probe_edge_prediction(model, graph)
            validity_results = probe_path_validity_understanding(model, graph)
            shortest_results = probe_shortest_path_preference(model, graph)
            multihop_results = probe_multi_hop_reasoning(model, graph)
            
            all_results[ckpt] = {
                **edge_results,
                **validity_results,
                **shortest_results,
                **multihop_results
            }
            
            # 打印当前结果
            print(f"  Edge prediction AUC: {edge_results['edge_prediction_auc']:.3f}")
            print(f"  Path validity discrimination: {validity_results['path_validity_discrimination']:.3f}")
            print(f"  Shortest path preference: {shortest_results['shortest_path_preference']:.3f}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error probing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果（转换为可序列化格式）
    serializable_results = convert_to_serializable(all_results)
    with open(os.path.join(output_dir, 'graph_understanding_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 创建可视化
    create_understanding_plots(all_results, output_dir)
    
    print(f"\nGraph understanding probe complete! Results saved to {output_dir}/")

def create_understanding_plots(results, output_dir):
    """创建图理解能力的可视化"""
    if not results:
        print("No results to plot")
        return
        
    checkpoints = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Edge prediction AUC
    ax = axes[0, 0]
    aucs = [results[ckpt]['edge_prediction_auc'] for ckpt in checkpoints]
    ax.plot(checkpoints, aucs, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('AUC')
    ax.set_title('Edge Prediction Capability')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    # 2. Path validity discrimination
    ax = axes[0, 1]
    discriminations = [results[ckpt]['path_validity_discrimination'] for ckpt in checkpoints]
    ax.plot(checkpoints, discriminations, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Valid - Invalid Score')
    ax.set_title('Path Validity Understanding')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. Shortest path preference
    ax = axes[1, 0]
    preferences = [results[ckpt]['shortest_path_preference'] for ckpt in checkpoints]
    ax.plot(checkpoints, preferences, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Preference Score')
    ax.set_title('Shortest Path Preference')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='No Preference')
    ax.set_ylim(0.3, 0.8)
    
    # 4. Multi-hop reasoning
    ax = axes[1, 1]
    if '1_hop_accuracy' in results[checkpoints[0]]:
        hop1 = [results[ckpt].get('1_hop_accuracy', 0) for ckpt in checkpoints]
        hop2 = [results[ckpt].get('2_hop_accuracy', 0) for ckpt in checkpoints]
        hop3 = [results[ckpt].get('3_hop_accuracy', 0) for ckpt in checkpoints]
        
        ax.plot(checkpoints, hop1, marker='o', label='1-hop', linewidth=2)
        ax.plot(checkpoints, hop2, marker='s', label='2-hop', linewidth=2)
        ax.plot(checkpoints, hop3, marker='^', label='3-hop', linewidth=2)
        ax.legend()
    else:
        # 如果没有多跳数据，显示综合得分
        combined_scores = []
        for ckpt in checkpoints:
            score = (
                results[ckpt]['edge_prediction_auc'] * 0.3 +
                min(results[ckpt]['path_validity_discrimination'] / 10, 1.0) * 0.4 +
                results[ckpt]['shortest_path_preference'] * 0.3
            )
            combined_scores.append(score)
        
        ax.plot(checkpoints, combined_scores, marker='o', linewidth=2, markersize=8, color='green')
        
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Multi-hop Reasoning / Overall Understanding')
    ax.grid(True, alpha=0.3)
    
    # 标记相变点
    for ax in axes.flat:
        if 140000 in checkpoints:
            ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_understanding_evolution.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()