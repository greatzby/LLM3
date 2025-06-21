"""
参数饱和度分析 - 证明第一个箭头的前提
测量: weight gap, gradient响应性, embedding similarity
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
import networkx as nx
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

def analyze_weight_gap(model, graph_path):
    """分析edge vs non-edge权重差异"""
    # 加载图
    graph = nx.read_graphml(graph_path)
    edges = set(graph.edges())
    
    # 获取embedding权重
    wte = model.transformer.wte.weight.data.cpu().numpy()
    num_nodes = 100  # 假设100个节点
    
    edge_weights = []
    non_edge_weights = []
    
    # 计算所有节点对的embedding相似度
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # 将节点id转换为token id（节点0-99对应token 2-101）
            token_i = i + 2
            token_j = j + 2
            
            if token_i < wte.shape[0] and token_j < wte.shape[0]:
                similarity = np.dot(wte[token_i], wte[token_j])
                
                if (str(i), str(j)) in edges or (str(j), str(i)) in edges:
                    edge_weights.append(similarity)
                else:
                    non_edge_weights.append(similarity)
    
    return {
        'edge_mean': float(np.mean(edge_weights)) if edge_weights else 0.0,
        'non_edge_mean': float(np.mean(non_edge_weights)) if non_edge_weights else 0.0,
        'weight_gap': float(np.mean(edge_weights) if edge_weights else 0) - float(np.mean(non_edge_weights) if non_edge_weights else 0),
        'edge_std': float(np.std(edge_weights)) if edge_weights else 0.0,
        'non_edge_std': float(np.std(non_edge_weights)) if non_edge_weights else 0.0
    }

def analyze_gradient_responsiveness(model, test_loader, device, num_samples=100):
    """测试不同节点的梯度响应性"""
    model.train()  # 需要梯度
    
    # 记录每个token的梯度大小
    token_gradients = {}
    
    for idx, (inputs, targets) in enumerate(test_loader):
        if idx >= num_samples:
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        model.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        
        # 记录embedding层的梯度
        if model.transformer.wte.weight.grad is not None:
            grad_norms = model.transformer.wte.weight.grad.norm(dim=1).cpu().numpy()
            
            # 累积每个token的梯度
            for i, grad_norm in enumerate(grad_norms):
                if i not in token_gradients:
                    token_gradients[i] = []
                token_gradients[i].append(float(grad_norm))
    
    # 计算平均梯度响应
    avg_gradients = {}
    for token_id, grads in token_gradients.items():
        avg_gradients[token_id] = float(np.mean(grads))
    
    # 分析常见vs罕见token的梯度差异
    # 假设token 2-11是最常见的路径节点（节点0-9）
    common_tokens = range(2, 12)  # token id
    rare_tokens = range(52, 62)   # token id (节点50-59)
    
    common_grad = float(np.mean([avg_gradients.get(t, 0) for t in common_tokens]))
    rare_grad = float(np.mean([avg_gradients.get(t, 0) for t in rare_tokens]))
    
    model.eval()  # 恢复eval模式
    
    return {
        'common_token_grad': common_grad,
        'rare_token_grad': rare_grad,
        'grad_ratio': float(rare_grad / (common_grad + 1e-8))
    }

def analyze_embedding_similarity(model):
    """分析embedding空间的聚类程度"""
    wte = model.transformer.wte.weight.data.cpu().numpy()
    
    # 只分析节点token（2-101）
    node_embeddings = wte[2:102]
    
    # 计算平均相似度
    similarities = []
    for i in range(len(node_embeddings)):
        for j in range(i+1, len(node_embeddings)):
            sim = np.dot(node_embeddings[i], node_embeddings[j]) / (
                np.linalg.norm(node_embeddings[i]) * np.linalg.norm(node_embeddings[j]) + 1e-8
            )
            similarities.append(float(sim))
    
    # 计算embedding norms
    norms = [float(np.linalg.norm(emb)) for emb in node_embeddings]
    
    return {
        'mean_similarity': float(np.mean(similarities)) if similarities else 0.0,
        'std_similarity': float(np.std(similarities)) if similarities else 0.0,
        'mean_norm': float(np.mean(norms)) if norms else 0.0,
        'std_norm': float(np.std(norms)) if norms else 0.0,
        'norm_variance': float(np.var(norms)) if norms else 0.0
    }

def prepare_test_loader(data_path, batch_size=64):
    """准备测试数据加载器 - 与训练代码一致"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # 加载验证数据
    val_data = np.memmap(os.path.join(data_path, 'val.bin'), dtype=np.uint16, mode='r')
    
    # 获取block_size
    meta = load_meta(data_path)
    block_size = meta['block_size']
    
    # 创建数据集 - 与训练代码中get_batch逻辑一致
    data_size = block_size + 1
    num_samples = (len(val_data) - data_size) // data_size
    
    # 创建所有可能的索引
    all_indices = np.arange(num_samples) * data_size
    
    # 采样一部分用于测试
    sample_size = min(1000, num_samples)
    sampled_indices = np.random.choice(all_indices, size=sample_size, replace=False)
    
    inputs = []
    targets = []
    for idx in sampled_indices:
        x = val_data[idx:idx+block_size].astype(np.int64)
        y = val_data[idx+1:idx+1+block_size].astype(np.int64)
        inputs.append(x)
        targets.append(y)
    
    inputs = torch.from_numpy(np.stack(inputs))
    targets = torch.from_numpy(np.stack(targets))
    
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def main():
    # 配置
    checkpoints = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    base_dir = 'out/simple_graph_1_1_120_100_original_seed42'
    graph_path = 'data/simple_graph/100/path_graph.graphml'
    data_path = 'data/simple_graph/100'
    output_dir = 'analysis_results/parameter_saturation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备测试数据
    test_loader = prepare_test_loader(data_path)
    
    results = {}
    
    for ckpt in tqdm(checkpoints, desc="Analyzing checkpoints"):
        ckpt_path = os.path.join(base_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found, skipping...")
            continue
        
        try:
            model = load_model(ckpt_path)
            
            # 分析weight gap
            weight_analysis = analyze_weight_gap(model, graph_path)
            
            # 分析梯度响应性
            grad_analysis = analyze_gradient_responsiveness(model, test_loader, 'cuda:0')
            
            # 分析embedding相似度
            embedding_analysis = analyze_embedding_similarity(model)
            
            results[ckpt] = {
                'weight_gap': weight_analysis,
                'gradient_response': grad_analysis,
                'embedding': embedding_analysis
            }
            
            # 打印当前结果
            print(f"\nCheckpoint {ckpt}:")
            print(f"  Weight gap: {weight_analysis['weight_gap']:.6f}")
            print(f"  Embedding similarity: {embedding_analysis['mean_similarity']:.6f}")
            print(f"  Gradient ratio: {grad_analysis['grad_ratio']:.2f}")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果（转换为可序列化格式）
    serializable_results = convert_to_serializable(results)
    with open(os.path.join(output_dir, 'parameter_saturation_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 创建可视化
    create_parameter_plots(results, output_dir)
    
    print(f"\nParameter saturation analysis complete! Results saved to {output_dir}/")

def create_parameter_plots(results, output_dir):
    """创建参数分析图表"""
    if not results:
        print("No results to plot")
        return
        
    checkpoints = sorted(results.keys())
    
    # 1. Weight Gap Evolution
    plt.figure(figsize=(10, 6))
    weight_gaps = [results[ckpt]['weight_gap']['weight_gap'] for ckpt in checkpoints]
    plt.plot(checkpoints, weight_gaps, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Training Iteration')
    plt.ylabel('Weight Gap (Edge - Non-edge)')
    plt.title('Weight Gap Evolution: Approaching Zero')
    plt.grid(True, alpha=0.3)
    
    # 标记相变点
    if 140000 in checkpoints:
        plt.axvline(x=140000, color='red', linestyle='--', label='Phase Transition')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_gap_evolution.png'), dpi=150)
    plt.close()
    
    # 2. Gradient Response Ratio
    plt.figure(figsize=(10, 6))
    grad_ratios = [results[ckpt]['gradient_response']['grad_ratio'] for ckpt in checkpoints]
    plt.plot(checkpoints, grad_ratios, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Training Iteration')
    plt.ylabel('Gradient Ratio (Rare/Common)')
    plt.title('Gradient Response Asymmetry')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_response_ratio.png'), dpi=150)
    plt.close()
    
    # 3. Embedding Similarity
    plt.figure(figsize=(10, 6))
    similarities = [results[ckpt]['embedding']['mean_similarity'] for ckpt in checkpoints]
    plt.plot(checkpoints, similarities, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Training Iteration')
    plt.ylabel('Mean Embedding Similarity')
    plt.title('Embedding Space Clustering')
    plt.grid(True, alpha=0.3)
    
    # 标记相变点
    if 140000 in checkpoints:
        plt.axvline(x=140000, color='red', linestyle='--', label='Phase Transition')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_similarity_evolution.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()