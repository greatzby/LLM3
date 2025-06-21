"""
Diversity机制完整分析脚本
分析diversity训练如何防止相变
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

# 导入模型相关
from model import GPTConfig, GPT

# 路径配置
DIVERSITY_PATH = "out/spurious_rewards/diversity_alpha0.5_div0.1_seed42_20250619_003757"
STANDARD_PATH = "out/simple_graph_1_1_120_100_original_seed42"
OUTPUT_DIR = "diversity_analysis_results"

# Diversity的checkpoint
DIVERSITY_CHECKPOINTS = [50000, 100000, 150000, 200000]

# Standard的关键checkpoint（如果存在）
STANDARD_CHECKPOINTS = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]

def ensure_output_dir():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # 加载权重
    state_dict = checkpoint['model']
    # 处理可能的前缀问题
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, checkpoint

def analyze_parameters(model):
    """分析模型参数"""
    results = {}
    
    # 1. 计算embedding similarity
    embeddings = model.transformer.wte.weight.data.cpu().numpy()
    num_nodes = min(100, embeddings.shape[0] - 2)  # 去掉特殊token
    
    similarities = []
    for i in range(2, num_nodes + 2):
        for j in range(i + 1, num_nodes + 2):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            )
            similarities.append(sim)
    
    results['embedding_similarity'] = {
        'mean': np.mean(similarities),
        'std': np.std(similarities),
        'max': np.max(similarities),
        'min': np.min(similarities)
    }
    
    # 2. 计算embedding norms
    norms = [np.linalg.norm(embeddings[i]) for i in range(2, num_nodes + 2)]
    results['embedding_norms'] = {
        'mean': np.mean(norms),
        'std': np.std(norms),
        'max': np.max(norms),
        'min': np.min(norms)
    }
    
    # 3. 分析lm_head权重（用于预测edge的权重）
    lm_head_weight = model.lm_head.weight.data.cpu().numpy()
    
    # 计算weight gap（需要知道哪些是edge，哪些不是）
    # 这里简化：分析所有权重的分布
    results['lm_head_stats'] = {
        'mean': float(np.mean(lm_head_weight)),
        'std': float(np.std(lm_head_weight)),
        'norm': float(np.linalg.norm(lm_head_weight))
    }
    
    # 4. 各层权重范数
    layer_norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_norms[name] = float(torch.norm(param).item())
    results['layer_norms'] = layer_norms
    
    return results

def analyze_output_distribution(model, num_samples=100, device='cuda'):
    """分析模型输出分布（熵、置信度等）"""
    model.to(device)
    results = {
        'entropies': [],
        'max_probs': [],
        'top5_probs': [],
        'confidence_ratios': []  # max_prob / second_max_prob
    }
    
    # 生成一些测试输入
    # 格式：source target source（标准的图路径格式）
    test_inputs = []
    for _ in range(num_samples):
        source = np.random.randint(0, 90)
        target = np.random.randint(source + 5, min(source + 15, 99))
        # +2 因为token偏移
        test_inputs.append([source + 2, target + 2, source + 2])
    
    with torch.no_grad():
        for inp in test_inputs:
            x = torch.tensor([inp], device=device)
            logits, _ = model(x)
            
            # 分析最后一个位置的输出
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=0)
            
            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            results['entropies'].append(entropy)
            
            # Top概率
            top5_probs, _ = torch.topk(probs, 5)
            results['max_probs'].append(top5_probs[0].item())
            results['top5_probs'].append(torch.sum(top5_probs).item())
            
            # 置信度比率
            if len(top5_probs) >= 2:
                ratio = top5_probs[0].item() / (top5_probs[1].item() + 1e-8)
                results['confidence_ratios'].append(ratio)
    
    # 计算统计量
    stats = {}
    for key, values in results.items():
        if values:
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    model.cpu()
    return stats

def compare_models(diversity_iter, standard_iter=None):
    """比较diversity和standard模型"""
    results = {
        'iteration': diversity_iter,
        'diversity': {},
        'standard': {}
    }
    
    # 加载diversity模型
    div_path = os.path.join(DIVERSITY_PATH, f"ckpt_{diversity_iter}.pt")
    div_model, div_ckpt = load_checkpoint(div_path)
    
    # 分析diversity
    print(f"\nAnalyzing diversity checkpoint {diversity_iter}...")
    results['diversity']['parameters'] = analyze_parameters(div_model)
    results['diversity']['output_dist'] = analyze_output_distribution(div_model)
    
    # 如果有对应的standard checkpoint
    if standard_iter:
        std_path = os.path.join(STANDARD_PATH, f"{standard_iter}_ckpt_20.pt")
        if os.path.exists(std_path):
            print(f"Analyzing standard checkpoint {standard_iter}...")
            std_model, std_ckpt = load_checkpoint(std_path)
            results['standard']['parameters'] = analyze_parameters(std_model)
            results['standard']['output_dist'] = analyze_output_distribution(std_model)
    
    return results

def visualize_comparison(all_results):
    """生成对比可视化"""
    # 提取数据用于绘图
    iterations = []
    div_entropy = []
    div_similarity = []
    div_max_prob = []
    
    std_entropy = []
    std_similarity = []
    std_max_prob = []
    
    for result in all_results:
        iterations.append(result['iteration'])
        
        # Diversity数据
        div_entropy.append(result['diversity']['output_dist']['entropies']['mean'])
        div_similarity.append(result['diversity']['parameters']['embedding_similarity']['mean'])
        div_max_prob.append(result['diversity']['output_dist']['max_probs']['mean'])
        
        # Standard数据（如果有）
        if result['standard'] and 'output_dist' in result['standard']:
            std_entropy.append(result['standard']['output_dist']['entropies']['mean'])
            std_similarity.append(result['standard']['parameters']['embedding_similarity']['mean'])
            std_max_prob.append(result['standard']['output_dist']['max_probs']['mean'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 输出熵对比
    ax = axes[0, 0]
    ax.plot(iterations, div_entropy, 'g-', marker='o', linewidth=2, markersize=8, label='Diversity')
    if std_entropy:
        ax.plot(iterations[:len(std_entropy)], std_entropy, 'r-', marker='s', linewidth=2, markersize=8, label='Standard')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Output Entropy')
    ax.set_title('Output Entropy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Embedding相似度
    ax = axes[0, 1]
    ax.plot(iterations, div_similarity, 'g-', marker='o', linewidth=2, markersize=8, label='Diversity')
    if std_similarity:
        ax.plot(iterations[:len(std_similarity)], std_similarity, 'r-', marker='s', linewidth=2, markersize=8, label='Standard')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Embedding Similarity')
    ax.set_title('Embedding Similarity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 最大概率
    ax = axes[0, 2]
    ax.plot(iterations, div_max_prob, 'g-', marker='o', linewidth=2, markersize=8, label='Diversity')
    if std_max_prob:
        ax.plot(iterations[:len(std_max_prob)], std_max_prob, 'r-', marker='s', linewidth=2, markersize=8, label='Standard')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Max Probability')
    ax.set_title('Output Confidence Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4-6. 可以添加更多可视化
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diversity_vs_standard_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Visualization saved to {OUTPUT_DIR}/diversity_vs_standard_comparison.png")

def generate_report(all_results):
    """生成文字报告"""
    report = []
    report.append("# Diversity机制分析报告\n")
    report.append("## 1. 关键发现\n")
    
    # 分析熵的变化
    div_entropies = [r['diversity']['output_dist']['entropies']['mean'] for r in all_results]
    report.append(f"### Diversity训练的熵保持:\n")
    report.append(f"- 50k: {div_entropies[0]:.3f}\n")
    report.append(f"- 100k: {div_entropies[1]:.3f}\n") 
    report.append(f"- 150k: {div_entropies[2]:.3f}\n")
    report.append(f"- 200k: {div_entropies[3]:.3f}\n")
    report.append(f"- 平均: {np.mean(div_entropies):.3f}\n")
    report.append(f"- 变化: {max(div_entropies) - min(div_entropies):.3f}\n\n")
    
    # 参数稳定性
    report.append("### 参数稳定性:\n")
    for i in range(len(all_results)-1):
        iter1 = all_results[i]['iteration']
        iter2 = all_results[i+1]['iteration']
        sim1 = all_results[i]['diversity']['parameters']['embedding_similarity']['mean']
        sim2 = all_results[i+1]['diversity']['parameters']['embedding_similarity']['mean']
        report.append(f"- {iter1}-{iter2}: 相似度变化 {sim2-sim1:.4f}\n")
    
    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'diversity_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    
    print(f"Report saved to {report_path}")

def main():
    """主函数"""
    print("=== Diversity机制完整分析 ===")
    
    # 确保输出目录存在
    ensure_output_dir()
    
    # 分析所有checkpoint
    all_results = []
    
    for div_iter in DIVERSITY_CHECKPOINTS:
        # 找到对应的standard checkpoint（如果存在）
        std_iter = div_iter if div_iter in STANDARD_CHECKPOINTS else None
        
        result = compare_models(div_iter, std_iter)
        all_results.append(result)
    
    # 保存原始结果
    results_path = os.path.join(OUTPUT_DIR, 'raw_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")
    
    # 生成可视化
    print("\nGenerating visualizations...")
    visualize_comparison(all_results)
    
    # 生成报告
    print("\nGenerating report...")
    generate_report(all_results)
    
    print("\n=== 分析完成 ===")
    print(f"所有结果保存在: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()