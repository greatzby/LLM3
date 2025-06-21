"""
预测分布细节分析 - 深入分析预测概率分布的特性
运行方式: python distribution_analysis.py --num_nodes 100
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
import torch.nn.functional as F
from scipy.stats import entropy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoints', type=int, nargs='+', 
                       default=[0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000])
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def compute_gini_coefficient(probs):
    """计算Gini系数"""
    sorted_probs = np.sort(probs)
    n = len(probs)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n)

def compute_effective_choices(probs):
    """计算有效选择数（perplexity）"""
    return float(np.exp(entropy(probs)))

def analyze_prediction_distribution(model, test_sequences, vocab_size, device):
    """深入分析预测分布"""
    model.eval()
    
    all_metrics = {
        'entropy': [],
        'top1_prob': [],
        'top5_prob': [],
        'top10_prob': [],
        'gini_coefficient': [],
        'kl_from_uniform': [],
        'effective_choices': [],
        'max_to_second_ratio': [],
        'concentration_index': []
    }
    
    # 按位置收集指标
    position_metrics = {}
    
    with torch.no_grad():
        for sequence in tqdm(test_sequences, desc="Analyzing distributions"):
            if len(sequence) < 4:
                continue
            
            # 获取context
            context = torch.tensor(sequence[:-1], device=device).unsqueeze(0)
            
            # 获取预测
            logits, _ = model(context)
            logits = logits[0, :, :]  # [seq_len, vocab_size]
            
            # 计算概率
            probs = F.softmax(logits, dim=-1)
            
            # 对每个位置计算指标
            for pos in range(min(probs.shape[0], 20)):  # 最多分析前20个位置
                pos_probs = probs[pos].cpu().numpy()
                
                # 基本指标
                ent = entropy(pos_probs)
                all_metrics['entropy'].append(float(ent))
                
                # Top-k概率
                sorted_probs = np.sort(pos_probs)[::-1]
                all_metrics['top1_prob'].append(float(sorted_probs[0]))
                all_metrics['top5_prob'].append(float(np.sum(sorted_probs[:5])))
                all_metrics['top10_prob'].append(float(np.sum(sorted_probs[:10])))
                
                # Gini系数
                gini = compute_gini_coefficient(pos_probs)
                all_metrics['gini_coefficient'].append(gini)
                
                # KL散度（与均匀分布的距离）
                uniform_probs = np.ones(vocab_size) / vocab_size
                kl_div = float(np.sum(pos_probs * np.log(pos_probs / uniform_probs + 1e-10)))
                all_metrics['kl_from_uniform'].append(kl_div)
                
                # 有效选择数
                eff_choices = compute_effective_choices(pos_probs)
                all_metrics['effective_choices'].append(eff_choices)
                
                # 最大概率与第二大概率的比率
                if sorted_probs[1] > 0:
                    ratio = float(sorted_probs[0] / sorted_probs[1])
                else:
                    ratio = float(sorted_probs[0] * 1000)  # 大数表示无穷大
                all_metrics['max_to_second_ratio'].append(ratio)
                
                # 集中度指标（前10个概率的平方和）
                concentration = float(np.sum(sorted_probs[:10] ** 2))
                all_metrics['concentration_index'].append(concentration)
                
                # 按位置收集
                if pos not in position_metrics:
                    position_metrics[pos] = {
                        'entropy': [],
                        'top1_prob': [],
                        'effective_choices': []
                    }
                position_metrics[pos]['entropy'].append(ent)
                position_metrics[pos]['top1_prob'].append(float(sorted_probs[0]))
                position_metrics[pos]['effective_choices'].append(eff_choices)
    
    # 计算统计量
    results = {}
    for key, values in all_metrics.items():
        if values:
            results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'percentiles': {
                    '10': float(np.percentile(values, 10)),
                    '25': float(np.percentile(values, 25)),
                    '75': float(np.percentile(values, 75)),
                    '90': float(np.percentile(values, 90))
                }
            }
    
    # 计算位置统计
    position_stats = {}
    for pos, metrics in position_metrics.items():
        position_stats[f'position_{pos}'] = {
            'entropy_mean': float(np.mean(metrics['entropy'])),
            'top1_prob_mean': float(np.mean(metrics['top1_prob'])),
            'effective_choices_mean': float(np.mean(metrics['effective_choices']))
        }
    
    results['position_stats'] = position_stats
    
    return results, all_metrics

def load_test_sequences(data_path, meta, max_samples=500):
    """加载测试序列"""
    stoi = meta['stoi']
    sequences = []
    
    try:
        with open(f'{data_path}/test.txt', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                tokens = line.strip().split()
                sequence = []
                for token in tokens:
                    if token in stoi:
                        sequence.append(stoi[token])
                
                if len(sequence) >= 4:
                    sequences.append(sequence)
    except:
        with open(f'{data_path}/test.txt', 'r', encoding='gbk') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                tokens = line.strip().split()
                sequence = []
                for token in tokens:
                    if token in stoi:
                        sequence.append(stoi[token])
                
                if len(sequence) >= 4:
                    sequences.append(sequence)
    
    return sequences

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    dataset = 'simple_graph'
    config = '1_1_120'
    data_path = f'data/{dataset}/{args.num_nodes}'
    meta_path = f'{data_path}/meta.pkl'
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    
    # 加载测试序列
    print("Loading test sequences...")
    test_sequences = load_test_sequences(data_path, meta, args.num_samples)
    print(f"Loaded {len(test_sequences)} test sequences")
    
    # 分析每个checkpoint
    results = {}
    raw_metrics = {}
    
    for ckpt_iter in args.checkpoints:
        print(f"\nAnalyzing checkpoint {ckpt_iter}")
        
        # 加载模型 - 更新路径
        out_dir = f'out/{dataset}_{config}_{args.num_nodes}_original_seed{args.seed}'
        ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_20.pt')
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping...")
            continue
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf)
            
            # 处理state dict
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            model.load_state_dict(state_dict)
            model.eval()
            model.to(args.device)
            
            # 分析分布
            ckpt_results, ckpt_raw = analyze_prediction_distribution(
                model, test_sequences, vocab_size, args.device)
            
            results[ckpt_iter] = ckpt_results
            raw_metrics[ckpt_iter] = ckpt_raw
            
            # 打印关键指标
            print(f"Key metrics for checkpoint {ckpt_iter}:")
            print(f"  Entropy: {ckpt_results['entropy']['mean']:.4f} ± {ckpt_results['entropy']['std']:.4f}")
            print(f"  Top-1 Prob: {ckpt_results['top1_prob']['mean']:.4f} ± {ckpt_results['top1_prob']['std']:.4f}")
            print(f"  Effective Choices: {ckpt_results['effective_choices']['mean']:.2f}")
            print(f"  Gini Coefficient: {ckpt_results['gini_coefficient']['mean']:.4f}")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt_iter}: {e}")
            continue
    
    # 保存结果
    with open(os.path.join(output_dir, 'distribution_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 创建可视化
    create_distribution_visualizations(results, raw_metrics, output_dir)
    
    print(f"\nDistribution analysis complete! Results saved to {output_dir}/")

def create_distribution_visualizations(results, raw_metrics, output_dir):
    """创建分布分析的可视化"""
    if not results:
        print("No results to plot")
        return
    
    checkpoints = sorted(results.keys())
    
    # 1. 主要指标的演化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['entropy', 'top1_prob', 'top5_prob', 'gini_coefficient', 
               'kl_from_uniform', 'effective_choices']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        means = []
        stds = []
        for ckpt in checkpoints:
            if metric in results[ckpt]:
                means.append(results[ckpt][metric]['mean'])
                stds.append(results[ckpt][metric]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        ax.errorbar(checkpoints, means, yerr=stds, marker='o', capsize=5)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric} Evolution')
        ax.grid(True, alpha=0.3)
        
        # 设置x轴标签
        ax.set_xticks(checkpoints[::2])  # 每隔一个显示
        ax.set_xticklabels([f'{ckpt//1000}k' for ckpt in checkpoints[::2]])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_metrics_evolution.png'), dpi=150)
    plt.close()
    
    # 2. 分布形状的可视化（箱线图）
    if raw_metrics:
        fig, axes = plt.subplots(2, len(checkpoints), figsize=(5*len(checkpoints), 10))
        if len(checkpoints) == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, ckpt in enumerate(checkpoints):
            if ckpt in raw_metrics:
                # 熵的箱线图
                ax = axes[0, idx]
                entropy_values = raw_metrics[ckpt].get('entropy', [])
                if entropy_values:
                    ax.boxplot(entropy_values, showfliers=False)
                    ax.set_title(f'Checkpoint {ckpt//1000}k')
                    ax.set_ylabel('Entropy')
                    ax.grid(True, alpha=0.3)
                
                # Top-1概率的箱线图
                ax = axes[1, idx]
                top1_values = raw_metrics[ckpt].get('top1_prob', [])
                if top1_values:
                    ax.boxplot(top1_values, showfliers=False)
                    ax.set_ylabel('Top-1 Probability')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_boxplots.png'), dpi=150)
        plt.close()
    
    # 3. 位置特定的分析
    positions = list(range(5))  # 分析前5个位置
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 熵随位置变化
    ax = axes[0]
    for ckpt in checkpoints[::2]:  # 每隔一个checkpoint
        if ckpt in results and 'position_stats' in results[ckpt]:
            pos_stats = results[ckpt]['position_stats']
            entropies = []
            for pos in positions:
                key = f'position_{pos}'
                if key in pos_stats:
                    entropies.append(pos_stats[key]['entropy_mean'])
                else:
                    entropies.append(0)
            if entropies:
                ax.plot(positions[:len(entropies)], entropies, marker='o', label=f'{ckpt//1000}k')
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Entropy by Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-1概率随位置变化
    ax = axes[1]
    for ckpt in checkpoints[::2]:
        if ckpt in results and 'position_stats' in results[ckpt]:
            pos_stats = results[ckpt]['position_stats']
            top1_probs = []
            for pos in positions:
                key = f'position_{pos}'
                if key in pos_stats:
                    top1_probs.append(pos_stats[key]['top1_prob_mean'])
                else:
                    top1_probs.append(0)
            if top1_probs:
                ax.plot(positions[:len(top1_probs)], top1_probs, marker='o', label=f'{ckpt//1000}k')
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Mean Top-1 Probability')
    ax.set_title('Top-1 Probability by Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 有效选择数随位置变化
    ax = axes[2]
    for ckpt in checkpoints[::2]:
        if ckpt in results and 'position_stats' in results[ckpt]:
            pos_stats = results[ckpt]['position_stats']
            eff_choices = []
            for pos in positions:
                key = f'position_{pos}'
                if key in pos_stats:
                    eff_choices.append(pos_stats[key]['effective_choices_mean'])
                else:
                    eff_choices.append(0)
            if eff_choices:
                ax.plot(positions[:len(eff_choices)], eff_choices, marker='o', label=f'{ckpt//1000}k')
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Mean Effective Choices')
    ax.set_title('Effective Choices by Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_specific_metrics.png'), dpi=150)
    plt.close()
    
    # 4. 集中度分析
    plt.figure(figsize=(10, 6))
    
    # 最大概率与第二大概率的比率
    ratios = []
    ratio_stds = []
    
    for ckpt in checkpoints:
        if ckpt in results and 'max_to_second_ratio' in results[ckpt]:
            ratios.append(results[ckpt]['max_to_second_ratio']['median'])  # 使用中位数
            ratio_stds.append(results[ckpt]['max_to_second_ratio']['std'])
        else:
            ratios.append(1)
            ratio_stds.append(0)
    
    plt.semilogy(checkpoints, ratios, marker='o', linewidth=2)
    plt.xlabel('Training Iteration')
    plt.ylabel('Max/Second Probability Ratio (log scale)')
    plt.title('Distribution Concentration Evolution')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴标签
    plt.xticks(checkpoints[::2], [f'{ckpt//1000}k' for ckpt in checkpoints[::2]])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concentration_evolution.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()