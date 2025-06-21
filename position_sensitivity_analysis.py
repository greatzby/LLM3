"""
位置敏感度分析 - 分析模型在不同token位置的行为
运行方式: python position_sensitivity_analysis.py --num_nodes 100
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoints', type=int, nargs='+', 
                       default=[0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000])
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def load_test_sequences(data_path, meta, max_samples=500):
    """加载并预处理测试序列"""
    stoi = meta['stoi']
    sequences = []
    
    try:
        with open(f'{data_path}/test.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        with open(f'{data_path}/test.txt', 'r', encoding='gbk') as f:
            lines = f.readlines()
    
    for i, line in enumerate(lines):
        if i >= max_samples:
            break
        
        tokens = line.strip().split()
        sequence = []
        for token in tokens:
            if token in stoi:
                sequence.append(stoi[token])
        
        if len(sequence) >= 4:  # 确保序列足够长
            sequences.append(sequence)
    
    return sequences

def analyze_position_predictions(model, test_sequences, position, device, meta):
    """分析特定位置的预测特性"""
    model.eval()
    
    accuracies = []
    entropies = []
    max_probs = []
    top5_probs = []
    
    with torch.no_grad():
        for sequence in test_sequences:
            if len(sequence) <= position:
                continue
            
            # 获取到position位置的context
            context = sequence[:position]
            target = sequence[position] if position < len(sequence) else None
            
            # 获取预测
            context_tensor = torch.tensor(context, device=device).unsqueeze(0)
            logits, _ = model(context_tensor)
            logits = logits[0, -1, :]  # 最后一个位置的logits
            
            # 计算概率
            probs = F.softmax(logits, dim=-1)
            
            # 计算指标
            if target is not None:
                pred = torch.argmax(probs).item()
                accuracies.append(1.0 if pred == target else 0.0)
            
            # 计算熵
            probs_np = probs.cpu().numpy()
            entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))
            entropies.append(float(entropy))
            
            # 最大概率和top-5概率
            sorted_probs, _ = torch.sort(probs, descending=True)
            max_probs.append(float(sorted_probs[0]))
            top5_probs.append(float(torch.sum(sorted_probs[:5])))
    
    return {
        'accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
        'accuracy_std': float(np.std(accuracies)) if accuracies else 0.0,
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'max_prob_mean': float(np.mean(max_probs)),
        'max_prob_std': float(np.std(max_probs)),
        'top5_prob_mean': float(np.mean(top5_probs)),
        'top5_prob_std': float(np.std(top5_probs)),
        'num_samples': len(accuracies)
    }

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'position_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载meta信息
    dataset = 'simple_graph'
    config = '1_1_120'
    data_path = f'data/{dataset}/{args.num_nodes}'
    meta_path = f'{data_path}/meta.pkl'
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # 加载测试数据
    print("Loading test sequences...")
    test_sequences = load_test_sequences(data_path, meta, args.max_samples)
    print(f"Loaded {len(test_sequences)} test sequences")
    
    # 分析每个checkpoint
    results = {}
    
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
            
            # 分析不同位置
            position_results = {}
            positions_to_analyze = [3, 4, 5, 6, 7, 8, 10, 12]  # 分析不同位置
            
            for pos in positions_to_analyze:
                print(f"  Analyzing position {pos}")
                pos_stats = analyze_position_predictions(model, test_sequences, pos, args.device, meta)
                position_results[f'position_{pos}'] = pos_stats
            
            results[ckpt_iter] = position_results
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt_iter}: {e}")
            continue
    
    # 保存结果
    with open(os.path.join(output_dir, 'position_sensitivity_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成可视化
    create_position_plots(results, output_dir)
    
    print(f"\nPosition analysis complete! Results saved to {output_dir}/")

def create_position_plots(results, output_dir):
    """创建位置分析的可视化图表"""
    if not results:
        print("No results to plot")
        return
    
    # 准备数据
    positions = [3, 4, 5, 6, 7, 8, 10, 12]
    checkpoints = sorted(results.keys())
    
    # 创建熵的热力图
    entropy_matrix = []
    accuracy_matrix = []
    
    for ckpt in checkpoints:
        entropy_row = []
        accuracy_row = []
        for pos in positions:
            key = f'position_{pos}'
            if key in results[ckpt]:
                entropy_row.append(results[ckpt][key]['entropy_mean'])
                accuracy_row.append(results[ckpt][key]['accuracy'])
            else:
                entropy_row.append(np.nan)
                accuracy_row.append(np.nan)
        entropy_matrix.append(entropy_row)
        accuracy_matrix.append(accuracy_row)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 熵热力图
    im1 = ax1.imshow(entropy_matrix, aspect='auto', cmap='viridis')
    ax1.set_xticks(range(len(positions)))
    ax1.set_xticklabels(positions)
    ax1.set_yticks(range(len(checkpoints)))
    ax1.set_yticklabels([f'{ckpt//1000}k' for ckpt in checkpoints])
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Training Iteration')
    ax1.set_title('Prediction Entropy by Position')
    plt.colorbar(im1, ax=ax1, label='Entropy')
    
    # 准确率热力图
    im2 = ax2.imshow(accuracy_matrix, aspect='auto', cmap='RdYlGn')
    ax2.set_xticks(range(len(positions)))
    ax2.set_xticklabels(positions)
    ax2.set_yticks(range(len(checkpoints)))
    ax2.set_yticklabels([f'{ckpt//1000}k' for ckpt in checkpoints])
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Training Iteration')
    ax2.set_title('Prediction Accuracy by Position')
    plt.colorbar(im2, ax=ax2, label='Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_analysis_heatmaps.png'), dpi=150)
    plt.close()
    
    # 创建线图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 位置3的演化
    ax = axes[0, 0]
    for metric in ['accuracy', 'entropy_mean']:
        values = [results[ckpt].get('position_3', {}).get(metric, 0) for ckpt in checkpoints]
        ax.plot(checkpoints, values, marker='o', label=metric)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Position 3 Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 位置5的演化
    ax = axes[0, 1]
    for metric in ['accuracy', 'entropy_mean']:
        values = [results[ckpt].get('position_5', {}).get(metric, 0) for ckpt in checkpoints]
        ax.plot(checkpoints, values, marker='o', label=metric)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Position 5 Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 不同位置的准确率比较
    ax = axes[1, 0]
    for pos in [3, 4, 5, 6]:
        values = [results[ckpt].get(f'position_{pos}', {}).get('accuracy', 0) for ckpt in checkpoints]
        ax.plot(checkpoints, values, marker='o', label=f'Position {pos}')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison by Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 最大概率的演化
    ax = axes[1, 1]
    for pos in [3, 5, 7]:
        values = [results[ckpt].get(f'position_{pos}', {}).get('max_prob_mean', 0) for ckpt in checkpoints]
        ax.plot(checkpoints, values, marker='o', label=f'Position {pos}')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Max Probability')
    ax.set_title('Maximum Probability Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_evolution_plots.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()