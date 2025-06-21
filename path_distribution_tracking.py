"""
路径分布细粒度追踪 - 完整版本
"""

import os
import torch
import numpy as np
import pandas as pd
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from model import GPTConfig, GPT
from utils import load_meta, convert_to_serializable, load_model, decode_tokens, encode_tokens, load_test_examples

def generate_full_path(model, prompt_tensor, device, max_length=20):
    """生成完整路径"""
    model.eval()
    generated = prompt_tensor.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            if generated.shape[1] >= model.config.block_size:
                break
                
            logits, _ = model(generated)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # 如果生成了换行符，停止
            if next_token.item() == 1:
                break
    
    return generated[0].cpu().numpy()

def extract_path_from_generation(token_ids):
    """从生成的token序列提取路径"""
    path = []
    # 跳过前3个token (prompt)
    for i in range(3, len(token_ids)):
        tid = token_ids[i]
        if tid == 1:  # 换行符
            break
        if 2 <= tid <= 101:  # 节点token
            path.append(tid - 2)
    return tuple(path)

def analyze_checkpoint_distribution(model, test_examples, device='cuda:0'):
    """分析单个checkpoint的路径分布"""
    model.eval()
    
    training_path_matches = 0
    total_examples = len(test_examples)
    path_counter = defaultdict(int)
    
    for prompt, target_str, target_path in tqdm(test_examples, desc="Analyzing"):
        # 生成完整路径
        prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
        generated_tokens = generate_full_path(model, prompt_tensor, device)
        
        # 提取生成的路径
        generated_path = extract_path_from_generation(generated_tokens)
        
        # 统计
        path_counter[generated_path] += 1
        
        # 检查是否匹配训练路径
        if generated_path == target_path:
            training_path_matches += 1
    
    # 计算统计信息
    training_path_ratio = training_path_matches / total_examples
    unique_paths = len(path_counter)
    
    # 找出最常见的路径
    most_common_paths = sorted(path_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'training_path_ratio': float(training_path_ratio),
        'unique_paths': unique_paths,
        'most_common_paths': most_common_paths,
        'path_distribution': dict(path_counter)
    }

def main():
    # 配置
    checkpoints = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    base_dir = 'out/simple_graph_1_1_120_100_original_seed42'
    data_path = 'data/simple_graph/100'
    output_dir = 'analysis_results/path_distribution'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载meta信息和测试数据
    meta = load_meta(data_path)
    test_examples = load_test_examples(data_path, meta, num_examples=100)
    print(f"Loaded {len(test_examples)} test examples")
    
    if len(test_examples) == 0:
        print("ERROR: No test examples loaded!")
        return
    
    # 分析每个checkpoint
    all_results = {}
    
    for ckpt in tqdm(checkpoints, desc="Analyzing checkpoints"):
        ckpt_path = os.path.join(base_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found, skipping...")
            continue
        
        try:
            model = load_model(ckpt_path)
            
            # 分析分布
            distribution_analysis = analyze_checkpoint_distribution(model, test_examples[:50])  # 用50个例子
            all_results[ckpt] = distribution_analysis
            
            # 打印结果
            print(f"\nCheckpoint {ckpt}:")
            print(f"  Training path match ratio: {distribution_analysis['training_path_ratio']:.4f}")
            print(f"  Unique paths generated: {distribution_analysis['unique_paths']}")
            print(f"  Most common paths:")
            for path, count in distribution_analysis['most_common_paths']:
                print(f"    {path}: {count} times")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    serializable_results = convert_to_serializable(all_results)
    with open(os.path.join(output_dir, 'path_distribution_analysis.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 创建可视化
    create_distribution_plots(all_results, output_dir)
    
    print(f"\nPath distribution analysis complete!")

def create_distribution_plots(results, output_dir):
    """创建分布分析图表"""
    if not results:
        return
    
    checkpoints = sorted(results.keys())
    
    # 1. Training path ratio over time
    plt.figure(figsize=(10, 6))
    
    ratios = [results[ckpt]['training_path_ratio'] for ckpt in checkpoints]
    plt.plot(checkpoints, ratios, marker='o', linewidth=2, markersize=8)
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Training Path Match Ratio')
    plt.title('Ratio of Predictions Matching Training Paths')
    plt.grid(True, alpha=0.3)
    
    if 140000 in checkpoints:
        plt.axvline(x=140000, color='red', linestyle='--', alpha=0.5, label='Phase Transition')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_path_ratio.png'), dpi=150)
    plt.close()
    
    # 2. Number of unique paths
    plt.figure(figsize=(10, 6))
    
    unique_paths = [results[ckpt]['unique_paths'] for ckpt in checkpoints]
    plt.plot(checkpoints, unique_paths, marker='s', linewidth=2, markersize=8, color='green')
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Number of Unique Paths Generated')
    plt.title('Path Generation Diversity')
    plt.grid(True, alpha=0.3)
    
    if 140000 in checkpoints:
        plt.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_diversity.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()