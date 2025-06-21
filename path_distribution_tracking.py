"""
路径分布细粒度追踪 - 证明第一个箭头到第二个箭头
追踪具体路径的概率变化，而不只是整体统计
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
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
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
    return model, gptconf.block_size

def decode_tokens(token_ids, itos):
    """解码token序列为字符串"""
    decoded = []
    for tid in token_ids:
        if tid == 1:  # 换行符
            break
        elif tid in itos:
            decoded.append(itos[tid])
    return ' '.join(decoded)

def extract_path_from_tokens(token_ids):
    """从token序列提取路径"""
    path = []
    # 跳过前3个token (source target source)
    for i in range(3, len(token_ids)):
        tid = token_ids[i]
        if tid == 1:  # 换行符
            break
        if 2 <= tid <= 101:  # 节点token范围
            path.append(tid - 2)  # 转换回节点id
    return tuple(path)

def load_test_examples(data_path, meta, num_examples=100):
    """加载测试样例 - 简化版本"""
    stoi = meta['stoi']
    examples = []
    
    # 读取测试文件
    test_file = os.path.join(data_path, 'test.txt')
    print(f"Loading test file from: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found!")
        return examples
    
    # 读取文件
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        try:
            with open(test_file, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except:
            print("ERROR: Cannot read test file")
            return examples
    
    print(f"Found {len(lines)} lines in test file")
    
    # 处理每一行
    count = 0
    for line in lines:
        if count >= num_examples:
            break
            
        line = line.strip()
        if not line:
            continue
        
        # 分割tokens
        tokens = line.split()
        if len(tokens) < 3:
            continue
        
        # 前3个token作为prompt
        prompt = []
        for i in range(3):
            if tokens[i] in stoi:
                prompt.append(stoi[tokens[i]])
            else:
                # 如果token不在词表中，跳过这一行
                break
        
        if len(prompt) == 3:
            # 提取路径（从第3个token开始）
            path = []
            for i in range(2, len(tokens)):
                if tokens[i].isdigit():
                    path.append(int(tokens[i]))
            
            if len(path) >= 2:  # 至少要有起点和终点
                examples.append((prompt, tuple(path)))
                count += 1
    
    print(f"Successfully loaded {len(examples)} examples")
    if len(examples) > 0:
        print(f"Example prompt: {examples[0][0]}")
        print(f"Example path: {examples[0][1]}")
    
    return examples

def track_path_probabilities(model, test_examples, device='cuda:0', top_k=5, block_size=32):
    """追踪每个测试样例的top-k路径概率"""
    model.eval()
    path_probabilities = {}
    
    for idx, (prompt, target_path) in enumerate(tqdm(test_examples, desc="Tracking paths")):
        # 准备输入
        prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
        
        # 获取第一个预测步骤的概率分布（简化版本）
        with torch.no_grad():
            logits, _ = model(prompt_tensor)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
        
        # 获取top-k个token及其概率
        top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[0]))
        
        # 简单记录top预测
        predictions = []
        for i in range(len(top_probs)):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            
            # 如果是节点token，记录
            if 2 <= token_id <= 101:
                node_id = token_id - 2
                predictions.append(((node_id,), float(prob)))
        
        # 记录结果
        path_probabilities[idx] = {
            'target_path': target_path,
            'predictions': predictions[:5]  # 最多保留5个
        }
    
    return path_probabilities

def analyze_distribution_shift(all_checkpoints_data):
    """分析分布如何变化"""
    results = {}
    
    for ckpt, data in all_checkpoints_data.items():
        if not data:
            results[ckpt] = {
                'mean_training_path_prob': 0.0,
                'mean_alternative_path_prob': 0.0,
                'num_unique_paths': 0,
                'entropy': 0.0,
                'top_path_frequency': 0.0,
                'training_path_in_top': 0.0
            }
            continue
        
        # 简化分析 - 只看第一步预测
        training_probs = []
        alternative_probs = []
        
        for example_data in data.values():
            target = example_data['target_path']
            predictions = example_data['predictions']
            
            if not predictions or not target:
                continue
            
            # 检查第一步预测是否匹配目标路径的第一步
            target_first = target[0] if len(target) > 0 else None
            
            found_target = False
            for pred_path, prob in predictions:
                if pred_path and pred_path[0] == target_first:
                    training_probs.append(prob)
                    found_target = True
                    break
            
            if not found_target and predictions:
                training_probs.append(0.0)
                
            # 记录最高概率的替代预测
            if predictions and predictions[0][0]:
                if not found_target or (predictions[0][0][0] != target_first):
                    alternative_probs.append(predictions[0][1])
        
        # 计算统计量
        results[ckpt] = {
            'mean_training_path_prob': float(np.mean(training_probs)) if training_probs else 0.0,
            'mean_alternative_path_prob': float(np.mean(alternative_probs)) if alternative_probs else 0.0,
            'num_unique_paths': len(data),
            'entropy': 0.0,  # 简化版本不计算熵
            'top_path_frequency': 0.0,
            'training_path_in_top': float(sum(1 for p in training_probs if p > 0) / len(training_probs)) if training_probs else 0.0
        }
    
    return results

def main():
    # 配置
    checkpoints = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    base_dir = 'out/simple_graph_1_1_120_100_original_seed42'
    data_path = 'data/simple_graph/100'
    output_dir = 'analysis_results/path_distribution'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载meta信息
    meta = load_meta(data_path)
    print(f"Vocabulary size: {len(meta['stoi'])}")
    print(f"Block size: {meta['block_size']}")
    
    # 准备测试样例
    test_examples = load_test_examples(data_path, meta, num_examples=100)
    
    if len(test_examples) == 0:
        print("ERROR: No test examples loaded. Please check:")
        print(f"1. Test file exists at: {os.path.join(data_path, 'test.txt')}")
        print(f"2. File format is correct (space-separated tokens)")
        print(f"3. Tokens are in vocabulary")
        return
    
    all_results = {}
    
    for ckpt in tqdm(checkpoints, desc="Analyzing checkpoints"):
        ckpt_path = os.path.join(base_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found, skipping...")
            continue
        
        try:
            # 加载模型
            model, block_size = load_model(ckpt_path)
            
            # 追踪路径概率
            path_probs = track_path_probabilities(model, test_examples, block_size=block_size)
            all_results[ckpt] = path_probs
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 分析分布变化
    distribution_shifts = analyze_distribution_shift(all_results)
    
    # 保存结果
    serializable_results = convert_to_serializable(all_results)
    with open(os.path.join(output_dir, 'path_probabilities.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    with open(os.path.join(output_dir, 'distribution_shifts.json'), 'w') as f:
        json.dump(distribution_shifts, f, indent=2)
    
    # 打印统计信息
    print("\nDistribution shift summary:")
    for ckpt in sorted(distribution_shifts.keys()):
        stats = distribution_shifts[ckpt]
        print(f"\nCheckpoint {ckpt}:")
        print(f"  Training path prob: {stats['mean_training_path_prob']:.4f}")
        print(f"  Alternative path prob: {stats['mean_alternative_path_prob']:.4f}")
    
    # 创建简单可视化
    if distribution_shifts:
        create_simple_plots(distribution_shifts, output_dir)
    
    print(f"\nPath distribution analysis complete! Results saved to {output_dir}/")

def create_simple_plots(results, output_dir):
    """创建简单的可视化"""
    checkpoints = sorted(results.keys())
    
    plt.figure(figsize=(10, 6))
    
    training_probs = [results[ckpt]['mean_training_path_prob'] for ckpt in checkpoints]
    
    plt.plot(checkpoints, training_probs, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Training Iteration')
    plt.ylabel('Mean Training Path Probability')
    plt.title('Training Path Probability Evolution')
    plt.grid(True, alpha=0.3)
    
    if 140000 in checkpoints:
        plt.axvline(x=140000, color='red', linestyle='--', alpha=0.5, label='Phase Transition')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_path_prob.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()