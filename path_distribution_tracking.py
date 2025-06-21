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

def find_third_number_position(number_string):
    """与训练代码一致的辅助函数"""
    numbers = number_string.split()
    third_number_index = 2
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index - 1
    return position

def load_test_examples(data_path, meta, num_examples=100):
    """加载测试样例 - 与训练代码一致"""
    stoi = meta['stoi']
    itos = meta['itos']
    simple_format = meta.get('simple_format', True)
    examples = []
    
    # 读取测试文件
    test_file = os.path.join(data_path, 'test.txt')
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        with open(test_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
    
    for i, line in enumerate(lines[:num_examples]):
        line = line.strip()
        if not line:
            continue
        
        # 根据simple_format处理
        if simple_format:
            pos = find_third_number_position(line)
            prompt_str = line[:pos]
        else:
            prompt_str = line.split(':')[0] + ':'
        
        # 编码prompt
        prompt_tokens = prompt_str.split()
        prompt = []
        for token in prompt_tokens:
            if token in stoi:
                prompt.append(stoi[token])
        
        if len(prompt) >= 3:
            # 提取完整路径
            full_path = []
            tokens = line.split()
            for j in range(2, len(tokens)):  # 从source开始
                if tokens[j].isdigit():
                    full_path.append(int(tokens[j]))
            
            examples.append((prompt, tuple(full_path)))
    
    return examples

def track_path_probabilities(model, test_examples, device='cuda:0', top_k=10, block_size=32):
    """追踪每个测试样例的top-k路径概率"""
    model.eval()
    path_probabilities = {}
    
    for idx, (prompt, target_path) in enumerate(tqdm(test_examples, desc="Tracking paths")):
        # 准备输入
        prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
        
        # 使用beam search获取top-k路径及其概率
        paths_with_probs = beam_search_paths(model, prompt_tensor, k=top_k, block_size=block_size)
        
        # 记录结果
        path_probabilities[idx] = {
            'target_path': target_path,
            'predictions': paths_with_probs
        }
    
    return path_probabilities

def beam_search_paths(model, prompt, k=10, block_size=32):
    """使用beam search找到top-k个路径及其概率"""
    device = prompt.device
    prompt_len = prompt.shape[1]
    max_new_tokens = min(block_size - prompt_len - 1, 20)  # 限制最大生成长度
    
    # 初始化beam
    beams = [(prompt, 0.0, False)]  # (sequence, log_prob, finished)
    finished_beams = []
    
    for step in range(max_new_tokens):
        if not beams:
            break
            
        new_beams = []
        for seq, log_prob, finished in beams:
            if finished:
                finished_beams.append((seq, log_prob))
                continue
            
            # 如果序列已经接近最大长度，强制结束
            if seq.shape[1] >= block_size - 1:
                finished_beams.append((seq, log_prob))
                continue
            
            # 获取下一个token的概率
            with torch.no_grad():
                logits, _ = model(seq)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
            
            # 获取top-k候选
            top_probs, top_indices = torch.topk(probs[0], min(k, probs.shape[-1]))
            
            for prob, idx in zip(top_probs, top_indices):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_log_prob = log_prob + torch.log(prob + 1e-10).item()
                
                # 检查是否结束（遇到换行符token=1）
                if idx.item() == 1:
                    finished_beams.append((new_seq, new_log_prob))
                else:
                    new_beams.append((new_seq, new_log_prob, False))
        
        # 保留top-k个beam
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:k]
    
    # 合并finished和ongoing beams
    all_beams = finished_beams + [(seq, log_prob) for seq, log_prob, _ in beams]
    all_beams.sort(key=lambda x: x[1], reverse=True)
    
    # 转换为路径和概率
    results = []
    for seq, log_prob in all_beams[:k]:
        path = extract_path_from_tokens(seq[0].cpu().numpy())
        prob = float(np.exp(log_prob))
        results.append((path, prob))
    
    return results

def analyze_distribution_shift(all_checkpoints_data):
    """分析分布如何变化"""
    results = {}
    
    for ckpt, data in all_checkpoints_data.items():
        # 统计每条路径的出现频率
        path_counts = defaultdict(int)
        training_path_probs = []
        alternative_path_probs = []
        
        for example_data in data.values():
            target = example_data['target_path']
            predictions = example_data['predictions']
            
            if not predictions:
                continue
                
            # 找出训练路径的概率
            target_prob = 0
            for path, prob in predictions:
                if path == target:
                    target_prob = prob
                    training_path_probs.append(prob)
                    break
            
            # 如果训练路径不在top-k中，概率为0
            if target_prob == 0:
                training_path_probs.append(0)
            
            # 记录最高概率的替代路径
            for path, prob in predictions:
                if path != target:
                    alternative_path_probs.append(prob)
                    break
            
            # 统计路径
            if predictions:
                top_path = predictions[0][0]
                path_counts[top_path] += 1
        
        # 计算统计量
        results[ckpt] = {
            'mean_training_path_prob': float(np.mean(training_path_probs)) if training_path_probs else 0.0,
            'mean_alternative_path_prob': float(np.mean(alternative_path_probs)) if alternative_path_probs else 0.0,
            'num_unique_paths': int(len(path_counts)),
            'entropy': float(calculate_entropy(list(path_counts.values()))),
            'top_path_frequency': float(max(path_counts.values()) / sum(path_counts.values())) if path_counts else 0.0,
            'training_path_in_top': float(len([p for p in training_path_probs if p > 0]) / len(training_path_probs)) if training_path_probs else 0.0
        }
    
    return results

def calculate_entropy(counts):
    """计算熵"""
    total = sum(counts)
    if total == 0:
        return 0
    probs = [c/total for c in counts]
    return -sum(p * np.log(p + 1e-10) for p in probs if p > 0)

def main():
    # 配置
    checkpoints = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    base_dir = 'out/simple_graph_1_1_120_100_original_seed42'
    data_path = 'data/simple_graph/100'
    output_dir = 'analysis_results/path_distribution'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载meta信息
    meta = load_meta(data_path)
    
    # 准备测试样例
    test_examples = load_test_examples(data_path, meta, num_examples=100)
    print(f"Loaded {len(test_examples)} test examples")
    
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
    
    # 保存结果（转换为可序列化格式）
    serializable_results = {}
    for ckpt, data in all_results.items():
        serializable_results[ckpt] = {}
        for idx, example_data in data.items():
            serializable_results[ckpt][idx] = {
                'target_path': list(example_data['target_path']),
                'predictions': [[list(path) if isinstance(path, tuple) else path, float(prob)] 
                              for path, prob in example_data['predictions']]
            }
    
    with open(os.path.join(output_dir, 'path_probabilities.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    with open(os.path.join(output_dir, 'distribution_shifts.json'), 'w') as f:
        json.dump(distribution_shifts, f, indent=2)
    
    # 打印一些统计信息
    print("\nDistribution shift summary:")
    for ckpt in sorted(distribution_shifts.keys()):
        stats = distribution_shifts[ckpt]
        print(f"\nCheckpoint {ckpt}:")
        print(f"  Training path prob: {stats['mean_training_path_prob']:.4f}")
        print(f"  Alternative path prob: {stats['mean_alternative_path_prob']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")
    
    # 创建可视化
    create_distribution_plots(distribution_shifts, output_dir)
    
    print(f"\nPath distribution analysis complete! Results saved to {output_dir}/")

def create_distribution_plots(results, output_dir):
    """创建分布变化图"""
    if not results:
        print("No results to plot")
        return
        
    checkpoints = sorted(results.keys())
    
    # 训练路径vs替代路径概率
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个子图：路径概率
    training_probs = [results[ckpt]['mean_training_path_prob'] for ckpt in checkpoints]
    alternative_probs = [results[ckpt]['mean_alternative_path_prob'] for ckpt in checkpoints]
    
    ax1.plot(checkpoints, training_probs, marker='o', label='Training Path Prob', linewidth=2, markersize=8)
    ax1.plot(checkpoints, alternative_probs, marker='s', label='Best Alternative Prob', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Average Probability')
    ax1.set_title('Training Path vs Alternative Path Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 标记相变点
    if 140000 in checkpoints:
        ax1.axvline(x=140000, color='red', linestyle='--', alpha=0.5, label='Phase Transition')
    
    # 第二个子图：熵的变化
    entropies = [results[ckpt]['entropy'] for ckpt in checkpoints]
    
    ax2.plot(checkpoints, entropies, marker='o', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Output Distribution Entropy')
    ax2.grid(True, alpha=0.3)
    
    if 140000 in checkpoints:
        ax2.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_probability_evolution.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()