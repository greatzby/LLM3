"""
快速修复版 - 处理推理时只返回最后一个位置logits的问题
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import networkx as nx

from model import GPTConfig, GPT

# ==================== 基础工具函数（保持不变）====================

def load_model(checkpoint_path):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('cuda:0')
    return model

def load_meta(data_path):
    """加载meta信息"""
    with open(f"{data_path}/meta.pkl", 'rb') as f:
        return pickle.load(f)

def load_graph(data_dir):
    """加载图结构"""
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    return nx.read_graphml(graph_path)

def extract_path_nodes(tokens):
    """从token序列提取节点序列"""
    nodes = []
    for t in tokens:
        if t == 1:  # 换行符
            break
        if 2 <= t <= 101:  # 节点范围
            nodes.append(t - 2)
    return nodes

def check_path_validity(graph, path):
    """检查路径是否有效"""
    if len(path) < 2:
        return False
    for i in range(len(path) - 1):
        if not graph.has_edge(str(path[i]), str(path[i+1])):
            return False
    return True

# ==================== 修复的分析函数 ====================

def get_full_logits(model, input_tensor, device='cuda:0'):
    """获取完整的logits（所有位置），通过传入dummy targets"""
    # 创建dummy targets来强制模型返回所有位置的logits
    dummy_targets = torch.zeros_like(input_tensor)
    
    with torch.no_grad():
        logits, _ = model(input_tensor, targets=dummy_targets)
    
    return logits

def analyze_position_3_distribution_fixed(model, test_data, device='cuda:0'):
    """修复版：分析位置3的分布"""
    position_3_stats = []
    
    model.eval()
    for prompt, _, _ in tqdm(test_data[:200], desc="Analyzing position 3"):
        prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
        
        # 获取完整的logits
        logits = get_full_logits(model, prompt_tensor, device)
        
        # 现在可以安全地访问位置2的logits（预测位置3）
        if logits.shape[1] >= 3:
            pos2_logits = logits[0, 2, :]
        else:
            # 如果序列太短，跳过
            continue
            
        probs = F.softmax(pos2_logits, dim=-1)
        
        # 计算统计量
        top1_prob = probs.max().item()
        top1_idx = probs.argmax().item()
        entropy_val = entropy(probs.cpu().numpy())
        
        # 计算top-5
        top5_probs, top5_indices = torch.topk(probs, min(5, probs.shape[0]))
        top5_cumsum = top5_probs.sum().item()
        
        # 计算有效节点（2-101）的概率总和
        valid_node_prob = probs[2:102].sum().item() if probs.shape[0] > 101 else 0
        
        position_3_stats.append({
            'top1_prob': top1_prob,
            'top1_token': top1_idx,
            'entropy': entropy_val,
            'top5_cumsum': top5_cumsum,
            'valid_node_prob': valid_node_prob,
            'is_valid_node': 2 <= top1_idx <= 101
        })
    
    if not position_3_stats:
        return {
            'position_3_top1': 0.0,
            'position_3_entropy': 0.0,
            'position_3_top5_cumsum': 0.0,
            'position_3_valid_node_prob': 0.0,
            'position_3_valid_prediction_rate': 0.0
        }
    
    df = pd.DataFrame(position_3_stats)
    return {
        'position_3_top1': float(df['top1_prob'].mean()),
        'position_3_entropy': float(df['entropy'].mean()),
        'position_3_top5_cumsum': float(df['top5_cumsum'].mean()),
        'position_3_valid_node_prob': float(df['valid_node_prob'].mean()),
        'position_3_valid_prediction_rate': float(df['is_valid_node'].mean())
    }

def analyze_teacher_forcing_accuracy_fixed(model, data_dir, meta, device='cuda:0', num_samples=500):
    """使用训练数据计算真实的TF准确率"""
    # 加载训练数据的一部分
    train_file = os.path.join(data_dir, 'train_20.txt')
    
    sequences = []
    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if line:
                tokens = line.split()
                sequence = [meta['stoi'][t] for t in tokens]
                sequences.append(sequence)
    
    total_correct = 0
    total_tokens = 0
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    model.eval()
    
    for seq in tqdm(sequences, desc="Computing TF accuracy"):
        if len(seq) < 2:
            continue
            
        # Teacher forcing: 输入[:-1]，预测[1:]
        input_seq = torch.tensor(seq[:-1], device=device).unsqueeze(0)
        target_seq = torch.tensor(seq[1:], device=device).unsqueeze(0)
        
        # 获取完整logits
        logits = get_full_logits(model, input_seq, device)
        predictions = logits.argmax(dim=-1)
        
        # 计算准确率
        correct = (predictions == target_seq).float()
        
        # 总体准确率
        total_correct += correct.sum().item()
        total_tokens += correct.numel()
        
        # 位置准确率（特别关注位置3）
        for pos in range(min(correct.shape[1], 10)):  # 只看前10个位置
            position_correct[pos] += correct[0, pos].item()
            position_total[pos] += 1
    
    # 计算结果
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    position_accuracy = {
        pos: position_correct[pos] / position_total[pos] 
        for pos in position_total if position_total[pos] > 0
    }
    
    return {
        'tf_accuracy': overall_accuracy,
        'position_accuracy': position_accuracy,
        'total_tokens_evaluated': total_tokens,
        'position_3_accuracy': position_accuracy.get(3, 0.0)  # 特别提取位置3
    }

# ==================== 简化的主分析函数 ====================

def run_simple_analysis(checkpoint_list, data_dir, output_dir):
    """运行简化的分析，专注于核心指标"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("Loading data...")
    meta = load_meta(data_dir)
    graph = load_graph(data_dir)
    
    # 加载测试数据
    test_data = []
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split()
                prompt = [meta['stoi'][t] for t in tokens[:3]]
                test_data.append((prompt, line, None))
    
    print(f"Loaded {len(test_data)} test examples")
    
    # 存储结果
    results_summary = {
        'checkpoints': [],
        'tf_accuracy': [],
        'position_3_accuracy': [],
        'position_3_entropy': [],
        'position_3_top1_prob': [],
        'valid_path_rate': []
    }
    
    for ckpt in checkpoint_list:
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint {ckpt}")
        print('='*60)
        
        ckpt_path = f'out/simple_graph_1_1_120_100_original_seed42/{ckpt}_ckpt_20.pt'
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found, skipping...")
            continue
        
        try:
            model = load_model(ckpt_path)
            
            # 1. Teacher Forcing准确率
            print("Computing TF accuracy...")
            tf_results = analyze_teacher_forcing_accuracy_fixed(model, data_dir, meta)
            
            # 2. Position 3分析
            print("Analyzing position 3...")
            pos3_results = analyze_position_3_distribution_fixed(model, test_data)
            
            # 3. 快速路径有效性检查
            print("Checking path validity...")
            valid_paths = 0
            for i, (prompt, _, _) in enumerate(test_data[:100]):
                if i >= 100:
                    break
                prompt_tensor = torch.tensor(prompt, device='cuda:0').unsqueeze(0)
                with torch.no_grad():
                    generated = model.generate(prompt_tensor, max_new_tokens=20, temperature=1.0)
                gen_path = extract_path_nodes(generated[0].cpu().tolist()[3:])
                if check_path_validity(graph, gen_path):
                    valid_paths += 1
            
            valid_path_rate = valid_paths / 100
            
            # 保存结果
            results_summary['checkpoints'].append(ckpt)
            results_summary['tf_accuracy'].append(tf_results['tf_accuracy'])
            results_summary['position_3_accuracy'].append(tf_results.get('position_3_accuracy', 0))
            results_summary['position_3_entropy'].append(pos3_results['position_3_entropy'])
            results_summary['position_3_top1_prob'].append(pos3_results['position_3_top1'])
            results_summary['valid_path_rate'].append(valid_path_rate)
            
            # 打印结果
            print(f"\nResults for checkpoint {ckpt}:")
            print(f"  TF accuracy: {tf_results['tf_accuracy']:.3f}")
            print(f"  Position 3 accuracy: {tf_results.get('position_3_accuracy', 0):.3f}")
            print(f"  Position 3 entropy: {pos3_results['position_3_entropy']:.3f}")
            print(f"  Position 3 top1 prob: {pos3_results['position_3_top1']:.3f}")
            print(f"  Valid path rate: {valid_path_rate:.3f}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存和可视化
    with open(os.path.join(output_dir, 'simple_analysis_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    create_simple_visualization(results_summary, output_dir)
    
    return results_summary

def create_simple_visualization(results, output_dir):
    """创建简单的可视化"""
    if not results['checkpoints']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    checkpoints = results['checkpoints']
    
    # 1. TF准确率
    ax = axes[0, 0]
    ax.plot(checkpoints, results['tf_accuracy'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Teacher Forcing Accuracy')
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5, label='Expected transition')
        ax.legend()
    
    # 2. Position 3准确率
    ax = axes[0, 1]
    ax.plot(checkpoints, results['position_3_accuracy'], 'o-', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Position 3 (First Generated Token) Accuracy')
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    # 3. Position 3熵和置信度
    ax = axes[1, 0]
    ax2 = ax.twinx()
    line1 = ax.plot(checkpoints, results['position_3_entropy'], 'o-', color='blue', label='Entropy')
    line2 = ax2.plot(checkpoints, results['position_3_top1_prob'], 's-', color='orange', label='Top1 Prob')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Entropy', color='blue')
    ax2.set_ylabel('Top1 Probability', color='orange')
    ax.set_title('Position 3 Distribution')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. 有效路径率
    ax = axes[1, 1]
    ax.plot(checkpoints, results['valid_path_rate'], 'o-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Valid Path Rate')
    ax.set_title('Autoregressive Valid Path Generation')
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Phase Transition Analysis - Key Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simple_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

def convert_to_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ==================== 主函数 ====================

def main():
    """主函数"""
    checkpoint_list = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    data_dir = 'data/simple_graph/100'
    output_dir = 'analysis_results/quick_fix'
    
    print("Starting quick fix analysis...")
    print("This version handles the inference-mode logits issue")
    print(f"Output directory: {output_dir}")
    
    results = run_simple_analysis(checkpoint_list, data_dir, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    # 检查是否有相变
    if len(results['checkpoints']) > 1:
        tf_acc = results['tf_accuracy']
        max_drop_idx = np.argmax([tf_acc[i-1] - tf_acc[i] for i in range(1, len(tf_acc))]) + 1
        if max_drop_idx < len(results['checkpoints']):
            max_drop = tf_acc[max_drop_idx-1] - tf_acc[max_drop_idx]
            if max_drop > 0.1:
                print(f"\nPHASE TRANSITION DETECTED!")
                print(f"Location: {results['checkpoints'][max_drop_idx]}")
                print(f"TF accuracy drop: {max_drop:.1%}")

if __name__ == "__main__":
    main()