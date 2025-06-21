"""
修正版相变分析脚本 - 正确处理位置3的分析
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

# ==================== 基础工具函数 ====================

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

# ==================== 修正的核心分析函数 ====================

def analyze_position_3_distribution(model, test_data, device='cuda:0'):
    """专门分析位置3（第一个生成位置）的分布"""
    position_3_stats = []
    
    model.eval()
    with torch.no_grad():
        for prompt, _, _ in tqdm(test_data[:200], desc="Analyzing position 3"):
            prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
            
            # 运行模型获取prompt位置的logits
            logits, _ = model(prompt_tensor)
            
            # 获取位置2（最后一个prompt token）的logits，这预测位置3的token
            pos2_logits = logits[0, 2, :]  # 这是预测下一个token（位置3）的logits
            probs = F.softmax(pos2_logits, dim=-1)
            
            # 计算统计量
            top1_prob = probs.max().item()
            top1_idx = probs.argmax().item()
            entropy_val = entropy(probs.cpu().numpy())
            
            # 计算top-5
            top5_probs, top5_indices = torch.topk(probs, 5)
            top5_cumsum = top5_probs.sum().item()
            
            # 计算有效节点（2-101）的概率总和
            valid_node_prob = probs[2:102].sum().item()
            
            position_3_stats.append({
                'top1_prob': top1_prob,
                'top1_token': top1_idx,
                'entropy': entropy_val,
                'top5_cumsum': top5_cumsum,
                'valid_node_prob': valid_node_prob,
                'is_valid_node': 2 <= top1_idx <= 101
            })
    
    df = pd.DataFrame(position_3_stats)
    return {
        'position_3_top1': float(df['top1_prob'].mean()),
        'position_3_entropy': float(df['entropy'].mean()),
        'position_3_top5_cumsum': float(df['top5_cumsum'].mean()),
        'position_3_valid_node_prob': float(df['valid_node_prob'].mean()),
        'position_3_valid_prediction_rate': float(df['is_valid_node'].mean())
    }

def analyze_path_generation_detailed(model, test_data, graph, meta, device='cuda:0'):
    """详细分析路径生成，包括格式问题"""
    results = {
        'format_matches': 0,  # 包含source的完整格式
        'skip_source_matches': 0,  # 跳过source的格式
        'valid_paths': 0,
        'correct_target': 0,
        'total': len(test_data),
        'path_examples': [],
        'token_level_accuracy': 0,
        'position_wise_accuracy': defaultdict(list)
    }
    
    itos = meta['itos']
    
    for idx, (prompt, target_str, _) in enumerate(tqdm(test_data[:100], desc="Detailed path analysis")):
        # 解析目标
        tokens = target_str.strip().split()
        source = int(tokens[0])
        target = int(tokens[1])
        full_target_path = [int(t) for t in tokens[2:]]  # 包含source的完整路径
        skip_source_path = full_target_path[1:] if full_target_path[0] == source else full_target_path
        
        # 生成
        prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_new_tokens=20, temperature=1.0, top_k=None)
        
        gen_tokens = generated[0].cpu().tolist()
        gen_path = extract_path_nodes(gen_tokens[3:])  # 跳过prompt部分
        
        # 检查不同的匹配类型
        format_match = (gen_path == full_target_path)
        skip_source_match = (gen_path == skip_source_path)
        is_valid = check_path_validity(graph, gen_path)
        reaches_target = len(gen_path) > 0 and gen_path[-1] == target
        
        if format_match:
            results['format_matches'] += 1
        if skip_source_match:
            results['skip_source_matches'] += 1
        if is_valid:
            results['valid_paths'] += 1
        if reaches_target:
            results['correct_target'] += 1
        
        # 计算token级准确率
        target_tokens = [meta['stoi'][t] for t in tokens]
        for i, (pred, tgt) in enumerate(zip(gen_tokens[3:], target_tokens[3:])):
            if i < 20:  # 限制长度
                results['position_wise_accuracy'][i].append(pred == tgt)
        
        # 保存例子
        if idx < 5:
            results['path_examples'].append({
                'source': source,
                'target': target,
                'full_target_path': full_target_path,
                'skip_source_path': skip_source_path,
                'generated_path': gen_path,
                'format_match': format_match,
                'skip_source_match': skip_source_match,
                'valid': is_valid,
                'reaches_target': reaches_target,
                'target_str': target_str,
                'generated_str': ' '.join([itos[t] for t in gen_tokens if t > 1])
            })
    
    # 计算位置准确率
    position_accuracy = {}
    for pos, accs in results['position_wise_accuracy'].items():
        if accs:
            position_accuracy[pos] = np.mean(accs)
    
    results['position_accuracy'] = position_accuracy
    results['format_match_rate'] = results['format_matches'] / results['total']
    results['skip_source_match_rate'] = results['skip_source_matches'] / results['total']
    results['valid_path_rate'] = results['valid_paths'] / results['total']
    results['target_reach_rate'] = results['correct_target'] / results['total']
    
    return results

def analyze_teacher_forcing_accuracy(model, test_data, meta, device='cuda:0', num_batches=10):
    """准确计算Teacher Forcing准确率"""
    total_correct = 0
    total_tokens = 0
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    model.eval()
    
    for batch_idx in range(num_batches):
        # 随机采样
        indices = np.random.choice(len(test_data), size=64, replace=False)
        
        for idx in indices:
            prompt, target_str, _ = test_data[idx]
            
            # 准备完整的输入序列（用于teacher forcing）
            tokens = target_str.strip().split()
            full_sequence = [meta['stoi'][t] for t in tokens]
            
            # Teacher forcing: 输入[:-1]，预测[1:]
            input_seq = torch.tensor(full_sequence[:-1], device=device).unsqueeze(0)
            target_seq = torch.tensor(full_sequence[1:], device=device).unsqueeze(0)
            
            with torch.no_grad():
                logits, _ = model(input_seq)
                predictions = logits.argmax(dim=-1)
                
                # 计算准确率
                correct = (predictions == target_seq).float()
                
                # 总体准确率
                total_correct += correct.sum().item()
                total_tokens += correct.numel()
                
                # 位置准确率
                for pos in range(correct.shape[1]):
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
        'total_tokens_evaluated': total_tokens
    }

# ==================== 主分析函数 ====================

def run_complete_analysis_fixed(checkpoint_list, data_dir, output_dir):
    """运行修正后的完整分析"""
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
    
    # 分析每个checkpoint
    all_results = {}
    
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
            
            # 运行各项分析
            print("1. Analyzing path generation...")
            path_results = analyze_path_generation_detailed(model, test_data, graph, meta)
            
            print("2. Analyzing position 3 distribution...")
            pos3_results = analyze_position_3_distribution(model, test_data)
            
            print("3. Analyzing teacher forcing accuracy...")
            tf_results = analyze_teacher_forcing_accuracy(model, test_data, meta)
            
            # 合并结果
            results = {
                'checkpoint': ckpt,
                'path_analysis': path_results,
                'position_3_analysis': pos3_results,
                'teacher_forcing': tf_results
            }
            
            all_results[ckpt] = results
            
            # 打印关键结果
            print(f"\nKey Results for checkpoint {ckpt}:")
            print(f"  Format match rate: {path_results['format_match_rate']:.3f}")
            print(f"  Skip-source match rate: {path_results['skip_source_match_rate']:.3f}")
            print(f"  Valid path rate: {path_results['valid_path_rate']:.3f}")
            print(f"  Position 3 top1 prob: {pos3_results['position_3_top1']:.3f}")
            print(f"  Position 3 entropy: {pos3_results['position_3_entropy']:.3f}")
            print(f"  Teacher forcing accuracy: {tf_results['tf_accuracy']:.3f}")
            
            # 打印路径例子
            if path_results['path_examples']:
                print("\nPath generation examples:")
                for i, ex in enumerate(path_results['path_examples'][:2]):
                    print(f"  Example {i+1}:")
                    print(f"    Target: {ex['target_str']}")
                    print(f"    Generated: {ex['generated_str']}")
                    print(f"    Skip-source match: {ex['skip_source_match']}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    with open(os.path.join(output_dir, 'fixed_analysis_results.json'), 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    # 创建可视化
    create_fixed_visualization(all_results, output_dir)
    
    return all_results

def create_fixed_visualization(results, output_dir):
    """创建修正后的可视化"""
    if not results:
        return
    
    checkpoints = sorted(results.keys())
    
    # 准备数据
    format_match = [results[c]['path_analysis']['format_match_rate'] for c in checkpoints]
    skip_source_match = [results[c]['path_analysis']['skip_source_match_rate'] for c in checkpoints]
    valid_path = [results[c]['path_analysis']['valid_path_rate'] for c in checkpoints]
    
    pos3_top1 = [results[c]['position_3_analysis']['position_3_top1'] for c in checkpoints]
    pos3_entropy = [results[c]['position_3_analysis']['position_3_entropy'] for c in checkpoints]
    
    tf_accuracy = [results[c]['teacher_forcing']['tf_accuracy'] for c in checkpoints]
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 路径匹配分析
    ax = axes[0, 0]
    ax.plot(checkpoints, format_match, 'o-', label='Format Match', linewidth=2)
    ax.plot(checkpoints, skip_source_match, 's-', label='Skip-source Match', linewidth=2)
    ax.plot(checkpoints, valid_path, '^-', label='Valid Path', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Rate')
    ax.set_title('Path Generation Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    # 2. Position 3 分析
    ax = axes[0, 1]
    ax.plot(checkpoints, pos3_top1, 'o-', color='green', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Top-1 Probability')
    ax.set_title('Position 3 (First Generated Token) Confidence')
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    # 3. Position 3 熵
    ax = axes[0, 2]
    ax.plot(checkpoints, pos3_entropy, 'o-', color='purple', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Entropy')
    ax.set_title('Position 3 Distribution Entropy')
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    # 4. Teacher Forcing准确率
    ax = axes[1, 0]
    ax.plot(checkpoints, tf_accuracy, 'o-', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Teacher Forcing Accuracy')
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5, label='Phase Transition')
        ax.legend()
    
    # 5. TF vs Path Match对比
    ax = axes[1, 1]
    ax.plot(checkpoints, tf_accuracy, 'o-', label='TF Accuracy', linewidth=2)
    ax.plot(checkpoints, skip_source_match, 's-', label='Path Match (skip-source)', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Rate')
    ax.set_title('TF Accuracy vs Path Matching')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    # 6. 相变指标
    ax = axes[1, 2]
    if len(checkpoints) > 1:
        tf_diff = np.diff(tf_accuracy)
        ax.plot(checkpoints[1:], tf_diff, 'o-', color='red', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('TF Accuracy Change')
        ax.set_title('Phase Transition Detection')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Fixed Phase Transition Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fixed_analysis.png'), dpi=150, bbox_inches='tight')
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
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

# ==================== 主函数 ====================

def main():
    """主函数"""
    # 配置
    checkpoint_list = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    data_dir = 'data/simple_graph/100'
    output_dir = 'analysis_results/fixed_phase_transition'
    
    print("Starting FIXED phase transition analysis...")
    print(f"This version correctly analyzes position 3 and path formats")
    print(f"Output directory: {output_dir}")
    
    # 运行分析
    results = run_complete_analysis_fixed(checkpoint_list, data_dir, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()