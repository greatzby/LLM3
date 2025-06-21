"""
错误模式分析 - 分析不同类型错误的演化
运行方式: python error_pattern_analysis.py --num_nodes 100
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
import networkx as nx
from model import GPTConfig, GPT
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoints', type=int, nargs='+', 
                       default=[0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000])
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def decode_sequence(token_ids, itos):
    """解码token序列"""
    decoded = []
    for tid in token_ids:
        if tid == 1:  # 换行符
            break
        elif tid in itos:
            decoded.append(itos[tid])
    return ' '.join(decoded)

def categorize_error(prediction, ground_truth, graph):
    """将错误分类"""
    # 提取数字
    pred_tokens = prediction.strip().split()
    gt_tokens = ground_truth.strip().split()
    
    pred_path = []
    for t in pred_tokens:
        if t.isdigit():
            pred_path.append(t)
        elif t == '\n':
            break
    
    gt_path = []
    for t in gt_tokens:
        if t.isdigit():
            gt_path.append(t)
    
    # 检查语法错误
    if len(pred_path) < 4:
        return 'syntax_error'
    
    # 检查source和target
    if len(gt_path) >= 2 and len(pred_path) >= 2:
        if pred_path[0] != gt_path[0] or pred_path[1] != gt_path[1]:
            return 'wrong_source_target'
    
    # 检查起始/结束匹配
    if len(pred_path) >= 4:
        if pred_path[2] != pred_path[0]:
            return 'start_mismatch'
        if pred_path[-1] != pred_path[1]:
            return 'end_mismatch'
    
    # 检查路径有效性 - 注意GraphML中节点是字符串
    invalid_edges = 0
    for i in range(2, len(pred_path) - 1):
        # 将字符串节点ID传给has_edge检查
        if not graph.has_edge(pred_path[i], pred_path[i + 1]):
            invalid_edges += 1
    
    if invalid_edges > 0:
        return 'invalid_edge'
    
    # 检查是否是正确路径
    if pred_path == gt_path:
        return 'correct'
    else:
        return 'valid_but_wrong'

def analyze_errors(model, test_data, graph, device, meta, args):
    """分析模型的错误模式"""
    model.eval()
    itos = meta['itos']
    
    error_counts = {
        'correct': 0,
        'syntax_error': 0,
        'wrong_source_target': 0,
        'start_mismatch': 0,
        'end_mismatch': 0,
        'invalid_edge': 0,
        'valid_but_wrong': 0
    }
    
    # 记录一些错误示例
    error_examples = {k: [] for k in error_counts.keys()}
    
    # 采样测试
    samples = min(args.num_samples, len(test_data))
    sampled_indices = np.random.choice(len(test_data), samples, replace=False)
    
    for idx in tqdm(sampled_indices, desc="Analyzing errors"):
        prompt, ground_truth = test_data[idx]
        
        # 生成预测
        with torch.no_grad():
            prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
            generated = model.generate(prompt_tensor, 
                                     max_new_tokens=args.max_new_tokens, 
                                     temperature=args.temperature,
                                     top_k=args.num_nodes + 2)  # vocab_size
        
        # 解码预测
        prediction = decode_sequence(generated[0].tolist(), itos)
        
        # 分类错误
        error_type = categorize_error(prediction, ground_truth, graph)
        error_counts[error_type] += 1
        
        # 保存错误示例（最多5个）
        if len(error_examples[error_type]) < 5 and error_type != 'correct':
            error_examples[error_type].append({
                'prediction': prediction,
                'ground_truth': ground_truth
            })
    
    # 转换为百分比
    total = sum(error_counts.values())
    error_percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in error_counts.items()}
    
    return error_percentages, error_examples

def load_test_data(data_path, meta):
    """加载测试数据"""
    stoi = meta['stoi']
    test_data = []
    
    try:
        with open(f'{data_path}/test.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        with open(f'{data_path}/test.txt', 'r', encoding='gbk') as f:
            lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        tokens = line.split()
        if len(tokens) >= 4:
            # 提取prompt（前3个token）
            prompt = []
            for i in range(3):
                if i < len(tokens) and tokens[i] in stoi:
                    prompt.append(stoi[tokens[i]])
            
            if len(prompt) == 3:
                test_data.append((prompt, line))
    
    return test_data

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'error_patterns')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    dataset = 'simple_graph'
    config = '1_1_120'
    data_path = f'data/{dataset}/{args.num_nodes}'
    
    # 加载图
    graph_path = f'{data_path}/path_graph.graphml'
    graph = nx.read_graphml(graph_path)
    
    # 加载meta信息
    meta_path = f'{data_path}/meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # 准备测试数据
    print("Loading test data...")
    test_data = load_test_data(data_path, meta)
    print(f"Loaded {len(test_data)} test samples")
    
    # 分析每个checkpoint
    results = {}
    all_examples = {}
    
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
            
            # 分析错误
            error_percentages, error_examples = analyze_errors(
                model, test_data, graph, args.device, meta, args)
            
            results[ckpt_iter] = error_percentages
            all_examples[ckpt_iter] = error_examples
            
            # 打印当前结果
            print(f"Checkpoint {ckpt_iter} results:")
            for error_type, percentage in error_percentages.items():
                print(f"  {error_type}: {percentage:.2f}%")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt_iter}: {e}")
            continue
    
    # 保存结果
    with open(os.path.join(output_dir, 'error_pattern_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, 'error_examples.json'), 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    # 创建可视化
    create_error_evolution_plot(results, output_dir)
    
    print(f"\nError pattern analysis complete! Results saved to {output_dir}/")

def create_error_evolution_plot(results, output_dir):
    """创建错误演化图"""
    if not results:
        print("No results to plot")
        return
    
    checkpoints = sorted(results.keys())
    error_types = ['syntax_error', 'start_mismatch', 'end_mismatch', 
                   'invalid_edge', 'valid_but_wrong', 'wrong_source_target']
    
    # 错误类型演化线图
    plt.figure(figsize=(12, 8))
    
    for error_type in error_types:
        percentages = [results[ckpt].get(error_type, 0) for ckpt in checkpoints]
        if any(p > 0 for p in percentages):  # 只绘制出现过的错误类型
            plt.plot(checkpoints, percentages, marker='o', label=error_type.replace('_', ' ').title())
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Error Percentage (%)')
    plt.title('Evolution of Error Types During Training')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_evolution.png'), dpi=150)
    plt.close()
    
    # 创建堆叠条形图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 准备数据
    bottom = np.zeros(len(checkpoints))
    colors = plt.cm.Set3(np.linspace(0, 1, len(error_types) + 1))
    
    for i, error_type in enumerate(error_types):
        percentages = [results[ckpt].get(error_type, 0) for ckpt in checkpoints]
        ax.bar(range(len(checkpoints)), percentages, bottom=bottom, 
               label=error_type.replace('_', ' ').title(), 
               color=colors[i], width=0.8)
        bottom += percentages
    
    # 添加正确率
    correct_percentages = [results[ckpt].get('correct', 0) for ckpt in checkpoints]
    ax.bar(range(len(checkpoints)), correct_percentages, bottom=bottom, 
           label='Correct', color=colors[-1], width=0.8)
    
    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([f'{ckpt//1000}k' for ckpt in checkpoints])
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Error Type Distribution Over Training')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution_stacked.png'), dpi=150)
    plt.close()
    
    # 正确率演化
    plt.figure(figsize=(10, 6))
    
    correct_percentages = [results[ckpt].get('correct', 0) for ckpt in checkpoints]
    plt.plot(checkpoints, correct_percentages, marker='o', linewidth=2, markersize=8, color='green')
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Correct Percentage (%)')
    plt.title('Model Accuracy Evolution')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    # 添加数值标签
    for i, (ckpt, pct) in enumerate(zip(checkpoints, correct_percentages)):
        if i % 2 == 0:  # 每隔一个标注，避免重叠
            plt.annotate(f'{pct:.1f}%', 
                        xy=(ckpt, pct), 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_evolution.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()