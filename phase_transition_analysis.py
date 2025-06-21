"""
完整的相变分析脚本 - 一次性运行所有分析
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

# 导入你的模型和工具
from model import GPTConfig, GPT
from utils import load_meta, convert_to_serializable

# ==================== 基础工具函数 ====================

def load_model(checkpoint_path):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    
    # 从checkpoint提取配置
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # 加载权重
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('cuda:0')
    
    return model

def load_graph(data_dir):
    """加载图结构"""
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    return nx.read_graphml(graph_path)

def decode_tokens(tokens, itos):
    """解码token序列"""
    decoded = []
    for t in tokens:
        if t == 1:  # 换行符
            break
        if t in itos:
            decoded.append(itos[t])
    return ' '.join(decoded)

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

# ==================== 核心分析函数 ====================

def analyze_path_generation(model, test_data, graph, device='cuda:0'):
    """分析路径生成的详细情况"""
    results = {
        'exact_matches': 0,
        'valid_alternatives': 0,
        'correct_target': 0,
        'invalid_paths': 0,
        'total': len(test_data),
        'path_examples': []
    }
    
    for idx, (prompt, target_str, _) in enumerate(tqdm(test_data[:100], desc="Analyzing paths")):
        # 解析目标
        tokens = target_str.strip().split()
        source = int(tokens[0])
        target = int(tokens[1])
        target_path = [int(t) for t in tokens[2:]]
        
        # 生成
        prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_new_tokens=20, temperature=1.0, top_k=None)
        
        # 提取生成的路径
        gen_tokens = generated[0].cpu().tolist()
        gen_path = extract_path_nodes(gen_tokens[3:])  # 跳过prompt部分
        
        # 分析
        is_exact_match = (gen_path == target_path)
        reaches_target = len(gen_path) > 0 and gen_path[-1] == target
        is_valid = check_path_validity(graph, gen_path)
        
        if is_exact_match:
            results['exact_matches'] += 1
        if is_valid and not is_exact_match:
            results['valid_alternatives'] += 1
        if reaches_target:
            results['correct_target'] += 1
        if not is_valid:
            results['invalid_paths'] += 1
            
        # 保存前10个例子用于调试
        if idx < 10:
            results['path_examples'].append({
                'source': source,
                'target': target,
                'target_path': target_path,
                'generated_path': gen_path,
                'exact_match': is_exact_match,
                'valid': is_valid,
                'reaches_target': reaches_target
            })
    
    # 计算比率
    results['exact_match_rate'] = results['exact_matches'] / results['total']
    results['valid_alternative_rate'] = results['valid_alternatives'] / results['total']
    results['target_reach_rate'] = results['correct_target'] / results['total']
    results['invalid_rate'] = results['invalid_paths'] / results['total']
    
    return results

def analyze_attention_patterns(model, test_data, device='cuda:0'):
    """分析注意力模式的变化"""
    attention_stats = {
        'entropy': [],
        'max_attention': [],
        'attention_std': []
    }
    
    model.eval()
    with torch.no_grad():
        for prompt, _, _ in tqdm(test_data[:50], desc="Analyzing attention"):
            prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
            
            # 获取中间层的注意力权重
            # 注意：这需要修改model.forward()来返回注意力权重
            # 这里我们用一个简化的方法
            logits, _ = model(prompt_tensor)
            
            # 获取最后一个token的logits分布
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            # 计算统计量
            attention_stats['entropy'].append(entropy(probs.cpu().numpy()))
            attention_stats['max_attention'].append(probs.max().item())
            attention_stats['attention_std'].append(probs.std().item())
    
    return {
        'mean_entropy': np.mean(attention_stats['entropy']),
        'std_entropy': np.std(attention_stats['entropy']),
        'mean_max_attention': np.mean(attention_stats['max_attention']),
        'mean_attention_std': np.mean(attention_stats['attention_std'])
    }

def analyze_embedding_dynamics(model):
    """分析embedding的动态变化"""
    embeddings = model.transformer.wte.weight.data.cpu().numpy()
    
    # 计算相似度矩阵
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    similarity_matrix = np.dot(normalized, normalized.T)
    
    # 只考虑节点embeddings (2-101)
    node_similarity = similarity_matrix[2:102, 2:102]
    
    return {
        'mean_similarity': float(np.mean(node_similarity[np.triu_indices_from(node_similarity, k=1)])),
        'max_similarity': float(np.max(node_similarity[np.triu_indices_from(node_similarity, k=1)])),
        'min_similarity': float(np.min(node_similarity[np.triu_indices_from(node_similarity, k=1)])),
        'similarity_std': float(np.std(node_similarity[np.triu_indices_from(node_similarity, k=1)])),
        'mean_norm': float(np.mean(norms[2:102])),
        'norm_std': float(np.std(norms[2:102]))
    }

def analyze_token_distribution(model, test_data, device='cuda:0'):
    """分析token预测分布的集中度"""
    distribution_stats = []
    
    model.eval()
    with torch.no_grad():
        for prompt, _, _ in tqdm(test_data[:100], desc="Analyzing distributions"):
            prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
            logits, _ = model(prompt_tensor)
            
            # 分析每个位置的分布
            for pos in range(logits.shape[1]):
                pos_logits = logits[0, pos, :]
                probs = F.softmax(pos_logits, dim=-1)
                
                # 计算分布统计
                top1_prob = probs.max().item()
                entropy_val = entropy(probs.cpu().numpy())
                
                # 计算top-k累积概率
                sorted_probs, _ = torch.sort(probs, descending=True)
                top5_cumsum = sorted_probs[:5].sum().item()
                
                distribution_stats.append({
                    'position': pos,
                    'top1_prob': top1_prob,
                    'entropy': entropy_val,
                    'top5_cumsum': top5_cumsum
                })
    
    df = pd.DataFrame(distribution_stats)
    return {
        'mean_top1_prob': float(df['top1_prob'].mean()),
        'mean_entropy': float(df['entropy'].mean()),
        'mean_top5_cumsum': float(df['top5_cumsum'].mean()),
        'position_3_top1': float(df[df['position'] == 3]['top1_prob'].mean()),  # 第一个决策点
        'position_3_entropy': float(df[df['position'] == 3]['entropy'].mean())
    }

# ==================== 主分析函数 ====================

def run_complete_analysis(checkpoint_list, data_dir, output_dir):
    """运行完整分析"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("Loading data...")
    meta = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
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
            # 加载模型
            model = load_model(ckpt_path)
            
            # 运行各项分析
            results = {
                'checkpoint': ckpt,
                'path_analysis': analyze_path_generation(model, test_data, graph),
                'attention_analysis': analyze_attention_patterns(model, test_data),
                'embedding_analysis': analyze_embedding_dynamics(model),
                'distribution_analysis': analyze_token_distribution(model, test_data)
            }
            
            all_results[ckpt] = results
            
            # 打印关键结果
            print(f"\nKey Results for checkpoint {ckpt}:")
            print(f"  Exact match rate: {results['path_analysis']['exact_match_rate']:.3f}")
            print(f"  Valid alternative rate: {results['path_analysis']['valid_alternative_rate']:.3f}")
            print(f"  Target reach rate: {results['path_analysis']['target_reach_rate']:.3f}")
            print(f"  Mean embedding similarity: {results['embedding_analysis']['mean_similarity']:.4f}")
            print(f"  Mean top1 probability: {results['distribution_analysis']['mean_top1_prob']:.3f}")
            print(f"  Mean entropy: {results['distribution_analysis']['mean_entropy']:.3f}")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    with open(os.path.join(output_dir, 'complete_analysis_results.json'), 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    # 创建可视化
    create_comprehensive_plots(all_results, output_dir)
    
    return all_results

def create_comprehensive_plots(results, output_dir):
    """创建综合可视化"""
    if not results:
        return
    
    checkpoints = sorted(results.keys())
    
    # 准备数据
    exact_match = [results[c]['path_analysis']['exact_match_rate'] for c in checkpoints]
    valid_alt = [results[c]['path_analysis']['valid_alternative_rate'] for c in checkpoints]
    target_reach = [results[c]['path_analysis']['target_reach_rate'] for c in checkpoints]
    
    emb_similarity = [results[c]['embedding_analysis']['mean_similarity'] for c in checkpoints]
    top1_prob = [results[c]['distribution_analysis']['mean_top1_prob'] for c in checkpoints]
    entropy_vals = [results[c]['distribution_analysis']['mean_entropy'] for c in checkpoints]
    
    # 创建4x2的子图
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    # 1. 路径匹配率
    ax = axes[0, 0]
    ax.plot(checkpoints, exact_match, 'o-', label='Exact Match', linewidth=2)
    ax.plot(checkpoints, valid_alt, 's-', label='Valid Alternative', linewidth=2)
    ax.plot(checkpoints, target_reach, '^-', label='Correct Target', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Rate')
    ax.set_title('Path Generation Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Embedding相似度
    ax = axes[0, 1]
    ax.plot(checkpoints, emb_similarity, 'o-', color='red', linewidth=2)
    ax.axhline(y=0.025, color='black', linestyle='--', alpha=0.5, label='Hypothesized threshold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Similarity')
    ax.set_title('Embedding Similarity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 分布集中度
    ax = axes[1, 0]
    ax.plot(checkpoints, top1_prob, 'o-', color='green', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Top-1 Probability')
    ax.set_title('Distribution Concentration')
    ax.grid(True, alpha=0.3)
    
    # 4. 熵变化
    ax = axes[1, 1]
    ax.plot(checkpoints, entropy_vals, 'o-', color='purple', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Prediction Entropy')
    ax.grid(True, alpha=0.3)
    
    # 5. 组合视图 - 寻找相关性
    ax = axes[2, 0]
    ax2 = ax.twinx()
    ax.plot(checkpoints, exact_match, 'b-', label='Exact Match Rate')
    ax2.plot(checkpoints, emb_similarity, 'r--', label='Embedding Similarity')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Exact Match Rate', color='b')
    ax2.set_ylabel('Embedding Similarity', color='r')
    ax.set_title('Correlation Analysis')
    ax.grid(True, alpha=0.3)
    
    # 6. 相变指标
    ax = axes[2, 1]
    # 计算变化率
    if len(checkpoints) > 1:
        exact_match_diff = np.diff(exact_match)
        emb_sim_diff = np.diff(emb_similarity)
        
        ax.plot(checkpoints[1:], exact_match_diff, 'o-', label='Δ Exact Match')
        ax.plot(checkpoints[1:], emb_sim_diff * 10, 's-', label='Δ Emb Similarity × 10')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Change Rate')
        ax.set_title('Phase Transition Indicators')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. 路径示例可视化
    ax = axes[3, 0]
    ax.text(0.1, 0.9, 'Path Examples:', transform=ax.transAxes, fontsize=12, weight='bold')
    
    # 显示几个具体例子
    if 140000 in results and len(results[140000]['path_analysis']['path_examples']) > 0:
        examples = results[140000]['path_analysis']['path_examples'][:3]
        y_pos = 0.8
        for ex in examples:
            text = f"S:{ex['source']}→T:{ex['target']}\n"
            text += f"Target: {ex['target_path'][:5]}...\n"
            text += f"Generated: {ex['generated_path'][:5]}...\n"
            text += f"Match: {ex['exact_match']}, Valid: {ex['valid']}\n"
            ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=9)
            y_pos -= 0.25
    ax.axis('off')
    
    # 8. 总结统计
    ax = axes[3, 1]
    ax.text(0.1, 0.9, 'Phase Transition Summary:', transform=ax.transAxes, fontsize=12, weight='bold')
    
    # 找出相变点
    if len(checkpoints) > 3:
        transition_idx = None
        max_drop = 0
        for i in range(1, len(exact_match)):
            drop = exact_match[i-1] - exact_match[i]
            if drop > max_drop:
                max_drop = drop
                transition_idx = i
        
        if transition_idx and max_drop > 0.1:
            text = f"Transition detected at: {checkpoints[transition_idx]}\n"
            text += f"Exact match drop: {max_drop:.3f}\n"
            text += f"Embedding similarity at transition: {emb_similarity[transition_idx]:.4f}\n"
            ax.text(0.1, 0.7, text, transform=ax.transAxes, fontsize=10)
    
    ax.axis('off')
    
    # 标记相变点
    if 140000 in checkpoints:
        for ax in axes.flat[:6]:
            if ax.lines:  # 只在有数据的图上标记
                ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/comprehensive_analysis.png")

# ==================== 主函数 ====================

def main():
    """主函数"""
    # 配置
    checkpoint_list = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    data_dir = 'data/simple_graph/100'
    output_dir = 'analysis_results/comprehensive_phase_transition'
    
    print("Starting comprehensive phase transition analysis...")
    print(f"Checkpoints to analyze: {checkpoint_list}")
    print(f"Output directory: {output_dir}")
    
    # 运行分析
    results = run_complete_analysis(checkpoint_list, data_dir, output_dir)
    
    # 打印最终总结
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    # 生成最终报告
    generate_final_report(results, output_dir)

def generate_final_report(results, output_dir):
    """生成最终的文字报告"""
    report = []
    report.append("# Phase Transition Analysis Report\n")
    report.append(f"Generated at: {pd.Timestamp.now()}\n\n")
    
    # 找出关键转换点
    checkpoints = sorted(results.keys())
    exact_matches = [results[c]['path_analysis']['exact_match_rate'] for c in checkpoints]
    
    # 检测最大下降
    max_drop = 0
    transition_point = None
    for i in range(1, len(checkpoints)):
        drop = exact_matches[i-1] - exact_matches[i]
        if drop > max_drop:
            max_drop = drop
            transition_point = checkpoints[i]
    
    report.append("## Key Findings\n\n")
    
    if transition_point and max_drop > 0.1:
        report.append(f"1. **Phase transition detected at iteration {transition_point}**\n")
        report.append(f"   - Exact match rate dropped by {max_drop:.1%}\n")
        report.append(f"   - Embedding similarity at transition: {results[transition_point]['embedding_analysis']['mean_similarity']:.4f}\n\n")
    
    # 总结每个阶段
    report.append("## Phase Analysis\n\n")
    
    for i, ckpt in enumerate(checkpoints):
        if ckpt not in results:
            continue
            
        r = results[ckpt]
        report.append(f"### Checkpoint {ckpt}\n")
        report.append(f"- Exact match rate: {r['path_analysis']['exact_match_rate']:.1%}\n")
        report.append(f"- Valid alternatives: {r['path_analysis']['valid_alternative_rate']:.1%}\n")
        report.append(f"- Embedding similarity: {r['embedding_analysis']['mean_similarity']:.4f}\n")
        report.append(f"- Distribution entropy: {r['distribution_analysis']['mean_entropy']:.3f}\n")
        
        # 标记特殊点
        if ckpt == transition_point:
            report.append("- **⚠️ PHASE TRANSITION POINT**\n")
        
        report.append("\n")
    
    # 写入文件
    with open(os.path.join(output_dir, 'analysis_report.md'), 'w') as f:
        f.writelines(report)
    
    print(f"\nReport saved to {output_dir}/analysis_report.md")

if __name__ == "__main__":
    main()