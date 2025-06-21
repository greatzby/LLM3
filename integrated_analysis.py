"""
综合分析 - 整合所有结果，构建完整的因果链
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def load_all_results():
    """加载所有分析结果"""
    base_dir = 'analysis_results'
    
    results = {}
    
    # 加载各个分析的结果
    analyses = [
        'parameter_saturation',
        'path_distribution', 
        'multilevel_performance',
        'graph_understanding'
    ]
    
    for analysis in analyses:
        result_file = os.path.join(base_dir, analysis, f'{analysis}_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[analysis] = json.load(f)
        else:
            print(f"Warning: {result_file} not found")
        
        # 特殊处理distribution shifts
        if analysis == 'path_distribution':
            shifts_file = os.path.join(base_dir, analysis, 'distribution_shifts.json')
            if os.path.exists(shifts_file):
                with open(shifts_file, 'r') as f:
                    results['distribution_shifts'] = json.load(f)
    
    return results

def build_causal_chain(results):
    """构建因果链数据框"""
    # 找到所有分析中共有的checkpoints
    all_checkpoints = set()
    for analysis_name, analysis_data in results.items():
        if isinstance(analysis_data, dict):
            all_checkpoints.update(int(k) for k in analysis_data.keys() if k.isdigit())
    
    checkpoints = sorted(all_checkpoints)
    
    data = []
    
    for ckpt in checkpoints:
        ckpt_str = str(ckpt)
        
        row = {'checkpoint': ckpt}
        
        # 参数饱和度指标
        if 'parameter_saturation' in results and ckpt_str in results['parameter_saturation']:
            ps = results['parameter_saturation'][ckpt_str]
            row.update({
                'weight_gap': ps.get('weight_gap', {}).get('weight_gap', np.nan),
                'embedding_similarity': ps.get('embedding', {}).get('mean_similarity', np.nan),
                'gradient_ratio': ps.get('gradient_response', {}).get('grad_ratio', np.nan),
            })
        
        # 分布变化指标
        if 'distribution_shifts' in results and ckpt_str in results['distribution_shifts']:
            ds = results['distribution_shifts'][ckpt_str]
            row.update({
                'training_path_prob': ds.get('mean_training_path_prob', np.nan),
                'entropy': ds.get('entropy', np.nan),
            })
        
        # 性能指标
        if 'multilevel_performance' in results and ckpt_str in results['multilevel_performance']:
            mp = results['multilevel_performance'][ckpt_str]
            row.update({
                'token_accuracy': mp.get('token_level_accuracy', np.nan),
                'path_accuracy': mp.get('path_level_accuracy', np.nan),
                'ar_accuracy': mp.get('ar_accuracy', np.nan),
            })
        
        # 图理解指标
        if 'graph_understanding' in results and ckpt_str in results['graph_understanding']:
            gu = results['graph_understanding'][ckpt_str]
            row.update({
                'edge_prediction_auc': gu.get('edge_prediction_auc', np.nan),
                'path_validity_score': gu.get('path_validity_discrimination', np.nan),
            })
        
        data.append(row)
    
    return pd.DataFrame(data)

def analyze_correlations(df):
    """分析变量间的相关性"""
    correlations = {}
    
    # 移除缺失值过多的行
    df_clean = df.dropna()
    
    if len(df_clean) < 3:
        print("Warning: Not enough data points for correlation analysis")
        return correlations
    
    # 参数饱和 → 分布变化
    if 'weight_gap' in df_clean.columns and 'training_path_prob' in df_clean.columns:
        correlations['weight_gap_vs_training_prob'] = pearsonr(
            df_clean['weight_gap'].values,
            df_clean['training_path_prob'].values
        )
    
    if 'embedding_similarity' in df_clean.columns and 'entropy' in df_clean.columns:
        correlations['embedding_sim_vs_entropy'] = pearsonr(
            df_clean['embedding_similarity'].values,
            df_clean['entropy'].values
        )
    
    # 分布变化 → 性能
    if 'training_path_prob' in df_clean.columns and 'token_accuracy' in df_clean.columns:
        correlations['training_prob_vs_token_acc'] = pearsonr(
            df_clean['training_path_prob'].values,
            df_clean['token_accuracy'].values
        )
    
    if 'entropy' in df_clean.columns and 'token_accuracy' in df_clean.columns:
        correlations['entropy_vs_token_acc'] = pearsonr(
            df_clean['entropy'].values,
            df_clean['token_accuracy'].values
        )
    
    return correlations

def create_causal_chain_visualization(df, output_dir):
    """创建因果链可视化"""
    # 按checkpoint排序
    df = df.sort_values('checkpoint')
    checkpoints = df['checkpoint'].values
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. 参数饱和度
    ax = axes[0]
    ax2 = ax.twinx()
    
    # 处理缺失值
    weight_gap = df['weight_gap'].values if 'weight_gap' in df else []
    embedding_sim = df['embedding_similarity'].values if 'embedding_similarity' in df else []
    
    if len(weight_gap) > 0 and not all(np.isnan(weight_gap)):
        line1 = ax.plot(checkpoints, weight_gap, 'b-', marker='o', label='Weight Gap', markersize=8)
        ax.set_ylabel('Weight Gap', color='b')
        ax.tick_params(axis='y', labelcolor='b')
    
    if len(embedding_sim) > 0 and not all(np.isnan(embedding_sim)):
        line2 = ax2.plot(checkpoints, embedding_sim, 'r-', marker='s', label='Embedding Similarity', markersize=8)
        ax2.set_ylabel('Embedding Similarity', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_xlabel('Training Iteration')
    ax.set_title('Chain 1: Parameter Saturation', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 合并图例
    if 'line1' in locals() and 'line2' in locals():
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    # 2. 分布变化
    ax = axes[1]
    ax2 = ax.twinx()
    
    training_prob = df['training_path_prob'].values if 'training_path_prob' in df else []
    entropy = df['entropy'].values if 'entropy' in df else []
    
    if len(training_prob) > 0 and not all(np.isnan(training_prob)):
        line1 = ax.plot(checkpoints, training_prob, 'g-', marker='o', label='Training Path Prob', markersize=8)
        ax.set_ylabel('Training Path Probability', color='g')
        ax.tick_params(axis='y', labelcolor='g')
    
    if len(entropy) > 0 and not all(np.isnan(entropy)):
        line2 = ax2.plot(checkpoints, entropy, 'm-', marker='s', label='Output Entropy', markersize=8)
        ax2.set_ylabel('Entropy', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
    
    ax.set_xlabel('Training Iteration')
    ax.set_title('Chain 2: Distribution Shift', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if 'line1' in locals() and 'line2' in locals():
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    # 3. 性能分歧
    ax = axes[2]
    
    token_acc = df['token_accuracy'].values if 'token_accuracy' in df else []
    path_acc = df['path_accuracy'].values if 'path_accuracy' in df else []
    ar_acc = df['ar_accuracy'].values if 'ar_accuracy' in df else []
    
    if len(token_acc) > 0 and not all(np.isnan(token_acc)):
        ax.plot(checkpoints, token_acc, 'b-', marker='o', label='Token Accuracy', linewidth=2, markersize=8)
    if len(path_acc) > 0 and not all(np.isnan(path_acc)):
        ax.plot(checkpoints, path_acc, 'g-', marker='s', label='Path Accuracy', linewidth=2, markersize=8)
    if len(ar_acc) > 0 and not all(np.isnan(ar_acc)):
        ax.plot(checkpoints, ar_acc, 'r--', marker='^', label='AR Accuracy', linewidth=2, markersize=8)
    
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Random Baseline')
    
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Chain 3: Performance Divergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 标记相变点
    for ax in axes:
        if 140000 in checkpoints:
            ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_chain_visualization.png'), dpi=150)
    plt.close()

def generate_summary_report(df, correlations, output_dir):
    """生成综合报告"""
    report = []
    
    report.append("# Causal Chain Analysis Summary\n")
    report.append(f"Generated on: {pd.Timestamp.now()}\n\n")
    
    # 1. 相变点识别
    report.append("## Phase Transition Detection\n")
    
    # 找到token accuracy下降最快的点
    if 'token_accuracy' in df.columns and len(df) > 1:
        df_sorted = df.sort_values('checkpoint')
        token_acc_diff = df_sorted['token_accuracy'].diff()
        if not token_acc_diff.isna().all():
            transition_idx = token_acc_diff.idxmin()
            transition_checkpoint = df.loc[transition_idx, 'checkpoint']
            
            report.append(f"- Phase transition detected at checkpoint: {transition_checkpoint}\n")
            report.append(f"- Token accuracy drop: {token_acc_diff[transition_idx]:.3f}\n")
            
            # 相变前后对比
            if transition_idx > 0:
                before_idx = df_sorted.index[df_sorted.index.get_loc(transition_idx) - 1]
                report.append(f"- Token accuracy: {df.loc[before_idx, 'token_accuracy']:.3f} → {df.loc[transition_idx, 'token_accuracy']:.3f}\n")
    
    # 2. 因果链证据
    report.append("\n## Causal Chain Evidence\n")
    
    report.append("### Chain 1: Parameter Saturation → Distribution Shift\n")
    if 'weight_gap' in df.columns:
        report.append(f"- Weight gap range: {df['weight_gap'].min():.5f} to {df['weight_gap'].max():.5f}\n")
    if 'embedding_similarity' in df.columns:
        report.append(f"- Embedding similarity range: {df['embedding_similarity'].min():.3f} to {df['embedding_similarity'].max():.3f}\n")
    
    if 'weight_gap_vs_training_prob' in correlations:
        r, p = correlations['weight_gap_vs_training_prob']
        report.append(f"- Correlation (weight_gap vs training_prob): r={r:.3f}, p={p:.3f}\n")
    
    report.append("\n### Chain 2: Distribution Shift → Performance Divergence\n")
    if 'training_path_prob' in df.columns:
        first_prob = df.loc[df['training_path_prob'].first_valid_index(), 'training_path_prob']
        last_prob = df.loc[df['training_path_prob'].last_valid_index(), 'training_path_prob']
        report.append(f"- Training path probability: {first_prob:.3f} → {last_prob:.3f}\n")
    
    if 'training_prob_vs_token_acc' in correlations:
        r, p = correlations['training_prob_vs_token_acc']
        report.append(f"- Correlation (training_prob vs token_acc): r={r:.3f}, p={p:.3f}\n")
    
    # 3. 关键发现
    report.append("\n## Key Findings\n")
    
    # 检查是否符合预期模式
    if 'token_accuracy' in df.columns:
        final_token_acc = df.loc[df['token_accuracy'].last_valid_index(), 'token_accuracy']
        if final_token_acc < 0.25:
            report.append(f"- ✓ Final token accuracy ({final_token_acc:.3f}) below random baseline (0.25)\n")
        else:
            report.append(f"- ✗ Final token accuracy ({final_token_acc:.3f}) above random baseline (0.25)\n")
    
    if 'ar_accuracy' in df.columns:
        final_ar = df.loc[df['ar_accuracy'].last_valid_index(), 'ar_accuracy']
        if final_ar > 0.9:
            report.append(f"- ✓ AR accuracy maintained high ({final_ar:.3f})\n")
    
    if 'weight_gap' in df.columns:
        final_gap = abs(df.loc[df['weight_gap'].last_valid_index(), 'weight_gap'])
        if final_gap < 0.001:
            report.append(f"- ✓ Weight gap approaches zero ({final_gap:.5f})\n")
    
    # 4. 图理解分析
    if 'edge_prediction_auc' in df.columns:
        report.append("\n## Graph Understanding Analysis\n")
        first_auc = df.loc[df['edge_prediction_auc'].first_valid_index(), 'edge_prediction_auc']
        last_auc = df.loc[df['edge_prediction_auc'].last_valid_index(), 'edge_prediction_auc']
        report.append(f"- Edge prediction AUC: {first_auc:.3f} → {last_auc:.3f}\n")
        
        if last_auc > first_auc:
            report.append("- ✓ Model's graph structure understanding improved\n")
    
    # 保存报告
    with open(os.path.join(output_dir, 'causal_chain_summary.txt'), 'w') as f:
        f.writelines(report)
    
    # 同时打印到控制台
    print(''.join(report))

def main():
    output_dir = 'analysis_results/integrated'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有结果
    print("Loading all analysis results...")
    results = load_all_results()
    
    if not results:
        print("No results found! Make sure to run other analysis scripts first.")
        return
    
    # 构建因果链数据
    print("Building causal chain...")
    df = build_causal_chain(results)
    
    if df.empty:
        print("No data to analyze!")
        return
    
    # 保存数据框
    df.to_csv(os.path.join(output_dir, 'causal_chain_data.csv'), index=False)
    print(f"Saved causal chain data with {len(df)} checkpoints")
    
    # 分析相关性
    print("Analyzing correlations...")
    correlations = analyze_correlations(df)
    
    # 创建可视化
    print("Creating visualizations...")
    create_causal_chain_visualization(df, output_dir)
    
    # 生成报告
    print("Generating summary report...")
    generate_summary_report(df, correlations, output_dir)
    
    print(f"\nIntegrated analysis complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main()