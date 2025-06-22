"""
analyze_phase_transition.py
分析神经网络相变的完整过程 - 适配1k间隔checkpoint
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import networkx as nx
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from model import GPTConfig, GPT

class PhaseTransitionAnalyzer:
    def __init__(self, checkpoint_dir, device='cuda'):
        self.checkpoint_dir = checkpoint_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 加载数据元信息
        self.load_metadata()
        
        # 创建输出目录
        self.output_dir = os.path.join(checkpoint_dir, 'phase_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_metadata(self):
        """加载图和词汇表等元信息"""
        # 假设使用100节点的图
        data_dir = 'data/simple_graph/100'
        
        # 加载meta
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = len(self.itos)
        self.block_size = meta['block_size']
        
        # 加载图
        self.graph = nx.read_graphml(os.path.join(data_dir, "path_graph.graphml"))
        
        # 加载验证数据
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), 
                                  dtype=np.uint16, mode='r')
    
    def load_checkpoint(self, iteration):
        """加载特定迭代的checkpoint"""
        ckpt_path = os.path.join(self.checkpoint_dir, f'ckpt_{iteration}.pt')
        if not os.path.exists(ckpt_path):
            return None
            
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 创建模型
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # 加载权重
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def analyze_output_distribution(self, model, num_samples=100):
        """分析模型的输出分布"""
        distributions = []
        
        # 从验证集采样
        for _ in range(num_samples):
            # 随机选择一个序列
            idx = np.random.randint(0, len(self.val_data) - self.block_size - 1)
            seq = self.val_data[idx:idx + self.block_size + 1]
            
            # 找到实际序列长度（第一个PAD之前）
            seq_len = np.where(seq == 0)[0]
            if len(seq_len) > 0:
                seq_len = seq_len[0]
            else:
                seq_len = len(seq)
            
            if seq_len < 4:  # 太短的序列跳过
                continue
            
            # 选择中间位置进行预测
            for pos in range(3, min(seq_len-1, 10)):
                input_seq = torch.tensor(seq[:pos], dtype=torch.long).unsqueeze(0).to(self.device)
                target = seq[pos]
                
                with torch.no_grad():
                    logits, _ = model(input_seq)
                    probs = torch.softmax(logits[0, -1, :], dim=0)
                    
                    # 获取top-k概率
                    top_k = 10
                    top_probs, top_indices = torch.topk(probs, top_k)
                    
                    distributions.append({
                        'position': pos,
                        'target': target,
                        'target_prob': probs[target].item(),
                        'top1_prob': top_probs[0].item(),
                        'top1_token': top_indices[0].item(),
                        'top2_prob': top_probs[1].item(),
                        'top2_token': top_indices[1].item(),
                        'entropy': -torch.sum(probs * torch.log(probs + 1e-8)).item(),
                        'confidence_ratio': top_probs[0].item() / (top_probs[1].item() + 1e-8),
                        'is_correct': top_indices[0].item() == target
                    })
        
        return distributions
    
    def track_specific_paths(self, iterations, num_test_cases=20):
        """追踪特定路径的概率变化"""
        print("\nTracking specific path probabilities across iterations...")
        
        # 选择一些测试用例
        test_cases = []
        for i in range(num_test_cases):
            idx = np.random.randint(0, len(self.val_data) - self.block_size - 1)
            seq = self.val_data[idx:idx + self.block_size + 1]
            
            # 找到有效序列
            seq_len = np.where(seq == 0)[0]
            if len(seq_len) > 0:
                seq_len = seq_len[0]
            else:
                seq_len = len(seq)
                
            if seq_len >= 6:  # 需要足够长的序列
                test_cases.append({
                    'sequence': seq[:seq_len],
                    'position': 4,  # 固定位置
                    'target': seq[4]
                })
        
        # 对每个iteration分析
        results = defaultdict(list)
        
        for iter_num in tqdm(iterations):
            model = self.load_checkpoint(iter_num)
            if model is None:
                continue
                
            for test_case in test_cases:
                input_seq = torch.tensor(test_case['sequence'][:test_case['position']], 
                                        dtype=torch.long).unsqueeze(0).to(self.device)
                target = test_case['target']
                
                with torch.no_grad():
                    logits, _ = model(input_seq)
                    probs = torch.softmax(logits[0, -1, :], dim=0)
                    
                    # 记录目标token的概率
                    results[iter_num].append({
                        'target_prob': probs[target].item(),
                        'top1_token': torch.argmax(probs).item(),
                        'top1_prob': torch.max(probs).item(),
                        'is_correct': torch.argmax(probs).item() == target
                    })
        
        return results, test_cases
    
    def analyze_phase_transition(self, start_iter=135000, end_iter=175000, step=1000):
        """分析相变过程"""
        iterations = list(range(start_iter, end_iter + 1, step))
        
        # 1. 收集每个iteration的统计信息
        print("\nAnalyzing output distributions...")
        all_stats = {}
        
        for iter_num in tqdm(iterations):
            model = self.load_checkpoint(iter_num)
            if model is None:
                continue
                
            # 分析输出分布
            distributions = self.analyze_output_distribution(model, num_samples=200)
            
            if distributions:
                all_stats[iter_num] = {
                    'avg_entropy': np.mean([d['entropy'] for d in distributions]),
                    'avg_confidence_ratio': np.mean([d['confidence_ratio'] for d in distributions]),
                    'avg_top1_prob': np.mean([d['top1_prob'] for d in distributions]),
                    'avg_target_prob': np.mean([d['target_prob'] for d in distributions]),
                    'accuracy': np.mean([d['is_correct'] for d in distributions]),
                    'distributions': distributions
                }
        
        # 2. 追踪特定路径
        path_results, test_cases = self.track_specific_paths(iterations)
        
        # 3. 生成可视化
        self.visualize_transition(all_stats, path_results, iterations)
        
        return all_stats, path_results
    
    def visualize_transition(self, all_stats, path_results, iterations):
        """可视化相变过程"""
        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        
        # 估计相变区域（基于准确率变化）
        iters = sorted(all_stats.keys())
        accuracies = [all_stats[i]['accuracy'] for i in iters]
        
        # 找到准确率下降最快的区域
        if len(accuracies) > 5:
            acc_changes = []
            for i in range(len(accuracies) - 1):
                acc_changes.append((iters[i], accuracies[i] - accuracies[i+1]))
            acc_changes.sort(key=lambda x: x[1], reverse=True)
            transition_start = acc_changes[0][0] - 2000
            transition_end = acc_changes[0][0] + 2000
        else:
            transition_start = 140000
            transition_end = 145000
        
        # 1. 平均熵的变化
        ax1 = plt.subplot(3, 3, 1)
        entropies = [all_stats[i]['avg_entropy'] for i in iters]
        ax1.plot(iters, entropies, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Entropy')
        ax1.set_title('Output Entropy Evolution')
        ax1.axvspan(transition_start, transition_end, alpha=0.2, color='red', label='Transition Zone')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 置信度比率
        ax2 = plt.subplot(3, 3, 2)
        conf_ratios = [all_stats[i]['avg_confidence_ratio'] for i in iters]
        ax2.semilogy(iters, conf_ratios, 'r-', linewidth=2, marker='s')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Confidence Ratio (log scale)')
        ax2.set_title('Top1/Top2 Probability Ratio')
        ax2.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax2.grid(True, alpha=0.3)
        
        # 3. 准确率变化
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(iters, accuracies, 'g-', linewidth=2, marker='^')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Prediction Accuracy')
        ax3.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        # 4. 特定路径概率追踪（热力图）
        ax4 = plt.subplot(3, 3, 4)
        if path_results:
            # 创建概率矩阵
            sorted_iters = sorted(path_results.keys())
            num_cases = len(path_results[sorted_iters[0]])
            prob_matrix = np.zeros((num_cases, len(sorted_iters)))
            
            for j, iter_num in enumerate(sorted_iters):
                for i in range(num_cases):
                    prob_matrix[i, j] = path_results[iter_num][i]['target_prob']
            
            # 绘制热力图
            im = ax4.imshow(prob_matrix, aspect='auto', cmap='RdYlBu_r', 
                           vmin=0, vmax=1)
            ax4.set_xlabel('Iteration Index')
            ax4.set_ylabel('Test Case')
            ax4.set_title('Target Token Probability Heatmap')
            
            # 设置x轴标签
            tick_indices = np.linspace(0, len(sorted_iters)-1, 7).astype(int)
            ax4.set_xticks(tick_indices)
            ax4.set_xticklabels([f'{sorted_iters[i]//1000}k' for i in tick_indices])
            
            plt.colorbar(im, ax=ax4)
        
        # 5. Top1概率分布
        ax5 = plt.subplot(3, 3, 5)
        top1_probs = [all_stats[i]['avg_top1_prob'] for i in iters]
        ax5.plot(iters, top1_probs, 'purple', linewidth=2, marker='o')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Average Top-1 Probability')
        ax5.set_title('Model Confidence Evolution')
        ax5.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax5.grid(True, alpha=0.3)
        
        # 6. 相变前后的概率分布对比
        ax6 = plt.subplot(3, 3, 6)
        # 选择相变前后的代表性迭代
        before_iter = transition_start - 5000
        after_iter = transition_end + 5000
        
        # 确保选择的迭代存在
        available_iters = sorted(all_stats.keys())
        before_iter = min(available_iters, key=lambda x: abs(x - before_iter))
        after_iter = min(available_iters, key=lambda x: abs(x - after_iter))
        
        if before_iter in all_stats and after_iter in all_stats:
            before_probs = [d['top1_prob'] for d in all_stats[before_iter]['distributions']]
            after_probs = [d['top1_prob'] for d in all_stats[after_iter]['distributions']]
            
            ax6.hist(before_probs, bins=30, alpha=0.5, label=f'Iter {before_iter}', 
                    density=True, color='blue')
            ax6.hist(after_probs, bins=30, alpha=0.5, label=f'Iter {after_iter}', 
                    density=True, color='red')
            ax6.set_xlabel('Top-1 Probability')
            ax6.set_ylabel('Density')
            ax6.set_title('Distribution of Top-1 Probabilities')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. 正确预测比例变化
        ax7 = plt.subplot(3, 3, 7)
        if path_results:
            sorted_iters = sorted(path_results.keys())
            correct_ratios = []
            for iter_num in sorted_iters:
                correct = sum(1 for r in path_results[iter_num] if r['is_correct'])
                total = len(path_results[iter_num])
                correct_ratios.append(correct / total)
            
            ax7.plot(sorted_iters, correct_ratios, 'orange', linewidth=2, marker='D')
            ax7.set_xlabel('Iteration')
            ax7.set_ylabel('Fraction Correct')
            ax7.set_title('Fraction of Correct Predictions (Fixed Test Cases)')
            ax7.axvspan(transition_start, transition_end, alpha=0.2, color='red')
            ax7.set_ylim([0, 1])
            ax7.grid(True, alpha=0.3)
        
        # 8. 目标概率 vs Top1概率
        ax8 = plt.subplot(3, 3, 8)
        target_probs = [all_stats[i]['avg_target_prob'] for i in iters]
        ax8.plot(iters, target_probs, 'cyan', linewidth=2, marker='o', label='Target Prob')
        ax8.plot(iters, top1_probs, 'magenta', linewidth=2, marker='s', label='Top-1 Prob')
        ax8.set_xlabel('Iteration')
        ax8.set_ylabel('Probability')
        ax8.set_title('Target vs Top-1 Probability')
        ax8.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. 相变速度分析
        ax9 = plt.subplot(3, 3, 9)
        if len(accuracies) > 1:
            # 计算准确率的导数（变化率）
            acc_changes = np.diff(accuracies)
            iter_diffs = np.diff(iters)
            acc_rates = acc_changes / iter_diffs * 1000  # 每1000 iteration的变化
            
            ax9.plot(iters[1:], acc_rates, 'brown', linewidth=2, marker='x')
            ax9.set_xlabel('Iteration')
            ax9.set_ylabel('Accuracy Change Rate (per 1k iter)')
            ax9.set_title('Rate of Accuracy Change')
            ax9.axhspan(-0.02, 0.02, alpha=0.2, color='gray', label='Stable Zone')
            ax9.axvspan(transition_start, transition_end, alpha=0.2, color='red')
            ax9.grid(True, alpha=0.3)
            ax9.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase_transition_analysis.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to {self.output_dir}/phase_transition_analysis.png")
        
        # 返回估计的相变区间供后续分析使用
        return transition_start, transition_end
    
    def analyze_critical_period(self, start=None, end=None, step=1000):
        """更细致地分析关键相变期"""
        # 如果没有指定范围，先做一次粗略分析找到相变区间
        if start is None or end is None:
            print("\nDetecting critical period automatically...")
            # 分析135k-175k找到相变区间
            coarse_stats, _ = self.analyze_phase_transition(135000, 175000, 5000)
            
            # 基于准确率变化找到关键期
            iters = sorted(coarse_stats.keys())
            accuracies = [coarse_stats[i]['accuracy'] for i in iters]
            
            # 找到准确率下降最快的区间
            max_drop = 0
            critical_iter = 140000
            for i in range(len(accuracies) - 1):
                drop = accuracies[i] - accuracies[i+1]
                if drop > max_drop:
                    max_drop = drop
                    critical_iter = iters[i]
            
            start = critical_iter - 3000
            end = critical_iter + 3000
        
        print(f"\nAnalyzing critical period ({start}-{end}) with step={step}...")
        
        iterations = list(range(start, end + 1, step))
        detailed_stats = []
        
        for iter_num in tqdm(iterations):
            model = self.load_checkpoint(iter_num)
            if model is None:
                continue
            
            # 分析大量样本
            distributions = self.analyze_output_distribution(model, num_samples=500)
            
            if distributions:
                stats = {
                    'iteration': iter_num,
                    'entropy': np.mean([d['entropy'] for d in distributions]),
                    'entropy_std': np.std([d['entropy'] for d in distributions]),
                    'top1_prob': np.mean([d['top1_prob'] for d in distributions]),
                    'top1_prob_std': np.std([d['top1_prob'] for d in distributions]),
                    'confidence_ratio': np.mean([d['confidence_ratio'] for d in distributions]),
                    'confidence_ratio_median': np.median([d['confidence_ratio'] for d in distributions]),
                    'accuracy': np.mean([d['is_correct'] for d in distributions]),
                    'num_samples': len(distributions)
                }
                detailed_stats.append(stats)
        
        # 创建详细报告
        df = pd.DataFrame(detailed_stats)
        df.to_csv(os.path.join(self.output_dir, 'critical_period_stats.csv'), index=False)
        
        # 绘制详细图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Entropy with error bars
        ax = axes[0, 0]
        ax.errorbar(df['iteration'], df['entropy'], yerr=df['entropy_std'], 
                   marker='o', capsize=5, capthick=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy')
        ax.set_title('Output Entropy During Critical Period')
        ax.grid(True, alpha=0.3)
        
        # Top-1 probability
        ax = axes[0, 1]
        ax.errorbar(df['iteration'], df['top1_prob'], yerr=df['top1_prob_std'], 
                   marker='s', color='red', capsize=5, capthick=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Top-1 Probability')
        ax.set_title('Model Confidence During Critical Period')
        ax.grid(True, alpha=0.3)
        
        # Confidence ratio (log scale)
        ax = axes[1, 0]
        ax.semilogy(df['iteration'], df['confidence_ratio'], 'g-', marker='^', linewidth=2)
        ax.semilogy(df['iteration'], df['confidence_ratio_median'], 'g--', 
                   linewidth=1, alpha=0.5, label='Median')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Confidence Ratio (log)')
        ax.set_title('Confidence Ratio Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[1, 1]
        ax.plot(df['iteration'], df['accuracy'], 'purple', marker='D', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.set_title('Prediction Accuracy During Transition')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 标记关键点
        if len(df) > 2:
            # 找到准确率下降最快的点
            acc_diff = np.diff(df['accuracy'])
            if len(acc_diff) > 0:
                min_idx = np.argmin(acc_diff)
                critical_iter = df['iteration'].iloc[min_idx + 1]
                
                for ax in axes.flat:
                    ax.axvline(critical_iter, color='red', linestyle='--', 
                             alpha=0.7, label=f'Critical point: {critical_iter}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'critical_period_detailed.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        return df
    
    def generate_report(self, all_stats, critical_df):
        """生成文字报告"""
        report_path = os.path.join(self.output_dir, 'phase_transition_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Phase Transition Analysis Report\n")
            f.write("="*60 + "\n\n")
            
            # 找到关键指标
            iters = sorted(all_stats.keys())
            accuracies = [all_stats[i]['accuracy'] for i in iters]
            entropies = [all_stats[i]['avg_entropy'] for i in iters]
            
            # 相变前后对比
            pre_transition = [i for i in iters if i <= 140000]
            post_transition = [i for i in iters if i >= 145000]
            
            if pre_transition and post_transition:
                pre_acc = np.mean([all_stats[i]['accuracy'] for i in pre_transition[-3:]])
                post_acc = np.mean([all_stats[i]['accuracy'] for i in post_transition[:3]])
                
                f.write(f"Accuracy before transition (≤140k): {pre_acc:.4f}\n")
                f.write(f"Accuracy after transition (≥145k): {post_acc:.4f}\n")
                f.write(f"Accuracy drop: {pre_acc - post_acc:.4f}\n\n")
                
                pre_ent = np.mean([all_stats[i]['avg_entropy'] for i in pre_transition[-3:]])
                post_ent = np.mean([all_stats[i]['avg_entropy'] for i in post_transition[:3]])
                
                f.write(f"Entropy before transition: {pre_ent:.4f}\n")
                f.write(f"Entropy after transition: {post_ent:.4f}\n")
                f.write(f"Entropy change: {post_ent - pre_ent:.4f}\n\n")
            
            # Critical period分析
            if critical_df is not None and len(critical_df) > 0:
                f.write("Critical Period Analysis:\n")
                f.write("-"*40 + "\n")
                
                # 找到最剧烈变化的点
                acc_changes = critical_df['accuracy'].diff()
                if not acc_changes.isna().all():
                    max_drop_idx = acc_changes.idxmin()
                    if not pd.isna(max_drop_idx):
                        critical_point = critical_df.loc[max_drop_idx]
                        f.write(f"Sharpest accuracy drop at iteration: {critical_point['iteration']}\n")
                        f.write(f"  Accuracy: {critical_point['accuracy']:.4f}\n")
                        f.write(f"  Entropy: {critical_point['entropy']:.4f}\n")
                        f.write(f"  Confidence ratio: {critical_point['confidence_ratio']:.1f}\n")
            
        print(f"\nReport saved to {report_path}")

def main():
    """主分析函数"""
    # 设置路径
    checkpoint_dir = "out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430"
    
    # 创建分析器
    analyzer = PhaseTransitionAnalyzer(checkpoint_dir)
    
    print("="*60)
    print("Phase Transition Analysis")
    print("="*60)
    
    # 1. 分析整体相变过程（135k-175k，每1k）
    print("\n1. Analyzing overall phase transition (135k-175k)...")
    all_stats, path_results = analyzer.analyze_phase_transition(
        start_iter=135000, 
        end_iter=175000, 
        step=1000  # 每1k分析一次
    )
    
    # 2. 基于初步分析，自动检测并分析关键期
    print("\n2. Detecting and analyzing critical period...")
    critical_df = analyzer.analyze_critical_period(
        start=None,  # 自动检测
        end=None,    # 自动检测
        step=1000    # 1k间隔
    )
    
    # 3. 生成报告
    print("\n3. Generating analysis report...")
    analyzer.generate_report(all_stats, critical_df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*60)
    
    # 打印一些关键发现
    if critical_df is not None and len(critical_df) > 0:
        print("\nKey findings:")
        print(f"- Initial accuracy: {critical_df.iloc[0]['accuracy']:.4f}")
        print(f"- Final accuracy: {critical_df.iloc[-1]['accuracy']:.4f}")
        print(f"- Total drop: {critical_df.iloc[0]['accuracy'] - critical_df.iloc[-1]['accuracy']:.4f}")
        
        # 找到下降最快的点
        acc_diff = critical_df['accuracy'].diff()
        if not acc_diff.isna().all():
            min_idx = acc_diff.idxmin()
            if not pd.isna(min_idx):
                print(f"- Sharpest drop at iteration: {critical_df.loc[min_idx]['iteration']}")

if __name__ == "__main__":
    main()