"""
analyze_critical_dynamics_complete.py
完整的临界动力学分析脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import os
import pickle
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT

class CriticalDynamicsAnalyzer:
    def __init__(self, checkpoint_dir, data_dir='data/simple_graph/100'):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载元数据
        self.load_metadata()
        
    def load_metadata(self):
        """加载必要的元数据"""
        with open(os.path.join(self.data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = len(self.itos)
        self.block_size = meta['block_size']
        
        # 加载验证数据
        self.val_data = np.memmap(os.path.join(self.data_dir, 'val.bin'), 
                                  dtype=np.uint16, mode='r')
    
    def prepare_test_inputs(self, num_samples=100):
        """准备测试输入"""
        test_inputs = []
        
        for _ in range(num_samples):
            # 随机采样
            idx = np.random.randint(0, len(self.val_data) - self.block_size - 1)
            seq = self.val_data[idx:idx + self.block_size + 1].astype(np.int64)
            
            # 找到有效长度
            seq_len = np.where(seq == 0)[0]
            seq_len = seq_len[0] if len(seq_len) > 0 else len(seq)
            
            if seq_len >= 5:  # 需要足够长度
                # 使用前4个token作为输入（预测第5个）
                test_inputs.append({
                    'input': seq[:4],
                    'target': seq[4] if seq_len > 4 else None,
                    'full_seq': seq[:seq_len]
                })
        
        return test_inputs
    
    def analyze_critical_dynamics(self, critical_range=(143000, 152000), step=200):
        """分析临界区域的详细动力学"""
        
        iterations = list(range(critical_range[0], critical_range[1] + 1, step))
        
        # 准备测试数据
        test_inputs = self.prepare_test_inputs(100)
        
        # 存储各种测量
        measurements = {
            'iteration': [],
            'entropy_mean': [],
            'entropy_std': [],
            'entropy_skew': [],
            'entropy_kurt': [],
            'prob_fluctuation': [],
            'path_diversity': [],
            'top1_prob_mean': [],
            'top1_prob_std': [],
            'confidence_ratio_mean': [],
            'temperature': []  # 有效温度
        }
        
        print(f"Analyzing critical dynamics from {critical_range[0]} to {critical_range[1]}...")
        
        for iter_num in tqdm(iterations):
            ckpt_path = os.path.join(self.checkpoint_dir, f'ckpt_{iter_num}.pt')
            if not os.path.exists(ckpt_path):
                continue
            
            # 加载模型
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model_args = checkpoint['model_args']
            model = GPT(GPTConfig(**model_args))
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            model.eval()
            
            # 收集统计数据
            entropies = []
            prob_distributions = []
            path_choices = []
            top1_probs = []
            confidence_ratios = []
            
            for test_case in test_inputs:
                input_seq = test_case['input']
                
                # 转换为tensor
                input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, _ = model(input_tensor)
                    
                    # 获取最后位置的输出分布
                    last_logits = logits[0, -1, :]
                    probs = torch.softmax(last_logits, dim=0).cpu().numpy()
                    
                    # 计算熵
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    entropies.append(entropy)
                    
                    # 记录概率分布
                    prob_distributions.append(probs)
                    
                    # 记录选择
                    chosen = np.argmax(probs)
                    path_choices.append(chosen)
                    
                    # Top-1概率
                    top1_probs.append(np.max(probs))
                    
                    # 置信度比率（top1/top2）
                    sorted_probs = np.sort(probs)[::-1]
                    if len(sorted_probs) > 1:
                        confidence_ratios.append(sorted_probs[0] / (sorted_probs[1] + 1e-8))
            
            # 计算统计量
            measurements['iteration'].append(iter_num)
            measurements['entropy_mean'].append(np.mean(entropies))
            measurements['entropy_std'].append(np.std(entropies))
            measurements['entropy_skew'].append(stats.skew(entropies))
            measurements['entropy_kurt'].append(stats.kurtosis(entropies))
            
            # 概率波动性
            prob_array = np.array(prob_distributions)
            prob_std = np.std(prob_array, axis=0)
            measurements['prob_fluctuation'].append(np.mean(prob_std))
            
            # 路径多样性
            unique_paths = len(set(path_choices))
            measurements['path_diversity'].append(unique_paths / len(path_choices))
            
            # Top-1概率统计
            measurements['top1_prob_mean'].append(np.mean(top1_probs))
            measurements['top1_prob_std'].append(np.std(top1_probs))
            
            # 置信度比率
            measurements['confidence_ratio_mean'].append(np.mean(confidence_ratios))
            
            # 有效温度（从分布宽度估算）
            effective_temp = 1.0 / (np.mean(top1_probs) + 0.01)  # 防止除零
            measurements['temperature'].append(effective_temp)
        
        return measurements
    
    def visualize_critical_dynamics(self, measurements, output_dir='critical_dynamics_results'):
        """可视化临界动力学"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # 定义相变区域
        transition_start = 145000
        transition_end = 150000
        
        # 1. 熵的演化（带误差条）
        ax = axes[0, 0]
        ax.errorbar(measurements['iteration'], measurements['entropy_mean'], 
                    yerr=measurements['entropy_std'], capsize=5, capthick=2,
                    marker='o', markersize=6, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy (nats)')
        ax.set_title('Output Entropy Evolution')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red', label='Phase transition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 熵的高阶矩
        ax = axes[0, 1]
        ax.plot(measurements['iteration'], measurements['entropy_skew'], 
                'r-', label='Skewness', linewidth=2, marker='s')
        ax.plot(measurements['iteration'], measurements['entropy_kurt'], 
                'b-', label='Kurtosis', linewidth=2, marker='^')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Higher Moments')
        ax.set_title('Entropy Distribution Shape')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 概率波动性
        ax = axes[0, 2]
        ax.plot(measurements['iteration'], measurements['prob_fluctuation'], 
                'g-', linewidth=3, marker='D', markersize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Probability STD')
        ax.set_title('Output Distribution Fluctuations')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax.grid(True, alpha=0.3)
        
        # 4. 路径多样性
        ax = axes[1, 0]
        ax.plot(measurements['iteration'], measurements['path_diversity'], 
                'purple', linewidth=3, marker='o', markersize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Path Diversity Ratio')
        ax.set_title('Solution Space Exploration')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 5. Top-1概率演化
        ax = axes[1, 1]
        ax.errorbar(measurements['iteration'], measurements['top1_prob_mean'],
                    yerr=measurements['top1_prob_std'], capsize=5,
                    color='orange', linewidth=2, marker='s')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Top-1 Probability')
        ax.set_title('Model Confidence Evolution')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 6. 置信度比率（对数尺度）
        ax = axes[1, 2]
        ax.semilogy(measurements['iteration'], measurements['confidence_ratio_mean'],
                    'brown', linewidth=2, marker='^')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Confidence Ratio (log)')
        ax.set_title('Top-1/Top-2 Probability Ratio')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax.grid(True, alpha=0.3)
        
        # 7. 有效温度
        ax = axes[2, 0]
        ax.plot(measurements['iteration'], measurements['temperature'],
                'red', linewidth=3, marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Effective Temperature')
        ax.set_title('System Temperature (1/confidence)')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax.grid(True, alpha=0.3)
        
        # 8. 相空间轨迹
        ax = axes[2, 1]
        # 绘制熵vs温度的相空间
        scatter = ax.scatter(measurements['entropy_mean'], measurements['temperature'],
                            c=measurements['iteration'], cmap='viridis', s=50)
        ax.set_xlabel('Mean Entropy')
        ax.set_ylabel('Effective Temperature')
        ax.set_title('Phase Space Trajectory')
        plt.colorbar(scatter, ax=ax, label='Iteration')
        
        # 连接相邻点
        for i in range(len(measurements['entropy_mean'])-1):
            ax.plot([measurements['entropy_mean'][i], measurements['entropy_mean'][i+1]],
                   [measurements['temperature'][i], measurements['temperature'][i+1]],
                   'k-', alpha=0.3, linewidth=0.5)
        
        # 9. 临界标度分析
        ax = axes[2, 2]
        # 找到熵波动的峰值
        entropy_std = measurements['entropy_std']
        if len(entropy_std) > 5:
            peak_idx = np.argmax(entropy_std)
            peak_iter = measurements['iteration'][peak_idx]
            
            # 计算距离临界点的距离
            distances = np.abs(np.array(measurements['iteration']) - peak_iter)
            
            # 只使用非零距离
            mask = distances > 0
            if np.sum(mask) > 3:
                log_dist = np.log(distances[mask])
                log_fluct = np.log(np.array(entropy_std)[mask])
                
                # 线性拟合找临界指数
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_dist, log_fluct)
                
                ax.loglog(distances[mask], np.array(entropy_std)[mask], 'o', markersize=8)
                
                # 绘制拟合线
                fit_x = np.logspace(np.log10(distances[mask].min()), 
                                   np.log10(distances[mask].max()), 100)
                fit_y = np.exp(intercept) * fit_x ** slope
                ax.loglog(fit_x, fit_y, 'r--', linewidth=2, 
                         label=f'γ = {slope:.2f}±{std_err:.2f}')
                
                ax.set_xlabel('|Iteration - Critical Point|')
                ax.set_ylabel('Entropy Fluctuation')
                ax.set_title('Critical Scaling Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'critical_dynamics_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        with open(os.path.join(output_dir, 'critical_dynamics_data.pkl'), 'wb') as f:
            pickle.dump(measurements, f)
        
        # 生成报告
        self.generate_report(measurements, output_dir)
    
    def generate_report(self, measurements, output_dir):
        """生成文本报告"""
        report_path = os.path.join(output_dir, 'critical_dynamics_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Critical Dynamics Analysis Report\n")
            f.write("="*60 + "\n\n")
            
            # 找到关键点
            entropy_mean = measurements['entropy_mean']
            peak_idx = np.argmax(measurements['entropy_std'])
            
            f.write("1. Critical Point Identification:\n")
            f.write("-"*40 + "\n")
            f.write(f"   Maximum fluctuation at: {measurements['iteration'][peak_idx]}\n")
            f.write(f"   Peak entropy STD: {measurements['entropy_std'][peak_idx]:.4f}\n")
            f.write(f"   Entropy at peak: {entropy_mean[peak_idx]:.4f}\n\n")
            
            # 相变特征
            f.write("2. Phase Transition Characteristics:\n")
            f.write("-"*40 + "\n")
            
            # 找到145k和150k的索引
            idx_145k = min(range(len(measurements['iteration'])), 
                          key=lambda i: abs(measurements['iteration'][i]-145000))
            idx_150k = min(range(len(measurements['iteration'])), 
                          key=lambda i: abs(measurements['iteration'][i]-150000))
            
            if idx_150k > idx_145k:
                # 熵变化
                entropy_change = (entropy_mean[idx_150k] - entropy_mean[idx_145k]) / entropy_mean[idx_145k] * 100
                f.write(f"   Entropy change (145k→150k): {entropy_change:+.1f}%\n")
                
                # 温度变化
                temp_change = measurements['temperature'][idx_150k] - measurements['temperature'][idx_145k]
                f.write(f"   Temperature change: {temp_change:+.2f}\n")
                
                # 多样性变化
                diversity_change = measurements['path_diversity'][idx_150k] - measurements['path_diversity'][idx_145k]
                f.write(f"   Path diversity change: {diversity_change:+.3f}\n")
            
            # 临界行为总结
            f.write("\n3. Critical Behavior Summary:\n")
            f.write("-"*40 + "\n")
            
            # 最大波动
            max_fluct = np.max(measurements['prob_fluctuation'])
            max_fluct_iter = measurements['iteration'][np.argmax(measurements['prob_fluctuation'])]
            f.write(f"   Maximum probability fluctuation: {max_fluct:.4f} at {max_fluct_iter}\n")
            
            # 最大多样性
            max_diversity = np.max(measurements['path_diversity'])
            max_div_iter = measurements['iteration'][np.argmax(measurements['path_diversity'])]
            f.write(f"   Maximum path diversity: {max_diversity:.3f} at {max_div_iter}\n")
            
            # 置信度崩溃
            min_conf = np.min(measurements['top1_prob_mean'])
            min_conf_iter = measurements['iteration'][np.argmin(measurements['top1_prob_mean'])]
            f.write(f"   Minimum confidence: {min_conf:.3f} at {min_conf_iter}\n")
        
        print(f"Report saved to: {report_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze critical dynamics during phase transition')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/simple_graph/100')
    parser.add_argument('--critical_start', type=int, default=143000)
    parser.add_argument('--critical_end', type=int, default=152000)
    parser.add_argument('--step', type=int, default=200)
    
    args = parser.parse_args()
    
    analyzer = CriticalDynamicsAnalyzer(args.checkpoint_dir, args.data_dir)
    
    print(f"Analyzing critical dynamics from {args.critical_start} to {args.critical_end}")
    
    measurements = analyzer.analyze_critical_dynamics(
        critical_range=(args.critical_start, args.critical_end),
        step=args.step
    )
    
    analyzer.visualize_critical_dynamics(measurements)
    
    print("Analysis complete! Results saved to: critical_dynamics_results/")

if __name__ == "__main__":
    main()