"""
analyze_phase_transition_v2.py
基于实际TF准确率数据分析相变过程
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd

class PhaseTransitionAnalyzerV2:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = os.path.join(checkpoint_dir, 'phase_analysis_v2')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_checkpoint_data(self, start=135000, end=160000, step=1000):
        """加载checkpoint中的训练历史数据"""
        data = {
            'iterations': [],
            'tf_accuracy': [],
            'ar_accuracy': [],
            'has_checkpoint': []
        }
        
        for iter_num in range(start, end + 1, step):
            ckpt_path = os.path.join(self.checkpoint_dir, f'ckpt_{iter_num}.pt')
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint {iter_num}...")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                
                # 获取历史数据
                tf_history = checkpoint.get('tf_history', [])
                ar_history = checkpoint.get('ar_history', [])
                
                # 找到当前iteration对应的准确率
                # 假设每1000步记录一次，所以index = iter_num // 1000
                idx = iter_num // 1000 - 1
                
                if idx < len(tf_history):
                    data['iterations'].append(iter_num)
                    data['tf_accuracy'].append(tf_history[idx])
                    data['ar_accuracy'].append(ar_history[idx] if idx < len(ar_history) else None)
                    data['has_checkpoint'].append(True)
        
        return data
    
    def analyze_model_states(self, critical_checkpoints):
        """分析关键checkpoint的模型状态"""
        results = {}
        
        for iter_num in critical_checkpoints:
            ckpt_path = os.path.join(self.checkpoint_dir, f'ckpt_{iter_num}.pt')
            if not os.path.exists(ckpt_path):
                continue
                
            print(f"\nAnalyzing model state at iteration {iter_num}...")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_state = checkpoint['model']
            
            # 分析权重统计
            weight_stats = self.analyze_weights(model_state)
            
            # 分析注意力权重
            attention_stats = self.analyze_attention(model_state)
            
            results[iter_num] = {
                'weight_stats': weight_stats,
                'attention_stats': attention_stats
            }
            
        return results
    
    def analyze_weights(self, model_state):
        """分析模型权重的统计特性"""
        stats = {}
        
        # 分析输出层权重
        if 'lm_head.weight' in model_state:
            lm_weights = model_state['lm_head.weight'].numpy()
            stats['lm_head'] = {
                'mean': np.mean(lm_weights),
                'std': np.std(lm_weights),
                'max': np.max(lm_weights),
                'min': np.min(lm_weights),
                'norm': np.linalg.norm(lm_weights)
            }
        
        # 分析embedding权重
        if 'transformer.wte.weight' in model_state:
            emb_weights = model_state['transformer.wte.weight'].numpy()
            stats['embedding'] = {
                'mean': np.mean(emb_weights),
                'std': np.std(emb_weights),
                'norm': np.linalg.norm(emb_weights, axis=1).mean()
            }
            
        return stats
    
    def analyze_attention(self, model_state):
        """分析注意力权重"""
        stats = {}
        
        # 找到所有注意力权重
        for key in model_state:
            if 'attn.c_attn.weight' in key:
                attn_weights = model_state[key].numpy()
                layer_name = key.split('.')[2]  # 提取层号
                
                stats[layer_name] = {
                    'mean': np.mean(attn_weights),
                    'std': np.std(attn_weights),
                    'norm': np.linalg.norm(attn_weights)
                }
                
        return stats
    
    def plot_phase_transition(self, data):
        """绘制相变过程的详细分析"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. TF准确率的详细变化
        ax1 = plt.subplot(2, 3, 1)
        iterations = data['iterations']
        tf_acc = data['tf_accuracy']
        
        ax1.plot(iterations, tf_acc, 'b-', linewidth=2, marker='o', markersize=8)
        
        # 标记关键点
        critical_points = [
            (138000, "Start decline"),
            (142000, "Accelerating"),
            (145000, "Cliff edge"),
            (150000, "Collapsed"),
            (155000, "Stabilizing")
        ]
        
        for iter_num, label in critical_points:
            if iter_num in iterations:
                idx = iterations.index(iter_num)
                ax1.axvline(iter_num, color='red', linestyle='--', alpha=0.5)
                ax1.annotate(label, xy=(iter_num, tf_acc[idx]), 
                           xytext=(iter_num, tf_acc[idx] + 0.1),
                           fontsize=8, ha='center',
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('TF Accuracy')
        ax1.set_title('Teacher Forcing Accuracy During Phase Transition')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 突出显示关键相变期
        ax1.axvspan(145000, 150000, alpha=0.2, color='red', label='Critical transition')
        ax1.legend()
        
        # 2. 下降速率分析
        ax2 = plt.subplot(2, 3, 2)
        if len(tf_acc) > 1:
            # 计算每1k步的变化率
            changes = []
            change_iters = []
            for i in range(1, len(tf_acc)):
                change = (tf_acc[i] - tf_acc[i-1]) * 100  # 转换为百分点
                changes.append(change)
                change_iters.append(iterations[i])
            
            colors = ['green' if c >= 0 else 'red' for c in changes]
            bars = ax2.bar(change_iters, changes, width=800, color=colors, alpha=0.7)
            
            # 标记最大下降
            min_idx = np.argmin(changes)
            ax2.axvline(change_iters[min_idx], color='darkred', linewidth=2, 
                       label=f'Max drop: {changes[min_idx]:.1f}pp')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('TF Accuracy Change (percentage points)')
            ax2.set_title('Rate of TF Accuracy Change')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. 相变阶段划分
        ax3 = plt.subplot(2, 3, 3)
        
        # 定义相变阶段
        phases = [
            (135000, 142000, 'Pre-transition\n(Stable)', 'green'),
            (142000, 145000, 'Early transition\n(Declining)', 'yellow'),
            (145000, 150000, 'Critical transition\n(Collapse)', 'red'),
            (150000, 155000, 'Late transition\n(Stabilizing)', 'orange'),
            (155000, 160000, 'Post-transition\n(New equilibrium)', 'blue')
        ]
        
        ax3.plot(iterations, tf_acc, 'k-', linewidth=2, marker='o')
        
        for start, end, label, color in phases:
            ax3.axvspan(start, end, alpha=0.3, color=color, label=label)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('TF Accuracy')
        ax3.set_title('Phase Transition Stages')
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. TF vs AR对比
        ax4 = plt.subplot(2, 3, 4)
        if data['ar_accuracy'][0] is not None:
            ar_acc = data['ar_accuracy']
            ax4.plot(iterations, tf_acc, 'b-', linewidth=2, marker='o', label='Teacher Forcing')
            ax4.plot(iterations, ar_acc, 'g-', linewidth=2, marker='s', label='Autoregressive')
            
            # 计算divergence
            divergence = [abs(tf - ar) for tf, ar in zip(tf_acc, ar_acc)]
            ax4_twin = ax4.twinx()
            ax4_twin.plot(iterations, divergence, 'r--', linewidth=1, alpha=0.5, label='|TF-AR|')
            ax4_twin.set_ylabel('|TF - AR|', color='red')
            ax4_twin.tick_params(axis='y', colors='red')
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('TF vs AR Accuracy')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
        
        # 5. 累积下降分析
        ax5 = plt.subplot(2, 3, 5)
        
        # 计算相对于初始值的累积下降
        initial_acc = tf_acc[0]
        cumulative_drop = [(initial_acc - acc) * 100 for acc in tf_acc]
        
        ax5.plot(iterations, cumulative_drop, 'purple', linewidth=2, marker='D')
        ax5.fill_between(iterations, 0, cumulative_drop, alpha=0.3, color='purple')
        
        # 标记关键阈值
        ax5.axhline(10, color='orange', linestyle='--', label='10pp drop')
        ax5.axhline(50, color='red', linestyle='--', label='50pp drop')
        ax5.axhline(75, color='darkred', linestyle='--', label='75pp drop')
        
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Cumulative Drop from Initial (pp)')
        ax5.set_title('Cumulative Accuracy Loss')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. 相变速度热力图
        ax6 = plt.subplot(2, 3, 6)
        
        # 创建一个2D表示：时间 vs 准确率区间
        acc_bins = np.linspace(0, 1, 21)  # 0-100%，每5%一个bin
        time_windows = []
        
        for i in range(len(iterations)-1):
            window_data = []
            for j in range(len(acc_bins)-1):
                if acc_bins[j] <= tf_acc[i] <= acc_bins[j+1]:
                    window_data.append(1)
                else:
                    window_data.append(0)
            time_windows.append(window_data)
        
        if time_windows:
            im = ax6.imshow(np.array(time_windows).T, aspect='auto', cmap='hot',
                           extent=[iterations[0], iterations[-1], 0, 100])
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('TF Accuracy (%)')
            ax6.set_title('Accuracy Distribution Over Time')
            plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase_transition_detailed.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 生成详细报告
        self.generate_detailed_report(data, critical_points)
    
    def generate_detailed_report(self, data, critical_points):
        """生成详细的文字报告"""
        report_path = os.path.join(self.output_dir, 'phase_transition_detailed_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED PHASE TRANSITION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # 总体统计
            iterations = data['iterations']
            tf_acc = data['tf_accuracy']
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Analysis range: {iterations[0]} - {iterations[-1]}\n")
            f.write(f"Initial TF accuracy: {tf_acc[0]:.4f} ({tf_acc[0]*100:.2f}%)\n")
            f.write(f"Final TF accuracy: {tf_acc[-1]:.4f} ({tf_acc[-1]*100:.2f}%)\n")
            f.write(f"Total drop: {(tf_acc[0] - tf_acc[-1])*100:.2f} percentage points\n")
            f.write(f"Relative drop: {((tf_acc[0] - tf_acc[-1])/tf_acc[0])*100:.1f}%\n\n")
            
            # 阶段分析
            f.write("PHASE-BY-PHASE ANALYSIS:\n")
            f.write("-"*40 + "\n")
            
            # 找到关键迭代的索引
            critical_indices = {}
            for iter_num, _ in critical_points:
                if iter_num in iterations:
                    critical_indices[iter_num] = iterations.index(iter_num)
            
            # Pre-transition (135k-142k)
            if 135000 in iterations and 142000 in iterations:
                idx_135k = iterations.index(135000)
                idx_142k = iterations.index(142000)
                f.write("\n1. PRE-TRANSITION PHASE (135k-142k):\n")
                f.write(f"   - Start: {tf_acc[idx_135k]:.4f} ({tf_acc[idx_135k]*100:.2f}%)\n")
                f.write(f"   - End: {tf_acc[idx_142k]:.4f} ({tf_acc[idx_142k]*100:.2f}%)\n")
                f.write(f"   - Change: {(tf_acc[idx_142k] - tf_acc[idx_135k])*100:.2f}pp\n")
                f.write("   - Status: Relatively stable, slight decline\n")
            
            # Critical transition (145k-150k)
            if 145000 in iterations and 150000 in iterations:
                idx_145k = iterations.index(145000)
                idx_150k = iterations.index(150000)
                f.write("\n2. CRITICAL TRANSITION PHASE (145k-150k):\n")
                f.write(f"   - Start: {tf_acc[idx_145k]:.4f} ({tf_acc[idx_145k]*100:.2f}%)\n")
                f.write(f"   - End: {tf_acc[idx_150k]:.4f} ({tf_acc[idx_150k]*100:.2f}%)\n")
                f.write(f"   - Change: {(tf_acc[idx_150k] - tf_acc[idx_145k])*100:.2f}pp\n")
                f.write(f"   - Drop rate: {((tf_acc[idx_145k] - tf_acc[idx_150k])*100/5):.2f}pp per 1k iterations\n")
                f.write("   - Status: CATASTROPHIC COLLAPSE\n")
            
            # Find steepest decline
            f.write("\n3. STEEPEST DECLINE ANALYSIS:\n")
            max_drop = 0
            max_drop_iter = 0
            for i in range(1, len(tf_acc)):
                drop = tf_acc[i-1] - tf_acc[i]
                if drop > max_drop:
                    max_drop = drop
                    max_drop_iter = iterations[i]
            
            f.write(f"   - Occurs at: {max_drop_iter} iterations\n")
            f.write(f"   - Magnitude: {max_drop*100:.2f}pp in 1k iterations\n")
            
            # Model behavior analysis
            f.write("\n4. MODEL BEHAVIOR INTERPRETATION:\n")
            f.write("-"*40 + "\n")
            f.write("   - 135k-140k: Model maintains memorization of training paths\n")
            f.write("   - 140k-145k: Increasing tension, gradient conflicts emerge\n")
            f.write("   - 145k-150k: Catastrophic reorganization, model abandons training paths\n")
            f.write("   - 150k+: New equilibrium, model prefers alternative valid paths\n")
            
        print(f"\nDetailed report saved to {report_path}")

def main():
    """主分析函数"""
    checkpoint_dir = "out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430"
    
    analyzer = PhaseTransitionAnalyzerV2(checkpoint_dir)
    
    print("="*60)
    print("Phase Transition Analysis V2 - Based on Actual TF Data")
    print("="*60)
    
    # 1. 加载checkpoint数据
    print("\n1. Loading checkpoint data...")
    data = analyzer.load_checkpoint_data(start=135000, end=175000, step=1000)
    
    if not data['iterations']:
        print("Error: No checkpoint data found!")
        return
    
    print(f"Loaded {len(data['iterations'])} checkpoints")
    
    # 2. 分析关键时期的模型状态
    print("\n2. Analyzing model states at critical points...")
    critical_checkpoints = [135000, 140000, 145000, 150000, 155000, 160000]
    model_analysis = analyzer.analyze_model_states(critical_checkpoints)
    
    # 3. 绘制详细分析
    print("\n3. Generating visualizations...")
    analyzer.plot_phase_transition(data)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()