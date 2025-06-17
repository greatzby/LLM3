"""
比较不同Spurious Rewards实验的结果
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_results(exp_dir):
    """加载实验结果"""
    metrics_path = os.path.join(exp_dir, 'final_metrics.pkl')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            return pickle.load(f)
    return None

def plot_comparison():
    """绘制对比图"""
    base_dir = 'out/spurious_rewards'
    
    # 收集所有实验结果
    experiments = {}
    for exp_name in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_name)
        if os.path.isdir(exp_path):
            results = load_experiment_results(exp_path)
            if results:
                # 解析实验类型
                if 'standard' in exp_name:
                    label = 'Standard (Baseline)'
                elif 'any_valid' in exp_name and 'mixed' not in exp_name:
                    label = 'Any Valid'
                elif 'mixed_alpha0.3' in exp_name:
                    label = 'Mixed (α=0.3)'
                elif 'mixed_alpha0.5' in exp_name:
                    label = 'Mixed (α=0.5)'
                elif 'mixed_alpha0.7' in exp_name:
                    label = 'Mixed (α=0.7)'
                elif 'diversity' in exp_name:
                    label = 'Diversity'
                elif 'phase_aware' in exp_name:
                    label = 'Phase-Aware'
                else:
                    label = exp_name
                
                experiments[label] = results
    
    # 创建对比图
    plt.figure(figsize=(20, 12))
    
    # 1. TF准确率对比
    plt.subplot(2, 3, 1)
    for label, results in experiments.items():
        plt.plot(results['iteration'], results['tf_accuracy'], 
                label=label, linewidth=2, alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Teacher Forcing Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 相变时机对比（放大100k-160k区间）
    plt.subplot(2, 3, 2)
    for label, results in experiments.items():
        iters = np.array(results['iteration'])
        tf_acc = np.array(results['tf_accuracy'])
        mask = (iters >= 100000) & (iters <= 160000)
        plt.plot(iters[mask], tf_acc[mask], label=label, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Phase Transition Period (100k-160k)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 最终TF准确率对比（柱状图）
    plt.subplot(2, 3, 3)
    labels = list(experiments.keys())
    final_tfs = [results['tf_accuracy'][-1] for results in experiments.values()]
    min_tfs = [min(results['tf_accuracy']) for results in experiments.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, final_tfs, width, label='Final TF', alpha=0.8)
    plt.bar(x + width/2, min_tfs, width, label='Min TF', alpha=0.8)
    plt.xlabel('Experiment')
    plt.ylabel('TF Accuracy')
    plt.title('Final vs Minimum TF Accuracy')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. 损失曲线对比
    plt.subplot(2, 3, 4)
    for label, results in experiments.items():
        plt.plot(results['iteration'], results['train_loss'], 
                label=label, linewidth=1, alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 5. Phase分布（堆叠面积图）
    plt.subplot(2, 3, 5)
    phase_names = ['memorization', 'transition_imminent', 'transitioning', 'post_transition']
    phase_colors = ['green', 'orange', 'red', 'purple']
    
    for exp_idx, (label, results) in enumerate(experiments.items()):
        if 'phase' in results:
            # 统计每个phase的持续时间
            phase_durations = {p: 0 for p in phase_names}
            for phase in results['phase']:
                if phase in phase_durations:
                    phase_durations[phase] += 1
            
            # 绘制
            bottom = exp_idx
            for phase, color in zip(phase_names, phase_colors):
                height = phase_durations[phase] / len(results['phase'])
                plt.bar(exp_idx, height, bottom=bottom, color=color, 
                       alpha=0.7, width=0.8)
                bottom += height
    
    plt.xlabel('Experiment')
    plt.ylabel('Phase Distribution')
    plt.title('Phase Distribution Across Experiments')
    plt.xticks(range(len(experiments)), list(experiments.keys()), 
              rotation=45, ha='right')
    
    # 创建图例
    for phase, color in zip(phase_names, phase_colors):
        plt.bar([], [], color=color, alpha=0.7, label=phase)
    plt.legend()
    
    # 6. 关键指标表格
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # 创建表格数据
    table_data = []
    headers = ['Experiment', 'Final TF', 'Min TF', 'Drop Iter', 'Recovery']
    
    for label, results in experiments.items():
        tf_acc = results['tf_accuracy']
        
        # 找到TF下降到50%的迭代
        drop_iter = None
        for i, (iter_num, tf) in enumerate(zip(results['iteration'], tf_acc)):
            if tf < 0.5 and i > 0 and tf_acc[i-1] >= 0.5:
                drop_iter = iter_num
                break
        
        # 检查是否恢复
        recovery = 'No'
        if drop_iter and min(tf_acc) < 0.5:
            # 检查后续是否恢复到>0.5
            drop_idx = results['iteration'].index(drop_iter)
            if any(tf > 0.5 for tf in tf_acc[drop_idx:]):
                recovery = 'Yes'
        
        table_data.append([
            label[:20],  # 截断长标签
            f"{tf_acc[-1]:.3f}",
            f"{min(tf_acc):.3f}",
            f"{drop_iter//1000}k" if drop_iter else "N/A",
            recovery
        ])
    
    # 绘制表格
    table = plt.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Key Metrics Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig('spurious_rewards_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_comparison()