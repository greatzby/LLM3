"""
可视化脚本 - 生成漂亮的图表
使用方法: python visualize_results.py
"""
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_experiment(out_dir):
    """为单个实验生成图表"""
    # 读取数据
    stats_file = os.path.join(out_dir, 'training_stats.pkl')
    if not os.path.exists(stats_file):
        print(f"跳过 {out_dir}: 没有找到数据")
        return
        
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
    
    # 读取分析结果
    analysis_file = os.path.join(out_dir, 'phase_analysis.json')
    phase_transitions = []
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
            phase_transitions = analysis.get('phase_transitions', [])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 准确率曲线
    iterations = stats['iterations']
    tf_accuracy = stats['tf_accuracy']
    ar_accuracy = stats['ar_accuracy']
    
    ax1.plot(iterations, tf_accuracy, 'b-', label='Teacher Forcing', linewidth=2)
    ax1.plot(iterations, ar_accuracy, 'g-', label='Autoregressive', linewidth=2)
    
    # 标记相变点
    for trans in phase_transitions:
        ax1.axvline(x=trans['iteration'], color='red', linestyle='--', alpha=0.5)
        ax1.text(trans['iteration'], 0.5, 'Phase\nTransition', 
                rotation=90, va='center', ha='right', color='red')
    
    # 标记不同阶段
    if phase_transitions:
        phase_iter = phase_transitions[0]['iteration']
        ax1.axvspan(0, 5000, alpha=0.1, color='red', label='Phase I')
        ax1.axvspan(5000, phase_iter, alpha=0.1, color='green', label='Phase II')
        ax1.axvspan(phase_iter, iterations[-1], alpha=0.1, color='blue', label='Phase III')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Training Progress - {os.path.basename(out_dir)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 图2: 损失曲线
    train_loss = stats['train_loss']
    train_iters = stats.get('train_iter_history', list(range(len(train_loss))))
    
    ax2.plot(train_iters, train_loss, 'r-', linewidth=0.5, alpha=0.7)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Training Loss')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(out_dir, 'phase_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_file}")
    plt.close()

def create_comparison_plot(experiment_dirs):
    """创建对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, out_dir in enumerate(experiment_dirs):
        stats_file = os.path.join(out_dir, 'training_stats.pkl')
        if not os.path.exists(stats_file):
            continue
            
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        
        label = os.path.basename(out_dir)
        color = colors[i % len(colors)]
        
        ax.plot(stats['iterations'], stats['tf_accuracy'], 
               color=color, label=f'{label} (TF)', linewidth=2)
        ax.plot(stats['iterations'], stats['ar_accuracy'], 
               color=color, label=f'{label} (AR)', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Experiment Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: experiment_comparison.png")
    plt.close()

def main():
    """主函数"""
    print("生成可视化图表")
    print("="*60)
    
    # 查找所有实验
    base_dir = 'out'
    experiment_dirs = []
    
    if os.path.exists(base_dir):
        for dir_name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.isdir(dir_path) and dir_name.startswith('simple_graph'):
                experiment_dirs.append(dir_path)
    
    if not experiment_dirs:
        print("没有找到实验结果!")
        return
    
    print(f"找到{len(experiment_dirs)}个实验")
    
    # 为每个实验生成图表
    for dir_path in experiment_dirs:
        print(f"\n处理: {dir_path}")
        plot_experiment(dir_path)
    
    # 生成对比图
    if len(experiment_dirs) > 1:
        create_comparison_plot(experiment_dirs)
    
    print("\n所有图表生成完成!")

if __name__ == "__main__":
    main()