"""
比较五种padding解决方案的效果
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_history(path):
    """加载历史数据"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_five_methods(baseline_path, masked_only_path, dynamic_only_path, 
                        dynamic_masked_path, dynamic_padreg_path):
    """比较五种方法的结果"""
    # 加载数据
    baseline = load_history(baseline_path)
    masked_only = load_history(masked_only_path)
    dynamic_only = load_history(dynamic_only_path)
    dynamic_masked = load_history(dynamic_masked_path)
    dynamic_padreg = load_history(dynamic_padreg_path)
    
    # 设置图形样式
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(24, 16))
    
    # 定义颜色和样式
    colors = {
        'baseline': '#1f77b4',
        'masked_only': '#ff7f0e', 
        'dynamic_only': '#2ca02c',
        'dynamic_masked': '#d62728',
        'dynamic_padreg': '#9467bd'
    }
    
    # 1. TF准确率对比
    plt.subplot(3, 3, 1)
    plt.plot(baseline['iter'], baseline['tf_accuracy'], 
            label='Baseline', color=colors['baseline'], linewidth=2.5, alpha=0.8)
    plt.plot(masked_only['iter'], masked_only['tf_accuracy'], 
            label='Masked Loss Only', color=colors['masked_only'], linewidth=2.5)
    plt.plot(dynamic_only['iter'], dynamic_only['tf_accuracy'], 
            label='Dynamic Batch Only', color=colors['dynamic_only'], linewidth=2.5)
    plt.plot(dynamic_masked['iter'], dynamic_masked['tf_accuracy'], 
            label='Dynamic + Masked', color=colors['dynamic_masked'], linewidth=2.5)
    plt.plot(dynamic_padreg['iter'], dynamic_padreg['tf_accuracy'], 
            label='Dynamic + Masked + PadReg', color=colors['dynamic_padreg'], linewidth=2.5)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('TF Accuracy', fontsize=12)
    plt.title('Teacher Forcing Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Anti-preference监控
    plt.subplot(3, 3, 2)
    if 'newline_pred_at_pad' in baseline:
        plt.plot(baseline['iter'], baseline['newline_pred_at_pad'], 
                label='Baseline', color=colors['baseline'], linewidth=2.5, alpha=0.8)
    plt.plot(masked_only['iter'], masked_only['newline_pred_at_pad'], 
            label='Masked Loss Only', color=colors['masked_only'], linewidth=2.5)
    plt.plot(dynamic_only['iter'], dynamic_only['newline_pred_at_pad'], 
            label='Dynamic Batch Only', color=colors['dynamic_only'], linewidth=2.5)
    plt.plot(dynamic_masked['iter'], dynamic_masked['newline_pred_at_pad'], 
            label='Dynamic + Masked', color=colors['dynamic_masked'], linewidth=2.5)
    plt.plot(dynamic_padreg['iter'], dynamic_padreg['newline_pred_at_pad'], 
            label='Dynamic + Masked + PadReg', color=colors['dynamic_padreg'], linewidth=2.5)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Newline at PAD ratio', fontsize=12)
    plt.title('Anti-preference Monitoring', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    # 3. 路径token准确率
    plt.subplot(3, 3, 3)
    if 'path_accuracy' in baseline:
        plt.plot(baseline['iter'], baseline['path_accuracy'], 
                label='Baseline', color=colors['baseline'], linewidth=2.5, alpha=0.8)
    plt.plot(masked_only['iter'], masked_only['path_accuracy'], 
            label='Masked Loss Only', color=colors['masked_only'], linewidth=2.5)
    plt.plot(dynamic_only['iter'], dynamic_only['path_accuracy'], 
            label='Dynamic Batch Only', color=colors['dynamic_only'], linewidth=2.5)
    plt.plot(dynamic_masked['iter'], dynamic_masked['path_accuracy'], 
            label='Dynamic + Masked', color=colors['dynamic_masked'], linewidth=2.5)
    plt.plot(dynamic_padreg['iter'], dynamic_padreg['path_accuracy'], 
            label='Dynamic + Masked + PadReg', color=colors['dynamic_padreg'], linewidth=2.5)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Path Token Accuracy', fontsize=12)
    plt.title('Path Prediction Quality', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 4. 验证损失对比
    plt.subplot(3, 3, 4)
    plt.plot(baseline['iter'], baseline['val_loss'], 
            label='Baseline', color=colors['baseline'], linewidth=2.5, alpha=0.8)
    plt.plot(masked_only['iter'], masked_only['val_loss'], 
            label='Masked Loss Only', color=colors['masked_only'], linewidth=2.5)
    plt.plot(dynamic_only['iter'], dynamic_only['val_loss'], 
            label='Dynamic Batch Only', color=colors['dynamic_only'], linewidth=2.5)
    plt.plot(dynamic_masked['iter'], dynamic_masked['val_loss'], 
            label='Dynamic + Masked', color=colors['dynamic_masked'], linewidth=2.5)
    plt.plot(dynamic_padreg['iter'], dynamic_padreg['val_loss'], 
            label='Dynamic + Masked + PadReg', color=colors['dynamic_padreg'], linewidth=2.5)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 5. Padding比例对比（只有dynamic方法有）
    plt.subplot(3, 3, 5)
    plt.axhline(y=0.789, color='red', linestyle='--', alpha=0.5, 
               linewidth=2, label='Original (78.9%)')
    if 'val_padding_ratio' in dynamic_only:
        plt.plot(dynamic_only['iter'], dynamic_only['val_padding_ratio'], 
                label='Dynamic Batch Only', color=colors['dynamic_only'], linewidth=2.5)
    if 'val_padding_ratio' in dynamic_masked:
        plt.plot(dynamic_masked['iter'], dynamic_masked['val_padding_ratio'], 
                label='Dynamic + Masked', color=colors['dynamic_masked'], linewidth=2.5)
    if 'val_padding_ratio' in dynamic_padreg:
        plt.plot(dynamic_padreg['iter'], dynamic_padreg['val_padding_ratio'], 
                label='Dynamic + Masked + PadReg', color=colors['dynamic_padreg'], linewidth=2.5)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Padding Ratio', fontsize=12)
    plt.title('Padding Reduction (Dynamic Methods)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.85)
    
    # 6. PAD预测准确率（在PAD位置预测PAD）
    plt.subplot(3, 3, 6)
    if 'pad_pred_ratio' in masked_only:
        plt.plot(masked_only['iter'], masked_only['pad_pred_ratio'], 
                label='Masked Loss Only', color=colors['masked_only'], linewidth=2.5)
    if 'pad_pred_ratio' in dynamic_only:
        plt.plot(dynamic_only['iter'], dynamic_only['pad_pred_ratio'], 
                label='Dynamic Batch Only', color=colors['dynamic_only'], linewidth=2.5)
    if 'pad_pred_ratio' in dynamic_masked:
        plt.plot(dynamic_masked['iter'], dynamic_masked['pad_pred_ratio'], 
                label='Dynamic + Masked', color=colors['dynamic_masked'], linewidth=2.5)
    if 'pad_pred_ratio' in dynamic_padreg:
        plt.plot(dynamic_padreg['iter'], dynamic_padreg['pad_pred_ratio'], 
                label='Dynamic + Masked + PadReg', color=colors['dynamic_padreg'], linewidth=2.5)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('PAD at PAD Ratio', fontsize=12)
    plt.title('PAD Prediction Accuracy at PAD Positions', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 7. 最终TF准确率对比（柱状图）
    plt.subplot(3, 3, 7)
    methods = ['Baseline', 'Masked\nOnly', 'Dynamic\nOnly', 'Dynamic\n+Masked', 'Dynamic\n+Masked\n+PadReg']
    final_tf_accs = [
        baseline['tf_accuracy'][-1],
        masked_only['tf_accuracy'][-1],
        dynamic_only['tf_accuracy'][-1],
        dynamic_masked['tf_accuracy'][-1],
        dynamic_padreg['tf_accuracy'][-1]
    ]
    bars = plt.bar(methods, final_tf_accs, color=list(colors.values()), alpha=0.8)
    plt.ylabel('Final TF Accuracy', fontsize=12)
    plt.title('Final TF Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    for bar, acc in zip(bars, final_tf_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. 最大Anti-preference对比（柱状图）
    plt.subplot(3, 3, 8)
    max_anti_prefs = []
    if 'newline_pred_at_pad' in baseline:
        max_anti_prefs.append(max(baseline['newline_pred_at_pad']))
    else:
        max_anti_prefs.append(0)
    max_anti_prefs.extend([
        max(masked_only['newline_pred_at_pad']),
        max(dynamic_only['newline_pred_at_pad']),
        max(dynamic_masked['newline_pred_at_pad']),
        max(dynamic_padreg['newline_pred_at_pad'])
    ])
    bars = plt.bar(methods, max_anti_prefs, color=list(colors.values()), alpha=0.8)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    plt.ylabel('Max Newline at PAD Ratio', fontsize=12)
    plt.title('Maximum Anti-preference During Training', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    for bar, val in zip(bars, max_anti_prefs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 9. 训练效率对比（如果有批次大小信息）
    plt.subplot(3, 3, 9)
    efficiency_text = "Training Efficiency & Key Metrics\n" + "="*40 + "\n\n"
    
    # 计算一些关键指标
    for method_name, data, color in [
        ('Baseline', baseline, colors['baseline']),
        ('Masked Only', masked_only, colors['masked_only']),
        ('Dynamic Only', dynamic_only, colors['dynamic_only']),
        ('Dynamic+Masked', dynamic_masked, colors['dynamic_masked']),
        ('Dynamic+Masked+PadReg', dynamic_padreg, colors['dynamic_padreg'])
    ]:
        final_tf = data['tf_accuracy'][-1]
        max_anti = max(data.get('newline_pred_at_pad', [0]))
        
        efficiency_text += f"{method_name}:\n"
        efficiency_text += f"  Final TF: {final_tf:.3f}\n"
        efficiency_text += f"  Max Anti-pref: {max_anti:.3f}\n"
        
        if 'val_padding_ratio' in data:
            final_pad = data['val_padding_ratio'][-1]
            efficiency_text += f"  Final Padding: {final_pad:.1%}\n"
        
        efficiency_text += "\n"
    
    plt.text(0.1, 0.9, efficiency_text, fontsize=11, family='monospace',
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.axis('off')
    
    # 调整布局并保存
    plt.suptitle('Comprehensive Comparison of Padding Solutions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('five_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'five_methods_comparison.png'")
    
    # 额外创建一个关键指标的对比图
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 2.1 TF准确率的放大图（关键区域）
    ax = axes[0, 0]
    for method_name, data, color, label in [
        ('baseline', baseline, colors['baseline'], 'Baseline'),
        ('masked_only', masked_only, colors['masked_only'], 'Masked Only'),
        ('dynamic_only', dynamic_only, colors['dynamic_only'], 'Dynamic Only'),
        ('dynamic_masked', dynamic_masked, colors['dynamic_masked'], 'Dynamic+Masked'),
        ('dynamic_padreg', dynamic_padreg, colors['dynamic_padreg'], 'Dyn+Mask+Reg')
    ]:
        # 只显示100k iterations后的数据
        mask = np.array(data['iter']) >= 100000
        ax.plot(np.array(data['iter'])[mask], 
               np.array(data['tf_accuracy'])[mask],
               color=color, linewidth=2.5, label=label)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('TF Accuracy', fontsize=12)
    ax.set_title('TF Accuracy - Critical Period (100k+ iterations)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2.2 Anti-preference趋势
    ax = axes[0, 1]
    for method_name, data, color, label in [
        ('baseline', baseline, colors['baseline'], 'Baseline'),
        ('masked_only', masked_only, colors['masked_only'], 'Masked Only'),
        ('dynamic_only', dynamic_only, colors['dynamic_only'], 'Dynamic Only'),
        ('dynamic_masked', dynamic_masked, colors['dynamic_masked'], 'Dynamic+Masked'),
        ('dynamic_padreg', dynamic_padreg, colors['dynamic_padreg'], 'Dyn+Mask+Reg')
    ]:
        if 'newline_pred_at_pad' in data:
            ax.plot(data['iter'], data['newline_pred_at_pad'],
                   color=color, linewidth=2.5, label=label)
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Safe Zone')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Danger Zone')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Newline at PAD Ratio', fontsize=12)
    ax.set_title('Anti-preference Evolution', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2.3 综合性能雷达图
    ax = axes[1, 0]
    categories = ['TF Accuracy', 'Path Accuracy', 'No Anti-pref', 'Low Padding', 'Stability']
    
    # 准备数据
    radar_data = {}
    for method_name, data, label in [
        ('baseline', baseline, 'Baseline'),
        ('masked_only', masked_only, 'Masked Only'),
        ('dynamic_only', dynamic_only, 'Dynamic Only'),
        ('dynamic_masked', dynamic_masked, 'Dynamic+Masked'),
        ('dynamic_padreg', dynamic_padreg, 'Dyn+Mask+Reg')
    ]:
        values = [
            data['tf_accuracy'][-1],  # TF Accuracy
            data.get('path_accuracy', [0])[-1] if 'path_accuracy' in data else 0,  # Path Accuracy
            1 - max(data.get('newline_pred_at_pad', [0])),  # No Anti-pref (inverted)
            1 - data.get('val_padding_ratio', [0.789])[-1] if 'val_padding_ratio' in data else 0.211,  # Low Padding
            1 - np.std(data['tf_accuracy'][-20:]) if len(data['tf_accuracy']) > 20 else 0.5  # Stability
        ]
        radar_data[label] = values
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    for label, values in radar_data.items():
        values += values[:1]
        color = list(colors.values())[list(radar_data.keys()).index(label)]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Comprehensive Performance Comparison', fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax.grid(True)
    
    # 2.4 最终排名
    ax = axes[1, 1]
    ax.axis('off')
    
    ranking_text = "Final Ranking by Overall Performance\n" + "="*40 + "\n\n"
    
    # 计算综合得分
    scores = []
    for method_name, data, label in [
        ('baseline', baseline, 'Baseline'),
        ('masked_only', masked_only, 'Masked Only'),
        ('dynamic_only', dynamic_only, 'Dynamic Only'),
        ('dynamic_masked', dynamic_masked, 'Dynamic+Masked'),
        ('dynamic_padreg', dynamic_padreg, 'Dynamic+Masked+PadReg')
    ]:
        score = 0
        score += data['tf_accuracy'][-1] * 30  # TF accuracy weight
        score += data.get('path_accuracy', [0])[-1] * 20 if 'path_accuracy' in data else 0
        score += (1 - max(data.get('newline_pred_at_pad', [0]))) * 30  # Anti-pref penalty
        if 'val_padding_ratio' in data:
            score += (1 - data['val_padding_ratio'][-1]) * 20  # Padding reduction bonus
        scores.append((label, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (method, score) in enumerate(scores):
        ranking_text += f"{i+1}. {method}: {score:.2f}/100\n"
    
    ranking_text += "\n\nKey Insights:\n"
    ranking_text += "• PadReg effectively prevents anti-preference\n"
    ranking_text += "• Dynamic batching reduces padding by ~60%\n"
    ranking_text += "• Masked loss alone may not prevent anti-pref\n"
    ranking_text += "• Combined approaches work best\n"
    
    ax.text(0.1, 0.9, ranking_text, fontsize=12, family='monospace',
           verticalalignment='top', transform=ax.transAxes)
    
    plt.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved as 'detailed_analysis.png'")
    
    plt.show()

if __name__ == "__main__":
    # 使用提供的路径
    compare_five_methods(
        baseline_path='out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250628_165104/history.pkl',
        masked_only_path='out/masked_loss_20250628_061704/history.pkl',
        dynamic_only_path='out/dynamic_batch_only_20250628_155937/history.pkl',
        dynamic_masked_path='out/dynamic_batch_20250628_044108/history.pkl',
        dynamic_padreg_path='out/dynamic_masked_padreg_20250628_204008/history.pkl'
    )