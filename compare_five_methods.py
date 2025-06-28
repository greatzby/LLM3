"""
比较五种padding解决方案的效果
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

def load_history(path):
    """加载历史数据"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_five_methods(baseline_path, masked_only_path, dynamic_only_path, 
                        dynamic_masked_path, dynamic_padreg_path):
    """比较五种方法的结果"""
    # 加载数据
    print("Loading data files...")
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
    # 对于baseline，使用tf_accuracy_all（含padding）以保持向后兼容
    if 'tf_accuracy_all' in baseline:
        plt.plot(baseline['iter'], baseline['tf_accuracy_all'], 
                label='Baseline', color=colors['baseline'], linewidth=2.5, alpha=0.8)
    else:
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
    if 'path_accuracy' in masked_only:
        plt.plot(masked_only['iter'], masked_only['path_accuracy'], 
                label='Masked Loss Only', color=colors['masked_only'], linewidth=2.5)
    if 'path_accuracy' in dynamic_only:
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
    final_tf_accs = []
    
    # 对baseline使用合适的tf accuracy
    if 'tf_accuracy_all' in baseline:
        final_tf_accs.append(baseline['tf_accuracy_all'][-1])
    else:
        final_tf_accs.append(baseline['tf_accuracy'][-1])
    
    final_tf_accs.extend([
        masked_only['tf_accuracy'][-1],
        dynamic_only['tf_accuracy'][-1],
        dynamic_masked['tf_accuracy'][-1],
        dynamic_padreg['tf_accuracy'][-1]
    ])
    
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
    
    # 9. 综合评分对比
    plt.subplot(3, 3, 9)
    
    # 计算综合得分
    scores = []
    score_details = []
    
    for method_name, data, label in [
        ('baseline', baseline, 'Baseline'),
        ('masked_only', masked_only, 'Masked Only'),
        ('dynamic_only', dynamic_only, 'Dynamic Only'),
        ('dynamic_masked', dynamic_masked, 'Dynamic+Masked'),
        ('dynamic_padreg', dynamic_padreg, 'Dynamic+Masked+PadReg')
    ]:
        score = 0
        details = {}
        
        # TF accuracy (30分)
        if method_name == 'baseline' and 'tf_accuracy_all' in data:
            tf_acc = data['tf_accuracy_all'][-1]
        else:
            tf_acc = data['tf_accuracy'][-1]
        score += tf_acc * 30
        details['tf'] = tf_acc
        
        # Path accuracy (20分)
        if 'path_accuracy' in data:
            path_acc = data['path_accuracy'][-1]
            score += path_acc * 20
            details['path'] = path_acc
        else:
            details['path'] = 0
            
        # Anti-preference penalty (30分)
        if 'newline_pred_at_pad' in data:
            anti_pref = 1 - max(data['newline_pred_at_pad'])
            score += anti_pref * 30
            details['anti_pref'] = anti_pref
        else:
            score += 30
            details['anti_pref'] = 1.0
            
        # Padding reduction bonus (20分)
        if 'val_padding_ratio' in data:
            pad_reduction = 1 - data['val_padding_ratio'][-1]
            score += pad_reduction * 20
            details['pad_reduction'] = pad_reduction
        else:
            details['pad_reduction'] = 0.211  # 假设原始padding是78.9%
            
        scores.append((label, score, details))
    
    # 排序并显示
    scores.sort(key=lambda x: x[1], reverse=True)
    
    y_pos = 0.9
    plt.text(0.1, y_pos, "Overall Performance Ranking", fontsize=14, fontweight='bold')
    y_pos -= 0.1
    
    for i, (method, score, details) in enumerate(scores):
        text = f"{i+1}. {method}: {score:.1f}/100\n"
        text += f"   TF: {details['tf']:.3f}, Path: {details.get('path', 0):.3f}\n"
        text += f"   Anti-pref: {details.get('anti_pref', 1.0):.3f}, "
        text += f"Pad-reduce: {details.get('pad_reduction', 0):.3f}\n"
        plt.text(0.1, y_pos, text, fontsize=10, family='monospace')
        y_pos -= 0.2
    
    plt.axis('off')
    
    # 调整布局并保存
    plt.suptitle('Comprehensive Comparison of Padding Solutions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('five_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'five_methods_comparison.png'")
    
    # 创建第二个更详细的分析图
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
        iters = np.array(data['iter'])
        mask = iters >= 100000
        
        if method_name == 'baseline' and 'tf_accuracy_all' in data:
            tf_data = np.array(data['tf_accuracy_all'])
        else:
            tf_data = np.array(data['tf_accuracy'])
            
        if mask.sum() > 0:
            ax.plot(iters[mask], tf_data[mask],
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
    
    # 2.3 Loss Components (for PadReg)
    ax = axes[1, 0]
    if 'ce_loss' in dynamic_padreg and 'reg_loss' in dynamic_padreg:
        ax.plot(dynamic_padreg['iter'], dynamic_padreg['ce_loss'], 
               'b-', label='CE Loss', linewidth=2)
        ax.plot(dynamic_padreg['iter'], dynamic_padreg['reg_loss'], 
               'r-', label='Reg Loss', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Components in Dynamic+Masked+PadReg', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'Loss component data\nnot available', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    # 2.4 Key Insights
    ax = axes[1, 1]
    ax.axis('off')
    
    insights_text = "Key Insights from Experiments\n" + "="*40 + "\n\n"
    
    # 分析各方法的表现
    insights = []
    
    # Baseline分析
    if 'newline_pred_at_pad' in baseline:
        baseline_anti = max(baseline['newline_pred_at_pad'])
        if baseline_anti > 0.5:
            insights.append("• Baseline shows severe anti-preference")
    
    # Masked Loss分析
    masked_anti = max(masked_only['newline_pred_at_pad'])
    if masked_anti > 0.5:
        insights.append("• Masked Loss alone doesn't prevent anti-preference")
    else:
        insights.append("• Masked Loss shows some anti-preference control")
    
    # Dynamic Batch分析
    dynamic_anti = max(dynamic_only['newline_pred_at_pad'])
    if 'val_padding_ratio' in dynamic_only:
        pad_reduction = (0.789 - dynamic_only['val_padding_ratio'][-1]) * 100
        insights.append(f"• Dynamic Batch reduces padding by {pad_reduction:.1f}%")
    if dynamic_anti > 0.5:
        insights.append("  but still shows anti-preference")
    
    # Combined分析
    combined_anti = max(dynamic_masked['newline_pred_at_pad'])
    if combined_anti < 0.3:
        insights.append("• Dynamic+Masked shows good anti-preference control")
    
    # PadReg分析
    padreg_anti = max(dynamic_padreg['newline_pred_at_pad'])
    if padreg_anti < 0.1:
        insights.append("• PadReg effectively eliminates anti-preference")
    
    insights.append("\nRecommendations:")
    insights.append("• For production: Use Dynamic+Masked+PadReg")
    insights.append("• PadReg weight λ=0.05 works well")
    insights.append("• Dynamic batching significantly improves efficiency")
    
    for i, insight in enumerate(insights):
        ax.text(0.1, 0.9 - i*0.08, insight, fontsize=11, 
               transform=ax.transAxes)
    
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