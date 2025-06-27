"""
比较不同padding解决方案的效果
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_history(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_experiments(baseline_path, masked_path, dynamic_path):
    """比较三种方法的结果"""
    # 加载数据
    baseline = load_history(baseline_path) if baseline_path else None
    masked = load_history(masked_path)
    dynamic = load_history(dynamic_path)
    
    plt.figure(figsize=(20, 12))
    
    # 1. TF准确率对比
    plt.subplot(2, 3, 1)
    if baseline:
        plt.plot(baseline['iter'], baseline['tf_accuracy'], 
                label='Baseline', linewidth=2, alpha=0.7)
    plt.plot(masked['iter'], masked['tf_accuracy'], 
            label='Masked Loss', linewidth=2)
    plt.plot(dynamic['iter'], dynamic['tf_accuracy'], 
            label='Dynamic Batch', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Teacher Forcing Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Anti-preference监控
    plt.subplot(2, 3, 2)
    plt.plot(masked['iter'], masked['newline_pred_at_pad'], 
            label='Masked Loss', linewidth=2)
    plt.plot(dynamic['iter'], dynamic['newline_pred_at_pad'], 
            label='Dynamic Batch', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Newline at PAD ratio')
    plt.title('Anti-preference Monitoring')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Padding比例（仅Dynamic Batch）
    plt.subplot(2, 3, 3)
    plt.plot(dynamic['iter'], dynamic['train_padding_ratio'], 
            label='Train', linewidth=2)
    plt.plot(dynamic['iter'], dynamic['val_padding_ratio'], 
            label='Val', linewidth=2, linestyle='--')
    plt.axhline(y=0.789, color='r', linestyle='--', alpha=0.5, 
               label='Original (78.9%)')
    plt.xlabel('Iteration')
    plt.ylabel('Padding Ratio')
    plt.title('Padding Reduction (Dynamic Batch)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 损失曲线
    plt.subplot(2, 3, 4)
    plt.plot(masked['iter'], masked['train_loss'], 
            label='Masked Train', linewidth=1, alpha=0.7)
    plt.plot(masked['iter'], masked['val_loss'], 
            label='Masked Val', linewidth=2)
    plt.plot(dynamic['iter'], dynamic['train_loss'], 
            label='Dynamic Train', linewidth=1, alpha=0.7, linestyle='--')
    plt.plot(dynamic['iter'], dynamic['val_loss'], 
            label='Dynamic Val', linewidth=2, linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 5. 路径token准确率
    plt.subplot(2, 3, 5)
    plt.plot(masked['iter'], masked['path_accuracy'], 
            label='Masked Loss', linewidth=2)
    plt.plot(dynamic['iter'], dynamic['path_accuracy'], 
            label='Dynamic Batch', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Path Token Accuracy')
    plt.title('Path Prediction Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 总结统计
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, "Summary Statistics", fontsize=16, fontweight='bold')
    
    # Masked Loss统计
    final_tf_masked = masked['tf_accuracy'][-1]
    max_anti_masked = max(masked['newline_pred_at_pad'])
    
    # Dynamic Batch统计
    final_tf_dynamic = dynamic['tf_accuracy'][-1]
    max_anti_dynamic = max(dynamic['newline_pred_at_pad'])
    avg_padding = np.mean(dynamic['train_padding_ratio'])
    
    stats_text = f"""
Masked Loss:
  Final TF Accuracy: {final_tf_masked:.3f}
  Max Anti-preference: {max_anti_masked:.3f}
  
Dynamic Batching:
  Final TF Accuracy: {final_tf_dynamic:.3f}
  Max Anti-preference: {max_anti_dynamic:.3f}
  Avg Padding: {avg_padding:.1%} (reduced from 78.9%)
  Padding Reduction: {(0.789 - avg_padding)*100:.1f}%
"""
    
    plt.text(0.1, 0.1, stats_text, fontsize=12, family='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('padding_solutions_comparison.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # 修改这些路径为你的实际实验结果路径
    compare_experiments(
        baseline_path=None,  # 如果有baseline结果
        masked_path='out/masked_loss_20250628_025248/history.pkl',
        dynamic_path='out/dynamic_batch_20250628_044108/history.pkl'
    )