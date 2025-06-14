"""
相变分析脚本 - 分析训练结果中的相变现象
使用方法: python analyze_phase.py
"""
import os
import pickle
import json
import numpy as np
import torch

def analyze_experiment(out_dir):
    """分析单个实验的结果"""
    print(f"\n分析实验目录: {out_dir}")
    print("="*60)
    
    # 1. 读取训练统计
    stats_file = os.path.join(out_dir, 'training_stats.pkl')
    if not os.path.exists(stats_file):
        print(f"错误: 找不到 {stats_file}")
        return None
        
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
    
    # 2. 分析相变
    tf_accuracy = stats['tf_accuracy']
    iterations = stats['iterations']
    
    # 找出最高准确率
    max_tf_acc = max(tf_accuracy)
    max_idx = tf_accuracy.index(max_tf_acc)
    max_iter = iterations[max_idx]
    
    # 找出最终准确率
    final_tf_acc = tf_accuracy[-1]
    
    print(f"\n关键指标:")
    print(f"  最高TF准确率: {max_tf_acc:.4f} (在{max_iter}次迭代)")
    print(f"  最终TF准确率: {final_tf_acc:.4f}")
    print(f"  准确率下降: {max_tf_acc - final_tf_acc:.4f}")
    
    # 3. 检测相变点
    phase_transitions = []
    for i in range(5, len(tf_accuracy)):
        # 检查前5个点的平均值
        recent_avg = np.mean(tf_accuracy[i-5:i])
        current = tf_accuracy[i]
        
        # 如果下降超过10%，认为是相变
        if recent_avg > 0.8 and current < recent_avg - 0.1:
            phase_transitions.append({
                'iteration': iterations[i],
                'before': recent_avg,
                'after': current,
                'drop': recent_avg - current
            })
    
    if phase_transitions:
        print(f"\n检测到相变:")
        for t in phase_transitions:
            print(f"  - {t['iteration']}次迭代: {t['before']:.4f} → {t['after']:.4f} (下降{t['drop']:.4f})")
    else:
        print(f"\n未检测到明显相变")
    
    # 4. 分析checkpoint
    print(f"\n可用的checkpoint:")
    checkpoints = []
    for file in os.listdir(out_dir):
        if file.endswith('_ckpt.pt') or file.endswith('_ckpt_20.pt'):
            try:
                iter_num = int(file.split('_')[0])
                checkpoints.append(iter_num)
            except:
                continue
    
    checkpoints.sort()
    print(f"  共{len(checkpoints)}个: {checkpoints}")
    
    # 5. 保存分析结果
    analysis_result = {
        'out_dir': out_dir,
        'max_tf_accuracy': float(max_tf_acc),
        'max_tf_iteration': int(max_iter),
        'final_tf_accuracy': float(final_tf_acc),
        'accuracy_drop': float(max_tf_acc - final_tf_acc),
        'phase_transitions': phase_transitions,
        'checkpoints': checkpoints
    }
    
    output_file = os.path.join(out_dir, 'phase_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    print(f"\n分析结果已保存到: {output_file}")
    
    return analysis_result

def main():
    """主函数"""
    # 查找所有实验目录
    base_dir = 'out'
    experiment_dirs = []
    
    if os.path.exists(base_dir):
        for dir_name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.isdir(dir_path) and dir_name.startswith('simple_graph'):
                experiment_dirs.append(dir_path)
    
    if not experiment_dirs:
        print("没有找到实验结果！")
        print("请先运行训练: python train.py --max_iters 200000")
        return
    
    print(f"找到{len(experiment_dirs)}个实验:")
    for dir_path in experiment_dirs:
        print(f"  - {dir_path}")
    
    # 分析每个实验
    all_results = []
    for dir_path in experiment_dirs:
        result = analyze_experiment(dir_path)
        if result:
            all_results.append(result)
    
    # 总结
    if all_results:
        print("\n" + "="*60)
        print("总结:")
        for r in all_results:
            print(f"\n{r['out_dir']}:")
            print(f"  最高准确率: {r['max_tf_accuracy']:.4f}")
            print(f"  相变: {'是' if r['phase_transitions'] else '否'}")
            if r['phase_transitions']:
                print(f"  相变时机: {r['phase_transitions'][0]['iteration']}次迭代")

if __name__ == "__main__":
    main()