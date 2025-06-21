"""
验证梯度饱和导致相变的假说
通过分析不同checkpoint在相同输入上的梯度响应
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pickle
import json
from tqdm import tqdm

from model import GPTConfig, GPT

# 配置
STANDARD_DIR = "out/simple_graph_1_1_120_100_original_seed42"
DATA_DIR = "data/simple_graph/100"
OUTPUT_DIR = "gradient_saturation_analysis"

# 关键checkpoint
CHECKPOINTS = [100000, 120000, 140000, 160000, 180000, 200000]

def load_model_checkpoint(iteration, device='cuda'):
    """加载特定迭代的模型"""
    ckpt_path = os.path.join(STANDARD_DIR, f"{iteration}_ckpt_20.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
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
    model.to(device)
    model.eval()
    
    return model

def prepare_test_samples(num_samples=100):
    """准备测试样本 - 使用验证集的真实数据"""
    # 加载元数据
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    
    # 加载验证数据
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    
    # 准备样本
    samples = []
    block_size = meta['block_size']
    data_size = block_size + 1
    
    # 随机选择样本
    num_sequences = (len(val_data) - data_size) // data_size
    selected_indices = np.random.choice(num_sequences, min(num_samples, num_sequences), replace=False)
    
    for idx in selected_indices:
        start = idx * data_size
        # 修复：先转换为int64
        sequence = val_data[start:start + block_size].astype(np.int64)
        target = val_data[start + 1:start + 1 + block_size].astype(np.int64)
        
        # 找到第4个位置（source target source之后的第一个预测位置）
        # 这是最关键的预测位置
        samples.append({
            'input': torch.tensor(sequence, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'key_position': 3,  # 第4个位置（0-indexed）
            'source': int(sequence[0]) - 2,  # 减2因为token偏移
            'destination': int(sequence[1]) - 2
        })
    
    return samples, stoi

def compute_gradient_distribution(model, samples, device='cuda'):
    """计算模型在样本上的梯度分布"""
    model.train()  # 需要梯度计算
    
    # 收集每个可能的下一跳节点的梯度
    node_gradients = defaultdict(list)
    node_probabilities = defaultdict(list)
    true_answer_gradients = []
    true_answer_probs = []
    
    for sample in tqdm(samples, desc="Computing gradients"):
        # 准备输入
        x = sample['input'].unsqueeze(0).to(device)
        y = sample['target'].unsqueeze(0).to(device)
        
        # 确保需要梯度
        x.requires_grad = False
        
        # 前向传播
        model.zero_grad()
        logits, _ = model(x)
        
        # 获取关键位置的logits
        key_logits = logits[0, sample['key_position'], :]
        key_probs = F.softmax(key_logits, dim=0)
        
        # 计算在真实目标上的损失
        true_next = y[0, sample['key_position']]
        loss = F.cross_entropy(key_logits.unsqueeze(0), true_next.unsqueeze(0))
        
        # 反向传播
        loss.backward()
        
        # 收集lm_head的梯度信息
        if model.lm_head.weight.grad is not None:
            lm_head_grad = model.lm_head.weight.grad.clone()
            
            # 分析top-k可能的下一跳
            top_k = 10
            top_probs, top_indices = torch.topk(key_probs, top_k)
            
            for i in range(top_k):
                node_id = top_indices[i].item()
                if node_id >= 2:  # 跳过特殊token
                    actual_node = node_id - 2
                    
                    # 记录该节点的梯度范数
                    grad_norm = torch.norm(lm_head_grad[node_id]).item()
                    node_gradients[actual_node].append(grad_norm)
                    
                    # 记录概率
                    prob = top_probs[i].item()
                    node_probabilities[actual_node].append(prob)
            
            # 特别记录真实答案的梯度
            if true_next >= 2:
                true_node = true_next.item() - 2
                true_grad_norm = torch.norm(lm_head_grad[true_next]).item()
                true_prob = key_probs[true_next].item()
                
                true_answer_gradients.append(true_grad_norm)
                true_answer_probs.append(true_prob)
    
    # 计算统计信息
    gradient_stats = {}
    for node, grads in node_gradients.items():
        if grads:
            gradient_stats[node] = {
                'mean_gradient': float(np.mean(grads)),
                'std_gradient': float(np.std(grads)),
                'max_gradient': float(np.max(grads)),
                'min_gradient': float(np.min(grads)),
                'count': len(grads),
                'mean_probability': float(np.mean(node_probabilities[node]))
            }
    
    # 添加真实答案的统计
    if true_answer_gradients:
        gradient_stats['_true_answers'] = {
            'mean_gradient': float(np.mean(true_answer_gradients)),
            'std_gradient': float(np.std(true_answer_gradients)),
            'mean_probability': float(np.mean(true_answer_probs)),
            'count': len(true_answer_gradients)
        }
    
    model.eval()  # 恢复eval模式
    return gradient_stats

def analyze_phase_transition():
    """分析相变过程中的梯度变化"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 准备测试样本
    print("Preparing test samples...")
    samples, stoi = prepare_test_samples(num_samples=200)
    
    # 存储所有checkpoint的结果
    all_results = {}
    
    # 对每个checkpoint进行分析
    for ckpt_iter in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint {ckpt_iter}...")
        
        try:
            # 加载模型
            model = load_model_checkpoint(ckpt_iter)
            
            # 计算梯度分布
            gradient_stats = compute_gradient_distribution(model, samples)
            
            # 保存结果
            all_results[ckpt_iter] = gradient_stats
            
            # 打印关键节点的梯度信息
            print(f"\nTop nodes by frequency at {ckpt_iter}:")
            
            # 过滤掉特殊键
            regular_nodes = {k: v for k, v in gradient_stats.items() if not str(k).startswith('_')}
            sorted_nodes = sorted(regular_nodes.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True)[:10]
            
            for node, stats in sorted_nodes:
                print(f"  Node {node}: "
                      f"grad={stats['mean_gradient']:.6f}, "
                      f"prob={stats['mean_probability']:.4f}, "
                      f"count={stats['count']}")
            
            # 打印真实答案的统计
            if '_true_answers' in gradient_stats:
                true_stats = gradient_stats['_true_answers']
                print(f"\nTrue answers stats:")
                print(f"  Mean gradient: {true_stats['mean_gradient']:.6f}")
                print(f"  Mean probability: {true_stats['mean_probability']:.4f}")
            
            # 清理GPU内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing checkpoint {ckpt_iter}: {e}")
            continue
    
    # 保存完整结果
    with open(os.path.join(OUTPUT_DIR, 'gradient_analysis.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 分析梯度演化
    analyze_gradient_evolution_patterns(all_results)
    
    # 生成可视化
    visualize_gradient_evolution(all_results)
    
    return all_results

def analyze_gradient_evolution_patterns(all_results):
    """分析梯度演化模式"""
    print(f"\n{'='*60}")
    print("Gradient Evolution Analysis:")
    
    # 追踪真实答案的梯度变化
    if all([k in all_results and '_true_answers' in all_results[k] for k in CHECKPOINTS]):
        print("\nTrue Answer Gradient Evolution:")
        for ckpt in CHECKPOINTS:
            if ckpt in all_results and '_true_answers' in all_results[ckpt]:
                stats = all_results[ckpt]['_true_answers']
                print(f"  {ckpt}: grad={stats['mean_gradient']:.6f}, prob={stats['mean_probability']:.4f}")
    
    # 找出梯度变化最大的节点
    print("\nNodes with largest gradient changes:")
    node_changes = {}
    
    # 计算每个节点的梯度变化
    all_nodes = set()
    for result in all_results.values():
        all_nodes.update([k for k in result.keys() if not str(k).startswith('_')])
    
    for node in all_nodes:
        gradients = []
        for ckpt in CHECKPOINTS:
            if ckpt in all_results and node in all_results[ckpt]:
                gradients.append(all_results[ckpt][node]['mean_gradient'])
        
        if len(gradients) >= 2:
            change = max(gradients) - min(gradients)
            node_changes[node] = {
                'change': change,
                'gradients': gradients
            }
    
    # 排序并打印top变化
    sorted_changes = sorted(node_changes.items(), key=lambda x: x[1]['change'], reverse=True)[:5]
    for node, info in sorted_changes:
        print(f"  Node {node}: change={info['change']:.6f}")

def visualize_gradient_evolution(all_results):
    """可视化梯度演化过程"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 真实答案的梯度和概率演化
    ax = axes[0, 0]
    
    true_grads = []
    true_probs = []
    iterations = []
    
    for ckpt in CHECKPOINTS:
        if ckpt in all_results and '_true_answers' in all_results[ckpt]:
            stats = all_results[ckpt]['_true_answers']
            true_grads.append(stats['mean_gradient'])
            true_probs.append(stats['mean_probability'])
            iterations.append(ckpt)
    
    if iterations:
        ax2 = ax.twinx()
        
        line1 = ax.plot(iterations, true_grads, 'b-', marker='o', markersize=8, 
                        linewidth=2, label='Gradient')
        line2 = ax2.plot(iterations, true_probs, 'r-', marker='s', markersize=8, 
                         linewidth=2, label='Probability')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Gradient', color='b')
        ax2.set_ylabel('Mean Probability', color='r')
        ax.set_title('True Answer: Gradient vs Probability')
        
        # 标记相变区域
        ax.axvspan(130000, 150000, alpha=0.2, color='gray', label='Transition zone')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        ax.grid(True, alpha=0.3)
    
    # 2. Top节点的梯度演化
    ax = axes[0, 1]
    
    # 找出最常见的节点
    node_counts = defaultdict(int)
    for result in all_results.values():
        for node in result:
            if not str(node).startswith('_'):
                node_counts[node] += 1
    
    top_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for node, _ in top_nodes:
        iterations = []
        gradients = []
        
        for ckpt in CHECKPOINTS:
            if ckpt in all_results and node in all_results[ckpt]:
                iterations.append(ckpt)
                gradients.append(all_results[ckpt][node]['mean_gradient'])
        
        if iterations:
            ax.plot(iterations, gradients, marker='o', label=f'Node {node}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Gradient Norm')
    ax.set_title('Gradient Evolution of Top Nodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 梯度vs概率散点图（140k时刻）
    ax = axes[1, 0]
    
    if 140000 in all_results:
        probs = []
        grads = []
        nodes = []
        
        for node, stats in all_results[140000].items():
            if not str(node).startswith('_'):
                probs.append(stats['mean_probability'])
                grads.append(stats['mean_gradient'])
                nodes.append(node)
        
        scatter = ax.scatter(probs, grads, alpha=0.6, s=100)
        
        # 标注一些关键点
        for i, node in enumerate(nodes):
            if probs[i] > 0.5 or grads[i] < 0.001:
                ax.annotate(f'{node}', (probs[i], grads[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Mean Probability')
        ax.set_ylabel('Mean Gradient Norm')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Gradient vs Probability at 140k (Phase Transition)')
        ax.grid(True, alpha=0.3)
    
    # 4. 梯度比率演化
    ax = axes[1, 1]
    
    ratios = []
    iterations = []
    
    for ckpt in CHECKPOINTS:
        if ckpt in all_results and '_true_answers' in all_results[ckpt]:
            true_grad = all_results[ckpt]['_true_answers']['mean_gradient']
            
            # 计算其他节点的平均梯度
            other_grads = []
            for node, stats in all_results[ckpt].items():
                if not str(node).startswith('_'):
                    other_grads.append(stats['mean_gradient'])
            
            if other_grads:
                avg_other = np.mean(other_grads)
                ratio = true_grad / (avg_other + 1e-8)
                ratios.append(ratio)
                iterations.append(ckpt)
    
    if iterations:
        ax.plot(iterations, ratios, 'g-', marker='o', markersize=8, linewidth=2)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('True Answer Gradient / Average Other Gradient')
        ax.set_title('Relative Gradient Strength')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradient_evolution.png'), dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {OUTPUT_DIR}/gradient_evolution.png")

def run_controlled_experiment():
    """运行控制实验：手动设置不同概率看梯度变化"""
    print("\n" + "="*60)
    print("Running controlled gradient experiment...")
    
    # 加载一个模型
    model = load_model_checkpoint(120000)
    model.train()  # 需要梯度
    
    # 创建合成输入
    # source=0, target=5, current=0
    synthetic_input = torch.tensor([[2, 7, 2]], dtype=torch.long).cuda()  # +2 for token offset
    
    # 测试不同的目标概率
    target_probs = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    
    results = []
    
    for target_prob in target_probs:
        model.zero_grad()
        
        # 前向传播
        logits, _ = model(synthetic_input)
        key_logits = logits[0, -1, :]  # 最后一个位置
        
        # 假设node 2 (token 4)是目标
        target_token = 4
        
        # 使用交叉熵损失
        # 创建one-hot目标（但是使用soft标签）
        soft_target = torch.zeros_like(key_logits)
        soft_target[target_token] = target_prob
        # 剩余概率均匀分布
        remaining_prob = (1 - target_prob) / (len(key_logits) - 1)
        soft_target[soft_target == 0] = remaining_prob
        
        # 计算损失
        log_probs = F.log_softmax(key_logits, dim=0)
        loss = -torch.sum(soft_target * log_probs)
        
        # 反向传播
        loss.backward()
        
        # 获取目标token的梯度
        target_grad = torch.norm(model.lm_head.weight.grad[target_token]).item()
        
        # 获取当前概率
        current_prob = F.softmax(key_logits, dim=0)[target_token].item()
        
        results.append({
            'target_prob': target_prob,
            'current_prob': current_prob,
            'gradient': target_grad,
            'loss': loss.item()
        })
        
        print(f"Target prob={target_prob:.3f}: current_prob={current_prob:.3f}, "
              f"grad={target_grad:.6f}, loss={loss.item():.6f}")
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    # 子图1：梯度vs目标概率
    plt.subplot(1, 2, 1)
    probs = [r['target_prob'] for r in results]
    grads = [r['gradient'] for r in results]
    
    plt.plot(probs, grads, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Target Probability')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Saturation Curve')
    plt.grid(True, alpha=0.3)
    
    # 标记饱和区域
    plt.axvspan(0.95, 1.0, alpha=0.2, color='red', label='Saturation Zone')
    plt.legend()
    
    # 子图2：损失vs目标概率
    plt.subplot(1, 2, 2)
    losses = [r['loss'] for r in results]
    
    plt.plot(probs, losses, 'ro-', markersize=8, linewidth=2)
    plt.xlabel('Target Probability')
    plt.ylabel('Loss')
    plt.title('Loss vs Target Probability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradient_saturation_curve.png'), dpi=150)
    plt.close()
    
    print(f"Saturation curve saved to {OUTPUT_DIR}/gradient_saturation_curve.png")
    
    model.eval()

if __name__ == "__main__":
    print("="*60)
    print("Gradient Saturation Verification")
    print("="*60)
    
    # 主分析
    results = analyze_phase_transition()
    
    # 控制实验
    run_controlled_experiment()
    
    print("\n" + "="*60)
    print("Analysis complete! Check the output directory for results.")