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
        sequence = val_data[start:start + block_size]
        target = val_data[start + 1:start + 1 + block_size]
        
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
    model.eval()  # 但我们仍需要梯度
    
    # 收集每个可能的下一跳节点的梯度
    node_gradients = defaultdict(list)
    node_probabilities = defaultdict(list)
    
    for sample in tqdm(samples, desc="Computing gradients"):
        # 准备输入
        x = sample['input'].unsqueeze(0).to(device)
        y = sample['target'].unsqueeze(0).to(device)
        
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
        lm_head_grad = model.lm_head.weight.grad
        
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
            if true_node not in [n for n, _ in enumerate(top_indices) if n >= 2]:
                node_gradients[true_node].append(true_grad_norm)
                node_probabilities[true_node].append(key_probs[true_next].item())
    
    # 计算统计信息
    gradient_stats = {}
    for node, grads in node_gradients.items():
        if grads:
            gradient_stats[node] = {
                'mean_gradient': np.mean(grads),
                'std_gradient': np.std(grads),
                'max_gradient': np.max(grads),
                'min_gradient': np.min(grads),
                'count': len(grads),
                'mean_probability': np.mean(node_probabilities[node])
            }
    
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
        
        # 加载模型
        model = load_model_checkpoint(ckpt_iter)
        
        # 计算梯度分布
        gradient_stats = compute_gradient_distribution(model, samples)
        
        # 保存结果
        all_results[ckpt_iter] = gradient_stats
        
        # 打印关键节点的梯度信息
        print(f"\nTop nodes by frequency at {ckpt_iter}:")
        sorted_nodes = sorted(gradient_stats.items(), 
                            key=lambda x: x[1]['count'], 
                            reverse=True)[:10]
        
        for node, stats in sorted_nodes:
            print(f"  Node {node}: "
                  f"grad={stats['mean_gradient']:.6f}, "
                  f"prob={stats['mean_probability']:.4f}, "
                  f"count={stats['count']}")
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache()
    
    # 保存完整结果
    with open(os.path.join(OUTPUT_DIR, 'gradient_analysis.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 分析node 2的梯度演化（假设node 2是最常见的）
    analyze_specific_node_evolution(all_results, target_node=2)
    
    # 生成可视化
    visualize_gradient_evolution(all_results)
    
    return all_results

def analyze_specific_node_evolution(all_results, target_node=2):
    """分析特定节点的梯度演化"""
    print(f"\n{'='*60}")
    print(f"Node {target_node} gradient evolution:")
    
    iterations = []
    gradients = []
    probabilities = []
    
    for ckpt_iter in CHECKPOINTS:
        if target_node in all_results[ckpt_iter]:
            stats = all_results[ckpt_iter][target_node]
            iterations.append(ckpt_iter)
            gradients.append(stats['mean_gradient'])
            probabilities.append(stats['mean_probability'])
            
            print(f"  {ckpt_iter}: grad={stats['mean_gradient']:.6f}, "
                  f"prob={stats['mean_probability']:.4f}")
    
    # 检测梯度饱和
    if len(gradients) > 1:
        # 计算梯度下降率
        grad_reduction = (gradients[0] - gradients[-1]) / gradients[0]
        print(f"\nGradient reduction: {grad_reduction:.2%}")
        
        # 检查是否存在梯度崩溃点
        for i in range(1, len(gradients)):
            if gradients[i] < gradients[i-1] * 0.1:  # 梯度下降90%以上
                print(f"WARNING: Gradient collapse detected between "
                      f"{iterations[i-1]} and {iterations[i]}!")

def visualize_gradient_evolution(all_results):
    """可视化梯度演化过程"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top节点的梯度演化
    ax = axes[0, 0]
    
    # 找出最常见的节点
    all_nodes = set()
    for result in all_results.values():
        all_nodes.update(result.keys())
    
    node_counts = {}
    for node in all_nodes:
        count = sum(1 for r in all_results.values() if node in r)
        node_counts[node] = count
    
    top_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for node, _ in top_nodes:
        iterations = []
        gradients = []
        
        for ckpt_iter in CHECKPOINTS:
            if node in all_results[ckpt_iter]:
                iterations.append(ckpt_iter)
                gradients.append(all_results[ckpt_iter][node]['mean_gradient'])
        
        if iterations:
            ax.plot(iterations, gradients, marker='o', label=f'Node {node}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Gradient Norm')
    ax.set_title('Gradient Evolution of Top Nodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 梯度vs概率散点图
    ax = axes[0, 1]
    
    for i, (ckpt_iter, color) in enumerate(zip([100000, 140000, 180000], ['blue', 'orange', 'red'])):
        if ckpt_iter in all_results:
            probs = []
            grads = []
            
            for node, stats in all_results[ckpt_iter].items():
                probs.append(stats['mean_probability'])
                grads.append(stats['mean_gradient'])
            
            ax.scatter(probs, grads, alpha=0.6, color=color, label=f'{ckpt_iter//1000}k')
    
    ax.set_xlabel('Mean Probability')
    ax.set_ylabel('Mean Gradient Norm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Gradient vs Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 梯度分布直方图
    ax = axes[1, 0]
    
    for ckpt_iter, color in zip([100000, 140000, 180000], ['blue', 'orange', 'red']):
        if ckpt_iter in all_results:
            all_gradients = []
            for stats in all_results[ckpt_iter].values():
                all_gradients.append(stats['mean_gradient'])
            
            ax.hist(all_gradients, bins=30, alpha=0.5, color=color, 
                   label=f'{ckpt_iter//1000}k', density=True)
    
    ax.set_xlabel('Gradient Norm')
    ax.set_ylabel('Density')
    ax.set_title('Gradient Distribution Evolution')
    ax.legend()
    ax.set_yscale('log')
    
    # 4. 相变指标
    ax = axes[1, 1]
    
    iterations = []
    total_gradient = []
    gradient_variance = []
    
    for ckpt_iter in CHECKPOINTS:
        if ckpt_iter in all_results:
            grads = [stats['mean_gradient'] for stats in all_results[ckpt_iter].values()]
            iterations.append(ckpt_iter)
            total_gradient.append(np.mean(grads))
            gradient_variance.append(np.var(grads))
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(iterations, total_gradient, 'b-', marker='o', label='Mean Gradient')
    line2 = ax2.plot(iterations, gradient_variance, 'r-', marker='s', label='Gradient Variance')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Gradient', color='b')
    ax2.set_ylabel('Gradient Variance', color='r')
    ax.set_title('Phase Transition Indicators')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    
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
    
    # 创建合成输入
    # source=0, target=5, current=0
    synthetic_input = torch.tensor([[2, 7, 2]], dtype=torch.long).cuda()  # +2 for token offset
    
    # 测试不同的目标概率
    target_probs = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]
    
    results = []
    
    for target_prob in target_probs:
        model.zero_grad()
        
        # 前向传播
        logits, _ = model(synthetic_input)
        key_logits = logits[0, -1, :]  # 最后一个位置
        
        # 创建目标分布：大部分概率给node 2 (token 4)
        target_dist = torch.zeros_like(key_logits)
        target_dist[4] = target_prob  # node 2
        target_dist[5] = (1 - target_prob) / 2  # node 3
        target_dist[6] = (1 - target_prob) / 2  # node 4
        
        # KL损失（模拟训练）
        log_probs = F.log_softmax(key_logits, dim=0)
        loss = F.kl_div(log_probs, target_dist, reduction='sum')
        
        # 反向传播
        loss.backward()
        
        # 获取node 2的梯度
        node2_grad = torch.norm(model.lm_head.weight.grad[4]).item()
        
        results.append({
            'target_prob': target_prob,
            'gradient': node2_grad,
            'loss': loss.item()
        })
        
        print(f"Target prob={target_prob:.3f}: grad={node2_grad:.6f}, loss={loss.item():.6f}")
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    
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
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradient_saturation_curve.png'), dpi=150)
    plt.close()
    
    print(f"Saturation curve saved to {OUTPUT_DIR}/gradient_saturation_curve.png")

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