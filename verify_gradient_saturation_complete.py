"""
验证梯度饱和导致相变的假说 - 完整版
包含数据格式探索和错误处理
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

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path

def load_model_checkpoint(iteration, device='cuda'):
    """加载特定迭代的模型"""
    if not torch.cuda.is_available():
        device = 'cpu'
        
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

def explore_data_format():
    """探索数据格式"""
    print("\n" + "="*60)
    print("Exploring data format...")
    
    # 加载元数据
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    print(f"Metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    
    # 检查数据文件
    val_path = os.path.join(DATA_DIR, 'val.bin')
    if os.path.exists(val_path):
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        print(f"\nValidation data shape: {val_data.shape}")
        print(f"First 20 tokens: {val_data[:20]}")
        
        # 尝试理解数据格式
        block_size = meta.get('block_size', 1)
        print(f"\nAssuming block_size={block_size}")
        
        # 显示几个样本
        print("\nFirst few sequences:")
        for i in range(min(3, len(val_data) // (block_size + 1))):
            start = i * (block_size + 1)
            end = start + block_size + 1
            seq = val_data[start:end]
            print(f"  Sequence {i}: {seq}")
            
            # 解码为实际节点（减去偏移）
            decoded = [int(x) - 2 if x >= 2 else x for x in seq]
            print(f"  Decoded: {decoded}")
    
    return meta

def create_synthetic_samples(num_samples=100, num_nodes=100):
    """创建合成样本进行测试"""
    print(f"\nCreating {num_samples} synthetic samples...")
    
    samples = []
    
    for _ in range(num_samples):
        # 创建一个简单的路径
        source = np.random.randint(0, num_nodes - 10)
        target = source + np.random.randint(5, 10)
        
        # 创建路径：source -> intermediate -> target
        path_length = np.random.randint(3, 6)
        path = [source]
        
        for i in range(path_length - 2):
            # 选择介于source和target之间的节点
            next_node = source + (i + 1) * (target - source) // (path_length - 1)
            path.append(next_node)
        
        path.append(target)
        
        # 转换为token序列（+2偏移）
        token_sequence = [source + 2, target + 2] + [p + 2 for p in path]
        
        # 创建多个预测点
        for pos in range(2, len(token_sequence) - 1):
            samples.append({
                'input': torch.tensor(token_sequence[:pos+1], dtype=torch.long),
                'target': token_sequence[pos+1],
                'position': pos,
                'source': source,
                'destination': target,
                'full_path': path
            })
    
    return samples

def compute_gradient_statistics(model, samples, device='cuda'):
    """计算梯度统计"""
    if not torch.cuda.is_available():
        device = 'cpu'
    
    model.to(device)
    model.eval()
    
    # 统计数据
    gradient_data = {
        'by_node': defaultdict(list),
        'by_position': defaultdict(lambda: defaultdict(list)),
        'true_vs_pred': [],
        'high_prob_gradients': [],
        'low_prob_gradients': []
    }
    
    print(f"\nAnalyzing {len(samples)} samples...")
    
    for sample in tqdm(samples):
        x = sample['input'].unsqueeze(0).to(device)
        target = sample['target']
        position = sample['position']
        
        try:
            # 清零梯度
            model.zero_grad()
            
            # 前向传播
            with torch.enable_grad():
                logits, _ = model(x)
                
                # 获取最后位置的输出
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=0)
                
                # 计算损失
                loss = F.cross_entropy(
                    last_logits.unsqueeze(0),
                    torch.tensor([target], device=device)
                )
                
                # 反向传播
                loss.backward()
                
                # 收集梯度信息
                if model.lm_head.weight.grad is not None:
                    grad_norms = torch.norm(model.lm_head.weight.grad, dim=1)
                    
                    # 获取预测和概率
                    pred_token = torch.argmax(probs).item()
                    pred_prob = probs[pred_token].item()
                    target_prob = probs[target].item()
                    
                    # 目标token的梯度
                    target_grad = grad_norms[target].item()
                    pred_grad = grad_norms[pred_token].item()
                    
                    # 记录数据
                    gradient_data['true_vs_pred'].append({
                        'target_prob': target_prob,
                        'target_grad': target_grad,
                        'pred_prob': pred_prob,
                        'pred_grad': pred_grad,
                        'correct': pred_token == target
                    })
                    
                    # 按概率分组
                    if target_prob > 0.9:
                        gradient_data['high_prob_gradients'].append(target_grad)
                    elif target_prob < 0.1:
                        gradient_data['low_prob_gradients'].append(target_grad)
                    
                    # 按节点记录
                    if target >= 2:
                        node = target - 2
                        gradient_data['by_node'][node].append({
                            'grad': target_grad,
                            'prob': target_prob
                        })
                    
                    # 按位置记录
                    gradient_data['by_position'][position]['grad'].append(target_grad)
                    gradient_data['by_position'][position]['prob'].append(target_prob)
                    
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    # 计算统计摘要
    summary = {
        'total_samples': len(samples),
        'avg_target_grad': np.mean([d['target_grad'] for d in gradient_data['true_vs_pred']]),
        'avg_target_prob': np.mean([d['target_prob'] for d in gradient_data['true_vs_pred']]),
        'accuracy': np.mean([d['correct'] for d in gradient_data['true_vs_pred']]),
    }
    
    if gradient_data['high_prob_gradients']:
        summary['high_prob_avg_grad'] = np.mean(gradient_data['high_prob_gradients'])
    
    if gradient_data['low_prob_gradients']:
        summary['low_prob_avg_grad'] = np.mean(gradient_data['low_prob_gradients'])
    
    return gradient_data, summary

def visualize_gradient_saturation(all_results):
    """可视化梯度饱和现象"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 梯度随迭代次数的变化
    ax = axes[0, 0]
    iterations = sorted(all_results.keys())
    avg_grads = [all_results[it]['summary']['avg_target_grad'] for it in iterations]
    avg_probs = [all_results[it]['summary']['avg_target_prob'] for it in iterations]
    
    ax2 = ax.twinx()
    line1 = ax.plot(iterations, avg_grads, 'b-', marker='o', linewidth=2, 
                    markersize=8, label='Avg Gradient')
    line2 = ax2.plot(iterations, avg_probs, 'r-', marker='s', linewidth=2,
                     markersize=8, label='Avg Probability')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Gradient', color='b')
    ax2.set_ylabel('Average Probability', color='r')
    ax.set_title('Gradient and Probability Evolution')
    
    # 标记相变区域
    ax.axvspan(130000, 150000, alpha=0.2, color='gray')
    ax.text(140000, max(avg_grads)*0.9, 'Phase Transition', 
            ha='center', fontsize=10, weight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)
    ax.grid(True, alpha=0.3)
    
    # 2. 梯度vs概率散点图（140k时刻）
    ax = axes[0, 1]
    if 140000 in all_results:
        data = all_results[140000]['data']['true_vs_pred']
        probs = [d['target_prob'] for d in data]
        grads = [d['target_grad'] for d in data]
        
        scatter = ax.scatter(probs, grads, alpha=0.5, s=30)
        ax.set_xlabel('Target Probability')
        ax.set_ylabel('Target Gradient')
        ax.set_title('Gradient vs Probability at 140k (Phase Transition)')
        
        # 添加趋势线
        if len(probs) > 10:
            z = np.polyfit(probs, grads, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(0, 1, 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        ax.grid(True, alpha=0.3)
    
    # 3. 高概率vs低概率梯度对比
    ax = axes[0, 2]
    high_grads = []
    low_grads = []
    iters = []
    
    for it in iterations:
        if 'high_prob_avg_grad' in all_results[it]['summary']:
            high_grads.append(all_results[it]['summary']['high_prob_avg_grad'])
            low_grads.append(all_results[it]['summary']['low_prob_avg_grad'])
            iters.append(it)
    
    if iters:
        ax.plot(iters, high_grads, 'r-', marker='o', linewidth=2, 
                label='High Prob (>0.9)')
        ax.plot(iters, low_grads, 'b-', marker='s', linewidth=2,
                label='Low Prob (<0.1)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Gradient')
        ax.set_title('Gradient by Probability Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. 准确率变化
    ax = axes[1, 0]
    accuracies = [all_results[it]['summary']['accuracy'] for it in iterations]
    ax.plot(iterations, accuracies, 'g-', marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Prediction Accuracy Evolution')
    ax.set_ylim([0, 1.1])
    ax.axvspan(130000, 150000, alpha=0.2, color='gray')
    ax.grid(True, alpha=0.3)
    
    # 5. 梯度分布直方图
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(iterations)))
    
    for i, it in enumerate(iterations):
        grads = [d['target_grad'] for d in all_results[it]['data']['true_vs_pred']]
        ax.hist(grads, bins=30, alpha=0.3, color=colors[i], 
                label=f'{it//1000}k', density=True)
    
    ax.set_xlabel('Gradient Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Gradient Distribution Evolution')
    ax.legend()
    ax.set_xlim([0, max([max(grads) for grads in 
                        [[d['target_grad'] for d in all_results[it]['data']['true_vs_pred']] 
                         for it in iterations]])])
    
    # 6. 梯度饱和曲线（使用140k的数据）
    ax = axes[1, 2]
    if 140000 in all_results:
        # 创建概率bins
        prob_bins = np.linspace(0, 1, 20)
        binned_grads = defaultdict(list)
        
        data = all_results[140000]['data']['true_vs_pred']
        for d in data:
            prob = d['target_prob']
            grad = d['target_grad']
            bin_idx = np.digitize(prob, prob_bins) - 1
            if 0 <= bin_idx < len(prob_bins) - 1:
                binned_grads[bin_idx].append(grad)
        
        # 计算每个bin的平均梯度
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
        avg_grads = []
        
        for i in range(len(bin_centers)):
            if i in binned_grads and binned_grads[i]:
                avg_grads.append(np.mean(binned_grads[i]))
            else:
                avg_grads.append(np.nan)
        
        # 绘制
        valid_idx = ~np.isnan(avg_grads)
        ax.plot(bin_centers[valid_idx], np.array(avg_grads)[valid_idx], 
                'ko-', linewidth=2, markersize=8)
        ax.set_xlabel('Probability')
        ax.set_ylabel('Average Gradient')
        ax.set_title('Gradient Saturation Curve at 140k')
        ax.axvspan(0.9, 1.0, alpha=0.2, color='red', label='Saturation Zone')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradient_saturation_analysis.png'), dpi=150)
    plt.close()
    print(f"\nVisualization saved to {OUTPUT_DIR}/gradient_saturation_analysis.png")

def main():
    """主分析函数"""
    ensure_dir(OUTPUT_DIR)
    
    # 1. 探索数据格式
    meta = explore_data_format()
    
    # 2. 创建合成样本（不依赖实际数据格式）
    samples = create_synthetic_samples(num_samples=500)
    
    # 3. 分析每个checkpoint
    all_results = {}
    
    for ckpt_iter in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint {ckpt_iter}...")
        
        try:
            # 加载模型
            model = load_model_checkpoint(ckpt_iter)
            
            # 计算梯度统计
            gradient_data, summary = compute_gradient_statistics(model, samples)
            
            # 保存结果
            all_results[ckpt_iter] = {
                'data': gradient_data,
                'summary': summary
            }
            
            # 打印摘要
            print(f"\nSummary for {ckpt_iter}:")
            for key, value in summary.items():
                print(f"  {key}: {value:.4f}")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing checkpoint {ckpt_iter}: {e}")
            continue
    
    # 4. 保存原始结果
    results_file = os.path.join(OUTPUT_DIR, 'gradient_analysis_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {results_file}")
    
    # 5. 生成可视化
    if all_results:
        visualize_gradient_saturation(all_results)
    
    # 6. 测试特定的梯度饱和行为
    print("\n" + "="*60)
    print("Testing gradient saturation behavior...")
    test_gradient_saturation_threshold()
    
    print("\n" + "="*60)
    print("Analysis complete!")

def test_gradient_saturation_threshold():
    """测试梯度饱和阈值"""
    # 加载相变前后的模型
    models = {
        'pre_transition': load_model_checkpoint(120000),
        'transition': load_model_checkpoint(140000),
        'post_transition': load_model_checkpoint(180000)
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建测试序列
    test_input = torch.tensor([[2, 7, 2, 4]], device=device)  # source=0, target=5, via 2
    
    results = {}
    
    for name, model in models.items():
        model.to(device)
        model.eval()
        
        # 测试不同的目标概率
        target_probs = np.linspace(0.5, 0.99, 10)
        gradients = []
        
        for target_prob in target_probs:
            model.zero_grad()
            
            # 前向传播
            with torch.enable_grad():
                logits, _ = model(test_input)
                last_logits = logits[0, -1, :]
                
                # 创建目标分布
                target_dist = torch.zeros_like(last_logits)
                target_dist[4] = target_prob  # node 2 (token 4)
                target_dist[target_dist == 0] = (1 - target_prob) / (len(target_dist) - 1)
                
                # KL损失
                log_probs = F.log_softmax(last_logits, dim=0)
                loss = F.kl_div(log_probs, target_dist, reduction='sum')
                loss.backward()
                
                # 记录梯度
                grad_norm = torch.norm(model.lm_head.weight.grad[4]).item()
                gradients.append(grad_norm)
        
        results[name] = {
            'target_probs': target_probs,
            'gradients': gradients
        }
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    
    colors = {'pre_transition': 'blue', 'transition': 'orange', 'post_transition': 'red'}
    
    for name, data in results.items():
        plt.plot(data['target_probs'], data['gradients'], 
                color=colors[name], marker='o', linewidth=2,
                label=name.replace('_', ' ').title())
    
    plt.xlabel('Target Probability')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Saturation Curves Across Training')
    plt.axvspan(0.9, 1.0, alpha=0.2, color='red', label='Saturation Zone')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'saturation_threshold_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Saturation curves saved to {OUTPUT_DIR}/saturation_threshold_comparison.png")

if __name__ == "__main__":
    print("="*60)
    print("Gradient Saturation Analysis - Complete Version")
    print("="*60)
    
    main()