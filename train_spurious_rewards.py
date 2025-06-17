"""
Spurious Rewards 实验脚本
测试不同奖励信号对相变现象的影响
"""
import os
import time
import math
import re
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from contextlib import nullcontext
import matplotlib.pyplot as plt
import random
from datetime import datetime

from model import GPTConfig, GPT
from logger import get_logger
import logging

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Spurious Rewards Experiments')
    # 基础参数
    parser.add_argument('--dataset', type=str, default='simple_graph', help='Dataset name')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=120, help='Embedding size')
    parser.add_argument('--max_iters', type=int, default=200000, help='Total iterations')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair')
    parser.add_argument('--test_interval', type=int, default=1000, help='Test interval')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature')
    parser.add_argument('--num_eval_batches', type=int, default=10, help='Eval batches')
    
    # Spurious Rewards相关参数
    parser.add_argument('--reward_type', type=str, default='standard', 
                      choices=['standard', 'any_valid', 'mixed', 'diversity', 'phase_aware'],
                      help='Type of reward function')
    parser.add_argument('--mixed_alpha', type=float, default=0.5, 
                      help='Mixing ratio for mixed reward (0=all any_valid, 1=all standard)')
    parser.add_argument('--diversity_weight', type=float, default=0.1,
                      help='Weight for entropy regularization in diversity reward')
    parser.add_argument('--phase_aware_transition', type=int, default=120000,
                      help='Iteration to switch reward in phase-aware training')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    
    return parser.parse_args()

# ================== Spurious Reward Functions ==================

def compute_any_valid_reward_loss(model, X, Y, graph, stoi, itos, device):
    """
    奖励任何有效路径，而不只是训练路径
    
    关键思想：对于每个时间步，如果预测的是任何有效的下一步节点，就给低损失
    """
    logits, _ = model(X, Y)
    batch_size, seq_len, vocab_size = logits.shape
    
    # 创建损失张量
    loss = torch.zeros(batch_size, device=device)
    
    for b in range(batch_size):
        seq_loss = 0
        valid_steps = 0
        
        # 解析起点和终点
        # 格式：source target path_nodes...
        source_token = X[b, 0].item()
        target_token = X[b, 1].item()
        
        # token转节点号（减2因为0是PAD，1是\n）
        if source_token >= 2 and target_token >= 2:
            source_node = str(source_token - 2)
            target_node = str(target_token - 2)
            
            # 从第3个位置开始是路径
            current_node = source_node
            
            for t in range(2, seq_len):
                if Y[b, t].item() == 0:  # PAD token
                    break
                    
                # 获取当前节点的所有有效邻居
                try:
                    neighbors = list(graph.successors(current_node))
                    valid_tokens = [int(n) + 2 for n in neighbors]  # 转回token id
                    
                    if len(valid_tokens) > 0:
                        # 计算选择任何有效节点的概率
                        probs = F.softmax(logits[b, t], dim=-1)
                        valid_prob = sum(probs[token] for token in valid_tokens if token < vocab_size)
                        
                        # 损失 = -log(选择有效节点的概率)
                        seq_loss += -torch.log(valid_prob + 1e-8)
                        valid_steps += 1
                    
                    # 更新current_node为实际的下一个节点
                    next_token = Y[b, t].item()
                    if next_token >= 2:
                        current_node = str(next_token - 2)
                        
                except:
                    # 如果节点不存在或其他错误，使用标准损失
                    seq_loss += F.cross_entropy(logits[b, t], Y[b, t], reduction='none')
                    valid_steps += 1
        
        if valid_steps > 0:
            loss[b] = seq_loss / valid_steps
        else:
            # 如果没有有效步骤，使用标准损失
            loss[b] = F.cross_entropy(logits[b], Y[b], reduction='mean')
    
    return loss.mean()

def compute_mixed_reward_loss(model, X, Y, graph, stoi, itos, device, alpha=0.5):
    """
    混合标准损失和any_valid损失
    alpha=1: 完全标准损失
    alpha=0: 完全any_valid损失
    """
    # 标准损失
    logits, _ = model(X, Y)
    standard_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                   Y.view(-1), ignore_index=0)
    
    # Any valid损失
    any_valid_loss = compute_any_valid_reward_loss(model, X, Y, graph, stoi, itos, device)
    
    # 混合
    return alpha * standard_loss + (1 - alpha) * any_valid_loss

def compute_diversity_reward_loss(model, X, Y, graph, stoi, itos, device, diversity_weight=0.1):
    """
    Any valid + 熵正则化鼓励多样性
    """
    # 基础any valid损失
    base_loss = compute_any_valid_reward_loss(model, X, Y, graph, stoi, itos, device)
    
    # 计算输出分布的熵
    logits, _ = model(X, Y)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    
    # Mask掉padding
    mask = (Y != 0).float()
    masked_entropy = entropy * mask
    avg_entropy = masked_entropy.sum() / mask.sum()
    
    # 负熵作为损失（因为我们想最大化熵）
    diversity_loss = -diversity_weight * avg_entropy
    
    return base_loss + diversity_loss

def compute_phase_aware_loss(model, X, Y, graph, stoi, itos, device, iteration, transition_iter=120000):
    """
    根据训练阶段自动切换损失函数
    """
    if iteration < transition_iter:
        # 早期：使用标准损失建立基础
        logits, _ = model(X, Y)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), 
                             Y.view(-1), ignore_index=0)
    else:
        # 后期：切换到any_valid避免过拟合
        return compute_any_valid_reward_loss(model, X, Y, graph, stoi, itos, device)

# ================== Phase Detection ==================

class PhaseDetector:
    def __init__(self, window_size=5000):
        self.window_size = window_size
        self.tf_history = []
        self.loss_history = []
        
    def update(self, tf_acc, loss):
        self.tf_history.append(tf_acc)
        self.loss_history.append(loss)
        
        # 只保留最近的历史
        if len(self.tf_history) > self.window_size:
            self.tf_history.pop(0)
            self.loss_history.pop(0)
    
    def detect_phase(self):
        if len(self.tf_history) < 100:
            return "early", {}
        
        # 计算最近的TF下降速度
        recent_window = min(1000, len(self.tf_history) // 10)
        recent_tf = self.tf_history[-recent_window:]
        tf_slope = (recent_tf[-1] - recent_tf[0]) / len(recent_tf)
        
        # 当前TF准确率
        current_tf = self.tf_history[-1]
        
        # Phase判断
        if current_tf > 0.85:
            phase = "memorization"
        elif current_tf > 0.5 and tf_slope < -0.0001:
            phase = "transition_imminent"
        elif current_tf < 0.5 and current_tf > 0.2:
            phase = "transitioning"
        elif current_tf < 0.2:
            phase = "post_transition"
        else:
            phase = "stable"
        
        metrics = {
            "current_tf": current_tf,
            "tf_slope": tf_slope,
            "phase": phase
        }
        
        return phase, metrics

# ================== Main Training Loop ==================

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 创建实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.reward_type}_alpha{args.mixed_alpha}_div{args.diversity_weight}_seed{args.seed}"
    
    # 设置输出目录
    out_dir = f'out/spurious_rewards/{experiment_name}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置logger
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    # 打印配置
    logger.info("="*60)
    logger.info(f"Spurious Rewards Experiment: {args.reward_type}")
    logger.info(f"Configuration: {vars(args)}")
    logger.info("="*60)
    
    # 加载数据和元信息
    data_dir = os.path.join('data', f'{args.dataset}/{args.num_nodes}')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = len(itos)
    
    # 加载图
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    G = nx.read_graphml(graph_path)
    
    # 加载数据
    if args.num_of_paths == 0:
        train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        train_data = np.memmap(os.path.join(data_dir, f'train_{args.num_of_paths}.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # 初始化模型
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=block_size,
        bias=False,
        vocab_size=vocab_size,
        dropout=0.0
    )
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(args.device)
    
    # 优化器
    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in args.device else 'cpu'
    )
    
    # 训练参数
    train_batch_size = 1024
    val_batch_size = 64
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Phase检测器
    phase_detector = PhaseDetector()
    
    # 数据获取函数
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        bs = train_batch_size if split == 'train' else val_batch_size
        data_size = block_size + 1
        
        ix = torch.randint((len(data) - data_size) // data_size, (bs,)) * data_size
        
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        x, y = x.to(args.device), y.to(args.device)
        return x, y
    
    # 评估函数（与原始train.py相同）
    @torch.no_grad()
    def test_model():
        total_correct = 0
        total_count = 0
        
        for _ in range(args.num_eval_batches):
            X, Y = get_batch('val')
            with ctx:
                logits, _ = model(X, Y)
            preds = torch.argmax(logits, dim=-1)
            
            total_correct += (preds == Y).float().sum().item()
            total_count += Y.numel()
        
        return total_correct / total_count
    
    # 训练历史记录
    metrics_history = {
        'iteration': [],
        'train_loss': [],
        'val_loss': [],
        'tf_accuracy': [],
        'ar_accuracy': [],
        'phase': [],
        'tf_slope': []
    }
    
    # 训练循环
    logger.info("Starting training...")
    model.train()
    
    for iter_num in range(args.max_iters + 1):
        # 学习率调度
        lr = args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 获取批次
        X, Y = get_batch('train')
        
        # 计算损失（根据reward_type选择）
        with ctx:
            if args.reward_type == 'standard':
                logits, loss = model(X, Y)
            elif args.reward_type == 'any_valid':
                loss = compute_any_valid_reward_loss(model, X, Y, G, stoi, itos, args.device)
            elif args.reward_type == 'mixed':
                loss = compute_mixed_reward_loss(model, X, Y, G, stoi, itos, args.device, args.mixed_alpha)
            elif args.reward_type == 'diversity':
                loss = compute_diversity_reward_loss(model, X, Y, G, stoi, itos, args.device, args.diversity_weight)
            elif args.reward_type == 'phase_aware':
                loss = compute_phase_aware_loss(model, X, Y, G, stoi, itos, args.device, iter_num, args.phase_aware_transition)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 定期评估
        if iter_num % args.test_interval == 0:
            model.eval()
            
            # 计算验证损失
            val_losses = []
            for _ in range(10):
                X_val, Y_val = get_batch('val')
                with torch.no_grad():
                    _, val_loss = model(X_val, Y_val)
                val_losses.append(val_loss.item())
            val_loss = np.mean(val_losses)
            
            # 计算TF准确率
            tf_acc = test_model()
            
            # 更新phase检测器
            phase_detector.update(tf_acc, loss.item())
            phase, phase_metrics = phase_detector.detect_phase()
            
            # 记录
            metrics_history['iteration'].append(iter_num)
            metrics_history['train_loss'].append(loss.item())
            metrics_history['val_loss'].append(val_loss)
            metrics_history['tf_accuracy'].append(tf_acc)
            metrics_history['phase'].append(phase)
            metrics_history['tf_slope'].append(phase_metrics.get('tf_slope', 0))
            
            # 打印
            logger.info(f"Iter {iter_num}: loss={loss.item():.4f}, val_loss={val_loss:.4f}, "
                       f"TF={tf_acc:.4f}, phase={phase}, slope={phase_metrics.get('tf_slope', 0):.6f}")
            
            # 保存checkpoint
            if iter_num % 10000 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'metrics_history': metrics_history,
                    'args': vars(args)
                }
                torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_{iter_num}.pt'))
            
            model.train()
    
    # 保存最终结果
    with open(os.path.join(out_dir, 'final_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_history, f)
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    # 1. 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(metrics_history['iteration'], metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['iteration'], metrics_history['val_loss'], label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - {args.reward_type}')
    plt.legend()
    plt.grid(True)
    
    # 2. TF准确率
    plt.subplot(2, 3, 2)
    plt.plot(metrics_history['iteration'], metrics_history['tf_accuracy'])
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0.15, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Teacher Forcing Accuracy')
    plt.grid(True)
    
    # 3. Phase变化
    plt.subplot(2, 3, 3)
    phase_colors = {
        'early': 'blue',
        'memorization': 'green',
        'transition_imminent': 'orange',
        'transitioning': 'red',
        'post_transition': 'purple',
        'stable': 'gray'
    }
    
    for i in range(len(metrics_history['iteration'])-1):
        plt.axvspan(metrics_history['iteration'][i], 
                   metrics_history['iteration'][i+1],
                   color=phase_colors.get(metrics_history['phase'][i], 'gray'),
                   alpha=0.3)
    plt.plot(metrics_history['iteration'], metrics_history['tf_accuracy'], 'k-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Phase Transitions')
    plt.grid(True)
    
    # 4. TF斜率
    plt.subplot(2, 3, 4)
    plt.plot(metrics_history['iteration'], metrics_history['tf_slope'])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('TF Slope')
    plt.title('TF Accuracy Slope')
    plt.grid(True)
    
    # 5. 对比标准训练（如果有baseline数据）
    plt.subplot(2, 3, 5)
    plt.plot(metrics_history['iteration'], metrics_history['tf_accuracy'], 
             label=f'{args.reward_type}', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Comparison with Baseline')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_results.png'), dpi=150)
    plt.close()
    
    # 打印总结
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info(f"Final TF Accuracy: {metrics_history['tf_accuracy'][-1]:.4f}")
    logger.info(f"Minimum TF Accuracy: {min(metrics_history['tf_accuracy']):.4f}")
    logger.info(f"Results saved to: {out_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()