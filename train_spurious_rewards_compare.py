"""
Spurious Rewards 实验脚本 - 增强版
添加了详细的token级别统计，以便与Masked Loss和Dynamic Batching实验比较
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
    parser.add_argument('--test_interval', type=int, default=2000, help='Test and print interval')
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
    parser.add_argument('--checkpoint_interval', type=int, default=50000, help='Checkpoint save interval')
    
    return parser.parse_args()

# ================== 优化的Spurious Reward Functions ==================

# 全局缓存
NEIGHBOR_CACHE = {}

def precompute_neighbor_masks(graph, num_nodes, vocab_size, device):
    """预计算所有节点的邻居mask，用于向量化"""
    print("Precomputing neighbor masks for fast any-valid reward...")
    
    # 创建邻接矩阵
    adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    for i in range(num_nodes):
        neighbors = list(graph.successors(str(i)))
        for j in neighbors:
            adj_matrix[i, int(j)] = True
    
    # 创建vocab mask矩阵
    neighbor_masks = torch.zeros(num_nodes, vocab_size, dtype=torch.bool)
    for i in range(num_nodes):
        neighbors = list(graph.successors(str(i)))
        for j in neighbors:
            if int(j) + 2 < vocab_size:  # +2 因为token偏移
                neighbor_masks[i, int(j) + 2] = True
    
    NEIGHBOR_CACHE['adj_matrix'] = adj_matrix.to(device)
    NEIGHBOR_CACHE['neighbor_masks'] = neighbor_masks.to(device)
    NEIGHBOR_CACHE['num_nodes'] = num_nodes
    
    print("Neighbor masks computed and cached!")

def compute_any_valid_reward_loss_vectorized(model, X, Y, graph, stoi, itos, device):
    """
    完全向量化的any-valid reward - 无Python循环
    """
    logits, _ = model(X, Y)
    batch_size, seq_len, vocab_size = logits.shape
    
    # 获取缓存的邻居信息
    neighbor_masks = NEIGHBOR_CACHE['neighbor_masks']
    
    # 计算标准CE loss作为基准
    ce_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size), 
        Y.reshape(-1), 
        ignore_index=0, 
        reduction='none'
    ).reshape(batch_size, seq_len)
    
    # 创建位置mask
    valid_mask = (Y != 0).float()
    
    # 找到每个序列的长度
    seq_lengths = valid_mask.sum(dim=1)
    
    # 创建位置索引
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 找到需要应用any-valid的位置（位置3到倒数第二个）
    # 使用广播创建end_positions
    end_positions = seq_lengths.unsqueeze(1) - 1
    middle_mask = (positions >= 3) & (positions < end_positions) & valid_mask.bool()
    
    # 如果没有中间位置需要处理，直接返回CE loss
    if not middle_mask.any():
        return (ce_loss * valid_mask).sum() / valid_mask.sum()
    
    # 获取前一个位置的节点（当前节点）来查找邻居
    # 向左shift Y来获取前一个token
    prev_Y = torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)
    current_nodes = (prev_Y - 2).clamp(0, NEIGHBOR_CACHE['num_nodes'] - 1)
    
    # 批量计算所有位置的any-valid loss
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 创建一个大的mask矩阵 (batch_size, seq_len, vocab_size)
    # 表示每个位置的有效下一跳
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
    seq_indices = torch.arange(seq_len, device=device).view(1, -1, 1)
    
    # 使用高级索引获取每个位置的邻居mask
    position_neighbor_masks = neighbor_masks[current_nodes]  # (batch_size, seq_len, vocab_size)
    
    # 只在middle_mask位置计算any-valid loss
    # 创建一个very negative的值用于mask
    masked_log_probs = log_probs.clone()
    
    # 在需要计算any-valid的位置，将非邻居的概率设为-inf
    masked_log_probs[middle_mask] = torch.where(
        position_neighbor_masks[middle_mask],
        log_probs[middle_mask],
        torch.tensor(float('-inf'), device=device)
    )
    
    # 使用logsumexp计算any-valid loss
    any_valid_log_probs = torch.logsumexp(masked_log_probs, dim=-1)
    any_valid_loss = -any_valid_log_probs
    
    # 混合策略：根据原始CE loss决定混合权重
    # 使用向量化的条件
    mix_weights = torch.where(
        ce_loss > 2.0,
        torch.tensor(0.7, device=device),
        torch.where(
            ce_loss > 1.0,
            torch.tensor(0.3, device=device),
            torch.tensor(0.0, device=device)
        )
    )
    
    # 只在middle_mask位置应用混合
    final_loss = ce_loss.clone()
    final_loss[middle_mask] = (
        (1 - mix_weights[middle_mask]) * ce_loss[middle_mask] + 
        mix_weights[middle_mask] * any_valid_loss[middle_mask]
    )
    
    # 应用有效mask并返回平均值
    masked_loss = final_loss * valid_mask
    return masked_loss.sum() / valid_mask.sum()

def compute_mixed_reward_loss(model, X, Y, graph, stoi, itos, device, alpha=0.5):
    """混合标准损失和any_valid损失"""
    logits, _ = model(X, Y)
    
    # 标准CE loss
    standard_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                   Y.view(-1), ignore_index=0)
    
    # Any valid loss（使用向量化版本）
    any_valid_loss = compute_any_valid_reward_loss_vectorized(model, X, Y, graph, stoi, itos, device)
    
    # 混合
    return alpha * standard_loss + (1 - alpha) * any_valid_loss

def compute_diversity_reward_loss(model, X, Y, graph, stoi, itos, device, diversity_weight=0.1):
    """Any valid + 熵正则化鼓励多样性"""
    # 基础any valid损失
    base_loss = compute_any_valid_reward_loss_vectorized(model, X, Y, graph, stoi, itos, device)
    
    # 计算熵
    logits, _ = model(X, Y)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    
    # Mask掉padding
    mask = (Y != 0).float()
    masked_entropy = entropy * mask
    avg_entropy = masked_entropy.sum() / mask.sum()
    
    # 负熵作为损失（我们想最大化熵）
    diversity_loss = -diversity_weight * avg_entropy
    
    return base_loss + diversity_loss

def compute_phase_aware_loss(model, X, Y, graph, stoi, itos, device, iteration, transition_iter=120000):
    """根据训练阶段自动切换损失函数"""
    if iteration < transition_iter:
        # 早期：使用标准损失建立基础
        logits, _ = model(X, Y)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), 
                             Y.view(-1), ignore_index=0)
    else:
        # 后期：切换到any_valid避免过拟合
        return compute_any_valid_reward_loss_vectorized(model, X, Y, graph, stoi, itos, device)

# ================== 评估函数 ==================

def encode(s, stoi):
    ss = s.split(" ")
    return [stoi[token] for token in ss if token in stoi]

def decode(l, itos):
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]

def find_third_number_position(number_string):  
    numbers = number_string.split()  
    third_number_index = 2 
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index - 1 
    return position 

def check_path(G, gen_str):
    """检查生成的路径是否有效"""
    path = re.findall(r'\d+', gen_str)
    if len(path) < 4:
        return 'wrong syntax'
    for node in path:
        if int(node) >= 100 or int(node) < 0:
            return 'wrong syntax'
    if path[2] != path[0] or path[-1] != path[1]:
        return 'incorrect start/end'
    for i in range(2, len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return f'non-existence path {(path[i], path[i + 1])}'
    return ''

# ================== 增强的评估函数 ==================

@torch.no_grad()
def evaluate_detailed(model, get_batch_fn, num_eval_batches, device):
    """
    详细的评估函数，包含所有需要的统计信息
    返回与Masked Loss和Dynamic Batching兼容的结果
    """
    model.eval()
    
    # 初始化统计量
    total_loss = 0
    
    # TF准确率（两种计算方式）
    tf_correct_all = 0      # 包含padding
    tf_total_all = 0
    tf_correct_masked = 0   # 不含padding
    tf_total_masked = 0
    
    # 分类统计
    path_correct = 0        # 路径节点 (token > 1)
    path_total = 0
    newline_correct = 0     # newline (token = 1)
    newline_total = 0
    pad_correct = 0         # padding (token = 0)
    pad_total = 0
    
    # Padding预测分析
    pad_pred_pad = 0        # 在padding位置预测padding
    pad_pred_newline = 0    # 在padding位置预测newline
    pad_pred_path = 0       # 在padding位置预测路径
    pad_positions_total = 0
    
    # 批次统计
    batch_accuracies = []
    val_losses = []
    
    for batch_idx in range(num_eval_batches):
        X, Y = get_batch_fn('val')
        
        # 计算损失
        logits, loss = model(X, Y)
        val_losses.append(loss.item())
        
        # 预测
        preds = torch.argmax(logits, dim=-1)
        
        # 批次准确率（包含所有位置）
        batch_correct = (preds == Y).float().sum().item()
        batch_total = Y.numel()
        batch_accuracy = batch_correct / batch_total
        batch_accuracies.append(batch_accuracy)
        
        # 1. 原始TF（包含所有token）
        tf_correct_all += batch_correct
        tf_total_all += batch_total
        
        # 2. Masked TF（只算有效token）
        mask = Y != 0
        if mask.sum() > 0:
            tf_correct_masked += (preds[mask] == Y[mask]).sum().item()
            tf_total_masked += mask.sum().item()
        
        # 3. 分类统计
        # 路径节点
        path_mask = Y > 1
        if path_mask.sum() > 0:
            path_correct += (preds[path_mask] == Y[path_mask]).sum().item()
            path_total += path_mask.sum().item()
        
        # Newline
        newline_mask = Y == 1
        if newline_mask.sum() > 0:
            newline_correct += (preds[newline_mask] == Y[newline_mask]).sum().item()
            newline_total += newline_mask.sum().item()
        
        # Padding
        pad_mask = Y == 0
        if pad_mask.sum() > 0:
            pad_correct += (preds[pad_mask] == Y[pad_mask]).sum().item()
            pad_total += pad_mask.sum().item()
            
            # 分析padding位置的预测
            pad_preds = preds[pad_mask]
            pad_pred_pad += (pad_preds == 0).sum().item()
            pad_pred_newline += (pad_preds == 1).sum().item()
            pad_pred_path += (pad_preds > 1).sum().item()
            pad_positions_total += pad_mask.sum().item()
    
    # 计算最终统计
    results = {
        # 损失
        'loss': np.mean(val_losses),
        
        # TF准确率（两种）
        'tf_accuracy_all': tf_correct_all / tf_total_all if tf_total_all > 0 else 0,
        'tf_accuracy': tf_correct_masked / tf_total_masked if tf_total_masked > 0 else 0,  # 主要指标，不含padding
        'tf_accuracy_std': np.std(batch_accuracies),
        
        # 分类准确率
        'path_accuracy': path_correct / path_total if path_total > 0 else 0,
        'newline_accuracy': newline_correct / newline_total if newline_total > 0 else 0,
        'pad_accuracy': pad_correct / pad_total if pad_total > 0 else 0,
        
        # Anti-preference关键指标
        'pad_pred_ratio': pad_pred_pad / pad_positions_total if pad_positions_total > 0 else 0,
        'newline_pred_at_pad': pad_pred_newline / pad_positions_total if pad_positions_total > 0 else 0,
        'path_pred_at_pad': pad_pred_path / pad_positions_total if pad_positions_total > 0 else 0,
        
        # Token分布
        'padding_ratio': pad_total / tf_total_all if tf_total_all > 0 else 0,
        'path_ratio': path_total / tf_total_all if tf_total_all > 0 else 0,
        'newline_ratio': newline_total / tf_total_all if tf_total_all > 0 else 0,
    }
    
    model.train()
    return results

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
    print("="*60)
    print(f"Spurious Rewards Experiment: {args.reward_type}")
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)
    
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
    simple_format = meta.get('simple_format', False)
    
    # 加载图
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    G = nx.read_graphml(graph_path)
    
    # 预计算邻居信息（如果使用any_valid相关的reward）
    if args.reward_type in ['any_valid', 'mixed', 'diversity', 'phase_aware']:
        precompute_neighbor_masks(G, args.num_nodes, vocab_size, args.device)
    
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
    
    print(f"Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(args.device)
    
    # 打印参数量
    print(f"number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
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
    
    # Autoregressive评估
    @torch.no_grad()
    def evaluate_autoregressive_lenient():
        test_file = os.path.join(data_dir, 'test.txt')
        try:
            with open(test_file, encoding='gbk') as f:
                lines = f.readlines()
        except:
            try:
                with open(test_file, encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                print(f"Failed to read test file {test_file}")
                return 0.0, {"wrong syntax": 0, "incorrect start/end": 0, "non-existence path": 0}
        
        # 处理测试数据
        encode_texts = []
        ground_truth = []
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            if not simple_format:
                prompt = line.split(':')[0] + ':'
            else:
                pos = find_third_number_position(line)
                prompt = line[:pos]
            encode_texts.append(encode(prompt, stoi))
            ground_truth.append(line)
        
        if len(encode_texts) == 0:
            return 0.0, {"wrong syntax": 0, "incorrect start/end": 0, "non-existence path": 0}
        
        # 转换为Tensor
        encode_texts = torch.tensor(encode_texts, dtype=torch.long, device=args.device)
        batch_size_eval = min(1000, len(encode_texts))
        num_samples = encode_texts.shape[0]
        num_iters = 10
        total_correct = 0
        total_count = 0
        error_wrong_syntax = 0
        error_incorrect_start_end = 0
        error_nonexistence = 0
        
        for _ in range(num_iters):
            # 随机采样
            ix = torch.randint(num_samples, (batch_size_eval,))
            x = encode_texts[ix]
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=block_size, 
                                 temperature=args.temperature, top_k=vocab_size)
            
            y_pred = [decode(y[t].tolist(), itos).split('\n')[0] for t in range(batch_size_eval)]
            for pred in y_pred:
                symbol = check_path(G, pred)
                total_count += 1
                if symbol == "":
                    total_correct += 1
                else:
                    if symbol == "wrong syntax":
                        error_wrong_syntax += 1
                    elif symbol == "incorrect start/end":
                        error_incorrect_start_end += 1
                    elif symbol.startswith("non-existence path"):
                        error_nonexistence += 1
        
        accuracy = total_correct / total_count if total_count > 0 else 0.0
        error_counts = {
            "wrong syntax": error_wrong_syntax,
            "incorrect start/end": error_incorrect_start_end,
            "non-existence path": error_nonexistence
        }
        return accuracy, error_counts
    
    # 使用统一的历史记录格式（与Masked/Dynamic兼容）
    history = {
        'iter': [],
        'train_loss': [],
        'val_loss': [],
        'tf_accuracy': [],  # 不含padding的准确率
        'tf_accuracy_all': [],  # 含padding的准确率（兼容旧版）
        'tf_accuracy_std': [],
        'ar_accuracy': [],
        'path_accuracy': [],
        'newline_accuracy': [],
        'pad_accuracy': [],
        'pad_pred_ratio': [],
        'newline_pred_at_pad': [],
        'path_pred_at_pad': [],
        'padding_ratio': [],
        'phase_history': []
    }
    
    # Phase检测器
    class PhaseDetector:
        def __init__(self):
            self.tf_history = []
            
        def update(self, tf_acc):
            self.tf_history.append(tf_acc)
            if len(self.tf_history) > 100:
                self.tf_history.pop(0)
        
        def detect_phase(self):
            if len(self.tf_history) < 5:
                return "early"
            
            current_tf = self.tf_history[-1]
            
            # 计算斜率
            if len(self.tf_history) >= 5:
                recent_tf = self.tf_history[-5:]
                tf_slope = (recent_tf[-1] - recent_tf[0]) / 4
            else:
                tf_slope = 0
            
            # 判断phase
            if current_tf > 0.85:
                return "memorization"
            elif current_tf > 0.5 and tf_slope < -0.02:
                return "transition_imminent"
            elif 0.2 < current_tf < 0.5:
                return "transitioning"
            elif current_tf < 0.2:
                return "post_transition"
            else:
                return "stable"
    
    phase_detector = PhaseDetector()
    
    # 学习率调度
    def get_lr(it):
        # warmup
        warmup_iters = args.max_iters // 20
        if it < warmup_iters:
            return args.learning_rate * it / warmup_iters
        # cosine decay
        if it > args.max_iters:
            return args.learning_rate / 10
        decay_ratio = (it - warmup_iters) / (args.max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.learning_rate / 10 + coeff * (args.learning_rate - args.learning_rate / 10)
    
    # 训练循环
    print("\nStarting training...")
    model.train()
    running_loss = 0
    loss_count = 0
    
    # 计时器（用于调试）
    if args.reward_type == 'any_valid':
        loss_times = []
    
    for iter_num in range(args.max_iters + 1):
        # 设置学习率
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 定期评估并打印
        if iter_num % args.test_interval == 0 and iter_num > 0:
            model.eval()
            
            # 计算平均训练损失
            avg_train_loss = running_loss / loss_count if loss_count > 0 else 0
            
            # 详细评估
            eval_results = evaluate_detailed(model, get_batch, args.num_eval_batches, args.device)
            
            # 更新phase检测器（使用包含padding的TF准确率以保持兼容性）
            phase_detector.update(eval_results['tf_accuracy_all'])
            current_phase = phase_detector.detect_phase()
            
            # Autoregressive准确率
            ar_acc, error_counts = evaluate_autoregressive_lenient()
            
            # 记录历史
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(eval_results['loss'])
            history['tf_accuracy'].append(eval_results['tf_accuracy'])  # 不含padding
            history['tf_accuracy_all'].append(eval_results['tf_accuracy_all'])  # 含padding
            history['tf_accuracy_std'].append(eval_results['tf_accuracy_std'])
            history['ar_accuracy'].append(ar_acc)
            history['path_accuracy'].append(eval_results['path_accuracy'])
            history['newline_accuracy'].append(eval_results['newline_accuracy'])
            history['pad_accuracy'].append(eval_results['pad_accuracy'])
            history['pad_pred_ratio'].append(eval_results['pad_pred_ratio'])
            history['newline_pred_at_pad'].append(eval_results['newline_pred_at_pad'])
            history['path_pred_at_pad'].append(eval_results['path_pred_at_pad'])
            history['padding_ratio'].append(eval_results['padding_ratio'])
            history['phase_history'].append(current_phase)
            
            # 打印信息
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={eval_results['loss']:.4f}")
            print(f"\nTeacher Forcing Accuracy:")
            print(f"  TF (with padding): {eval_results['tf_accuracy_all']:.4f} (±{eval_results['tf_accuracy_std']:.4f})")
            print(f"  TF (w/o padding): {eval_results['tf_accuracy']:.4f}")
            print(f"\nToken-wise Accuracy:")
            print(f"  Path tokens: {eval_results['path_accuracy']:.4f}")
            print(f"  Newline: {eval_results['newline_accuracy']:.4f}")
            print(f"  Padding: {eval_results['pad_accuracy']:.4f}")
            print(f"\nAnti-preference Analysis:")
            print(f"  Model predicts PAD at PAD: {eval_results['pad_pred_ratio']:.2%}")
            print(f"  Model predicts Newline at PAD: {eval_results['newline_pred_at_pad']:.2%}")
            print(f"  Model predicts Path at PAD: {eval_results['path_pred_at_pad']:.2%}")
            
            # Anti-preference警告
            if eval_results['newline_pred_at_pad'] > 0.5:
                print(f"  ⚠️  WARNING: Anti-preference detected! Newline at PAD = {eval_results['newline_pred_at_pad']:.2%}")
            
            print(f"\nAutoregressive: {ar_acc:.4f}")
            print(f"  Errors: syntax={error_counts['wrong syntax']}, "
                  f"start/end={error_counts['incorrect start/end']}, "
                  f"path={error_counts['non-existence path']}")
            print(f"\nPhase: {current_phase}")
            
            # 如果是any_valid，打印平均计算时间
            if args.reward_type == 'any_valid' and loss_times:
                avg_time = np.mean(loss_times[-1000:])  # 最近1000次的平均
                print(f"  Avg loss computation time: {avg_time*1000:.2f}ms")
            
            # 记录到日志
            logger.info(f"Iter {iter_num}: train_loss={avg_train_loss:.4f}, val_loss={eval_results['loss']:.4f}, "
                       f"TF_all={eval_results['tf_accuracy_all']:.4f}, TF_masked={eval_results['tf_accuracy']:.4f}, "
                       f"AR={ar_acc:.4f}, phase={current_phase}, "
                       f"newline_at_pad={eval_results['newline_pred_at_pad']:.2%}")
            
            # 特殊时期提醒
            if 110000 <= iter_num <= 150000 and iter_num % 10000 == 0:
                print(f"  *** Critical transition period - monitoring phase changes ***")
            
            # 重置损失累计
            running_loss = 0
            loss_count = 0
            
            model.train()
        
        # 保存checkpoint
        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'reward_type': args.reward_type,
                'history': history,  # 保存完整历史
            }
            ckpt_path = os.path.join(out_dir, f'ckpt_{iter_num}.pt')
            torch.save(checkpoint, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")
        
        if iter_num == 0:
            continue
        
        # 训练步骤
        X, Y = get_batch('train')
        
        # 计算损失（带计时）
        if args.reward_type == 'any_valid':
            t_start = time.time()
        
        with ctx:
            if args.reward_type == 'standard':
                logits, loss = model(X, Y)
            elif args.reward_type == 'any_valid':
                loss = compute_any_valid_reward_loss_vectorized(model, X, Y, G, stoi, itos, args.device)
            elif args.reward_type == 'mixed':
                loss = compute_mixed_reward_loss(model, X, Y, G, stoi, itos, args.device, args.mixed_alpha)
            elif args.reward_type == 'diversity':
                loss = compute_diversity_reward_loss(model, X, Y, G, stoi, itos, args.device, args.diversity_weight)
            elif args.reward_type == 'phase_aware':
                loss = compute_phase_aware_loss(model, X, Y, G, stoi, itos, args.device, iter_num, args.phase_aware_transition)
        
        if args.reward_type == 'any_valid':
            loss_times.append(time.time() - t_start)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 累计损失
        running_loss += loss.item()
        loss_count += 1
    
    # 保存最终结果（统一格式）
    with open(os.path.join(out_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 同时保存旧格式以保持兼容性
    final_results = {
        'iterations': history['iter'],
        'tf_accuracy': history['tf_accuracy_all'],  # 使用包含padding的版本以保持兼容
        'tf_accuracy_std': history['tf_accuracy_std'],
        'ar_accuracy': history['ar_accuracy'],
        'phase_history': history['phase_history'],
        'train_loss': [],  # 这里为空，因为我们只记录了test interval的平均值
        'train_iter': [],
        'config': vars(args)
    }
    
    with open(os.path.join(out_dir, 'final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    
    # 绘制增强的训练曲线
    plt.figure(figsize=(20, 15))
    
    # 1. 训练和验证损失
    plt.subplot(3, 4, 1)
    plt.plot(history['iter'], history['train_loss'], 'b-', label='Train', linewidth=2)
    plt.plot(history['iter'], history['val_loss'], 'r-', label='Val', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss - {args.reward_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 两种TF准确率对比
    plt.subplot(3, 4, 2)
    plt.plot(history['iter'], history['tf_accuracy_all'], 'b-', label='TF (w/ padding)', linewidth=2)
    plt.plot(history['iter'], history['tf_accuracy'], 'g-', label='TF (w/o padding)', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0.15, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Teacher Forcing Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. AR准确率
    plt.subplot(3, 4, 3)
    plt.plot(history['iter'], history['ar_accuracy'], 'g-', marker='s', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('AR Accuracy')
    plt.title('Autoregressive Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 4. Token-wise准确率
    plt.subplot(3, 4, 4)
    plt.plot(history['iter'], history['path_accuracy'], 'b-', label='Path', linewidth=2)
    plt.plot(history['iter'], history['newline_accuracy'], 'g-', label='Newline', linewidth=2)
    plt.plot(history['iter'], history['pad_accuracy'], 'r-', label='Padding', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Token-wise Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Anti-preference监控
    plt.subplot(3, 4, 5)
    plt.plot(history['iter'], history['newline_pred_at_pad'], 'r-', linewidth=2)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50%')
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('Anti-preference: Newline at PAD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Padding位置的预测分布
    plt.subplot(3, 4, 6)
    plt.plot(history['iter'], history['pad_pred_ratio'], 'b-', label='Pred PAD', linewidth=2)
    plt.plot(history['iter'], history['newline_pred_at_pad'], 'r-', label='Pred Newline', linewidth=2)
    plt.plot(history['iter'], history['path_pred_at_pad'], 'g-', label='Pred Path', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('Predictions at PAD Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. TF vs AR对比
    plt.subplot(3, 4, 7)
    plt.plot(history['iter'], history['tf_accuracy_all'], 'b-', marker='o', label='TF (w/ pad)', markersize=4)
    plt.plot(history['iter'], history['tf_accuracy'], 'g-', marker='^', label='TF (w/o pad)', markersize=4)
    plt.plot(history['iter'], history['ar_accuracy'], 'r-', marker='s', label='AR', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('All Accuracies Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Phase变化
    plt.subplot(3, 4, 8)
    phase_colors = {
        'early': 'gray',
        'memorization': 'green',
        'transition_imminent': 'orange',
        'transitioning': 'red',
        'post_transition': 'purple',
        'stable': 'blue'
    }
    
    # 绘制phase背景
    for i in range(len(history['iter'])-1):
        if i < len(history['phase_history']):
            plt.axvspan(history['iter'][i], history['iter'][i+1],
                       color=phase_colors.get(history['phase_history'][i], 'gray'),
                       alpha=0.3)
    
    plt.plot(history['iter'], history['tf_accuracy_all'], 'k-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy (w/ padding)')
    plt.title('Learning Phases')
    
    # 添加phase图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.3, label=phase)
                      for phase, color in phase_colors.items()]
    plt.legend(handles=legend_elements, loc='best')
    plt.grid(True, alpha=0.3)
    
    # 9. 关键区间放大
    plt.subplot(3, 4, 9)
    critical_iters = [(i, tf_all, tf, ar) for i, tf_all, tf, ar in 
                      zip(history['iter'], history['tf_accuracy_all'], 
                          history['tf_accuracy'], history['ar_accuracy'])
                      if 100000 <= i <= 160000]
    if critical_iters:
        iters, tfs_all, tfs, ars = zip(*critical_iters)
        plt.plot(iters, tfs_all, 'b-', marker='o', label='TF (w/ pad)', markersize=4)
        plt.plot(iters, tfs, 'g-', marker='^', label='TF (w/o pad)', markersize=4)
        plt.plot(iters, ars, 'r-', marker='s', label='AR', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Critical Period (100k-160k)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 10. Anti-preference vs TF准确率
    plt.subplot(3, 4, 10)
    plt.scatter(history['newline_pred_at_pad'], history['tf_accuracy_all'], alpha=0.6)
    plt.xlabel('Newline at PAD Ratio')
    plt.ylabel('TF Accuracy (w/ padding)')
    plt.title('Anti-preference vs TF Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 11. 训练进度总结
    plt.subplot(3, 4, 11)
    plt.text(0.1, 0.9, f"Experiment: {args.reward_type}", fontsize=14, fontweight='bold')
    
    summary_text = f"""
Final Results:
  TF (w/ padding): {history['tf_accuracy_all'][-1]:.4f}
  TF (w/o padding): {history['tf_accuracy'][-1]:.4f}
  AR Accuracy: {history['ar_accuracy'][-1]:.4f}
  
Token Accuracy:
  Path: {history['path_accuracy'][-1]:.4f}
  Newline: {history['newline_accuracy'][-1]:.4f}
  Padding: {history['pad_accuracy'][-1]:.4f}
  
Anti-preference:
  Newline at PAD: {history['newline_pred_at_pad'][-1]:.2%}
  Max during training: {max(history['newline_pred_at_pad']):.2%}
  
Phase Transitions:
  Final phase: {history['phase_history'][-1]}
"""
    
    plt.text(0.1, 0.1, summary_text, fontsize=11, family='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves_enhanced.png'), dpi=150)
    plt.close()
    
    # 打印总结
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Experiment: {args.reward_type}")
    print(f"\nFinal TF Accuracy:")
    print(f"  With padding: {history['tf_accuracy_all'][-1]:.4f} (±{history['tf_accuracy_std'][-1]:.4f})")
    print(f"  Without padding: {history['tf_accuracy'][-1]:.4f}")
    print(f"\nFinal AR Accuracy: {history['ar_accuracy'][-1]:.4f}")
    print(f"\nToken-wise Accuracy:")
    print(f"  Path: {history['path_accuracy'][-1]:.4f}")
    print(f"  Newline: {history['newline_accuracy'][-1]:.4f}")
    print(f"  Padding: {history['pad_accuracy'][-1]:.4f}")
    print(f"\nAnti-preference:")
    print(f"  Final newline at PAD: {history['newline_pred_at_pad'][-1]:.2%}")
    print(f"  Maximum during training: {max(history['newline_pred_at_pad']):.2%}")
    
    # 找到相变点
    for i in range(1, len(history['tf_accuracy_all'])):
        if history['tf_accuracy_all'][i-1] > 0.5 and history['tf_accuracy_all'][i] < 0.5:
            print(f"\nPhase transition detected at iteration: {history['iter'][i]}")
            if history['newline_pred_at_pad'][i] > 0.5:
                print(f"  Confirmed: Anti-preference at transition = {history['newline_pred_at_pad'][i]:.2%}")
            break
    
    print(f"\nResults saved to: {out_dir}")
    print("  - history.pkl: Complete training history (compatible format)")
    print("  - final_results.pkl: Legacy format for backward compatibility")
    print("  - training_curves_enhanced.png: Detailed visualization")
    print("="*60)

if __name__ == "__main__":
    main()