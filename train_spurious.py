"""
Spurious Rewards 实验脚本 - 真正的向量化版本
完全消除Python循环，大幅提升速度
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
                      choices=['standard', 'any_valid', 'mixed', 'diversity', 'phase_aware', 'standard_entropy'],
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

def compute_standard_entropy_loss(model, X, Y, device, entropy_weight=0.1):
    """标准CE损失 + 熵正则化"""
    logits, _ = model(X, Y)
    
    # 标准CE loss
    standard_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        Y.view(-1), 
        ignore_index=0
    )
    
    # 计算熵
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    
    # Mask掉padding
    mask = (Y != 0).float()
    masked_entropy = entropy * mask
    avg_entropy = masked_entropy.sum() / mask.sum()
    
    # 负熵作为损失（我们想最大化熵）
    entropy_loss = -entropy_weight * avg_entropy
    
    return standard_loss + entropy_loss

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
    
    # Teacher Forcing评估
    @torch.no_grad()
    def test_model():
        total_correct = 0
        total_count = 0
        batch_accuracies = []
        
        for batch_idx in range(args.num_eval_batches):
            X, Y = get_batch('val')
            with ctx:
                logits, _ = model(X, Y)
            preds = torch.argmax(logits, dim=-1)
            
            batch_correct = (preds == Y).float().sum().item()
            batch_total = Y.numel()
            batch_accuracy = batch_correct / batch_total
            batch_accuracies.append(batch_accuracy)
            
            total_correct += batch_correct
            total_count += batch_total
        
        overall_accuracy = total_correct / total_count
        accuracy_std = np.std(batch_accuracies)
        
        return overall_accuracy, accuracy_std
    
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
    
    # 训练历史记录
    train_loss_history = []
    train_iter_history = []
    tf_accuracy_history = []
    tf_accuracy_std_history = []
    ar_accuracy_history = []
    test_iter_history = []
    phase_history = []
    
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
            
            # 计算验证损失
            val_losses = []
            for _ in range(10):
                X_val, Y_val = get_batch('val')
                with torch.no_grad():
                    _, val_loss = model(X_val, Y_val)
                val_losses.append(val_loss.item())
            val_loss = np.mean(val_losses)
            
            # Teacher Forcing准确率
            tf_acc, tf_std = test_model()
            tf_accuracy_history.append(tf_acc)
            tf_accuracy_std_history.append(tf_std)
            
            # 更新phase检测器
            phase_detector.update(tf_acc)
            current_phase = phase_detector.detect_phase()
            phase_history.append(current_phase)
            
            # Autoregressive准确率
            ar_acc, error_counts = evaluate_autoregressive_lenient()
            ar_accuracy_history.append(ar_acc)
            
            test_iter_history.append(iter_num)
            
            # 打印信息
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={val_loss:.4f}")
            print(f"  Teacher Forcing: {tf_acc:.4f} (±{tf_std:.4f})")
            print(f"  Autoregressive: {ar_acc:.4f}")
            print(f"  Phase: {current_phase}")
            print(f"  Errors: syntax={error_counts['wrong syntax']}, "
                  f"start/end={error_counts['incorrect start/end']}, "
                  f"path={error_counts['non-existence path']}")
            
            # 如果是any_valid，打印平均计算时间
            if args.reward_type == 'any_valid' and loss_times:
                avg_time = np.mean(loss_times[-1000:])  # 最近1000次的平均
                print(f"  Avg loss computation time: {avg_time*1000:.2f}ms")
            
            # 记录到日志
            logger.info(f"Iter {iter_num}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                       f"TF={tf_acc:.4f}±{tf_std:.4f}, AR={ar_acc:.4f}, phase={current_phase}")
            
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
                'tf_history': tf_accuracy_history,
                'ar_history': ar_accuracy_history,
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
            elif args.reward_type == 'standard_entropy':
                loss = compute_standard_entropy_loss(model, X, Y, args.device, args.diversity_weight)
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
        
        # 记录完整历史
        train_loss_history.append(loss.item())
        train_iter_history.append(iter_num)
    
    # 保存最终结果
    final_results = {
        'iterations': test_iter_history,
        'tf_accuracy': tf_accuracy_history,
        'tf_accuracy_std': tf_accuracy_std_history,
        'ar_accuracy': ar_accuracy_history,
        'phase_history': phase_history,
        'train_loss': train_loss_history,
        'train_iter': train_iter_history,
        'config': vars(args)
    }
    
    with open(os.path.join(out_dir, 'final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    
    # 绘制训练曲线
    plt.figure(figsize=(16, 10))
    
    # 1. 训练损失
    plt.subplot(2, 3, 1)
    plt.plot(train_iter_history, train_loss_history, 'b-', linewidth=0.5, alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss - {args.reward_type}')
    plt.grid(True, alpha=0.3)
    
    # 2. TF准确率
    plt.subplot(2, 3, 2)
    plt.errorbar(test_iter_history, tf_accuracy_history, yerr=tf_accuracy_std_history,
                marker='o', capsize=5, markersize=4)
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    plt.axhline(y=0.15, color='r', linestyle='--', alpha=0.5, label='15%')
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Teacher Forcing Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. AR准确率
    plt.subplot(2, 3, 3)
    plt.plot(test_iter_history, ar_accuracy_history, 'g-', marker='s', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('AR Accuracy')
    plt.title('Autoregressive Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 4. TF vs AR对比
    plt.subplot(2, 3, 4)
    plt.plot(test_iter_history, tf_accuracy_history, 'b-', marker='o', label='TF', markersize=4)
    plt.plot(test_iter_history, ar_accuracy_history, 'g-', marker='s', label='AR', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('TF vs AR Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Phase变化
    plt.subplot(2, 3, 5)
    phase_colors = {
        'early': 'gray',
        'memorization': 'green',
        'transition_imminent': 'orange',
        'transitioning': 'red',
        'post_transition': 'purple',
        'stable': 'blue'
    }
    
    # 绘制phase背景
    for i in range(len(test_iter_history)-1):
        if i < len(phase_history):
            plt.axvspan(test_iter_history[i], test_iter_history[i+1],
                       color=phase_colors.get(phase_history[i], 'gray'),
                       alpha=0.3)
    
    plt.plot(test_iter_history, tf_accuracy_history, 'k-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Learning Phases')
    
    # 添加phase图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.3, label=phase)
                      for phase, color in phase_colors.items()]
    plt.legend(handles=legend_elements, loc='best')
    plt.grid(True, alpha=0.3)
    
    # 6. 关键区间放大
    plt.subplot(2, 3, 6)
    critical_iters = [(i, tf, ar) for i, tf, ar in 
                      zip(test_iter_history, tf_accuracy_history, ar_accuracy_history)
                      if 100000 <= i <= 160000]
    if critical_iters:
        iters, tfs, ars = zip(*critical_iters)
        plt.plot(iters, tfs, 'b-', marker='o', label='TF', markersize=4)
        plt.plot(iters, ars, 'g-', marker='s', label='AR', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Critical Period (100k-160k)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # 打印总结
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Experiment: {args.reward_type}")
    print(f"Final TF Accuracy: {tf_accuracy_history[-1]:.4f} (±{tf_accuracy_std_history[-1]:.4f})")
    print(f"Final AR Accuracy: {ar_accuracy_history[-1]:.4f}")
    print(f"Minimum TF Accuracy: {min(tf_accuracy_history):.4f}")
    print(f"Maximum TF drop: {max(tf_accuracy_history) - min(tf_accuracy_history):.4f}")
    
    # 找到相变点
    for i in range(1, len(tf_accuracy_history)):
        if tf_accuracy_history[i-1] > 0.5 and tf_accuracy_history[i] < 0.5:
            print(f"Phase transition detected at iteration: {test_iter_history[i]}")
            break
    
    print(f"Results saved to: {out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()