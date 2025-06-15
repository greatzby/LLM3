"""
修复版SimPO训练脚本 - 确保可重复性和正确的参数解析
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

from model import GPTConfig, GPT
from logger import get_logger
import logging

# 固定随机种子函数
def set_seed(seed=42):
    """固定所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description='SimPO-Inspired Training for Path Planning')
    parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=120, help='Size of the embeddings')
    parser.add_argument('--max_iters', type=int, default=50000, help='Total number of training iterations')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths')
    parser.add_argument('--test_interval', type=int, default=100, help='Interval (in iterations) for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--num_eval_batches', type=int, default=10, help='Number of batches for TF evaluation')
    
    # SimPO相关参数 - 修复布尔值解析
    parser.add_argument('--simpo_beta', type=float, default=10.0, help='SimPO beta parameter')
    parser.add_argument('--use_simpo', action='store_true', help='Use SimPO-inspired loss')
    parser.add_argument('--no_simpo', dest='use_simpo', action='store_false', help='Use original loss')
    parser.set_defaults(use_simpo=True)
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

args = parse_args()

# 立即设置随机种子
set_seed(args.seed)

# 解析参数
dataset      = args.dataset
n_layer      = args.n_layer
n_head       = args.n_head
n_embd       = args.n_embd
max_iters    = args.max_iters
num_nodes    = args.num_nodes
num_of_paths = args.num_of_paths
test_interval= args.test_interval
device       = args.device
temperature  = args.temperature
num_eval_batches = args.num_eval_batches
simpo_beta   = args.simpo_beta
use_simpo    = args.use_simpo

# 打印配置
print("="*60)
print(f"Training Configuration:")
print(f"  Use SimPO: {use_simpo}")
print(f"  SimPO Beta: {simpo_beta}")
print(f"  Random Seed: {args.seed}")
print(f"  Max Iterations: {max_iters}")
print(f"  Test Interval: {test_interval}")
print("="*60)

# 数据和模型初始化代码...
data_dir = os.path.join('data', f'{dataset}/{num_nodes}')
meta_path = os.path.join(data_dir, 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos  = meta['stoi'], meta['itos']
block_size  = meta['block_size']
top_k       = len(itos)
simple_format = meta.get('simple_format', False)

# 修改输出目录名称
if use_simpo:
    out_dir = f'out/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}_simpo_beta{simpo_beta}_seed{args.seed}'
else:
    out_dir = f'out/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}_original_seed{args.seed}'
os.makedirs(out_dir, exist_ok=True)

logger = get_logger(os.path.join(out_dir, "train.log"))

# 训练参数
gradient_accumulation_steps = 1
train_batch_size = 1024
val_batch_size   = 64
batch_size       = train_batch_size

master_process = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 加载数据
if num_of_paths == 0:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data   = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_of_paths}.bin'), dtype=np.uint16, mode='r')
    val_data   = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    """获取批次数据 - 使用固定的随机种子"""
    data = train_data if split == 'train' else val_data
    bs = train_batch_size if split == 'train' else val_batch_size
    data_size = block_size + 1
    
    # 使用torch的随机数生成器（已经固定种子）
    ix = torch.randint((len(data) - data_size) // data_size, (bs,)) * data_size
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# [插入其他函数：test_model, encode, decode等...]

# 模型初始化
init_from = 'scratch'
meta_vocab_size = meta.get('vocab_size', None)
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
                  dropout=0.0)
                  
print("Initializing a new model from scratch")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
model.train()

# 优化器
scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))
weight_decay   = 1e-1
learning_rate  = 5e-4
beta1, beta2   = 0.9, 0.95
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# 学习率调度
decay_lr = True
warmup_iters = max_iters // 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# SimPO损失函数
def compute_simpo_loss(model, X, Y, iteration, beta=10.0):
    """SimPO启发的损失函数"""
    logits, _ = model(X, Y)
    
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        Y.view(-1), 
        ignore_index=0,
        reduction='none'
    )
    ce_loss = ce_loss.view(Y.shape)
    
    # 长度归一化
    mask = (Y != 0).float()
    lengths = mask.sum(dim=1).clamp(min=1)
    normalized_loss = (ce_loss * mask).sum(dim=1) / lengths
    
    # 动态beta
    if iteration < 100000:
        current_beta = 2.0
    elif iteration < 120000:
        progress = (iteration - 100000) / 20000
        current_beta = 2.0 + (beta - 2.0) * progress
    else:
        current_beta = beta
        
    scaled_loss = normalized_loss.mean() * current_beta
    return scaled_loss, current_beta

# 训练循环
print("\n开始训练...")
iter_num = 0
t0 = time.time()

# [记录变量初始化...]
train_loss_history = []
train_iter_history = []
tf_accuracy_history = []
ar_accuracy_history = []
test_iter_history = []

while True:
    # 学习率调整
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # [评估和checkpoint保存代码...]
    
    # 训练步骤
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with ctx:
            if use_simpo:
                # 使用SimPO损失
                loss, current_beta = compute_simpo_loss(model, X, Y, iter_num, simpo_beta)
            else:
                # 使用原始损失
                logits, loss = model(X, Y)
                current_beta = 1.0
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    
    # 优化器更新
    grad_clip = 1.0
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # 记录
    if iter_num % 100 == 0:
        loss_val = loss.item() * gradient_accumulation_steps
        if use_simpo:
            print(f"iter {iter_num}: loss {loss_val:.4f}, beta {current_beta:.2f}")
        else:
            print(f"iter {iter_num}: loss {loss_val:.4f}")
    
    iter_num += 1
    if iter_num > max_iters:
        break

print("\nTraining completed!")