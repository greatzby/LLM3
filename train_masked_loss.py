"""
Masked Loss Training - 只在实际路径token上计算损失
解决padding主导学习信号的问题
"""
import os
import time
import math
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from contextlib import nullcontext
import matplotlib.pyplot as plt
from datetime import datetime

from model import GPTConfig, GPT
from logger import get_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Masked Loss Training')
    parser.add_argument('--dataset', type=str, default='simple_graph')
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--max_iters', type=int, default=300000)
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--num_of_paths', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--checkpoint_interval', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_masked_loss(model, X, Y, pad_token=0):
    """
    计算masked loss - 完全忽略padding位置
    
    关键改进：
    1. 使用ignore_index确保padding不参与loss计算
    2. 计算准确的token级别mask
    3. 返回详细的统计信息用于监控
    """
    logits, _ = model(X, Y)
    batch_size, seq_len, vocab_size = logits.shape
    
    # 创建mask: True表示需要计算loss的位置
    mask = (Y != pad_token)
    
    # 计算loss时ignore padding
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1),
        ignore_index=pad_token,
        reduction='mean'  # 注意：这里用mean会自动处理mask
    )
    
    # 计算一些统计信息
    with torch.no_grad():
        # 有效token数量
        valid_tokens = mask.sum()
        total_tokens = mask.numel()
        padding_ratio = 1 - (valid_tokens.float() / total_tokens)
        
        # 分别计算不同类型token的准确率
        preds = torch.argmax(logits, dim=-1)
        
        # 路径token准确率（排除padding和newline）
        path_mask = (Y > 1) & mask  # token > 1是实际的节点
        if path_mask.sum() > 0:
            path_acc = (preds[path_mask] == Y[path_mask]).float().mean()
        else:
            path_acc = torch.tensor(0.0)
        
        # Newline准确率
        newline_mask = (Y == 1) & mask
        if newline_mask.sum() > 0:
            newline_acc = (preds[newline_mask] == Y[newline_mask]).float().mean()
        else:
            newline_acc = torch.tensor(0.0)
        
        # Padding预测分布（看模型在padding位置预测了什么）
        if (~mask).sum() > 0:  # 如果有padding
            padding_preds = preds[~mask]
            pad_pred_ratio = (padding_preds == pad_token).float().mean()
            newline_pred_ratio = (padding_preds == 1).float().mean()
        else:
            pad_pred_ratio = torch.tensor(0.0)
            newline_pred_ratio = torch.tensor(0.0)
    
    stats = {
        'loss': loss.item(),
        'padding_ratio': padding_ratio.item(),
        'path_accuracy': path_acc.item(),
        'newline_accuracy': newline_acc.item(),
        'pad_pred_ratio': pad_pred_ratio.item(),
        'newline_pred_at_pad': newline_pred_ratio.item()
    }
    
    return loss, stats

def get_batch_with_stats(data, batch_size, block_size, device):
    """获取batch并返回padding统计"""
    data_size = block_size + 1
    ix = torch.randint((len(data) - data_size) // data_size, (batch_size,)) * data_size
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # 计算padding比例
    padding_ratio = (y == 0).float().mean().item()
    
    return x.to(device), y.to(device), padding_ratio

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/masked_loss_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置logger
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print("Masked Loss Training")
    print("="*60)
    
    # 加载数据和元信息
    data_dir = os.path.join('data', f'{args.dataset}/{args.num_nodes}')
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = len(itos)
    
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
    model = GPT(gptconf).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 优化器
    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in args.device else 'cpu'
    )
    
    # 训练历史
    history = {
        'iter': [],
        'train_loss': [],
        'val_loss': [],
        'tf_accuracy': [],
        'ar_accuracy': [],
        'path_accuracy': [],
        'newline_accuracy': [],
        'pad_pred_ratio': [],
        'newline_pred_at_pad': []
    }
    
    # 评估函数
    @torch.no_grad()
    def evaluate():
        model.eval()
        
        # 验证集loss和统计
        val_losses = []
        val_stats = []
        for _ in range(10):
            X_val, Y_val, _ = get_batch_with_stats(val_data, 64, block_size, args.device)
            loss, stats = compute_masked_loss(model, X_val, Y_val)
            val_losses.append(loss.item())
            val_stats.append(stats)
        
        # 平均统计
        avg_stats = {}
        for key in val_stats[0].keys():
            avg_stats[key] = np.mean([s[key] for s in val_stats])
        
        # Teacher Forcing准确率（只在非padding位置）
        tf_correct = 0
        tf_total = 0
        for _ in range(10):
            X_val, Y_val, _ = get_batch_with_stats(val_data, 64, block_size, args.device)
            logits, _ = model(X_val, Y_val)
            preds = torch.argmax(logits, dim=-1)
            mask = Y_val != 0
            tf_correct += (preds[mask] == Y_val[mask]).sum().item()
            tf_total += mask.sum().item()
        
        tf_accuracy = tf_correct / tf_total if tf_total > 0 else 0
        
        model.train()
        return avg_stats, tf_accuracy
    
    # 训练循环
    print("\nStarting training with masked loss...")
    running_loss = 0
    loss_count = 0
    
    for iter_num in range(args.max_iters + 1):
        # 学习率调度
        lr = args.learning_rate
        if iter_num < 2000:
            lr = args.learning_rate * iter_num / 2000
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 定期评估
        if iter_num % args.test_interval == 0:
            avg_train_loss = running_loss / loss_count if loss_count > 0 else 0
            stats, tf_acc = evaluate()
            
            # 记录历史
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(stats['loss'])
            history['tf_accuracy'].append(tf_acc)
            history['path_accuracy'].append(stats['path_accuracy'])
            history['newline_accuracy'].append(stats['newline_accuracy'])
            history['pad_pred_ratio'].append(stats['pad_pred_ratio'])
            history['newline_pred_at_pad'].append(stats['newline_pred_at_pad'])
            
            # 打印
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={stats['loss']:.4f}")
            print(f"  TF Accuracy: {tf_acc:.4f}")
            print(f"  Path Token Acc: {stats['path_accuracy']:.4f}")
            print(f"  Newline Token Acc: {stats['newline_accuracy']:.4f}")
            print(f"  Model predicts PAD at PAD positions: {stats['pad_pred_ratio']:.2%}")
            print(f"  Model predicts Newline at PAD positions: {stats['newline_pred_at_pad']:.2%}")
            
            # 检测反偏好现象
            if stats['newline_pred_at_pad'] > 0.5:
                print("  ⚠️  WARNING: Anti-preference detected! Model predicting newline at padding positions!")
            
            running_loss = 0
            loss_count = 0
        
        # 保存checkpoint
        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'history': history
            }
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
        
        if iter_num == 0:
            continue
        
        # 训练步
        X, Y, batch_pad_ratio = get_batch_with_stats(train_data, args.batch_size, block_size, args.device)
        
        loss, stats = compute_masked_loss(model, X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        loss_count += 1
    
    # 保存最终结果和绘图
    with open(os.path.join(out_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['iter'], history['train_loss'], label='Train')
    plt.plot(history['iter'], history['val_loss'], label='Val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['iter'], history['tf_accuracy'])
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Teacher Forcing Accuracy')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['iter'], history['path_accuracy'], label='Path Tokens')
    plt.plot(history['iter'], history['newline_accuracy'], label='Newline')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Token-wise Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(history['iter'], history['newline_pred_at_pad'])
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('Newline Predictions at PAD Positions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'))
    
    print(f"\nTraining complete! Results saved to {out_dir}")

if __name__ == "__main__":
    main()