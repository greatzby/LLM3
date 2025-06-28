"""
Dynamic Batching (Only) - 通过智能分组最小化padding
纯Dynamic Batching，不使用Masked Loss
输出格式与Masked Loss和Dynamic+Masked完全兼容
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import random

from model import GPTConfig, GPT
from logger import get_logger

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Dynamic Batching Training (Without Masked Loss)')
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
    parser.add_argument('--target_batch_size', type=int, default=1024, 
                      help='Target total tokens per batch')
    parser.add_argument('--max_padding_ratio', type=float, default=0.2,
                      help='Maximum allowed padding ratio in a batch')
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

class DynamicBatchDataset:
    """
    动态批处理数据集，按长度分组以最小化padding
    """
    def __init__(self, data_path, block_size, device, 
                 target_batch_size=1024, max_padding_ratio=0.2):
        self.device = device
        self.block_size = block_size
        self.target_batch_size = target_batch_size
        self.max_padding_ratio = max_padding_ratio
        
        # 加载数据
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # 预处理：按实际长度分组
        self._preprocess_data()
        
    def _preprocess_data(self):
        """预处理数据，按长度分组"""
        print("Preprocessing data for dynamic batching...")
        
        self.length_groups = defaultdict(list)
        data_size = self.block_size + 1
        num_sequences = len(self.data) // data_size
        
        for i in range(num_sequences):
            start_idx = i * data_size
            seq = self.data[start_idx:start_idx + data_size]
            
            # 找到实际长度（第一个padding之前）
            actual_length = data_size
            for j, token in enumerate(seq):
                if token == 0:  # padding token
                    actual_length = j
                    break
            
            # 按长度分组存储索引
            self.length_groups[actual_length].append(i)
        
        # 转换为列表并排序
        self.length_keys = sorted(self.length_groups.keys())
        
        # 统计信息
        total_seqs = sum(len(indices) for indices in self.length_groups.values())
        print(f"Total sequences: {total_seqs}")
        print(f"Length distribution:")
        for length in self.length_keys[:10]:  # 显示前10个长度
            count = len(self.length_groups[length])
            print(f"  Length {length}: {count} sequences ({count/total_seqs*100:.1f}%)")
        
    def get_dynamic_batch(self):
        """
        获取一个动态批次
        策略：选择相近长度的序列组成批次，最小化padding
        """
        # 随机选择一个起始长度
        start_length_idx = np.random.randint(len(self.length_keys))
        target_length = self.length_keys[start_length_idx]
        
        batch_indices = []
        batch_tokens = 0
        current_padding_ratio = 0
        
        # 从选定长度开始，尝试填充批次
        for length_offset in range(len(self.length_keys)):
            # 尝试相近的长度
            for direction in [0, 1, -1]:  # 当前长度，更长，更短
                idx = start_length_idx + direction * length_offset
                if 0 <= idx < len(self.length_keys):
                    length = self.length_keys[idx]
                    available_indices = self.length_groups[length]
                    
                    if not available_indices:
                        continue
                    
                    # 计算如果添加这个长度的序列，padding比例
                    max_length = max(target_length, length) if batch_indices else length
                    
                    # 随机选择一些序列
                    num_to_add = min(
                        len(available_indices),
                        (self.target_batch_size - batch_tokens) // max_length
                    )
                    
                    if num_to_add > 0:
                        selected = np.random.choice(available_indices, num_to_add, replace=False)
                        
                        # 计算新的padding比例
                        total_actual = batch_tokens + length * num_to_add
                        total_padded = (len(batch_indices) + num_to_add) * max_length
                        new_padding_ratio = 1 - total_actual / total_padded
                        
                        # 如果padding比例可接受，添加到批次
                        if new_padding_ratio <= self.max_padding_ratio:
                            batch_indices.extend([(idx, max_length) for idx in selected])
                            batch_tokens = total_padded
                            current_padding_ratio = new_padding_ratio
                            target_length = max_length
                        
                        # 如果批次够大了，停止
                        if batch_tokens >= self.target_batch_size * 0.8:
                            break
            
            if batch_tokens >= self.target_batch_size * 0.8:
                break
        
        # 如果批次太小，使用标准方法
        if len(batch_indices) < 4:
            return self._get_standard_batch()
        
        # 构建批次张量
        data_size = self.block_size + 1
        xs, ys = [], []
        
        for seq_idx, padded_length in batch_indices:
            start = seq_idx * data_size
            seq = self.data[start:start + data_size]
            
            # 截取到padded_length
            x = seq[:padded_length-1]
            y = seq[1:padded_length]
            
            # 转换为tensor
            x_tensor = torch.from_numpy(x.astype(np.int64))
            y_tensor = torch.from_numpy(y.astype(np.int64))
            
            xs.append(x_tensor)
            ys.append(y_tensor)
        
        # Pad到相同长度
        x_batch = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
        y_batch = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
        
        return x_batch.to(self.device), y_batch.to(self.device), current_padding_ratio
    
    def _get_standard_batch(self):
        """后备方法：标准批处理"""
        batch_size = 32  # 使用较小的批次
        data_size = self.block_size + 1
        
        ix = torch.randint(len(self.data) // data_size, (batch_size,)) * data_size
        
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) 
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64)) 
            for i in ix
        ])
        
        padding_ratio = (y == 0).float().mean().item()
        
        return x.to(self.device), y.to(self.device), padding_ratio

def compute_stats_standard_loss(model, X, Y, pad_token=0):
    """
    使用标准loss计算，但返回与masked loss兼容的统计信息
    """
    logits, loss = model(X, Y)
    
    with torch.no_grad():
        # 预测
        preds = torch.argmax(logits, dim=-1)
        
        # 创建masks
        mask = Y != pad_token  # 非padding位置
        path_mask = Y > 1  # 路径节点
        newline_mask = Y == 1  # newline
        pad_mask = Y == pad_token  # padding
        
        # 计算padding比例
        padding_ratio = pad_mask.float().mean()
        
        # 路径准确率
        if path_mask.sum() > 0:
            path_accuracy = (preds[path_mask] == Y[path_mask]).float().mean()
        else:
            path_accuracy = torch.tensor(0.0)
        
        # Newline准确率
        if newline_mask.sum() > 0:
            newline_accuracy = (preds[newline_mask] == Y[newline_mask]).float().mean()
        else:
            newline_accuracy = torch.tensor(0.0)
        
        # Padding位置的预测分析
        if pad_mask.sum() > 0:
            pad_preds = preds[pad_mask]
            pad_pred_ratio = (pad_preds == pad_token).float().mean()
            newline_pred_at_pad = (pad_preds == 1).float().mean()
        else:
            pad_pred_ratio = torch.tensor(0.0)
            newline_pred_at_pad = torch.tensor(0.0)
    
    stats = {
        'loss': loss.item(),
        'padding_ratio': padding_ratio.item(),
        'path_accuracy': path_accuracy.item(),
        'newline_accuracy': newline_accuracy.item(),
        'pad_pred_ratio': pad_pred_ratio.item(),
        'newline_pred_at_pad': newline_pred_at_pad.item()
    }
    
    return loss, stats

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/dynamic_batch_only_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print("Dynamic Batching Training (Without Masked Loss)")
    print(f"Target batch size: {args.target_batch_size} tokens")
    print(f"Max padding ratio: {args.max_padding_ratio}")
    print("="*60)
    
    # 加载元信息
    data_dir = os.path.join('data', f'{args.dataset}/{args.num_nodes}')
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = len(itos)
    
    # 创建动态批处理数据集
    if args.num_of_paths == 0:
        train_path = os.path.join(data_dir, 'train.bin')
        val_path = os.path.join(data_dir, 'val.bin')
    else:
        train_path = os.path.join(data_dir, f'train_{args.num_of_paths}.bin')
        val_path = os.path.join(data_dir, 'val.bin')
    
    train_dataset = DynamicBatchDataset(
        train_path, block_size, args.device,
        args.target_batch_size, args.max_padding_ratio
    )
    val_dataset = DynamicBatchDataset(
        val_path, block_size, args.device,
        args.target_batch_size, args.max_padding_ratio
    )
    
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
    
    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in args.device else 'cpu'
    )
    
    # 训练历史 - 完全匹配其他两个版本的格式
    history = {
        'iter': [],
        'train_loss': [],
        'val_loss': [],
        'tf_accuracy': [],  # 这应该是不含padding的准确率
        'train_padding_ratio': [],
        'val_padding_ratio': [],
        'batch_sizes': [],
        'path_accuracy': [],
        'newline_accuracy': [],
        'pad_pred_ratio': [],
        'newline_pred_at_pad': [],
        # 可选：添加AR准确率以匹配masked loss版本
        'ar_accuracy': []
    }
    
    # 评估函数
    @torch.no_grad()
    def evaluate():
        model.eval()
        
        val_losses = []
        val_stats = []
        val_pad_ratios = []
        batch_sizes = []
        
        # TF准确率计算（不含padding）
        tf_correct = 0
        tf_total = 0
        
        for _ in range(20):
            X_val, Y_val, pad_ratio = val_dataset.get_dynamic_batch()
            loss, stats = compute_stats_standard_loss(model, X_val, Y_val)
            
            val_losses.append(stats['loss'])
            val_stats.append(stats)
            val_pad_ratios.append(pad_ratio)
            batch_sizes.append(X_val.shape[0])
            
            # 计算TF准确率（只在非padding位置）
            logits, _ = model(X_val, Y_val)
            preds = torch.argmax(logits, dim=-1)
            mask = Y_val != 0
            if mask.sum() > 0:
                tf_correct += (preds[mask] == Y_val[mask]).sum().item()
                tf_total += mask.sum().item()
        
        # 计算平均值
        avg_stats = {}
        for key in val_stats[0].keys():
            avg_stats[key] = np.mean([s[key] for s in val_stats])
        
        tf_accuracy = tf_correct / tf_total if tf_total > 0 else 0
        
        model.train()
        
        return {
            'val_loss': np.mean(val_losses),
            'tf_accuracy': tf_accuracy,
            'val_padding_ratio': np.mean(val_pad_ratios),
            'batch_size': np.mean(batch_sizes),
            'path_accuracy': avg_stats['path_accuracy'],
            'newline_accuracy': avg_stats['newline_accuracy'],
            'pad_pred_ratio': avg_stats['pad_pred_ratio'],
            'newline_pred_at_pad': avg_stats['newline_pred_at_pad']
        }
    
    # 训练
    print("\nStarting training with dynamic batching (standard loss)...")
    running_loss = 0
    running_padding = []
    running_batch_sizes = []
    loss_count = 0
    
    for iter_num in range(args.max_iters + 1):
        # 学习率调度
        lr = args.learning_rate
        if iter_num < 2000:
            lr = args.learning_rate * iter_num / 2000
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 评估
        if iter_num % args.test_interval == 0:
            avg_train_loss = running_loss / loss_count if loss_count > 0 else 0
            avg_train_padding = np.mean(running_padding) if running_padding else 0
            avg_batch_size = np.mean(running_batch_sizes) if running_batch_sizes else 0
            
            # 验证集评估
            eval_results = evaluate()
            
            # 记录历史
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(eval_results['val_loss'])
            history['tf_accuracy'].append(eval_results['tf_accuracy'])
            history['train_padding_ratio'].append(avg_train_padding)
            history['val_padding_ratio'].append(eval_results['val_padding_ratio'])
            history['batch_sizes'].append(avg_batch_size)
            history['path_accuracy'].append(eval_results['path_accuracy'])
            history['newline_accuracy'].append(eval_results['newline_accuracy'])
            history['pad_pred_ratio'].append(eval_results['pad_pred_ratio'])
            history['newline_pred_at_pad'].append(eval_results['newline_pred_at_pad'])
            history['ar_accuracy'].append(0.0)  # 占位符，保持格式一致
            
            # 打印
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={eval_results['val_loss']:.4f}")
            print(f"  TF Accuracy (w/o padding): {eval_results['tf_accuracy']:.4f}")
            print(f"  Path Accuracy: {eval_results['path_accuracy']:.4f}")
            print(f"  Newline Accuracy: {eval_results['newline_accuracy']:.4f}")
            print(f"  Avg batch size: {avg_batch_size:.1f} sequences")
            print(f"  Padding ratio: train={avg_train_padding:.2%}, val={eval_results['val_padding_ratio']:.2%}")
            print(f"  Newline at PAD: {eval_results['newline_pred_at_pad']:.2%}")
            
            if eval_results['newline_pred_at_pad'] > 0.5:
                print("  ⚠️  Anti-preference detected!")
            
            logger.info(f"Iter {iter_num}: train_loss={avg_train_loss:.4f}, "
                       f"val_loss={eval_results['val_loss']:.4f}, "
                       f"TF={eval_results['tf_accuracy']:.4f}, "
                       f"path_acc={eval_results['path_accuracy']:.4f}, "
                       f"newline_at_pad={eval_results['newline_pred_at_pad']:.2%}")
            
            running_loss = 0
            running_padding = []
            running_batch_sizes = []
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
            print(f"  Checkpoint saved")
        
        if iter_num == 0:
            continue
        
        # 训练步
        X, Y, pad_ratio = train_dataset.get_dynamic_batch()
        
        # 使用标准loss
        loss, _ = compute_stats_standard_loss(model, X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        running_padding.append(pad_ratio)
        running_batch_sizes.append(X.shape[0])
        loss_count += 1
    
    # 保存最终结果
    with open(os.path.join(out_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 绘图（与其他版本类似的布局）
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['iter'], history['train_loss'], label='Train')
    plt.plot(history['iter'], history['val_loss'], label='Val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss with Dynamic Batching')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['iter'], history['train_padding_ratio'], label='Train')
    plt.plot(history['iter'], history['val_padding_ratio'], label='Val')
    plt.axhline(y=0.789, color='r', linestyle='--', alpha=0.5, label='Original (78.9%)')
    plt.xlabel('Iteration')
    plt.ylabel('Padding Ratio')
    plt.title('Padding Ratio Reduction')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['iter'], history['path_accuracy'])
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Path Token Accuracy')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(history['iter'], history['newline_pred_at_pad'])
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('Anti-preference Monitor')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(history['iter'], history['tf_accuracy'])
    plt.xlabel('Iteration')
    plt.ylabel('TF Accuracy')
    plt.title('Teacher Forcing Accuracy (w/o padding)')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(history['iter'], history['batch_sizes'])
    plt.xlabel('Iteration')
    plt.ylabel('Batch Size')
    plt.title('Average Batch Size (sequences)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dynamic_batch_results.png'))
    
    print(f"\nTraining complete! Results saved to {out_dir}")
    print(f"Average padding reduction: {78.9 - np.mean(history['train_padding_ratio'])*100:.1f}%")
    
    # 检查anti-preference
    if max(history['newline_pred_at_pad']) > 0.5:
        print(f"\n⚠️  WARNING: Anti-preference detected!")
        print(f"  Maximum newline at PAD: {max(history['newline_pred_at_pad']):.2%}")

if __name__ == "__main__":
    main()