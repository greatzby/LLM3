"""
Dynamic Batching - 通过智能分组最小化padding
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

from model import GPTConfig, GPT
from logger import get_logger
from train_masked_loss import compute_masked_loss, set_seed

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Dynamic Batching Training')
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
    parser.add_argument('--checkpoint_interval', type=int, default=50000)
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

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/dynamic_batch_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print("Dynamic Batching Training")
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
        'train_padding_ratio': [],
        'val_padding_ratio': [],
        'batch_sizes': [],
        'path_accuracy': [],
        'newline_pred_at_pad': []
    }
    
    # 评估函数
    @torch.no_grad()
    def evaluate():
        model.eval()
        
        val_losses = []
        val_stats = []
        val_pad_ratios = []
        
        for _ in range(20):
            X_val, Y_val, pad_ratio = val_dataset.get_dynamic_batch()
            loss, stats = compute_masked_loss(model, X_val, Y_val)
            val_losses.append(loss.item())
            val_stats.append(stats)
            val_pad_ratios.append(pad_ratio)
        
        avg_stats = {}
        for key in val_stats[0].keys():
            if key != 'loss':
                avg_stats[key] = np.mean([s[key] for s in val_stats])
        
        model.train()
        
        return np.mean(val_losses), avg_stats, np.mean(val_pad_ratios)
    
    # 训练
    print("\nStarting training with dynamic batching...")
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
            
            val_loss, val_stats, val_pad_ratio = evaluate()
            
            # 记录
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['tf_accuracy'].append(val_stats.get('path_accuracy', 0))
            history['train_padding_ratio'].append(avg_train_padding)
            history['val_padding_ratio'].append(val_pad_ratio)
            history['batch_sizes'].append(avg_batch_size)
            history['path_accuracy'].append(val_stats.get('path_accuracy', 0))
            history['newline_pred_at_pad'].append(val_stats.get('newline_pred_at_pad', 0))
            
            # 打印
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={val_loss:.4f}")
            print(f"  Path Accuracy: {val_stats.get('path_accuracy', 0):.4f}")
            print(f"  Avg batch size: {avg_batch_size:.1f} sequences")
            print(f"  Padding ratio: train={avg_train_padding:.2%}, val={val_pad_ratio:.2%}")
            print(f"  Newline at PAD: {val_stats.get('newline_pred_at_pad', 0):.2%}")
            
            if val_stats.get('newline_pred_at_pad', 0) > 0.5:
                print("  ⚠️  Anti-preference detected!")
            
            running_loss = 0
            running_padding = []
            running_batch_sizes = []
            loss_count = 0
        
        if iter_num == 0:
            continue
        
        # 训练步
        X, Y, pad_ratio = train_dataset.get_dynamic_batch()
        
        loss, _ = compute_masked_loss(model, X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        running_padding.append(pad_ratio)
        running_batch_sizes.append(X.shape[0])
        loss_count += 1
    
    # 保存结果
    with open(os.path.join(out_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 绘图
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dynamic_batch_results.png'))
    
    print(f"\nTraining complete! Results saved to {out_dir}")
    print(f"Average padding reduction: {78.9 - np.mean(history['train_padding_ratio'])*100:.1f}%")

if __name__ == "__main__":
    main()