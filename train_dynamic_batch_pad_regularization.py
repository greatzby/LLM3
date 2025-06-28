"""
Dynamic Batching + Masked CE + PAD正则化
生产级配置：CE_masked + λ × KL(pred_PAD || one-hot_PAD)
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
    parser = argparse.ArgumentParser(description='Dynamic Batching + Masked CE + PAD Regularization')
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
    
    # PAD正则化参数
    parser.add_argument('--pad_reg_weight', type=float, default=0.05,
                      help='Weight for PAD regularization (λ in the formula)')
    parser.add_argument('--pad_reg_start_iter', type=int, default=0,
                      help='Start iteration for PAD regularization (0 means from beginning)')
    
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

def compute_masked_ce_with_pad_reg(model, X, Y, pad_token=0, pad_reg_weight=0.05, compute_reg=True):
    """
    计算Masked CE Loss + PAD正则化
    CE_masked + λ × KL(pred_PAD || one-hot_PAD)
    
    Args:
        model: 模型
        X: 输入
        Y: 目标
        pad_token: padding token ID (默认0)
        pad_reg_weight: PAD正则化权重λ
        compute_reg: 是否计算正则化项
    """
    logits, _ = model(X, Y)
    batch_size, seq_len, vocab_size = logits.shape
    
    # 创建mask
    mask = (Y != pad_token)
    pad_mask = (Y == pad_token)
    
    # 1. Masked CE Loss (主损失)
    ce_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1),
        ignore_index=pad_token,
        reduction='mean'
    )
    
    # 2. PAD正则化项 (KL散度)
    if compute_reg and pad_mask.sum() > 0 and pad_reg_weight > 0:
        # 获取padding位置的预测
        pad_logits = logits[pad_mask]  # (num_pad_positions, vocab_size)
        pad_probs = F.softmax(pad_logits, dim=-1)
        
        # 创建one-hot目标分布 (所有概率在pad_token上)
        pad_target = torch.zeros_like(pad_probs)
        pad_target[:, pad_token] = 1.0
        
        # 计算KL散度: KL(pred || target) = sum(pred * log(pred/target))
        # 为了数值稳定性，使用log_softmax
        pad_log_probs = F.log_softmax(pad_logits, dim=-1)
        kl_div = F.kl_div(pad_log_probs, pad_target, reduction='batchmean')
        
        # 组合损失
        total_loss = ce_loss + pad_reg_weight * kl_div
        reg_loss = kl_div.item()
    else:
        total_loss = ce_loss
        reg_loss = 0.0
    
    # 计算统计信息
    with torch.no_grad():
        # 预测
        preds = torch.argmax(logits, dim=-1)
        
        # 有效token数量
        valid_tokens = mask.sum()
        total_tokens = mask.numel()
        padding_ratio = 1 - (valid_tokens.float() / total_tokens)
        
        # 路径token准确率（排除padding和newline）
        path_mask = (Y > 1) & mask
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
        
        # Padding预测分析
        if pad_mask.sum() > 0:
            padding_preds = preds[pad_mask]
            pad_pred_ratio = (padding_preds == pad_token).float().mean()
            newline_pred_ratio = (padding_preds == 1).float().mean()
        else:
            pad_pred_ratio = torch.tensor(0.0)
            newline_pred_ratio = torch.tensor(0.0)
    
    stats = {
        'loss': total_loss.item(),
        'ce_loss': ce_loss.item(),
        'reg_loss': reg_loss,
        'padding_ratio': padding_ratio.item(),
        'path_accuracy': path_acc.item(),
        'newline_accuracy': newline_acc.item(),
        'pad_pred_ratio': pad_pred_ratio.item(),
        'newline_pred_at_pad': newline_pred_ratio.item()
    }
    
    return total_loss, stats

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/dynamic_masked_padreg_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print("Dynamic Batching + Masked CE + PAD Regularization")
    print(f"Target batch size: {args.target_batch_size} tokens")
    print(f"Max padding ratio: {args.max_padding_ratio}")
    print(f"PAD regularization weight (λ): {args.pad_reg_weight}")
    print("="*60)
    
    # 记录配置
    logger.info("="*60)
    logger.info("Dynamic Batching + Masked CE + PAD Regularization")
    logger.info(f"Configuration: {vars(args)}")
    logger.info("="*60)
    
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
        'newline_accuracy': [],
        'pad_pred_ratio': [],
        'newline_pred_at_pad': [],
        'ce_loss': [],
        'reg_loss': []
    }
    
    # 评估函数
    @torch.no_grad()
    def evaluate():
        model.eval()
        
        val_losses = []
        val_stats = []
        val_pad_ratios = []
        
        # TF准确率统计
        tf_correct = 0
        tf_total = 0
        
        for _ in range(20):
            X_val, Y_val, pad_ratio = val_dataset.get_dynamic_batch()
            
            # 评估时不使用正则化
            loss, stats = compute_masked_ce_with_pad_reg(
                model, X_val, Y_val, 
                pad_reg_weight=0,  # 评估时不计算正则化
                compute_reg=False
            )
            
            val_losses.append(loss.item())
            val_stats.append(stats)
            val_pad_ratios.append(pad_ratio)
            
            # 计算TF准确率（只在非padding位置）
            logits, _ = model(X_val, Y_val)
            preds = torch.argmax(logits, dim=-1)
            mask = Y_val != 0
            if mask.sum() > 0:
                tf_correct += (preds[mask] == Y_val[mask]).sum().item()
                tf_total += mask.sum().item()
        
        # 平均统计
        avg_stats = {}
        for key in val_stats[0].keys():
            if key != 'loss':
                avg_stats[key] = np.mean([s[key] for s in val_stats])
        
        tf_accuracy = tf_correct / tf_total if tf_total > 0 else 0
        
        model.train()
        
        return np.mean(val_losses), avg_stats, np.mean(val_pad_ratios), tf_accuracy
    
    # 训练
    print("\nStarting training with Dynamic Batching + Masked CE + PAD Regularization...")
    running_loss = 0
    running_ce_loss = 0
    running_reg_loss = 0
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
            avg_ce_loss = running_ce_loss / loss_count if loss_count > 0 else 0
            avg_reg_loss = running_reg_loss / loss_count if loss_count > 0 else 0
            avg_train_padding = np.mean(running_padding) if running_padding else 0
            avg_batch_size = np.mean(running_batch_sizes) if running_batch_sizes else 0
            
            val_loss, val_stats, val_pad_ratio, tf_accuracy = evaluate()
            
            # 记录
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['tf_accuracy'].append(tf_accuracy)
            history['train_padding_ratio'].append(avg_train_padding)
            history['val_padding_ratio'].append(val_pad_ratio)
            history['batch_sizes'].append(avg_batch_size)
            history['path_accuracy'].append(val_stats['path_accuracy'])
            history['newline_accuracy'].append(val_stats['newline_accuracy'])
            history['pad_pred_ratio'].append(val_stats['pad_pred_ratio'])
            history['newline_pred_at_pad'].append(val_stats['newline_pred_at_pad'])
            history['ce_loss'].append(avg_ce_loss)
            history['reg_loss'].append(avg_reg_loss)
            
            # 打印
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Total Loss: train={avg_train_loss:.4f}, val={val_loss:.4f}")
            print(f"  CE Loss: {avg_ce_loss:.4f}, Reg Loss: {avg_reg_loss:.6f}")
            print(f"  TF Accuracy (w/o padding): {tf_accuracy:.4f}")
            print(f"  Path Accuracy: {val_stats['path_accuracy']:.4f}")
            print(f"  Newline Accuracy: {val_stats['newline_accuracy']:.4f}")
            print(f"  Avg batch size: {avg_batch_size:.1f} sequences")
            print(f"  Padding ratio: train={avg_train_padding:.2%}, val={val_pad_ratio:.2%}")
            print(f"  PAD predictions at PAD: {val_stats['pad_pred_ratio']:.2%}")
            print(f"  Newline at PAD: {val_stats['newline_pred_at_pad']:.2%}")
            
            if val_stats['newline_pred_at_pad'] > 0.5:
                print("  ⚠️  Anti-preference detected! (But should be controlled by regularization)")
            elif val_stats['newline_pred_at_pad'] < 0.1 and val_stats['pad_pred_ratio'] > 0.8:
                print("  ✅ PAD regularization working well!")
            
            logger.info(f"Iter {iter_num}: loss={avg_train_loss:.4f}, "
                       f"CE={avg_ce_loss:.4f}, Reg={avg_reg_loss:.6f}, "
                       f"TF={tf_accuracy:.4f}, path_acc={val_stats['path_accuracy']:.4f}, "
                       f"newline_at_pad={val_stats['newline_pred_at_pad']:.2%}")
            
            running_loss = 0
            running_ce_loss = 0
            running_reg_loss = 0
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
                'history': history,
                'config': vars(args)
            }
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
            print(f"  Checkpoint saved")
        
        if iter_num == 0:
            continue
        
        # 训练步
        X, Y, pad_ratio = train_dataset.get_dynamic_batch()
        
        # 决定是否使用正则化
        use_reg = iter_num >= args.pad_reg_start_iter
        
        loss, stats = compute_masked_ce_with_pad_reg(
            model, X, Y, 
            pad_reg_weight=args.pad_reg_weight if use_reg else 0,
            compute_reg=use_reg
        )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += stats['loss']
        running_ce_loss += stats['ce_loss']
        running_reg_loss += stats['reg_loss']
        running_padding.append(pad_ratio)
        running_batch_sizes.append(X.shape[0])
        loss_count += 1
    
    # 保存结果
    with open(os.path.join(out_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 绘图
    plt.figure(figsize=(18, 12))
    
    # 1. 总损失
    plt.subplot(3, 3, 1)
    plt.plot(history['iter'], history['train_loss'], label='Train Total')
    plt.plot(history['iter'], history['val_loss'], label='Val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. CE损失和正则化损失
    plt.subplot(3, 3, 2)
    plt.plot(history['iter'], history['ce_loss'], label='CE Loss')
    plt.plot(history['iter'], history['reg_loss'], label='Reg Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 3. Padding比例
    plt.subplot(3, 3, 3)
    plt.plot(history['iter'], history['train_padding_ratio'], label='Train')
    plt.plot(history['iter'], history['val_padding_ratio'], label='Val')
    plt.axhline(y=0.789, color='r', linestyle='--', alpha=0.5, label='Original (78.9%)')
    plt.xlabel('Iteration')
    plt.ylabel('Padding Ratio')
    plt.title('Padding Ratio Reduction')
    plt.legend()
    plt.grid(True)
    
    # 4. TF准确率
    plt.subplot(3, 3, 4)
    plt.plot(history['iter'], history['tf_accuracy'], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Teacher Forcing Accuracy (w/o padding)')
    plt.grid(True)
    
    # 5. Token级别准确率
    plt.subplot(3, 3, 5)
    plt.plot(history['iter'], history['path_accuracy'], label='Path', linewidth=2)
    plt.plot(history['iter'], history['newline_accuracy'], label='Newline', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Token-wise Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 6. Anti-preference监控
    plt.subplot(3, 3, 6)
    plt.plot(history['iter'], history['newline_pred_at_pad'], 'r-', label='Newline at PAD', linewidth=2)
    plt.plot(history['iter'], history['pad_pred_ratio'], 'g-', label='PAD at PAD', linewidth=2)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('PAD Position Predictions')
    plt.legend()
    plt.grid(True)
    
    # 7. 批次大小
    plt.subplot(3, 3, 7)
    plt.plot(history['iter'], history['batch_sizes'])
    plt.xlabel('Iteration')
    plt.ylabel('Batch Size (sequences)')
    plt.title('Average Batch Size')
    plt.grid(True)
    
    # 8. 正则化效果
    plt.subplot(3, 3, 8)
    # 计算anti-preference score
    anti_pref_score = np.array(history['newline_pred_at_pad']) - np.array(history['pad_pred_ratio'])
    plt.plot(history['iter'], anti_pref_score)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Anti-preference Score')
    plt.title('Newline@PAD - PAD@PAD (lower is better)')
    plt.grid(True)
    
    # 9. 总结
    plt.subplot(3, 3, 9)
    plt.text(0.1, 0.9, "Training Summary", fontsize=14, fontweight='bold')
    
    summary_text = f"""
Configuration:
  Dynamic Batching: max_pad_ratio={args.max_padding_ratio}
  PAD Regularization: λ={args.pad_reg_weight}
  
Final Results:
  TF Accuracy: {history['tf_accuracy'][-1]:.4f}
  Path Accuracy: {history['path_accuracy'][-1]:.4f}
  
Padding Statistics:
  Initial: 78.9%
  Final: {history['val_padding_ratio'][-1]:.1%}
  Reduction: {(0.789 - history['val_padding_ratio'][-1])*100:.1f}%
  
Anti-preference Control:
  Newline at PAD: {history['newline_pred_at_pad'][-1]:.2%}
  PAD at PAD: {history['pad_pred_ratio'][-1]:.2%}
  Max anti-pref: {max(history['newline_pred_at_pad']):.2%}
"""
    
    plt.text(0.1, 0.05, summary_text, fontsize=10, family='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # 打印最终总结
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Results saved to: {out_dir}")
    print(f"\nFinal Performance:")
    print(f"  TF Accuracy (w/o padding): {history['tf_accuracy'][-1]:.4f}")
    print(f"  Path Token Accuracy: {history['path_accuracy'][-1]:.4f}")
    print(f"  Padding Reduction: {(0.789 - history['val_padding_ratio'][-1])*100:.1f}%")
    print(f"\nAnti-preference Control:")
    print(f"  Newline predictions at PAD: {history['newline_pred_at_pad'][-1]:.2%}")
    print(f"  PAD predictions at PAD: {history['pad_pred_ratio'][-1]:.2%}")
    
    if history['newline_pred_at_pad'][-1] < 0.1 and history['pad_pred_ratio'][-1] > 0.8:
        print("\n✅ SUCCESS: PAD regularization effectively prevented anti-preference!")
    elif history['newline_pred_at_pad'][-1] > 0.5:
        print("\n⚠️  WARNING: Anti-preference still present. Consider increasing λ.")
    
    logger.info(f"Training complete. Final TF accuracy: {history['tf_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()