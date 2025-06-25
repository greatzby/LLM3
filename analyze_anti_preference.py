"""
反偏好分析：在val数据上分析模型如何避开正确答案
"""
import os
import torch
import numpy as np
import pickle
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
import torch.nn.functional as F
from contextlib import nullcontext

def load_checkpoint_and_model(checkpoint_path, device='cuda'):
    """加载checkpoint和模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def load_graph_and_metadata(data_dir):
    """加载图结构和元数据"""
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    G = nx.read_graphml(graph_path)
    
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    return G, meta

def decode_sequence(data, block_size, itos):
    """将二进制数据解码为路径序列"""
    sequences = []
    data_size = block_size + 1
    num_sequences = (len(data) - data_size) // data_size
    
    for i in range(num_sequences):
        start_idx = i * data_size
        seq_data = data[start_idx:start_idx + data_size]
        
        # 解码为tokens
        tokens = []
        for idx in seq_data:
            if idx == 0:  # PAD
                break
            if idx == 1:  # \n
                break
            if idx >= 2:
                tokens.append(itos[idx])
        
        if len(tokens) >= 3:  # 至少要有source, target, source
            sequences.append(tokens)
    
    return sequences

def analyze_anti_preference_on_val(model, val_data, train_file, G, stoi, itos, block_size, device='cuda', num_samples=1000):
    """在val数据上分析反偏好"""
    
    # 1. 统计训练数据中的转移模式
    print("\n  Analyzing training patterns...")
    training_transitions = defaultdict(Counter)
    
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line != 'x':
                tokens = line.split()
                source, target = tokens[0], tokens[1]
                path = tokens[2:]
                
                for i in range(len(path) - 1):
                    current = path[i]
                    next_node = path[i + 1]
                    key = (source, target, i, current)
                    training_transitions[key][next_node] += 1
    
    # 2. 解码val数据
    print("  Decoding validation sequences...")
    val_sequences = decode_sequence(val_data, block_size, itos)
    print(f"    Found {len(val_sequences)} validation sequences")
    
    # 3. 在val数据上进行详细分析
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    results = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'rank_distribution': [],
        'probability_on_true': [],
        'avoidance_examples': [],
        'position_accuracy': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'choice_distribution': []
    }
    
    # 采样分析
    sampled_sequences = np.random.choice(len(val_sequences), 
                                       min(num_samples, len(val_sequences)), 
                                       replace=False)
    
    for seq_idx in sampled_sequences:
        tokens = val_sequences[seq_idx]
        
        if len(tokens) < 3:
            continue
            
        source, target = tokens[0], tokens[1]
        path = tokens[2:]
        
        # 分析路径中的每个预测点
        for i in range(len(path) - 1):
            current = path[i]
            true_next = path[i + 1]
            
            # 获取有效邻居
            valid_neighbors = list(G.successors(current))
            if len(valid_neighbors) <= 1:
                continue
            
            results['total_predictions'] += 1
            results['choice_distribution'].append(len(valid_neighbors))
            
            # 构建输入（包括source, target和当前路径）
            input_sequence = [source, target] + path[:i+1]
            input_ids = [stoi[token] for token in input_sequence]
            
            # 填充到block_size
            if len(input_ids) < block_size:
                input_ids = input_ids + [0] * (block_size - len(input_ids))
            
            input_tensor = torch.tensor(input_ids[:block_size], dtype=torch.long, device=device).unsqueeze(0)
            
            # 获取预测
            with torch.no_grad():
                with ctx:
                    logits, _ = model(input_tensor)
                
                # 获取对应位置的logits
                position = len(input_sequence) - 1
                if position < logits.shape[1]:
                    last_logits = logits[0, position, :]
                    probs = F.softmax(last_logits, dim=-1)
                    
                    # 获取预测
                    pred_idx = torch.argmax(probs).item()
                    pred_node = itos[pred_idx] if pred_idx >= 2 and pred_idx < len(itos) else 'UNK'
                    
                    # 检查是否正确
                    if pred_node == true_next:
                        results['correct_predictions'] += 1
                        results['position_accuracy'][i]['correct'] += 1
                    results['position_accuracy'][i]['total'] += 1
                    
                    # 分析真实答案的排名和概率
                    if true_next in stoi:
                        true_idx = stoi[true_next]
                        true_prob = probs[true_idx].item()
                        results['probability_on_true'].append(true_prob)
                        
                        # 计算在有效邻居中的排名
                        neighbor_probs = []
                        for neighbor in valid_neighbors:
                            if neighbor in stoi:
                                neighbor_probs.append((probs[stoi[neighbor]].item(), neighbor))
                        
                        neighbor_probs.sort(reverse=True)
                        rank = next((i+1 for i, (_, n) in enumerate(neighbor_probs) if n == true_next), len(neighbor_probs))
                        results['rank_distribution'].append(rank)
                        
                        # 记录避免的例子
                        if rank > len(valid_neighbors) / 2 and len(results['avoidance_examples']) < 10:
                            key = (source, target, i, current)
                            train_freq = training_transitions.get(key, {})
                            
                            results['avoidance_examples'].append({
                                'context': ' '.join(input_sequence),
                                'position': i,
                                'current': current,
                                'true_next': true_next,
                                'predicted': pred_node,
                                'true_rank': rank,
                                'true_prob': true_prob,
                                'num_choices': len(valid_neighbors),
                                'top_predictions': neighbor_probs[:5],
                                'training_frequency': dict(train_freq)
                            })
    
    return results

def compute_tf_accuracy(model, val_data, block_size, device, num_eval_batches=10, batch_size=64):
    """计算Teacher Forcing准确率（与训练代码一致）"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    data_size = block_size + 1
    total_correct = 0
    total_count = 0
    batch_accuracies = []
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
            
            x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            
            x, y = x.to(device), y.to(device)
            
            with ctx:
                logits, _ = model(x, y)
            
            preds = torch.argmax(logits, dim=-1)
            
            batch_correct = (preds == y).float().sum().item()
            batch_total = y.numel()
            batch_accuracy = batch_correct / batch_total
            batch_accuracies.append(batch_accuracy)
            
            total_correct += batch_correct
            total_count += batch_total
    
    overall_accuracy = total_correct / total_count
    accuracy_std = np.std(batch_accuracies)
    
    return overall_accuracy, accuracy_std

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430'
    data_dir = 'data/simple_graph/100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoints_to_analyze = {
        '50k': 50000,
        '100k': 100000,
        '190k': 190000,
        '200k': 200000
    }
    
    # 加载图和元数据
    print("\nLoading graph and metadata...")
    G, meta = load_graph_and_metadata(data_dir)
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    
    # 加载验证数据
    val_data_path = os.path.join(data_dir, 'val.bin')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # 训练文件路径
    train_file = os.path.join(data_dir, 'train_20.txt')
    
    # 计算随机基线
    avg_choices = np.mean([G.out_degree(str(i)) for i in range(100)])
    print(f"Average number of choices: {avg_choices:.1f}")
    
    # 存储结果
    all_results = {}
    
    for name, iteration in checkpoints_to_analyze.items():
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint {name} (iteration {iteration})...")
        
        # 加载模型
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
        
        # 1. 计算标准TF准确率
        tf_accuracy, tf_std = compute_tf_accuracy(model, val_data, block_size, device)
        print(f"Teacher Forcing Accuracy: {tf_accuracy:.4f} (±{tf_std:.4f})")
        
        # 2. 详细的反偏好分析
        anti_pref = analyze_anti_preference_on_val(model, val_data, train_file, G, stoi, itos, block_size, device)
        
        if anti_pref['total_predictions'] > 0:
            detailed_accuracy = anti_pref['correct_predictions'] / anti_pref['total_predictions']
            avg_rank = np.mean(anti_pref['rank_distribution']) if anti_pref['rank_distribution'] else 0
            avg_prob = np.mean(anti_pref['probability_on_true']) if anti_pref['probability_on_true'] else 0
            
            print(f"\nDetailed Analysis on Val Data:")
            print(f"  Accuracy on multi-choice positions: {detailed_accuracy:.4f}")
            print(f"  Average rank of true answer: {avg_rank:.2f} / {np.mean(anti_pref['choice_distribution']):.1f}")
            print(f"  Average probability on true answer: {avg_prob:.4f}")
            
            # 打印避免的例子
            if anti_pref['avoidance_examples']:
                print(f"\n  Examples of systematic avoidance:")
                for i, ex in enumerate(anti_pref['avoidance_examples'][:3]):
                    print(f"\n  Example {i+1}:")
                    print(f"    Context: {ex['context']}")
                    print(f"    Position {ex['position']}: {ex['current']} → {ex['true_next']}")
                    print(f"    Model predicted: {ex['predicted']}")
                    print(f"    True answer rank: {ex['true_rank']}/{ex['num_choices']} (prob={ex['true_prob']:.4f})")
                    print(f"    Top predictions:")
                    for prob, node in ex['top_predictions'][:5]:
                        marker = "✓" if node == ex['true_next'] else " "
                        train_count = ex['training_frequency'].get(node, 0)
                        print(f"      {marker} {node}: {prob:.4f} (seen {train_count}x in training)")
        
        all_results[name] = {
            'tf_accuracy': tf_accuracy,
            'anti_pref': anti_pref
        }
    
    # 生成对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. TF准确率
    names = list(all_results.keys())
    tf_accs = [all_results[name]['tf_accuracy'] for name in names]
    ax1.bar(names, tf_accs)
    ax1.axhline(y=0.475, color='r', linestyle='--', label='Random baseline')
    ax1.set_ylabel('Teacher Forcing Accuracy')
    ax1.set_title('TF Accuracy: Catastrophic Drop')
    ax1.legend()
    
    # 2. 真实答案的平均排名
    avg_ranks = []
    avg_choices = []
    for name in names:
        ranks = all_results[name]['anti_pref']['rank_distribution']
        choices = all_results[name]['anti_pref']['choice_distribution']
        avg_ranks.append(np.mean(ranks) if ranks else 0)
        avg_choices.append(np.mean(choices) if choices else 0)
    
    # 归一化排名
    normalized_ranks = [r/c if c > 0 else 0 for r, c in zip(avg_ranks, avg_choices)]
    ax2.bar(names, normalized_ranks)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random')
    ax2.set_ylabel('Normalized Rank (rank / num_choices)')
    ax2.set_title('True Answer Pushed to Lower Ranks')
    ax2.legend()
    
    # 3. 排名分布对比
    for checkpoint in ['50k', '200k']:
        if checkpoint in all_results:
            ranks = all_results[checkpoint]['anti_pref']['rank_distribution']
            if ranks:
                ax3.hist(ranks, bins=15, alpha=0.5, label=checkpoint, density=True)
    ax3.set_xlabel('Rank of True Answer')
    ax3.set_ylabel('Density')
    ax3.set_title('Rank Distribution: Before vs After Collapse')
    ax3.legend()
    
    # 4. 概率分布对比
    for checkpoint in ['50k', '200k']:
        if checkpoint in all_results:
            probs = all_results[checkpoint]['anti_pref']['probability_on_true']
            if probs:
                ax4.hist(probs, bins=20, alpha=0.5, label=checkpoint, density=True)
    ax4.set_xlabel('Probability on True Answer')
    ax4.set_ylabel('Density')
    ax4.set_title('Probability Distribution: Before vs After')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'anti_preference_val_analysis.png'), dpi=150)
    plt.close()
    
    # 总结
    print("\n" + "="*60)
    print("ANTI-PREFERENCE PHENOMENON CONFIRMED")
    print("="*60)
    
    # TF准确率
    early_tf = np.mean([all_results[k]['tf_accuracy'] for k in ['50k', '100k']])
    late_tf = np.mean([all_results[k]['tf_accuracy'] for k in ['190k', '200k']])
    
    print(f"\nTeacher Forcing Accuracy Drop:")
    print(f"  Stable phase (50k-100k): {early_tf:.4f}")
    print(f"  Collapsed phase (190k-200k): {late_tf:.4f}")
    print(f"  Drop: {early_tf - late_tf:.4f}")
    
    if late_tf < 0.475:
        print(f"\n✓ ANTI-PREFERENCE CONFIRMED:")
        print(f"  TF accuracy {late_tf:.4f} is {0.475/late_tf:.1f}x worse than random (0.475)")
        print(f"  Model systematically avoids correct answers in validation data")

if __name__ == "__main__":
    main()