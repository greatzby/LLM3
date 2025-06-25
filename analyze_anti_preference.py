"""
分析不同训练阶段模型的输出分布，验证反偏好现象
增强版：添加更多调试信息和改进的反偏好检测
"""
import os
import torch
import numpy as np
import pickle
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
import torch.nn.functional as F
from contextlib import nullcontext

def load_checkpoint_and_model(checkpoint_path, device='cuda'):
    """加载checkpoint和模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从checkpoint获取模型配置
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def load_graph_and_metadata(data_dir):
    """加载图结构和元数据"""
    # 加载图
    graph_path = os.path.join(data_dir, "path_graph.graphml")
    G = nx.read_graphml(graph_path)
    
    # 加载元数据
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    return G, meta

def compute_tf_accuracy(model, val_data, block_size, device, num_eval_batches=10, batch_size=64):
    """计算Teacher Forcing准确率（与训练代码完全一致）"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    data_size = block_size + 1
    total_correct = 0
    total_count = 0
    batch_accuracies = []
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            # 获取批次数据（与训练代码保持一致）
            ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
            
            x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            
            x, y = x.to(device), y.to(device)
            
            # 获取模型预测
            with ctx:
                logits, _ = model(x, y)
            
            # 计算准确率（与训练代码完全一致，包括padding）
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

def analyze_path_preferences_detailed(model, train_file, test_file, G, stoi, itos, device='cuda', num_samples=1000):
    """更详细的路径偏好分析"""
    
    # 1. 从训练文件构建训练路径字典
    print("\n  Building training path dictionary...")
    training_paths = defaultdict(set)  # (source, target) -> {paths}
    training_transitions = defaultdict(set)  # (source, target, position, current) -> {next_nodes}
    
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line != 'x':
                tokens = line.split()
                source, target = tokens[0], tokens[1]
                path = tokens[2:]
                
                # 记录完整路径
                training_paths[(source, target)].add(tuple(path))
                
                # 记录每个位置的转移
                for i in range(len(path) - 1):
                    current = path[i]
                    next_node = path[i + 1]
                    # 包含位置信息，因为同一个节点在不同位置可能有不同的下一跳
                    key = (source, target, i, current)
                    training_transitions[key].add(next_node)
    
    print(f"    Found {len(training_paths)} source-target pairs in training")
    print(f"    Found {len(training_transitions)} unique transitions in training")
    
    # 2. 分析测试数据
    results = {
        'total_positions': 0,
        'positions_with_choice': 0,
        'prefers_training': 0,
        'prefers_alternative': 0,
        'no_valid_alternative': 0,
        'examples': [],
        'probability_shifts': []
    }
    
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    print(f"  Analyzing {min(num_samples, len(test_lines))} test examples...")
    
    # 随机采样
    sampled_lines = np.random.choice(test_lines, min(num_samples, len(test_lines)), replace=False)
    
    for line_idx, line in enumerate(sampled_lines):
        tokens = line.split()
        source, target = tokens[0], tokens[1]
        path = tokens[2:]
        
        # 检查这个source-target对是否在训练集中
        is_new_pair = (source, target) not in training_paths
        
        # 分析路径中的每个预测点
        for i in range(len(path) - 1):
            current = path[i]
            true_next = path[i + 1]
            results['total_positions'] += 1
            
            # 构建输入
            input_sequence = [source, target] + path[:i+1]
            input_ids = [stoi[token] for token in input_sequence]
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # 获取模型预测分布
            with torch.no_grad():
                logits, _ = model(input_tensor)
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
            
            # 获取当前节点的所有有效邻居
            valid_neighbors = set(G.successors(current))
            
            # 获取训练中这个位置见过的下一跳
            key = (source, target, i, current)
            training_next_nodes = training_transitions.get(key, set())
            
            # 计算有效的替代路径
            alternative_nodes = valid_neighbors - training_next_nodes
            
            if len(alternative_nodes) == 0:
                results['no_valid_alternative'] += 1
                continue
            
            if len(training_next_nodes) == 0:
                # 这是一个全新的上下文
                continue
            
            results['positions_with_choice'] += 1
            
            # 计算概率
            training_prob = sum(probs[stoi[node]].item() for node in training_next_nodes if node in stoi)
            alternative_prob = sum(probs[stoi[node]].item() for node in alternative_nodes if node in stoi)
            
            # 记录偏好
            if training_prob > alternative_prob:
                results['prefers_training'] += 1
            else:
                results['prefers_alternative'] += 1
            
            # 记录概率差
            prob_diff = alternative_prob - training_prob
            results['probability_shifts'].append(prob_diff)
            
            # 记录例子
            if len(results['examples']) < 5:
                top_k = 10
                top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                predictions = []
                
                for idx, prob in zip(top_indices, top_probs):
                    if idx >= 2 and idx < len(itos):
                        pred_node = itos[idx.item()]
                        if pred_node in training_next_nodes:
                            pred_type = 'training'
                        elif pred_node in valid_neighbors:
                            pred_type = 'alternative'
                        else:
                            pred_type = 'invalid'
                        predictions.append((pred_node, prob.item(), pred_type))
                
                results['examples'].append({
                    'context': ' '.join(input_sequence),
                    'position': i,
                    'current': current,
                    'true_next': true_next,
                    'training_options': sorted(list(training_next_nodes)),
                    'alternative_options': sorted(list(alternative_nodes)),
                    'training_prob': training_prob,
                    'alternative_prob': alternative_prob,
                    'top_predictions': predictions[:5],
                    'is_new_pair': is_new_pair
                })
    
    return results

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430'
    data_dir = 'data/simple_graph/100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 要分析的checkpoints
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
    
    # 加载验证数据（用于TF准确率计算）
    val_data_path = os.path.join(data_dir, 'val.bin')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    
    # 文件路径
    train_file = os.path.join(data_dir, 'train_20.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    
    # 存储结果
    all_results = {}
    
    for name, iteration in checkpoints_to_analyze.items():
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint {name} (iteration {iteration})...")
        
        # 加载模型
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
        
        # 1. 计算TF准确率
        tf_accuracy, tf_std = compute_tf_accuracy(model, val_data, block_size, device)
        print(f"Teacher Forcing Accuracy: {tf_accuracy:.4f} (±{tf_std:.4f})")
        
        # 2. 分析路径偏好
        preferences = analyze_path_preferences_detailed(model, train_file, test_file, G, stoi, itos, device)
        
        print(f"\nPath Preference Analysis:")
        print(f"  Total positions analyzed: {preferences['total_positions']}")
        print(f"  Positions with choice: {preferences['positions_with_choice']}")
        print(f"  No valid alternatives: {preferences['no_valid_alternative']}")
        
        if preferences['positions_with_choice'] > 0:
            pref_training = preferences['prefers_training'] / preferences['positions_with_choice']
            pref_alternative = preferences['prefers_alternative'] / preferences['positions_with_choice']
            avg_prob_shift = np.mean(preferences['probability_shifts']) if preferences['probability_shifts'] else 0
            
            print(f"\n  Decision Analysis (when alternatives exist):")
            print(f"    Prefers training path: {pref_training:.1%}")
            print(f"    Prefers alternative path: {pref_alternative:.1%}")
            print(f"    Average probability shift to alternatives: {avg_prob_shift:.3f}")
            
            # 打印例子
            print(f"\n  Example predictions:")
            for i, ex in enumerate(preferences['examples']):
                print(f"\n  Example {i+1}:")
                print(f"    Context: {ex['context']}")
                print(f"    Position {ex['position']}: {ex['current']} → ?")
                print(f"    Training options: {ex['training_options']}")
                print(f"    Alternative options: {ex['alternative_options']}")
                print(f"    Probability mass on training paths: {ex['training_prob']:.3f}")
                print(f"    Probability mass on alternatives: {ex['alternative_prob']:.3f}")
                print(f"    Top 5 predictions:")
                for node, prob, ptype in ex['top_predictions']:
                    marker = "→" if node == ex['true_next'] else " "
                    print(f"      {marker} {node}: {prob:.3f} ({ptype})")
        
        all_results[name] = {
            'tf_accuracy': tf_accuracy,
            'tf_std': tf_std,
            'preferences': preferences
        }
    
    # 生成对比图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. TF准确率
    names = list(all_results.keys())
    tf_accs = [all_results[name]['tf_accuracy'] for name in names]
    
    ax1.bar(names, tf_accs)
    ax1.set_ylabel('Teacher Forcing Accuracy')
    ax1.set_title('TF Accuracy Drop')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.475, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.legend()
    
    # 2. 偏好变化
    pref_alternative = []
    for name in names:
        if all_results[name]['preferences']['positions_with_choice'] > 0:
            ratio = all_results[name]['preferences']['prefers_alternative'] / \
                   all_results[name]['preferences']['positions_with_choice']
            pref_alternative.append(ratio)
        else:
            pref_alternative.append(0.5)
    
    ax2.bar(names, pref_alternative)
    ax2.set_ylabel('Preference for Alternative Paths')
    ax2.set_title('Path Preference Shift')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    
    # 3. 概率偏移分布
    for i, name in enumerate(names):
        shifts = all_results[name]['preferences']['probability_shifts']
        if shifts:
            ax3.hist(shifts, bins=30, alpha=0.5, label=name, density=True)
    
    ax3.set_xlabel('Probability Shift (Alternative - Training)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Probability Shifts')
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'anti_preference_detailed.png'), dpi=150)
    plt.close()
    
    # 总结
    print("\n" + "="*60)
    print("ANTI-PREFERENCE ANALYSIS SUMMARY")
    print("="*60)
    
    # 检测反偏好
    early_pref = np.mean([all_results[k]['preferences']['prefers_alternative'] / 
                         max(all_results[k]['preferences']['positions_with_choice'], 1)
                         for k in ['50k', '100k']])
    late_pref = np.mean([all_results[k]['preferences']['prefers_alternative'] / 
                        max(all_results[k]['preferences']['positions_with_choice'], 1)
                        for k in ['190k', '200k']])
    
    print(f"\nPreference for alternatives:")
    print(f"  Stable phase (50k-100k): {early_pref:.1%}")
    print(f"  Collapsed phase (190k-200k): {late_pref:.1%}")
    print(f"  Change: {late_pref - early_pref:+.1%}")
    
    if late_pref > early_pref + 0.1:
        print("\n✓ ANTI-PREFERENCE CONFIRMED: Model increasingly prefers alternative paths!")
    
    # TF准确率总结
    early_tf = np.mean([all_results[k]['tf_accuracy'] for k in ['50k', '100k']])
    late_tf = np.mean([all_results[k]['tf_accuracy'] for k in ['190k', '200k']])
    
    print(f"\nTeacher Forcing accuracy:")
    print(f"  Stable phase: {early_tf:.4f}")
    print(f"  Collapsed phase: {late_tf:.4f}")
    print(f"  Drop: {early_tf - late_tf:.4f}")
    
    if late_tf < 0.475:  # 低于随机基线
        print(f"\n✓ TF BELOW RANDOM: {late_tf:.4f} < 0.475 (random baseline)")

if __name__ == "__main__":
    main()