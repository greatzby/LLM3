"""
反偏好分析：直接分析TF准确率下降的原因
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

def detailed_tf_analysis(model, val_data, block_size, G, stoi, itos, device, num_batches=20, batch_size=32):
    """详细分析TF准确率，理解每个位置的预测"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    data_size = block_size + 1
    
    # 统计信息
    position_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'examples': []})
    token_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    prediction_analysis = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # 获取数据（与训练时一致）
            ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
            
            x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            
            x, y = x.to(device), y.to(device)
            
            # 获取预测
            with ctx:
                logits, _ = model(x, y)
            
            preds = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            
            # 分析每个位置
            for b in range(batch_size):
                for pos in range(block_size):
                    true_token = y[b, pos].item()
                    pred_token = preds[b, pos].item()
                    
                    # 跳过padding
                    if true_token == 0:
                        continue
                    
                    is_correct = (pred_token == true_token)
                    
                    # 更新位置统计
                    position_stats[pos]['total'] += 1
                    if is_correct:
                        position_stats[pos]['correct'] += 1
                    
                    # 分析token类型
                    if true_token == 1:  # \n
                        token_type = 'newline'
                    elif true_token >= 2:  # 节点
                        token_type = 'node'
                    else:
                        token_type = 'other'
                    
                    token_type_stats[token_type]['total'] += 1
                    if is_correct:
                        token_type_stats[token_type]['correct'] += 1
                    
                    # 对于节点预测，进行更详细的分析
                    if token_type == 'node' and pos >= 3:  # 跳过source, target, source
                        # 获取上下文
                        context_tokens = []
                        for i in range(pos + 1):
                            t = x[b, i].item()
                            if t > 0:
                                context_tokens.append(itos.get(t, 'UNK'))
                        
                        if len(context_tokens) >= 3:
                            source = context_tokens[0]
                            target = context_tokens[1]
                            current = context_tokens[-1]
                            true_next = itos.get(true_token, 'UNK')
                            pred_next = itos.get(pred_token, 'UNK')
                            
                            # 检查是否是有效的邻居
                            valid_neighbors = list(G.successors(current)) if current != 'UNK' else []
                            
                            if len(valid_neighbors) > 1:  # 只分析有多个选择的情况
                                true_prob = probs[b, pos, true_token].item()
                                
                                # 计算真实答案在有效邻居中的排名
                                neighbor_probs = []
                                for neighbor in valid_neighbors:
                                    if neighbor in stoi:
                                        neighbor_idx = stoi[neighbor]
                                        neighbor_prob = probs[b, pos, neighbor_idx].item()
                                        neighbor_probs.append((neighbor_prob, neighbor))
                                
                                neighbor_probs.sort(reverse=True)
                                true_rank = next((i+1 for i, (_, n) in enumerate(neighbor_probs) 
                                                if n == true_next), len(neighbor_probs) + 1)
                                
                                prediction_analysis.append({
                                    'position': pos,
                                    'context': context_tokens,
                                    'current': current,
                                    'true_next': true_next,
                                    'pred_next': pred_next,
                                    'is_correct': is_correct,
                                    'true_prob': true_prob,
                                    'true_rank': true_rank,
                                    'num_choices': len(valid_neighbors),
                                    'top_predictions': neighbor_probs[:5]
                                })
                                
                                # 记录错误的例子
                                if not is_correct and len(position_stats[pos]['examples']) < 3:
                                    position_stats[pos]['examples'].append({
                                        'context': ' -> '.join(context_tokens),
                                        'true': true_next,
                                        'pred': pred_next,
                                        'true_prob': true_prob,
                                        'true_rank': f"{true_rank}/{len(valid_neighbors)}"
                                    })
    
    return position_stats, token_type_stats, prediction_analysis

def analyze_training_patterns(train_file):
    """分析训练数据中的路径模式"""
    path_counter = Counter()
    transition_counter = Counter()
    
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line != 'x':
                tokens = line.split()
                source, target = tokens[0], tokens[1]
                path = tokens[2:]
                
                # 统计完整路径
                path_key = (source, target, tuple(path))
                path_counter[path_key] += 1
                
                # 统计转移
                for i in range(len(path) - 1):
                    transition = (source, target, i, path[i], path[i+1])
                    transition_counter[transition] += 1
    
    return path_counter, transition_counter

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
    
    # 分析训练数据
    train_file = os.path.join(data_dir, 'train_20.txt')
    print("\nAnalyzing training patterns...")
    path_counter, transition_counter = analyze_training_patterns(train_file)
    print(f"  Found {len(path_counter)} unique paths in training")
    print(f"  Found {len(transition_counter)} unique transitions in training")
    
    # 存储所有结果
    all_results = {}
    
    for name, iteration in checkpoints_to_analyze.items():
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint {name} (iteration {iteration})...")
        
        # 加载模型
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
        
        # 详细TF分析
        position_stats, token_type_stats, prediction_analysis = detailed_tf_analysis(
            model, val_data, block_size, G, stoi, itos, device
        )
        
        # 计算整体准确率
        total_correct = sum(stats['correct'] for stats in position_stats.values())
        total_count = sum(stats['total'] for stats in position_stats.values())
        overall_accuracy = total_correct / total_count if total_count > 0 else 0
        
        print(f"\nOverall TF Accuracy: {overall_accuracy:.4f}")
        
        # Token类型准确率
        print(f"\nAccuracy by token type:")
        for token_type, stats in token_type_stats.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                print(f"  {token_type}: {acc:.4f} ({stats['correct']}/{stats['total']})")
        
        # 位置准确率（前10个位置）
        print(f"\nAccuracy by position:")
        for pos in range(min(10, block_size)):
            if position_stats[pos]['total'] > 0:
                acc = position_stats[pos]['correct'] / position_stats[pos]['total']
                print(f"  Position {pos}: {acc:.4f} ({position_stats[pos]['correct']}/{position_stats[pos]['total']})")
                
                # 打印错误例子
                if position_stats[pos]['examples'] and acc < 0.5:
                    print(f"    Error examples:")
                    for ex in position_stats[pos]['examples'][:2]:
                        print(f"      Context: {ex['context']}")
                        print(f"      True: {ex['true']}, Pred: {ex['pred']}, True prob: {ex['true_prob']:.4f}, Rank: {ex['true_rank']}")
        
        # 分析预测
        if prediction_analysis:
            # 平均排名
            avg_rank = np.mean([p['true_rank'] for p in prediction_analysis])
            avg_choices = np.mean([p['num_choices'] for p in prediction_analysis])
            correct_rate = np.mean([p['is_correct'] for p in prediction_analysis])
            
            print(f"\nPrediction Analysis (multi-choice positions):")
            print(f"  Accuracy: {correct_rate:.4f}")
            print(f"  Average true answer rank: {avg_rank:.2f} / {avg_choices:.1f}")
            
            # 找最差的预测
            worst_predictions = sorted(prediction_analysis, 
                                     key=lambda x: x['true_prob'])[:5]
            
            print(f"\n  Worst predictions (lowest probability on true answer):")
            for i, pred in enumerate(worst_predictions):
                print(f"\n  Example {i+1}:")
                print(f"    Context: {' -> '.join(pred['context'])}")
                print(f"    True: {pred['true_next']}, Pred: {pred['pred_next']}")
                print(f"    True probability: {pred['true_prob']:.6f}")
                print(f"    True rank: {pred['true_rank']}/{pred['num_choices']}")
                print(f"    Top predictions:")
                for prob, node in pred['top_predictions'][:3]:
                    marker = "✓" if node == pred['true_next'] else " "
                    print(f"      {marker} {node}: {prob:.4f}")
        
        all_results[name] = {
            'overall_accuracy': overall_accuracy,
            'position_stats': position_stats,
            'token_type_stats': token_type_stats,
            'prediction_analysis': prediction_analysis
        }
    
    # 对比分析
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    # 比较稳定期和崩溃期
    if len(prediction_analysis) > 0:
        stable_ranks = []
        collapsed_ranks = []
        
        for name in ['50k', '100k']:
            if name in all_results:
                ranks = [p['true_rank'] for p in all_results[name]['prediction_analysis']]
                stable_ranks.extend(ranks)
        
        for name in ['190k', '200k']:
            if name in all_results:
                ranks = [p['true_rank'] for p in all_results[name]['prediction_analysis']]
                collapsed_ranks.extend(ranks)
        
        if stable_ranks and collapsed_ranks:
            print(f"\nTrue answer ranking:")
            print(f"  Stable phase: avg rank = {np.mean(stable_ranks):.2f}")
            print(f"  Collapsed phase: avg rank = {np.mean(collapsed_ranks):.2f}")
            
            # 可视化
            plt.figure(figsize=(10, 6))
            plt.hist(stable_ranks, bins=20, alpha=0.5, label='Stable (50k-100k)', density=True)
            plt.hist(collapsed_ranks, bins=20, alpha=0.5, label='Collapsed (190k-200k)', density=True)
            plt.xlabel('Rank of True Answer')
            plt.ylabel('Density')
            plt.title('Distribution of True Answer Rankings')
            plt.legend()
            plt.savefig(os.path.join(base_dir, 'rank_distribution.png'), dpi=150)
            plt.close()
    
    # TF准确率总结
    early_tf = np.mean([all_results[k]['overall_accuracy'] for k in ['50k', '100k']])
    late_tf = np.mean([all_results[k]['overall_accuracy'] for k in ['190k', '200k']])
    
    print(f"\nTeacher Forcing Accuracy Summary:")
    print(f"  Stable phase (50k-100k): {early_tf:.4f}")
    print(f"  Collapsed phase (190k-200k): {late_tf:.4f}")
    print(f"  Drop: {early_tf - late_tf:.4f}")
    
    if late_tf < 0.475:
        print(f"\n✓ ANTI-PREFERENCE CONFIRMED:")
        print(f"  Model performs {0.475/late_tf:.1f}x worse than random")
        print(f"  This indicates systematic avoidance of correct paths")

if __name__ == "__main__":
    main()