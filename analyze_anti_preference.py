"""
分析不同训练阶段模型的输出分布，验证反偏好现象
修正版本：与训练代码完全一致的TF准确率计算
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

def analyze_path_preferences(model, train_file, test_file, G, stoi, itos, device='cuda', num_samples=500):
    """分析模型对不同类型路径的偏好"""
    
    # 1. 从训练文件提取所有训练中使用的具体路径转移
    training_transitions = defaultdict(set)  # (source, target, current) -> {next_nodes}
    
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line != 'x':
                tokens = line.split()
                source, target = tokens[0], tokens[1]
                path = tokens[2:]
                
                # 记录路径中的每个转移
                for i in range(len(path) - 1):
                    current = path[i]
                    next_node = path[i + 1]
                    key = (source, target, current)
                    training_transitions[key].add(next_node)
    
    # 2. 分析测试数据上的预测
    results = {
        'prefers_training': 0,
        'prefers_alternative': 0,
        'total_decisions': 0,
        'examples': []
    }
    
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    # 随机采样
    sampled_lines = np.random.choice(test_lines, min(num_samples, len(test_lines)), replace=False)
    
    for line in sampled_lines:
        tokens = line.split()
        source, target = tokens[0], tokens[1]
        path = tokens[2:]
        
        # 分析路径中的每个预测点
        for i in range(len(path) - 1):
            current = path[i]
            true_next = path[i + 1]
            
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
            valid_neighbors = list(G.successors(current))
            
            # 获取训练中见过的下一跳
            key = (source, target, current)
            training_next_nodes = training_transitions.get(key, set())
            
            # 计算不同类别的概率总和
            training_prob = 0.0
            alternative_prob = 0.0
            
            for neighbor in valid_neighbors:
                if neighbor in stoi:
                    idx = stoi[neighbor]
                    prob = probs[idx].item()
                    
                    if neighbor in training_next_nodes:
                        training_prob += prob
                    else:
                        alternative_prob += prob
            
            # 只有当既有训练路径又有替代路径时才计算偏好
            if len(training_next_nodes) > 0 and len(valid_neighbors) > len(training_next_nodes):
                results['total_decisions'] += 1
                
                if training_prob > alternative_prob:
                    results['prefers_training'] += 1
                else:
                    results['prefers_alternative'] += 1
                
                # 记录具体例子
                if len(results['examples']) < 10:
                    top_k = 5
                    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                    predictions = []
                    
                    for idx, prob in zip(top_indices, top_probs):
                        if idx >= 2 and idx < len(itos):
                            pred_node = itos[idx.item()]
                            pred_type = 'training' if pred_node in training_next_nodes else \
                                       'alternative' if pred_node in valid_neighbors else \
                                       'invalid'
                            predictions.append((pred_node, prob.item(), pred_type))
                    
                    results['examples'].append({
                        'context': input_sequence,
                        'training_options': list(training_next_nodes),
                        'all_valid_options': valid_neighbors,
                        'training_prob': training_prob,
                        'alternative_prob': alternative_prob,
                        'top_predictions': predictions
                    })
    
    return results

def main():
    # 配置
    base_dir = 'out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430'
    data_dir = 'data/simple_graph/100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 要分析的checkpoints - 检查正确的文件名
    print("Checking available checkpoints...")
    available_checkpoints = {}
    for filename in os.listdir(base_dir):
        if filename.startswith('ckpt_') and filename.endswith('.pt'):
            iteration = int(filename[5:-3])
            available_checkpoints[iteration] = filename
    
    print(f"Available checkpoints: {sorted(available_checkpoints.keys())}")
    
    # 根据实际的checkpoint选择要分析的
    # 从你的训练日志看，checkpoint_interval是1000，不是50000
    checkpoints_to_analyze = {}
    
    # 选择稳定期的checkpoint（TF > 0.9）
    for iter_num in [50000, 100000]:
        if iter_num in available_checkpoints:
            checkpoints_to_analyze[f'{iter_num//1000}k'] = iter_num
    
    # 选择崩溃后的checkpoint（TF < 0.2）
    for iter_num in [190000, 200000]:
        if iter_num in available_checkpoints:
            checkpoints_to_analyze[f'{iter_num//1000}k'] = iter_num
    
    print(f"Will analyze: {checkpoints_to_analyze}")
    
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
        
        # 1. 计算TF准确率（与训练时完全一致）
        tf_accuracy, tf_std = compute_tf_accuracy(model, val_data, block_size, device)
        print(f"Teacher Forcing Accuracy: {tf_accuracy:.4f} (±{tf_std:.4f})")
        
        # 查看checkpoint中保存的TF历史（如果有）
        if 'tf_history' in checkpoint:
            tf_history = checkpoint['tf_history']
            if len(tf_history) > 0:
                # 找到最接近当前iteration的记录
                test_iters = checkpoint.get('test_iters', list(range(2000, len(tf_history)*2000+1, 2000)))
                closest_idx = min(range(len(test_iters)), key=lambda i: abs(test_iters[i] - iteration))
                print(f"  (Training log TF at nearest iteration: {tf_history[closest_idx]:.4f})")
        
        # 2. 分析路径偏好
        preferences = analyze_path_preferences(model, train_file, test_file, G, stoi, itos, device)
        
        if preferences['total_decisions'] > 0:
            pref_training_ratio = preferences['prefers_training'] / preferences['total_decisions']
            pref_alternative_ratio = preferences['prefers_alternative'] / preferences['total_decisions']
            
            print(f"\nPath Preference Analysis:")
            print(f"  Total decision points analyzed: {preferences['total_decisions']}")
            print(f"  Prefers training paths: {preferences['prefers_training']} ({pref_training_ratio:.1%})")
            print(f"  Prefers alternative paths: {preferences['prefers_alternative']} ({pref_alternative_ratio:.1%})")
            
            # 打印一些例子
            print(f"\nExample predictions:")
            for i, example in enumerate(preferences['examples'][:3]):
                print(f"\n  Example {i+1}:")
                print(f"    Context: {' '.join(example['context'])}")
                print(f"    Training options: {example['training_options']}")
                print(f"    All valid options: {example['all_valid_options']}")
                print(f"    Training path probability: {example['training_prob']:.3f}")
                print(f"    Alternative path probability: {example['alternative_prob']:.3f}")
                print(f"    Top predictions:")
                for node, prob, ptype in example['top_predictions'][:3]:
                    print(f"      {node}: {prob:.3f} ({ptype})")
        
        all_results[name] = {
            'tf_accuracy': tf_accuracy,
            'tf_std': tf_std,
            'preferences': preferences
        }
    
    # 绘制可视化
    if len(all_results) >= 4:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # TF准确率对比
        names = list(all_results.keys())
        tf_accs = [all_results[name]['tf_accuracy'] for name in names]
        tf_stds = [all_results[name]['tf_std'] for name in names]
        
        ax1.bar(names, tf_accs, yerr=tf_stds, capsize=10)
        ax1.set_ylabel('Teacher Forcing Accuracy')
        ax1.set_title('TF Accuracy Across Checkpoints')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 路径偏好对比
        pref_ratios = []
        for name in names:
            if all_results[name]['preferences']['total_decisions'] > 0:
                ratio = all_results[name]['preferences']['prefers_alternative'] / \
                       all_results[name]['preferences']['total_decisions']
                pref_ratios.append(ratio)
            else:
                pref_ratios.append(0)
        
        ax2.bar(names, pref_ratios)
        ax2.set_ylabel('Preference for Alternative Paths')
        ax2.set_title('Path Preference Analysis')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'anti_preference_analysis.png'), dpi=150)
        plt.close()
    
    # 总结
    print("\n" + "="*60)
    print("SUMMARY: Anti-Preference Analysis")
    print("="*60)
    
    # TF准确率对比
    print("\nTeacher Forcing Accuracy:")
    for name, results in all_results.items():
        print(f"  {name}: {results['tf_accuracy']:.4f} (±{results['tf_std']:.4f})")
    
    # 检测是否有明显的TF下降
    if len(all_results) >= 4:
        early_keys = [k for k in all_results.keys() if int(k[:-1]) <= 100]
        late_keys = [k for k in all_results.keys() if int(k[:-1]) >= 190]
        
        if early_keys and late_keys:
            early_tf = np.mean([all_results[k]['tf_accuracy'] for k in early_keys])
            late_tf = np.mean([all_results[k]['tf_accuracy'] for k in late_keys])
            
            print(f"\nPhase Analysis:")
            print(f"  Early phase average TF: {early_tf:.4f}")
            print(f"  Late phase average TF: {late_tf:.4f}")
            print(f"  TF Drop: {early_tf - late_tf:.4f}")
            
            if early_tf - late_tf > 0.3:  # 显著下降
                print("\n✓ SIGNIFICANT TF DROP DETECTED!")

if __name__ == "__main__":
    main()