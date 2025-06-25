"""
分析不同训练阶段模型的输出分布，验证反偏好现象
修正版本：正确计算TF准确率和分析反偏好
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
    """计算Teacher Forcing准确率（与训练代码一致）"""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    data_size = block_size + 1
    total_correct = 0
    total_count = 0
    
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
            
            # 计算准确率
            preds = torch.argmax(logits, dim=-1)
            
            # 只计算非padding位置的准确率
            mask = (y != 0)
            correct = (preds == y) & mask
            
            total_correct += correct.sum().item()
            total_count += mask.sum().item()
    
    return total_correct / total_count if total_count > 0 else 0

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
    
    # 要分析的checkpoints
    checkpoints_to_analyze = {
        '50k': 50000,
        '100k': 100000,
        '190k': 190000,
        '200k': 200000
    }
    
    # 加载图和元数据
    print("Loading graph and metadata...")
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
        
        # 1. 计算TF准确率（与训练时一致）
        tf_accuracy = compute_tf_accuracy(model, val_data, block_size, device)
        print(f"Teacher Forcing Accuracy: {tf_accuracy:.4f}")
        
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
            'preferences': preferences
        }
    
    # 总结
    print("\n" + "="*60)
    print("SUMMARY: Anti-Preference Analysis")
    print("="*60)
    
    # TF准确率对比
    print("\nTeacher Forcing Accuracy:")
    for name, results in all_results.items():
        print(f"  {name}: {results['tf_accuracy']:.4f}")
    
    # 路径偏好对比
    print("\nPath Preferences (when both training and alternative paths exist):")
    for name, results in all_results.items():
        if results['preferences']['total_decisions'] > 0:
            pref_ratio = results['preferences']['prefers_alternative'] / results['preferences']['total_decisions']
            print(f"  {name}: {pref_ratio:.1%} prefer alternatives")
    
    # 检测反偏好
    stable_tf = (all_results['50k']['tf_accuracy'] + all_results['100k']['tf_accuracy']) / 2
    collapsed_tf = (all_results['190k']['tf_accuracy'] + all_results['200k']['tf_accuracy']) / 2
    
    print(f"\nPhase Transition Detection:")
    print(f"  Stable phase TF: {stable_tf:.4f}")
    print(f"  Collapsed phase TF: {collapsed_tf:.4f}")
    print(f"  TF Drop: {stable_tf - collapsed_tf:.4f}")
    
    if collapsed_tf < stable_tf * 0.5:  # TF下降超过50%
        print("\n✓ PHASE TRANSITION CONFIRMED!")
        
        # 检查是否有反偏好
        stable_pref = (all_results['50k']['preferences']['prefers_alternative'] + 
                      all_results['100k']['preferences']['prefers_alternative']) / \
                     (all_results['50k']['preferences']['total_decisions'] + 
                      all_results['100k']['preferences']['total_decisions'])
        
        collapsed_pref = (all_results['190k']['preferences']['prefers_alternative'] + 
                         all_results['200k']['preferences']['prefers_alternative']) / \
                        (all_results['190k']['preferences']['total_decisions'] + 
                         all_results['200k']['preferences']['total_decisions'])
        
        if collapsed_pref > stable_pref:
            print(f"✓ ANTI-PREFERENCE DETECTED: Alternative preference increased from {stable_pref:.1%} to {collapsed_pref:.1%}")
    
    print("="*60)

if __name__ == "__main__":
    main()