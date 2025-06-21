"""
多层次性能分析 - 证明第二个箭头到第三个箭头
同时测量token-level和path-level准确率，以及position-wise分析
"""

import os
import torch
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import networkx as nx
from model import GPTConfig, GPT
import re

def convert_to_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def load_meta(data_path):
    """加载meta信息"""
    with open(os.path.join(data_path, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)

def load_model(checkpoint_path, device='cuda:0'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def decode_tokens(token_ids, itos):
    """解码token序列为字符串"""
    decoded = []
    for tid in token_ids:
        if tid == 1:  # 换行符
            decoded.append('\n')
            break
        elif tid in itos:
            decoded.append(itos[tid])
    return ' '.join(decoded)

def find_third_number_position(number_string):
    """与训练代码一致的辅助函数"""
    numbers = number_string.split()
    third_number_index = 2
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index - 1
    return position

def calculate_token_level_accuracy(predictions, targets, start_pos=3):
    """计算token级别准确率（与训练时一致）"""
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        pred_tokens = pred.strip().split()
        target_tokens = target.strip().split()
        
        # 从指定位置开始比较（跳过prompt）
        for i in range(start_pos, min(len(pred_tokens), len(target_tokens))):
            if pred_tokens[i] == '\n' or target_tokens[i] == '\n':
                break
                
            if pred_tokens[i] == target_tokens[i]:
                correct += 1
            total += 1
    
    return float(correct / total) if total > 0 else 0.0

def calculate_path_level_accuracy(predictions, targets):
    """计算路径级别准确率"""
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        pred_path = extract_path(pred)
        target_path = extract_path(target)
        
        if pred_path == target_path:
            correct += 1
    
    return float(correct / total) if total > 0 else 0.0

def extract_path(sequence_str):
    """从序列字符串中提取路径"""
    tokens = sequence_str.strip().split()
    path = []
    
    # 跳过前2个token（source target），从第3个开始是路径
    for i in range(2, len(tokens)):
        if tokens[i] == '\n':
            break
        if tokens[i].isdigit():
            path.append(int(tokens[i]))
    
    return tuple(path)

def calculate_position_wise_accuracy(predictions, targets, max_positions=20):
    """计算每个位置的准确率"""
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    for pred, target in zip(predictions, targets):
        pred_tokens = pred.strip().split()
        target_tokens = target.strip().split()
        
        for pos in range(min(len(pred_tokens), len(target_tokens), max_positions)):
            if pred_tokens[pos] == '\n' or target_tokens[pos] == '\n':
                break
                
            if pred_tokens[pos] == target_tokens[pos]:
                position_correct[pos] += 1
            position_total[pos] += 1
    
    # 计算每个位置的准确率
    position_accuracy = {}
    for pos in range(max_positions):
        if position_total[pos] > 0:
            position_accuracy[pos] = float(position_correct[pos] / position_total[pos])
        else:
            position_accuracy[pos] = 0.0
    
    return position_accuracy

def check_path(G, gen_str):
    """与训练代码一致的路径检查函数"""
    path = re.findall(r'\d+', gen_str)
    if len(path) < 4:
        return 'wrong syntax'
    for node in path:
        if int(node) >= 100 or int(node) < 0:  # 假设100个节点
            return 'wrong syntax'
    if path[2] != path[0] or path[-1] != path[1]:
        return 'incorrect start/end'
    for i in range(2, len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return f'non-existence path {(path[i], path[i + 1])}'
    return ''

def calculate_ar_accuracy(predictions, graph):
    """计算autoregressive准确率（路径是否有效）- 与训练代码一致"""
    valid_count = 0
    total_count = len(predictions)
    
    error_counts = {
        "wrong syntax": 0,
        "incorrect start/end": 0,
        "non-existence path": 0
    }
    
    for pred in predictions:
        symbol = check_path(graph, pred)
        if symbol == "":
            valid_count += 1
        else:
            if symbol == "wrong syntax":
                error_counts["wrong syntax"] += 1
            elif symbol == "incorrect start/end":
                error_counts["incorrect start/end"] += 1
            elif symbol.startswith("non-existence path"):
                error_counts["non-existence path"] += 1
    
    return float(valid_count / total_count) if total_count > 0 else 0.0, error_counts

def load_test_data(data_path, meta, num_samples=500):
    """加载测试数据 - 与训练代码一致"""
    test_data = []
    stoi = meta['stoi']
    simple_format = meta.get('simple_format', True)
    
    test_file = os.path.join(data_path, 'test.txt')
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        with open(test_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
    
    for line in lines[:num_samples]:
        line = line.strip()
        if not line:
            continue
        
        # 根据simple_format处理
        if simple_format:
            pos = find_third_number_position(line)
            prompt_str = line[:pos]
        else:
            prompt_str = line.split(':')[0] + ':'
        
        # 编码prompt
        prompt_tokens = prompt_str.split()
        prompt = []
        for token in prompt_tokens:
            if token in stoi:
                prompt.append(stoi[token])
        
        if len(prompt) >= 3:
            test_data.append((prompt, line))
    
    return test_data

def test_teacher_forcing(model, data_path, meta, device, num_eval_batches=10):
    """使用teacher forcing评估 - 与训练代码一致"""
    val_data = np.memmap(os.path.join(data_path, 'val.bin'), dtype=np.uint16, mode='r')
    block_size = meta['block_size']
    batch_size = 64
    
    def get_batch():
        data_size = block_size + 1
        ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
        x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)
    
    total_correct = 0
    total_count = 0
    batch_accuracies = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_eval_batches):
            X, Y = get_batch()
            logits, _ = model(X, Y)
            preds = torch.argmax(logits, dim=-1)
            
            batch_correct = (preds == Y).float().sum().item()
            batch_total = Y.numel()
            batch_accuracy = batch_correct / batch_total
            batch_accuracies.append(batch_accuracy)
            
            total_correct += batch_correct
            total_count += batch_total
    
    overall_accuracy = total_correct / total_count
    accuracy_std = np.std(batch_accuracies)
    
    return overall_accuracy, accuracy_std

def analyze_performance(model, test_data, device, meta, graph, max_new_tokens=None):
    """全面分析模型性能"""
    model.eval()
    itos = meta['itos']
    top_k = len(itos)
    
    # 获取block_size
    block_size = model.config.block_size
    if max_new_tokens is None:
        max_new_tokens = block_size  # 与训练代码一致
    
    predictions = []
    targets = []
    
    # 生成预测
    for prompt, target in tqdm(test_data, desc="Generating predictions"):
        with torch.no_grad():
            prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)
            generated = model.generate(prompt_tensor, max_new_tokens=max_new_tokens, 
                                     temperature=1.0, top_k=top_k)
        
        # 解码预测
        pred_str = decode_tokens(generated[0].cpu().numpy(), itos).split('\n')[0]  # 取第一行
        predictions.append(pred_str)
        targets.append(target)
    
    # 计算各种指标
    ar_acc, error_counts = calculate_ar_accuracy(predictions, graph)
    
    results = {
        'token_level_accuracy': calculate_token_level_accuracy(predictions, targets),
        'path_level_accuracy': calculate_path_level_accuracy(predictions, targets),
        'position_wise_accuracy': calculate_position_wise_accuracy(predictions, targets),
        'ar_accuracy': ar_acc,
        'error_counts': error_counts
    }
    
    return results, predictions

def save_prediction_samples(predictions, targets, filepath, num_samples=50):
    """保存预测样例"""
    with open(filepath, 'w') as f:
        for i, (pred, target) in enumerate(zip(predictions[:num_samples], targets[:num_samples])):
            f.write(f"=== Example {i+1} ===\n")
            f.write(f"Target:     {target}\n")
            f.write(f"Prediction: {pred}\n")
            f.write("\n")

def main():
    # 配置
    checkpoints = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]
    base_dir = 'out/simple_graph_1_1_120_100_original_seed42'
    data_path = 'data/simple_graph/100'
    graph_path = 'data/simple_graph/100/path_graph.graphml'
    output_dir = 'analysis_results/multilevel_performance'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载meta信息和图
    meta = load_meta(data_path)
    graph = nx.read_graphml(graph_path)
    
    # 加载测试数据
    test_data = load_test_data(data_path, meta, num_samples=500)
    print(f"Loaded {len(test_data)} test samples")
    
    all_results = {}
    
    for ckpt in tqdm(checkpoints, desc="Analyzing checkpoints"):
        ckpt_path = os.path.join(base_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found, skipping...")
            continue
        
        try:
            model = load_model(ckpt_path)
            
            # Teacher forcing评估
            tf_acc, tf_std = test_teacher_forcing(model, data_path, meta, 'cuda:0')
            
            # Autoregressive评估
            results, predictions = analyze_performance(model, test_data, 'cuda:0', meta, graph)
            
            # 合并结果
            results['tf_accuracy'] = tf_acc
            results['tf_accuracy_std'] = tf_std
            
            all_results[ckpt] = results
            
            # 打印当前结果
            print(f"\nCheckpoint {ckpt}:")
            print(f"  Teacher Forcing accuracy: {tf_acc:.4f} (±{tf_std:.4f})")
            print(f"  Token-level accuracy: {results['token_level_accuracy']:.4f}")
            print(f"  Path-level accuracy: {results['path_level_accuracy']:.4f}")
            print(f"  AR accuracy: {results['ar_accuracy']:.4f}")
            print(f"  Error counts: {results['error_counts']}")
            
            # 保存一些预测样例
            if ckpt in [100000, 140000, 200000]:  # 关键checkpoint
                save_prediction_samples(predictions, [t for _, t in test_data], 
                                      os.path.join(output_dir, f'predictions_{ckpt}.txt'))
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果（转换为可序列化格式）
    serializable_results = convert_to_serializable(all_results)
    with open(os.path.join(output_dir, 'performance_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 创建可视化
    create_performance_plots(all_results, output_dir)
    
    print(f"\nMultilevel performance analysis complete! Results saved to {output_dir}/")

def create_performance_plots(results, output_dir):
    """创建性能对比图"""
    if not results:
        print("No results to plot")
        return
        
    checkpoints = sorted(results.keys())
    
    # 1. Token vs Path level accuracy对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    tf_acc = [results[ckpt]['tf_accuracy'] for ckpt in checkpoints]
    token_acc = [results[ckpt]['token_level_accuracy'] for ckpt in checkpoints]
    path_acc = [results[ckpt]['path_level_accuracy'] for ckpt in checkpoints]
    ar_acc = [results[ckpt]['ar_accuracy'] for ckpt in checkpoints]
    
    ax.plot(checkpoints, tf_acc, marker='o', label='Teacher Forcing (TF)', linewidth=2, markersize=8)
    ax.plot(checkpoints, token_acc, marker='s', label='Token-level (AR)', linewidth=2, markersize=8)
    ax.plot(checkpoints, path_acc, marker='^', label='Path-level (AR)', linewidth=2, markersize=8)
    ax.plot(checkpoints, ar_acc, marker='d', label='AR Valid Paths', linewidth=2, markersize=8)
    
    # 添加25%随机基线
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Random Baseline (25%)')
    
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Multiple Accuracy Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 标记相变点
    if 140000 in checkpoints:
        ax.axvline(x=140000, color='red', linestyle='--', alpha=0.5, label='Phase Transition')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Position-wise accuracy heatmap
    positions = list(range(20))
    position_data = []
    
    for ckpt in checkpoints:
        pos_acc = results[ckpt]['position_wise_accuracy']
        position_data.append([pos_acc.get(p, 0) for p in positions])
    
    # 创建热图
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(position_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(positions)
    ax.set_yticks(range(len(checkpoints)))
    ax.set_yticklabels([f'{ckpt//1000}k' for ckpt in checkpoints])
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Training Iteration')
    ax.set_title('Position-wise Accuracy Heatmap')
    
    # 添加垂直线标记关键位置
    ax.axvline(x=2.5, color='white', linestyle='--', alpha=0.7)  # prompt结束
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_accuracy_heatmap.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()