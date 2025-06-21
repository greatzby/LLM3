import torch
import numpy as np
import pickle
from model import GPTConfig, GPT
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_position_wise_accuracy(model, val_data, block_size, num_samples=1000):
    """分析每个位置的准确率，找出问题在哪"""
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    # 采样分析
    data_size = block_size + 1
    for _ in range(num_samples):
        idx = np.random.randint(0, (len(val_data) - data_size) // data_size) * data_size
        
        x = torch.from_numpy(val_data[idx:idx+block_size].astype(np.int64)).unsqueeze(0).cuda()
        y = torch.from_numpy(val_data[idx+1:idx+1+block_size].astype(np.int64)).unsqueeze(0).cuda()
        
        # 获取完整logits
        dummy = torch.zeros_like(x)
        with torch.no_grad():
            logits, _ = model(x, targets=dummy)
            preds = logits.argmax(dim=-1)
        
        # 逐位置统计
        for pos in range(y.shape[1]):
            if y[0, pos] != 0:  # 忽略padding
                correct = (preds[0, pos] == y[0, pos]).item()
                position_correct[pos] += correct
                position_total[pos] += 1
    
    # 计算每个位置的准确率
    position_accuracy = {}
    for pos in sorted(position_total.keys()):
        if position_total[pos] > 0:
            position_accuracy[pos] = position_correct[pos] / position_total[pos]
    
    return position_accuracy

def analyze_prediction_patterns(model, val_data, meta, num_samples=100):
    """分析模型的预测模式，看看它在预测什么"""
    stoi = meta['stoi']
    itos = meta['itos']
    block_size = meta['block_size']
    
    patterns = {
        'skip_source': 0,  # 跳过source节点
        'correct_format': 0,  # 完整格式
        'wrong_position': defaultdict(int),  # 错位统计
        'examples': []
    }
    
    data_size = block_size + 1
    
    for i in range(num_samples):
        idx = np.random.randint(0, (len(val_data) - data_size) // data_size) * data_size
        
        # 获取完整序列
        full_seq = val_data[idx:idx+data_size]
        
        # 解析格式 (假设格式: source target source node1 node2 ... target)
        tokens = [itos[t] for t in full_seq if t < len(itos)]
        
        if len(tokens) < 4:
            continue
            
        source = tokens[0]
        target = tokens[1]
        
        # 预测
        x = torch.from_numpy(full_seq[:-1].astype(np.int64)).unsqueeze(0).cuda()
        y = torch.from_numpy(full_seq[1:].astype(np.int64)).unsqueeze(0).cuda()
        
        dummy = torch.zeros_like(x)
        with torch.no_grad():
            logits, _ = model(x, targets=dummy)
            preds = logits.argmax(dim=-1)
        
        pred_tokens = [itos[p.item()] for p in preds[0] if p.item() < len(itos)]
        
        # 分析预测模式
        if len(pred_tokens) >= 2 and len(tokens) >= 4:
            # 检查是否跳过了source
            if pred_tokens[2] != tokens[2] and pred_tokens[2] == tokens[3]:
                patterns['skip_source'] += 1
                patterns['wrong_position'][3] += 1
            elif pred_tokens[2] == tokens[2]:
                patterns['correct_format'] += 1
        
        # 保存前几个例子
        if i < 5:
            patterns['examples'].append({
                'input': tokens[:3],
                'target': tokens[3:],
                'predicted': pred_tokens[2:],
                'full_target': tokens,
                'full_pred': pred_tokens
            })
    
    patterns['skip_rate'] = patterns['skip_source'] / num_samples if num_samples > 0 else 0
    patterns['correct_rate'] = patterns['correct_format'] / num_samples if num_samples > 0 else 0
    
    return patterns

def main():
    data_dir = 'data/simple_graph/100'
    
    # 加载数据
    with open(f'{data_dir}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    val_data = np.memmap(f'{data_dir}/val.bin', dtype=np.uint16, mode='r')
    
    # 分析关键checkpoints
    checkpoints = [120000, 140000, 160000]
    results = {}
    
    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint {ckpt}")
        print('='*60)
        
        # 加载模型
        ckpt_path = f'out/simple_graph_1_1_120_100_original_seed42/{ckpt}_ckpt_20.pt'
        checkpoint = torch.load(ckpt_path)
        model = GPT(GPTConfig(**checkpoint['model_args']))
        model.load_state_dict(checkpoint['model'])
        model.eval().cuda()
        
        # 1. 位置准确率分析
        print("Analyzing position-wise accuracy...")
        pos_acc = analyze_position_wise_accuracy(model, val_data, meta['block_size'])
        
        # 2. 预测模式分析
        print("Analyzing prediction patterns...")
        patterns = analyze_prediction_patterns(model, val_data, meta)
        
        results[ckpt] = {
            'position_accuracy': pos_acc,
            'patterns': patterns
        }
        
        # 打印结果
        print(f"\nPosition-wise accuracy:")
        for pos in sorted(list(pos_acc.keys()))[:10]:
            print(f"  Position {pos}: {pos_acc[pos]:.3f}")
        
        print(f"\nPrediction patterns:")
        print(f"  Skip source rate: {patterns['skip_rate']:.1%}")
        print(f"  Correct format rate: {patterns['correct_rate']:.1%}")
        
        print(f"\nExample predictions:")
        for i, ex in enumerate(patterns['examples'][:2]):
            print(f"  Example {i+1}:")
            print(f"    Input: {ex['input']}")
            print(f"    Target sequence: {ex['target'][:5]}...")
            print(f"    Predicted: {ex['predicted'][:5]}...")
        
        del model
        torch.cuda.empty_cache()
    
    # 可视化位置准确率变化
    plt.figure(figsize=(10, 6))
    positions = list(range(10))
    
    for ckpt in checkpoints:
        pos_acc = results[ckpt]['position_accuracy']
        accuracies = [pos_acc.get(p, 0) for p in positions]
        plt.plot(positions, accuracies, 'o-', label=f'Checkpoint {ckpt}', linewidth=2, markersize=8)
    
    plt.xlabel('Token Position')
    plt.ylabel('Accuracy')
    plt.title('Position-wise Accuracy Across Checkpoints')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('position_accuracy_analysis.png', dpi=150)
    plt.close()
    
    # 保存详细结果
    import json
    with open('deep_phase_analysis_results.json', 'w') as f:
        # 转换defaultdict为普通dict以便序列化
        serializable_results = {}
        for ckpt, data in results.items():
            serializable_results[ckpt] = {
                'position_accuracy': dict(data['position_accuracy']),
                'patterns': {
                    'skip_rate': data['patterns']['skip_rate'],
                    'correct_rate': data['patterns']['correct_rate'],
                    'examples': data['patterns']['examples'][:3]  # 只保存前3个例子
                }
            }
        json.dump(serializable_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Analysis complete! Check position_accuracy_analysis.png")

if __name__ == "__main__":
    main()