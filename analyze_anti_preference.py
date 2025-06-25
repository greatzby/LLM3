"""
分析不同训练阶段模型的输出分布，验证反偏好现象
修正版本：正确区分训练路径和图结构
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

def extract_training_paths(train_file_path):
    """从训练文件中提取训练路径信息"""
    # 存储每个(source, target, current_position) -> next_node的映射
    training_next_nodes = {}
    
    with open(train_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line != 'x':
                tokens = line.split()
                source = tokens[0]
                target = tokens[1]
                path = tokens[2:]  # 完整路径
                
                # 对于路径中的每个位置，记录训练时的下一个节点
                for i in range(len(path) - 1):
                    # key是(source, target, 当前节点位置的上下文)
                    context = tuple(tokens[:2+i+1])  # source, target, path_so_far
                    next_node = path[i + 1]
                    training_next_nodes[context] = next_node
    
    return training_next_nodes

def analyze_predictions(model, test_data, G, training_next_nodes, stoi, itos, device='cuda', num_samples=1000):
    """分析模型在测试数据上的预测分布"""
    
    results = {
        'matches_training': [],  # 预测是否匹配训练路径
        'valid_alternative': [],  # 预测是否是有效的替代路径
        'invalid': [],  # 预测是否无效
        'predictions': [],
        'tf_correct': 0,
        'tf_total': 0
    }
    
    # 读取测试数据
    with open(test_data, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    # 随机采样
    sampled_lines = np.random.choice(test_lines, min(num_samples, len(test_lines)), replace=False)
    
    for line in sampled_lines:
        tokens = line.split()
        source, target = tokens[0], tokens[1]
        path = tokens[2:]
        
        # 对路径中的每个位置进行预测（除了最后一个）
        for i in range(len(path) - 1):
            current_node = path[i]
            true_next = path[i + 1]
            
            # 构建输入序列和上下文
            input_sequence = [source, target] + path[:i+1]
            context = tuple(input_sequence)
            
            # 获取训练时这个上下文对应的下一个节点
            training_next = training_next_nodes.get(context, None)
            
            input_ids = [stoi[token] for token in input_sequence]
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # 获取模型预测
            with torch.no_grad():
                logits, _ = model(input_tensor)
                # 获取最后一个位置的logits
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
            
            # 获取预测的节点
            pred_idx = torch.argmax(probs).item()
            pred_node = itos[pred_idx] if pred_idx < len(itos) else 'UNK'
            
            # 检查TF准确率（预测是否匹配测试集的真实下一个节点）
            results['tf_total'] += 1
            if pred_node == true_next:
                results['tf_correct'] += 1
            
            # 分析预测分布
            matches_training_prob = 0.0
            valid_alternative_prob = 0.0
            invalid_prob = 0.0
            
            # 获取当前节点的所有可能的下一跳
            current_node_neighbors = list(G.successors(current_node))
            
            # 遍历所有可能的预测
            for next_node_idx in range(2, len(itos)):  # 从2开始，跳过PAD和\n
                if next_node_idx >= len(probs):
                    break
                    
                next_node = itos[next_node_idx]
                prob = probs[next_node_idx].item()
                
                # 分类
                if training_next and next_node == training_next:
                    # 匹配训练路径
                    matches_training_prob += prob
                elif next_node in current_node_neighbors:
                    # 有效的替代路径
                    valid_alternative_prob += prob
                else:
                    # 无效路径
                    invalid_prob += prob
            
            results['matches_training'].append(matches_training_prob)
            results['valid_alternative'].append(valid_alternative_prob)
            results['invalid'].append(invalid_prob)
            
            # 记录预测细节
            top_k = 5
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            predictions = []
            for idx, prob in zip(top_indices, top_probs):
                if idx >= 2 and idx < len(itos):
                    pred_node_top = itos[idx.item()]
                    
                    # 判断类型
                    if training_next and pred_node_top == training_next:
                        edge_type = 'matches_training'
                    elif pred_node_top in current_node_neighbors:
                        edge_type = 'valid_alternative'
                    else:
                        edge_type = 'invalid'
                    
                    predictions.append({
                        'node': pred_node_top,
                        'prob': prob.item(),
                        'type': edge_type,
                        'is_true_next': pred_node_top == true_next
                    })
            
            results['predictions'].append({
                'context': input_sequence,
                'current': current_node,
                'true_next': true_next,
                'training_next': training_next,
                'predicted': pred_node,
                'top_predictions': predictions
            })
    
    return results

def plot_distribution_comparison(results_dict, save_path):
    """绘制不同checkpoint的分布对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Output Distribution Analysis: Stable vs Collapsed States', fontsize=16)
    
    checkpoints = list(results_dict.keys())
    
    for idx, (checkpoint, results) in enumerate(results_dict.items()):
        ax = axes[idx // 2, idx % 2]
        
        # 计算平均概率
        avg_matches = np.mean(results['matches_training'])
        avg_alternative = np.mean(results['valid_alternative'])
        avg_invalid = np.mean(results['invalid'])
        
        # 计算TF准确率
        tf_accuracy = results['tf_correct'] / results['tf_total'] if results['tf_total'] > 0 else 0
        
        # 绘制饼图
        labels = ['Matches Training', 'Valid Alternative', 'Invalid']
        sizes = [avg_matches, avg_alternative, avg_invalid]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        
        ax.set_title(f'Checkpoint {checkpoint}\n'
                    f'TF Accuracy: {tf_accuracy:.1%}\n'
                    f'Avg probs: Training={avg_matches:.3f}, Alternative={avg_alternative:.3f}, Invalid={avg_invalid:.3f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

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
    
    # 提取训练路径信息
    print("Extracting training paths...")
    train_file = os.path.join(data_dir, 'train_20.txt')
    training_next_nodes = extract_training_paths(train_file)
    print(f"Found {len(training_next_nodes)} training contexts")
    
    # 统计图信息
    total_edges = G.number_of_edges()
    print(f"Total edges in graph: {total_edges}")
    
    # 测试文件
    test_file = os.path.join(data_dir, 'test.txt')
    
    # 分析每个checkpoint
    results_dict = {}
    
    for name, iteration in checkpoints_to_analyze.items():
        print(f"\nAnalyzing checkpoint {name} (iteration {iteration})...")
        
        # 加载模型
        checkpoint_path = os.path.join(base_dir, f'ckpt_{iteration}.pt')
        model, checkpoint = load_checkpoint_and_model(checkpoint_path, device)
        
        # 分析预测
        results = analyze_predictions(model, test_file, G, training_next_nodes, stoi, itos, device)
        results_dict[name] = results
        
        # 计算TF准确率
        tf_accuracy = results['tf_correct'] / results['tf_total'] if results['tf_total'] > 0 else 0
        
        # 打印统计
        avg_matches = np.mean(results['matches_training'])
        avg_alternative = np.mean(results['valid_alternative'])
        avg_invalid = np.mean(results['invalid'])
        
        print(f"Teacher Forcing Accuracy: {tf_accuracy:.3f}")
        print(f"Average probabilities:")
        print(f"  Matches training path: {avg_matches:.3f}")
        print(f"  Valid alternative paths: {avg_alternative:.3f}")
        print(f"  Invalid paths: {avg_invalid:.3f}")
        
        # 打印一些具体的预测示例
        print(f"\nExample predictions:")
        for i in range(min(3, len(results['predictions']))):
            pred = results['predictions'][i]
            print(f"  Context: {' '.join(pred['context'])}")
            print(f"    True next: {pred['true_next']}, Training next: {pred['training_next']}, Predicted: {pred['predicted']}")
            for j, top_pred in enumerate(pred['top_predictions'][:3]):
                mark = "✓" if top_pred['is_true_next'] else ""
                print(f"    Top-{j+1}: {top_pred['node']} (p={top_pred['prob']:.3f}, type={top_pred['type']}) {mark}")
    
    # 绘制对比图
    print("\nGenerating visualization...")
    save_path = os.path.join(base_dir, 'anti_preference_analysis.png')
    plot_distribution_comparison(results_dict, save_path)
    
    # 保存详细结果
    results_save_path = os.path.join(base_dir, 'anti_preference_results.pkl')
    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  Plots: {save_path}")
    print(f"  Data: {results_save_path}")
    
    # 生成总结报告
    print("\n" + "="*60)
    print("ANTI-PREFERENCE ANALYSIS SUMMARY")
    print("="*60)
    
    # 计算稳定期和崩溃期的平均值
    stable_tf = (results_dict['50k']['tf_correct'] / results_dict['50k']['tf_total'] + 
                 results_dict['100k']['tf_correct'] / results_dict['100k']['tf_total']) / 2
    collapsed_tf = (results_dict['190k']['tf_correct'] / results_dict['190k']['tf_total'] + 
                    results_dict['200k']['tf_correct'] / results_dict['200k']['tf_total']) / 2
    
    stable_matches = np.mean([np.mean(results_dict['50k']['matches_training']), 
                             np.mean(results_dict['100k']['matches_training'])])
    collapsed_matches = np.mean([np.mean(results_dict['190k']['matches_training']), 
                                np.mean(results_dict['200k']['matches_training'])])
    
    stable_alternative = np.mean([np.mean(results_dict['50k']['valid_alternative']), 
                                 np.mean(results_dict['100k']['valid_alternative'])])
    collapsed_alternative = np.mean([np.mean(results_dict['190k']['valid_alternative']), 
                                    np.mean(results_dict['200k']['valid_alternative'])])
    
    print(f"\nStable Phase (50k-100k):")
    print(f"  Teacher Forcing accuracy: {stable_tf:.1%}")
    print(f"  Matches training path: {stable_matches:.1%}")
    print(f"  Valid alternatives: {stable_alternative:.1%}")
    
    print(f"\nCollapsed Phase (190k-200k):")
    print(f"  Teacher Forcing accuracy: {collapsed_tf:.1%}")
    print(f"  Matches training path: {collapsed_matches:.1%}")
    print(f"  Valid alternatives: {collapsed_alternative:.1%}")
    
    print(f"\nChange:")
    print(f"  TF Accuracy: {stable_tf:.1%} → {collapsed_tf:.1%} (Δ = {collapsed_tf - stable_tf:.1%})")
    print(f"  Training path preference: {stable_matches:.1%} → {collapsed_matches:.1%} (Δ = {collapsed_matches - stable_matches:.1%})")
    print(f"  Alternative paths: {stable_alternative:.1%} → {collapsed_alternative:.1%} (Δ = {collapsed_alternative - stable_alternative:.1%})")
    
    if collapsed_tf < stable_tf and collapsed_alternative > stable_alternative:
        print("\n✓ ANTI-PREFERENCE CONFIRMED: Model shifts from training paths to alternatives!")
    
    print("="*60)

if __name__ == "__main__":
    main()