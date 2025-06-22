"""
analyze_gradient_competition_complete.py
完整的梯度竞争分析脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
from collections import defaultdict
import networkx as nx

# 假设这些是你的项目结构
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT

class GradientCompetitionAnalyzer:
    def __init__(self, checkpoint_dir, data_dir='data/simple_graph/100'):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载必要的元数据
        self.load_metadata()
        
    def load_metadata(self):
        """加载图结构和词汇表"""
        # 加载meta信息
        with open(os.path.join(self.data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = len(self.itos)
        self.block_size = meta['block_size']
        
        # 加载图
        self.graph = nx.read_graphml(os.path.join(self.data_dir, "path_graph.graphml"))
        
        # 加载验证数据
        self.val_data = np.memmap(os.path.join(self.data_dir, 'val.bin'), 
                                  dtype=np.uint16, mode='r')
        
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def prepare_multi_path_sequences(self, num_sequences=50):
        """准备具有多个有效下一跳的测试序列"""
        sequences = []
        
        print("Preparing multi-path test sequences...")
        attempts = 0
        while len(sequences) < num_sequences and attempts < num_sequences * 10:
            attempts += 1
            
            # 从验证集随机采样
            idx = np.random.randint(0, len(self.val_data) - self.block_size - 1)
            seq = self.val_data[idx:idx + self.block_size + 1].astype(np.int64)
            
            # 找到序列长度（第一个PAD之前）
            seq_len = np.where(seq == 0)[0]
            seq_len = seq_len[0] if len(seq_len) > 0 else len(seq)
            
            # 我们需要至少5个token的序列（确保有路径选择）
            if seq_len < 5:
                continue
            
            # 检查位置3（通常是路径选择点）
            # 序列格式: source target source next1 next2 ... target
            if seq_len > 4:
                current_pos = 3  # 第4个位置
                current_token = seq[current_pos]
                
                # 将token转换为node（token = node + 2）
                current_node = current_token - 2
                
                if 0 <= current_node < 100:  # 确保是有效节点
                    # 获取所有可能的下一跳
                    node_str = str(current_node)
                    if node_str in self.graph:
                        neighbors = list(self.graph.successors(node_str))
                        
                        # 需要至少2个选择才有竞争
                        if len(neighbors) >= 2:
                            # 转换邻居节点为token
                            valid_next_tokens = [int(n) + 2 for n in neighbors]
                            
                            # 过滤掉超出词汇表的token
                            valid_next_tokens = [t for t in valid_next_tokens if t < self.vocab_size]
                            
                            if len(valid_next_tokens) >= 2:
                                sequences.append({
                                    'sequence': seq[:current_pos+1],  # 包含到当前位置
                                    'position': current_pos,
                                    'current_node': current_node,
                                    'valid_next_tokens': valid_next_tokens,
                                    'actual_next': seq[current_pos+1] if current_pos+1 < seq_len else None
                                })
        
        print(f"Prepared {len(sequences)} multi-path sequences")
        return sequences
    
    def analyze_gradient_competition(self, iterations, num_sequences=50):
        """分析不同路径选择的梯度竞争"""
        
        # 准备测试序列
        test_sequences = self.prepare_multi_path_sequences(num_sequences)
        
        if not test_sequences:
            print("Error: Could not prepare test sequences!")
            return None
        
        results = defaultdict(list)
        
        for iter_num in tqdm(iterations, desc="Analyzing gradient competition"):
            ckpt_path = os.path.join(self.checkpoint_dir, f'ckpt_{iter_num}.pt')
            if not os.path.exists(ckpt_path):
                continue
            
            # 加载checkpoint
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # 创建模型
            model_args = checkpoint['model_args']
            model = GPT(GPTConfig(**model_args))
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            model.train()  # 需要梯度
            
            # 分析每个序列的梯度
            iter_gradients = defaultdict(list)
            
            for test_case in test_sequences:
                seq = test_case['sequence']
                valid_tokens = test_case['valid_next_tokens']
                
                # 转换为tensor
                input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # 对每个有效的下一个token计算梯度
                for next_token in valid_tokens:
                    model.zero_grad()
                    
                    # Forward pass
                    logits, _ = model(input_tensor)
                    
                    # 只计算最后一个位置的loss
                    target = torch.tensor([next_token], dtype=torch.long).to(self.device)
                    loss = torch.nn.functional.cross_entropy(
                        logits[0, -1:, :], target
                    )
                    
                    # Backward pass
                    loss.backward()
                    
                    # 收集梯度大小
                    total_grad_norm = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += grad_norm ** 2
                    total_grad_norm = np.sqrt(total_grad_norm)
                    
                    # 记录这个token的梯度
                    iter_gradients[next_token].append(total_grad_norm)
            
            # 汇总这个iteration的结果
            if iter_gradients:
                # 计算每个token的平均梯度
                avg_gradients = {k: np.mean(v) for k, v in iter_gradients.items()}
                
                # 找出最强的几个
                sorted_tokens = sorted(avg_gradients.items(), key=lambda x: x[1], reverse=True)
                
                if len(sorted_tokens) >= 2:
                    strongest = sorted_tokens[0]
                    second = sorted_tokens[1]
                    
                    results['iteration'].append(iter_num)
                    results['strongest_token'].append(strongest[0])
                    results['strongest_gradient'].append(strongest[1])
                    results['second_token'].append(second[0])
                    results['second_gradient'].append(second[1])
                    results['gradient_ratio'].append(strongest[1] / (second[1] + 1e-8))
                    results['gradient_difference'].append(strongest[1] - second[1])
                    
                    # 记录前5个token的梯度
                    for i, (token, grad) in enumerate(sorted_tokens[:5]):
                        results[f'top{i+1}_token'].append(token)
                        results[f'top{i+1}_gradient'].append(grad)
        
        return dict(results)
    
    def visualize_results(self, results, output_dir='gradient_competition_results'):
        """可视化梯度竞争结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 1. 梯度强度对比
        ax = axes[0, 0]
        ax.plot(results['iteration'], results['strongest_gradient'], 'b-', 
                label='Strongest path', linewidth=2, marker='o')
        ax.plot(results['iteration'], results['second_gradient'], 'r-', 
                label='Second path', linewidth=2, marker='s')
        if 'top3_gradient' in results:
            ax.plot(results['iteration'], results['top3_gradient'], 'g--', 
                    label='Third path', linewidth=1, marker='^', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Gradient Competition Between Top Paths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 标记相变区域
        ax.axvspan(145000, 150000, alpha=0.2, color='red', label='Phase transition')
        
        # 2. 梯度比率（对数尺度）
        ax = axes[0, 1]
        ax.semilogy(results['iteration'], results['gradient_ratio'], 'g-', linewidth=2, marker='D')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Ratio (log scale)')
        ax.set_title('Dominance Factor: Strongest/Second')
        ax.axhline(1.0, color='black', linestyle='--', label='Equal competition')
        ax.axvspan(145000, 150000, alpha=0.2, color='red')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 梯度差异
        ax = axes[1, 0]
        ax.plot(results['iteration'], results['gradient_difference'], 'purple', linewidth=2, marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Difference')
        ax.set_title('Absolute Gradient Advantage')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvspan(145000, 150000, alpha=0.2, color='red')
        ax.grid(True, alpha=0.3)
        
        # 4. 最强路径的身份变化
        ax = axes[1, 1]
        # 检测路径切换
        path_changes = []
        for i in range(1, len(results['strongest_token'])):
            if results['strongest_token'][i] != results['strongest_token'][i-1]:
                path_changes.append((results['iteration'][i], 
                                   results['strongest_token'][i-1], 
                                   results['strongest_token'][i]))
        
        # 绘制最强token的身份
        unique_tokens = list(set(results['strongest_token']))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tokens)))
        token_to_color = {token: color for token, color in zip(unique_tokens, colors)}
        
        for i in range(len(results['iteration'])-1):
            ax.plot([results['iteration'][i], results['iteration'][i+1]], 
                   [0, 0], 
                   color=token_to_color[results['strongest_token'][i]], 
                   linewidth=10, solid_capstyle='butt')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Dominant Path Identity')
        ax.set_title(f'Path Preference Changes (Total switches: {len(path_changes)})')
        ax.set_ylim([-0.5, 0.5])
        ax.axvspan(145000, 150000, alpha=0.2, color='red')
        
        # 5. 所有top-5梯度的演化
        ax = axes[2, 0]
        for i in range(1, 6):
            if f'top{i}_gradient' in results:
                ax.plot(results['iteration'], results[f'top{i}_gradient'], 
                       label=f'Top-{i}', linewidth=2-i*0.2, alpha=1-i*0.15)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Top-5 Path Gradients Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvspan(145000, 150000, alpha=0.2, color='red')
        
        # 6. 竞争强度指标
        ax = axes[2, 1]
        # 计算竞争强度：所有top-5梯度的标准差
        competition_intensity = []
        for i in range(len(results['iteration'])):
            top_grads = []
            for j in range(1, 6):
                if f'top{j}_gradient' in results and i < len(results[f'top{j}_gradient']):
                    top_grads.append(results[f'top{j}_gradient'][i])
            if len(top_grads) > 1:
                competition_intensity.append(np.std(top_grads) / (np.mean(top_grads) + 1e-8))
            else:
                competition_intensity.append(0)
        
        ax.plot(results['iteration'], competition_intensity, 'orange', linewidth=2, marker='s')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Competition Intensity (CV of top-5)')
        ax.set_title('Gradient Competition Intensity')
        ax.axvspan(145000, 150000, alpha=0.2, color='red')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_competition_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存数值结果
        with open(os.path.join(output_dir, 'gradient_competition_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # 生成文本报告
        self.generate_text_report(results, path_changes, output_dir)
        
        return path_changes
    
    def generate_text_report(self, results, path_changes, output_dir):
        """生成文本分析报告"""
        report_path = os.path.join(output_dir, 'gradient_competition_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Gradient Competition Analysis Report\n")
            f.write("="*60 + "\n\n")
            
            # 找到关键迭代
            iterations = results['iteration']
            idx_140k = min(range(len(iterations)), key=lambda i: abs(iterations[i]-140000))
            idx_145k = min(range(len(iterations)), key=lambda i: abs(iterations[i]-145000))
            idx_150k = min(range(len(iterations)), key=lambda i: abs(iterations[i]-150000))
            
            f.write("Key Findings:\n")
            f.write("-"*40 + "\n")
            
            # 相变前后对比
            f.write(f"\n1. Gradient Competition Before Transition (140k):\n")
            f.write(f"   Strongest gradient: {results['strongest_gradient'][idx_140k]:.4f}\n")
            f.write(f"   Second gradient: {results['second_gradient'][idx_140k]:.4f}\n")
            f.write(f"   Ratio: {results['gradient_ratio'][idx_140k]:.2f}x\n")
            
            f.write(f"\n2. At Critical Point (145k):\n")
            f.write(f"   Strongest gradient: {results['strongest_gradient'][idx_145k]:.4f}\n")
            f.write(f"   Second gradient: {results['second_gradient'][idx_145k]:.4f}\n")
            f.write(f"   Ratio: {results['gradient_ratio'][idx_145k]:.2f}x\n")
            
            f.write(f"\n3. After Transition (150k):\n")
            f.write(f"   Strongest gradient: {results['strongest_gradient'][idx_150k]:.4f}\n")
            f.write(f"   Second gradient: {results['second_gradient'][idx_150k]:.4f}\n")
            f.write(f"   Ratio: {results['gradient_ratio'][idx_150k]:.2f}x\n")
            
            # 路径切换分析
            f.write(f"\n4. Path Switching Events:\n")
            f.write(f"   Total switches: {len(path_changes)}\n")
            if path_changes:
                f.write("   Switch details:\n")
                for iter_num, old_token, new_token in path_changes[:5]:  # 显示前5个
                    f.write(f"     - At {iter_num}: token {old_token} → {new_token}\n")
                if len(path_changes) > 5:
                    f.write(f"     ... and {len(path_changes)-5} more switches\n")
            
            # 竞争动态总结
            f.write(f"\n5. Competition Dynamics Summary:\n")
            max_ratio_idx = np.argmax(results['gradient_ratio'])
            min_ratio_idx = np.argmin(results['gradient_ratio'])
            
            f.write(f"   Maximum dominance: {results['gradient_ratio'][max_ratio_idx]:.2f}x at iteration {results['iteration'][max_ratio_idx]}\n")
            f.write(f"   Minimum dominance: {results['gradient_ratio'][min_ratio_idx]:.2f}x at iteration {results['iteration'][min_ratio_idx]}\n")
            
            # 相变期间的特征
            phase_start_idx = min(range(len(iterations)), key=lambda i: abs(iterations[i]-145000))
            phase_end_idx = min(range(len(iterations)), key=lambda i: abs(iterations[i]-150000))
            
            if phase_end_idx > phase_start_idx:
                phase_ratios = results['gradient_ratio'][phase_start_idx:phase_end_idx+1]
                f.write(f"\n6. Phase Transition Characteristics (145k-150k):\n")
                f.write(f"   Average competition ratio: {np.mean(phase_ratios):.2f}x\n")
                f.write(f"   Ratio volatility (std): {np.std(phase_ratios):.2f}\n")
                
                # 检查是否有路径切换
                phase_switches = [pc for pc in path_changes if 145000 <= pc[0] <= 150000]
                f.write(f"   Path switches during transition: {len(phase_switches)}\n")
        
        print(f"Text report saved to: {report_path}")

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze gradient competition during phase transition')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/simple_graph/100', help='Data directory')
    parser.add_argument('--start_iter', type=int, default=140000, help='Start iteration')
    parser.add_argument('--end_iter', type=int, default=155000, help='End iteration')
    parser.add_argument('--step', type=int, default=1000, help='Step between iterations')
    parser.add_argument('--num_sequences', type=int, default=50, help='Number of test sequences')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = GradientCompetitionAnalyzer(args.checkpoint_dir, args.data_dir)
    
    # 设置要分析的迭代
    iterations = list(range(args.start_iter, args.end_iter + 1, args.step))
    
    print(f"Analyzing gradient competition from {args.start_iter} to {args.end_iter}")
    print(f"Checkpoints to analyze: {len(iterations)}")
    
    # 运行分析
    results = analyzer.analyze_gradient_competition(iterations, args.num_sequences)
    
    if results:
        # 可视化结果
        path_changes = analyzer.visualize_results(results)
        print(f"\nAnalysis complete! Found {len(path_changes)} path preference changes.")
        print("Results saved to: gradient_competition_results/")
    else:
        print("Analysis failed - no results obtained")

if __name__ == "__main__":
    main()