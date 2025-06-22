"""
analyze_solution_topology_complete.py
完整的解空间拓扑分析脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from tqdm import tqdm
import pickle
import os
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT

class SolutionTopologyAnalyzer:
    def __init__(self, checkpoint_dir, data_dir='data/simple_graph/100'):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载元数据
        self.load_metadata()
        
    def load_metadata(self):
        """加载必要的元数据"""
        with open(os.path.join(self.data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = len(self.itos)
        self.block_size = meta['block_size']
        
        # 加载验证数据
        self.val_data = np.memmap(os.path.join(self.data_dir, 'val.bin'), 
                                  dtype=np.uint16, mode='r')
        
    def prepare_fixed_test_sequences(self, n_sequences=500):
        """准备固定的测试序列以确保一致性"""
        np.random.seed(42)  # 固定随机种子
        
        sequences = []
        positions = []  # 记录每个序列测试的位置
        
        for _ in range(n_sequences):
            idx = np.random.randint(0, len(self.val_data) - self.block_size - 1)
            seq = self.val_data[idx:idx + self.block_size + 1].astype(np.int64)
            
            # 找到有效长度
            seq_len = np.where(seq == 0)[0]
            seq_len = seq_len[0] if len(seq_len) > 0 else len(seq)
            
            if seq_len >= 5:
                # 测试多个位置
                test_positions = [3, 4]  # 测试第4和第5个位置
                for pos in test_positions:
                    if pos < seq_len - 1:
                        sequences.append(seq[:pos+1])
                        positions.append(pos)
        
        print(f"Prepared {len(sequences)} test sequences")
        return sequences, positions
    
    def extract_solution_fingerprints(self, model, test_sequences):
        """提取模型的解决方案指纹"""
        fingerprints = []
        
        model.eval()
        with torch.no_grad():
            for seq in test_sequences:
                input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # 获取输出分布
                logits, _ = model(input_tensor)
                probs = torch.softmax(logits[0, -1, :], dim=0).cpu().numpy()
                
                # 使用概率分布作为指纹
                fingerprints.append(probs)
        
        return np.array(fingerprints)
    
    def analyze_solution_topology(self, iterations, n_samples=500):
        """分析模型在解空间中的轨迹"""
        
        # 准备固定的测试集
        test_sequences, positions = self.prepare_fixed_test_sequences(n_samples)
        
        # 收集所有checkpoint的解分布
        solution_fingerprints = []
        iteration_labels = []
        model_weights = []  # 也收集模型权重用于额外分析
        
        print("Collecting solution distributions...")
        for iter_num in tqdm(iterations):
            ckpt_path = os.path.join(self.checkpoint_dir, f'ckpt_{iter_num}.pt')
            if not os.path.exists(ckpt_path):
                continue
            
            # 加载模型
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model_args = checkpoint['model_args']
            model = GPT(GPTConfig(**model_args))
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            
            # 提取解决方案指纹
            fingerprints = self.extract_solution_fingerprints(model, test_sequences)
            
            # 展平成一维向量
            flattened = fingerprints.flatten()
            solution_fingerprints.append(flattened)
            iteration_labels.append(iter_num)
            
            # 收集关键权重（例如输出层）
            if 'lm_head.weight' in checkpoint['model']:
                lm_weights = checkpoint['model']['lm_head.weight'].cpu().numpy()
                model_weights.append(lm_weights.flatten()[:1000])  # 只取前1000个防止太大
        
        solution_array = np.array(solution_fingerprints)
        weight_array = np.array(model_weights) if model_weights else None
        
        print(f"Solution array shape: {solution_array.shape}")
        
        return self.analyze_trajectories(solution_array, weight_array, iteration_labels)
    
    def analyze_trajectories(self, solution_array, weight_array, iteration_labels):
        """分析解空间轨迹"""
        results = {}
        
        # 1. PCA降维
        print("Performing PCA...")
        pca = PCA(n_components=min(50, solution_array.shape[0]))
        pca_result = pca.fit_transform(solution_array)
        results['pca'] = pca_result
        results['pca_explained_variance'] = pca.explained_variance_ratio_
        
        # 2. t-SNE嵌入
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(solution_array)-1), 
                    random_state=42, n_iter=1000)
        tsne_result = tsne.fit_transform(pca_result[:, :30])  # 使用PCA的前30个成分
        results['tsne'] = tsne_result
        
        # 3. 计算轨迹统计
        distances = []
        velocities = []
        accelerations = []
        
        for i in range(len(solution_array) - 1):
            # 欧氏距离
            dist = np.linalg.norm(solution_array[i+1] - solution_array[i])
            distances.append(dist)
            
            # 速度（距离/迭代数）
            iter_diff = iteration_labels[i+1] - iteration_labels[i]
            velocity = dist / iter_diff if iter_diff > 0 else 0
            velocities.append(velocity)
        
        # 加速度
        for i in range(len(velocities) - 1):
            acc = velocities[i+1] - velocities[i]
            accelerations.append(acc)
        
        results['distances'] = distances
        results['velocities'] = velocities
        results['accelerations'] = accelerations
        results['iterations'] = iteration_labels
        
        # 4. 如果有权重数据，也分析权重空间
        if weight_array is not None:
            print("Analyzing weight space...")
            weight_pca = PCA(n_components=3)
            weight_pca_result = weight_pca.fit_transform(weight_array)
            results['weight_pca'] = weight_pca_result
        
        return results
    
    def visualize_topology(self, results, output_dir='solution_topology_results'):
        """可视化解空间拓扑"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 16))
        
        # 定义相变区域
        transition_start = 145000
        transition_end = 150000
        
        # 1. PCA轨迹（3D）
        ax = fig.add_subplot(3, 3, 1, projection='3d')
        pca_result = results['pca']
        iterations = results['iterations']
        
        # 创建颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(iterations)))
        
        # 绘制轨迹
        for i in range(len(pca_result) - 1):
            ax.plot([pca_result[i, 0], pca_result[i+1, 0]],
                   [pca_result[i, 1], pca_result[i+1, 1]],
                   [pca_result[i, 2], pca_result[i+1, 2]],
                   color=colors[i], alpha=0.7, linewidth=2)
        
        # 标记关键点
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                           c=iterations, cmap='viridis', s=50)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('Solution Space Trajectory (PCA 3D)')
        
        # 2. t-SNE嵌入
        ax = fig.add_subplot(3, 3, 2)
        tsne_result = results['tsne']
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                           c=iterations, cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
        
        # 连接相邻点
        for i in range(len(tsne_result) - 1):
            ax.plot([tsne_result[i, 0], tsne_result[i+1, 0]],
                   [tsne_result[i, 1], tsne_result[i+1, 1]],
                   'k-', alpha=0.3, linewidth=1)
        
        # 标记相变区域的点
        phase_indices = [i for i, iter_num in enumerate(iterations) 
                        if transition_start <= iter_num <= transition_end]
        if phase_indices:
            ax.scatter(tsne_result[phase_indices, 0], tsne_result[phase_indices, 1],
                     color='red', s=200, marker='*', edgecolors='black', linewidth=2,
                     label='Phase transition', zorder=10)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Solution Space Topology (t-SNE)')
        plt.colorbar(scatter, ax=ax, label='Iteration')
        ax.legend()
        
        # 3. 解释方差
        ax = fig.add_subplot(3, 3, 3)
        explained_var = results['pca_explained_variance']
        cumsum_var = np.cumsum(explained_var)
        
        ax.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, label='Individual')
        ax.plot(range(1, len(explained_var)+1), cumsum_var, 'r-', marker='o', 
               linewidth=2, label='Cumulative')
        
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 找到90%方差的成分数
        n_components_90 = np.argmax(cumsum_var >= 0.9) + 1
        ax.axvline(n_components_90, color='green', linestyle='--', 
                  label=f'90% variance: {n_components_90} components')
        ax.legend()
        
        # 4. 移动速度
        ax = fig.add_subplot(3, 3, 4)
        velocities = results['velocities']
        ax.plot(iterations[:-1], velocities, 'b-', linewidth=2, marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Movement Speed')
        ax.set_title('Solution Space Movement Velocity')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red', label='Phase transition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 加速度
        ax = fig.add_subplot(3, 3, 5)
        if results['accelerations']:
            ax.plot(iterations[:-2], results['accelerations'], 'g-', linewidth=2, marker='s')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Acceleration')
            ax.set_title('Change in Movement Speed')
            ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        # 6. 累积距离
        ax = fig.add_subplot(3, 3, 6)
        cumulative_dist = np.cumsum(results['distances'])
        ax.plot(iterations[:-1], cumulative_dist, 'purple', linewidth=3, marker='D')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Distance')
        ax.set_title('Total Distance Traveled in Solution Space')
        ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
        ax.grid(True, alpha=0.3)
        
        # 7. PC1 vs PC2轨迹
        ax = fig.add_subplot(3, 3, 7)
        pc1 = pca_result[:, 0]
        pc2 = pca_result[:, 1]
        
        # 绘制带箭头的轨迹
        for i in range(len(pc1) - 1):
            ax.annotate('', xy=(pc1[i+1], pc2[i+1]), xytext=(pc1[i], pc2[i]),
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
        
        scatter = ax.scatter(pc1, pc2, c=iterations, cmap='viridis', s=100, 
                           edgecolors='black', linewidth=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Principal Component Trajectory')
        plt.colorbar(scatter, ax=ax, label='Iteration')
        
        # 8. 权重空间分析（如果有）
        if 'weight_pca' in results:
            ax = fig.add_subplot(3, 3, 8)
            weight_pca = results['weight_pca']
            scatter = ax.scatter(weight_pca[:, 0], weight_pca[:, 1],
                               c=iterations, cmap='plasma', s=100)
            ax.set_xlabel('Weight PC1')
            ax.set_ylabel('Weight PC2')
            ax.set_title('Weight Space Evolution')
            plt.colorbar(scatter, ax=ax, label='Iteration')
        
        # 9. 局部维度分析
        ax = fig.add_subplot(3, 3, 9)
        # 计算局部本征维度
        local_dims = []
        window = 3  # 局部窗口大小
        
        for i in range(window, len(pca_result) - window):
            # 取局部窗口
            local_data = pca_result[i-window:i+window+1]
            
            # 计算局部协方差
            local_cov = np.cov(local_data.T)
            eigvals = np.linalg.eigvalsh(local_cov)
            
            # 计算有效维度
            eigvals = eigvals[eigvals > 1e-10]
            if len(eigvals) > 0:
                eigvals_norm = eigvals / np.sum(eigvals)
                effective_dim = np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm + 1e-10)))
                local_dims.append(effective_dim)
        
        if local_dims:
            ax.plot(iterations[window:-window], local_dims, 'brown', linewidth=2, marker='h')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Local Effective Dimension')
            ax.set_title('Solution Space Local Complexity')
            ax.axvspan(transition_start, transition_end, alpha=0.2, color='red')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'solution_topology_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        with open(os.path.join(output_dir, 'topology_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # 生成报告
        self.generate_report(results, output_dir)
    
    def generate_report(self, results, output_dir):
        """生成分析报告"""
        report_path = os.path.join(output_dir, 'topology_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Solution Space Topology Analysis Report\n")
            f.write("="*60 + "\n\n")
            
            # 基本统计
            f.write("1. Trajectory Statistics:\n")
            f.write("-"*40 + "\n")
            f.write(f"   Total checkpoints analyzed: {len(results['iterations'])}\n")
            f.write(f"   Iteration range: {results['iterations'][0]} - {results['iterations'][-1]}\n")
            
            # 移动统计
            if results['distances']:
                total_distance = np.sum(results['distances'])
                avg_speed = np.mean(results['velocities'])
                max_speed_idx = np.argmax(results['velocities'])
                max_speed_iter = results['iterations'][max_speed_idx]
                
                f.write(f"\n2. Movement Analysis:\n")
                f.write("-"*40 + "\n")
                f.write(f"   Total distance traveled: {total_distance:.4f}\n")
                f.write(f"   Average speed: {avg_speed:.6f}\n")
                f.write(f"   Maximum speed: {results['velocities'][max_speed_idx]:.6f} at {max_speed_iter}\n")
            
            # PCA分析
            f.write(f"\n3. Dimensionality Analysis:\n")
            f.write("-"*40 + "\n")
            explained_var = results['pca_explained_variance']
            cumsum_var = np.cumsum(explained_var)
            
            n_comp_90 = np.argmax(cumsum_var >= 0.9) + 1
            n_comp_95 = np.argmax(cumsum_var >= 0.95) + 1
            n_comp_99 = np.argmax(cumsum_var >= 0.99) + 1
            
            f.write(f"   Components for 90% variance: {n_comp_90}\n")
            f.write(f"   Components for 95% variance: {n_comp_95}\n")
            f.write(f"   Components for 99% variance: {n_comp_99}\n")
            f.write(f"   Top 3 PC variance: {explained_var[:3]}\n")
            
            # 相变期间的特征
            f.write(f"\n4. Phase Transition Characteristics:\n")
            f.write("-"*40 + "\n")
            
            # 找到相变期间的索引
            phase_indices = []
            for i, iter_num in enumerate(results['iterations']):
                if 145000 <= iter_num <= 150000:
                    phase_indices.append(i)
            
            if phase_indices and len(phase_indices) > 1:
                # 相变期间的移动
                phase_distances = [results['distances'][i] for i in phase_indices[:-1] 
                                 if i < len(results['distances'])]
                if phase_distances:
                    phase_total_dist = np.sum(phase_distances)
                    phase_avg_speed = np.mean([results['velocities'][i] for i in phase_indices[:-1]
                                              if i < len(results['velocities'])])
                    
                    f.write(f"   Distance during transition: {phase_total_dist:.4f}\n")
                    f.write(f"   Average speed during transition: {phase_avg_speed:.6f}\n")
                    
                    # 与整体平均比较
                    overall_avg_speed = np.mean(results['velocities'])
                    speed_ratio = phase_avg_speed / overall_avg_speed
                    f.write(f"   Speed ratio (transition/overall): {speed_ratio:.2f}x\n")
        
        print(f"Report saved to: {report_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze solution space topology')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/simple_graph/100')
    parser.add_argument('--iterations', type=str, required=True, 
                       help='Comma-separated list of iterations')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of test sequences')
    
    args = parser.parse_args()
    
    # 解析迭代列表
    iterations = [int(x.strip()) for x in args.iterations.split(',')]
    
    analyzer = SolutionTopologyAnalyzer(args.checkpoint_dir, args.data_dir)
    
    print(f"Analyzing solution topology for iterations: {iterations}")
    
    results = analyzer.analyze_solution_topology(iterations, args.n_samples)
    
    analyzer.visualize_topology(results)
    
    print("Analysis complete! Results saved to: solution_topology_results/")

if __name__ == "__main__":
    main()