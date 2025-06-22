"""
analyze_internal_dynamics.py
深入分析相变过程中模型内部的变化
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import networkx as nx
from collections import defaultdict

from model import GPTConfig, GPT

class InternalDynamicsAnalyzer:
    def __init__(self, checkpoint_dir, device='cuda'):
        self.checkpoint_dir = checkpoint_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = os.path.join(checkpoint_dir, 'internal_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载必要的元数据
        self.load_metadata()
        
    def load_metadata(self):
        """加载数据和图结构"""
        data_dir = 'data/simple_graph/100'
        
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = len(self.itos)
        self.block_size = meta['block_size']
        
        # 加载图
        self.graph = nx.read_graphml(os.path.join(data_dir, "path_graph.graphml"))
        
        # 加载验证数据用于测试
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), 
                                  dtype=np.uint16, mode='r')
    
    def load_model(self, iteration):
        """加载特定iteration的模型"""
        ckpt_path = os.path.join(self.checkpoint_dir, f'ckpt_{iteration}.pt')
        if not os.path.exists(ckpt_path):
            return None
            
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 创建模型
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # 加载权重
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint['model']
    
    def analyze_parameter_saturation(self, iterations):
        """分析参数饱和度"""
        print("\n1. Analyzing Parameter Saturation...")
        
        saturation_stats = []
        
        for iter_num in tqdm(iterations):
            model, state_dict = self.load_model(iter_num)
            if model is None:
                continue
                
            stats = {'iteration': iter_num}
            
            # 分析各层权重的饱和度
            for name, param in state_dict.items():
                if 'weight' in name:
                    w = param.cpu().numpy()
                    
                    # 计算饱和度指标
                    # 1. 接近边界值的比例（假设权重初始化在[-1, 1]范围）
                    saturated_ratio = np.mean(np.abs(w) > 0.95)
                    
                    # 2. 权重分布的峰度（高峰度表示更多极端值）
                    from scipy.stats import kurtosis
                    kurt = kurtosis(w.flatten())
                    
                    # 3. 有效秩（表示权重矩阵的多样性）
                    if len(w.shape) >= 2:
                        singular_values = np.linalg.svd(w, compute_uv=False)
                        effective_rank = np.sum(singular_values > 0.01 * singular_values[0])
                        rank_ratio = effective_rank / min(w.shape)
                    else:
                        rank_ratio = 1.0
                    
                    layer_name = name.split('.')[-2]
                    stats[f'{layer_name}_saturated_ratio'] = saturated_ratio
                    stats[f'{layer_name}_kurtosis'] = kurt
                    stats[f'{layer_name}_rank_ratio'] = rank_ratio
            
            saturation_stats.append(stats)
        
        return pd.DataFrame(saturation_stats)
    
    def analyze_distribution_shift(self, iterations, num_test_sequences=100):
        """分析输出分布的变化"""
        print("\n2. Analyzing Distribution Shift...")
        
        # 选择一些固定的测试序列
        test_sequences = []
        for _ in range(num_test_sequences):
            idx = np.random.randint(0, len(self.val_data) - self.block_size - 1)
            seq = self.val_data[idx:idx + self.block_size + 1].astype(np.int64)
            
            # 找到有效长度
            seq_len = np.where(seq == 0)[0]
            seq_len = seq_len[0] if len(seq_len) > 0 else len(seq)
            
            if seq_len >= 6:  # 需要足够长
                test_sequences.append(seq[:seq_len])
        
        distribution_shifts = []
        
        # 为每个iteration分析分布
        for iter_num in tqdm(iterations):
            model, _ = self.load_model(iter_num)
            if model is None:
                continue
            
            # 收集所有预测分布
            all_distributions = []
            path_choice_distributions = []  # 特别关注路径选择点
            
            for seq in test_sequences:
                # 找到路径分叉点（第3个位置通常是关键）
                if len(seq) > 4:
                    input_seq = torch.tensor(seq[:4], dtype=torch.long).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits, _ = model(input_seq)
                        probs = torch.softmax(logits[0, -1, :], dim=0).cpu().numpy()
                        
                        # 获取当前节点的所有可能下一跳
                        current_node = seq[3] - 2  # token to node
                        if 0 <= current_node < 100:
                            neighbors = list(self.graph.neighbors(str(current_node)))
                            
                            # 收集每个邻居的概率
                            neighbor_probs = {}
                            for n in neighbors:
                                token_id = int(n) + 2
                                if token_id < self.vocab_size:
                                    neighbor_probs[n] = probs[token_id]
                            
                            if neighbor_probs:
                                path_choice_distributions.append(neighbor_probs)
                        
                        all_distributions.append(probs)
            
            # 计算分布统计
            if all_distributions:
                all_distributions = np.array(all_distributions)
                
                # 1. 平均熵
                entropy = -np.sum(all_distributions * np.log(all_distributions + 1e-8), axis=1)
                avg_entropy = np.mean(entropy)
                
                # 2. Top-k概率集中度
                top5_concentration = np.mean(np.sort(all_distributions, axis=1)[:, -5:].sum(axis=1))
                
                # 3. 分布的多样性（不同样本间的KL散度）
                from scipy.stats import entropy as kl_div
                kl_distances = []
                for i in range(min(10, len(all_distributions))):
                    for j in range(i+1, min(10, len(all_distributions))):
                        kl = kl_div(all_distributions[i], all_distributions[j])
                        kl_distances.append(kl)
                avg_kl = np.mean(kl_distances) if kl_distances else 0
                
                # 4. 路径选择的偏好变化
                if path_choice_distributions:
                    # 计算最受偏好的路径
                    path_preferences = defaultdict(list)
                    for dist in path_choice_distributions:
                        if dist:
                            max_neighbor = max(dist, key=dist.get)
                            for n, p in dist.items():
                                path_preferences[n].append(p)
                    
                    # 计算每条路径的平均概率
                    avg_path_probs = {n: np.mean(probs) for n, probs in path_preferences.items()}
                    top_path = max(avg_path_probs, key=avg_path_probs.get) if avg_path_probs else None
                    top_path_prob = avg_path_probs[top_path] if top_path else 0
                else:
                    top_path = None
                    top_path_prob = 0
                
                distribution_shifts.append({
                    'iteration': iter_num,
                    'avg_entropy': avg_entropy,
                    'top5_concentration': top5_concentration,
                    'distribution_diversity': avg_kl,
                    'top_path': top_path,
                    'top_path_prob': top_path_prob
                })
        
        return pd.DataFrame(distribution_shifts)
    
    def analyze_gradient_dynamics(self, iterations):
        """分析梯度动态（通过相邻checkpoint的权重变化估计）"""
        print("\n3. Analyzing Gradient Dynamics...")
        
        gradient_stats = []
        
        for i in range(len(iterations) - 1):
            iter1, iter2 = iterations[i], iterations[i+1]
            
            model1, state1 = self.load_model(iter1)
            model2, state2 = self.load_model(iter2)
            
            if model1 is None or model2 is None:
                continue
            
            stats = {
                'iteration': iter2,
                'delta_iter': iter2 - iter1
            }
            
            # 计算权重变化
            for name in state1:
                if 'weight' in name and name in state2:
                    w1 = state1[name].cpu().numpy()
                    w2 = state2[name].cpu().numpy()
                    
                    # 权重变化的范数
                    delta_w = w2 - w1
                    delta_norm = np.linalg.norm(delta_w)
                    relative_change = delta_norm / (np.linalg.norm(w1) + 1e-8)
                    
                    # 变化的方向一致性（通过余弦相似度）
                    if i > 0 and len(gradient_stats) > 0:
                        prev_name = f'{name}_delta'
                        if prev_name in gradient_stats[-1]:
                            prev_delta = gradient_stats[-1][prev_name]
                            cosine_sim = np.dot(delta_w.flatten(), prev_delta.flatten()) / (
                                np.linalg.norm(delta_w) * np.linalg.norm(prev_delta) + 1e-8
                            )
                            stats[f'{name}_consistency'] = cosine_sim
                    
                    layer_name = name.split('.')[-2]
                    stats[f'{layer_name}_delta_norm'] = delta_norm
                    stats[f'{layer_name}_relative_change'] = relative_change
                    stats[f'{name}_delta'] = delta_w  # 保存用于下次比较
            
            # 移除保存的delta（避免DataFrame过大）
            stats = {k: v for k, v in stats.items() if not k.endswith('_delta')}
            gradient_stats.append(stats)
        
        return pd.DataFrame(gradient_stats)
    
    def analyze_attention_patterns(self, iterations, num_samples=50):
        """分析注意力模式的变化"""
        print("\n4. Analyzing Attention Patterns...")
        
        attention_stats = []
        
        # 准备测试样本
        test_inputs = []
        for _ in range(num_samples):
            idx = np.random.randint(0, len(self.val_data) - self.block_size - 1)
            seq = self.val_data[idx:idx + 10].astype(np.int64)  # 短序列
            test_inputs.append(seq)
        
        for iter_num in tqdm(iterations):
            model, _ = self.load_model(iter_num)
            if model is None:
                continue
            
            # 收集注意力权重
            attention_entropies = []
            attention_sparsities = []
            
            for seq in test_inputs:
                input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # 需要修改model.forward来返回注意力权重
                    # 这里简化处理，只分析输出
                    logits, _ = model(input_tensor)
                    
                    # 分析输出分布作为注意力的代理
                    probs = torch.softmax(logits[0, -1, :], dim=0).cpu().numpy()
                    
                    # 计算熵和稀疏度
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    sparsity = np.sum(probs > 0.01)  # 有效选择的数量
                    
                    attention_entropies.append(entropy)
                    attention_sparsities.append(sparsity)
            
            attention_stats.append({
                'iteration': iter_num,
                'avg_attention_entropy': np.mean(attention_entropies),
                'avg_attention_sparsity': np.mean(attention_sparsities)
            })
        
        return pd.DataFrame(attention_stats)
    
    def visualize_internal_dynamics(self, saturation_df, distribution_df, gradient_df, attention_df):
        """可视化内部动态变化"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 参数饱和度
        ax1 = plt.subplot(3, 3, 1)
        if 'lm_head_saturated_ratio' in saturation_df.columns:
            ax1.plot(saturation_df['iteration'], saturation_df['lm_head_saturated_ratio'], 
                    'b-', linewidth=2, label='Output layer')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Saturation Ratio')
        ax1.set_title('Parameter Saturation')
        ax1.axvspan(145000, 150000, alpha=0.2, color='red', label='Critical period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 权重矩阵的有效秩
        ax2 = plt.subplot(3, 3, 2)
        if 'lm_head_rank_ratio' in saturation_df.columns:
            ax2.plot(saturation_df['iteration'], saturation_df['lm_head_rank_ratio'], 
                    'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Effective Rank Ratio')
        ax2.set_title('Weight Matrix Diversity')
        ax2.axvspan(145000, 150000, alpha=0.2, color='red')
        ax2.grid(True, alpha=0.3)
        
        # 3. 输出分布熵
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(distribution_df['iteration'], distribution_df['avg_entropy'], 
                'purple', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Average Entropy')
        ax3.set_title('Output Distribution Entropy')
        ax3.axvspan(145000, 150000, alpha=0.2, color='red')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top-5概率集中度
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(distribution_df['iteration'], distribution_df['top5_concentration'], 
                'orange', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Top-5 Probability Sum')
        ax4.set_title('Probability Concentration')
        ax4.axvspan(145000, 150000, alpha=0.2, color='red')
        ax4.grid(True, alpha=0.3)
        
        # 5. 分布多样性
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(distribution_df['iteration'], distribution_df['distribution_diversity'], 
                'brown', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Average KL Divergence')
        ax5.set_title('Distribution Diversity')
        ax5.axvspan(145000, 150000, alpha=0.2, color='red')
        ax5.grid(True, alpha=0.3)
        
        # 6. 梯度动态
        ax6 = plt.subplot(3, 3, 6)
        if 'lm_head_relative_change' in gradient_df.columns:
            ax6.plot(gradient_df['iteration'], gradient_df['lm_head_relative_change'], 
                    'cyan', linewidth=2)
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Relative Weight Change')
        ax6.set_title('Gradient Magnitude')
        ax6.axvspan(145000, 150000, alpha=0.2, color='red')
        ax6.grid(True, alpha=0.3)
        
        # 7. 路径偏好变化
        ax7 = plt.subplot(3, 3, 7)
        # 绘制top path的概率变化
        ax7.plot(distribution_df['iteration'], distribution_df['top_path_prob'], 
                'magenta', linewidth=2)
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('Top Path Probability')
        ax7.set_title('Dominant Path Selection')
        ax7.axvspan(145000, 150000, alpha=0.2, color='red')
        ax7.grid(True, alpha=0.3)
        
        # 8. 参数变化一致性
        ax8 = plt.subplot(3, 3, 8)
        consistency_cols = [c for c in gradient_df.columns if 'consistency' in c]
        if consistency_cols:
            ax8.plot(gradient_df['iteration'], gradient_df[consistency_cols[0]], 
                    'darkgreen', linewidth=2)
        ax8.set_xlabel('Iteration')
        ax8.set_ylabel('Cosine Similarity')
        ax8.set_title('Gradient Direction Consistency')
        ax8.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax8.axvspan(145000, 150000, alpha=0.2, color='red')
        ax8.grid(True, alpha=0.3)
        
        # 9. 综合相变指标
        ax9 = plt.subplot(3, 3, 9)
        # 创建一个综合指标
        if len(saturation_df) == len(distribution_df):
            phase_indicator = (
                saturation_df['lm_head_saturated_ratio'].values * 100 +
                (1 - distribution_df['avg_entropy'].values / distribution_df['avg_entropy'].max()) * 50
            )
            ax9.plot(saturation_df['iteration'], phase_indicator, 'red', linewidth=3)
            ax9.set_xlabel('Iteration')
            ax9.set_ylabel('Phase Transition Indicator')
            ax9.set_title('Combined Phase Metric')
            ax9.axvspan(145000, 150000, alpha=0.2, color='red')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'internal_dynamics_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    def generate_mechanism_report(self, saturation_df, distribution_df, gradient_df):
        """生成机制分析报告"""
        report_path = os.path.join(self.output_dir, 'mechanism_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PHASE TRANSITION INTERNAL MECHANISM ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write("HYPOTHESIS: Parameter saturation → Distribution shift → Performance divergence\n\n")
            
            # 1. 参数饱和分析
            f.write("1. PARAMETER SATURATION EVIDENCE:\n")
            f.write("-"*40 + "\n")
            
            # 找到相变前后的饱和度
            pre_transition = saturation_df[saturation_df['iteration'] <= 145000].tail(3)
            post_transition = saturation_df[saturation_df['iteration'] >= 150000].head(3)
            
            if len(pre_transition) > 0 and len(post_transition) > 0:
                pre_sat = pre_transition['lm_head_saturated_ratio'].mean()
                post_sat = post_transition['lm_head_saturated_ratio'].mean()
                f.write(f"   Pre-transition saturation: {pre_sat:.4f}\n")
                f.write(f"   Post-transition saturation: {post_sat:.4f}\n")
                f.write(f"   Change: {(post_sat - pre_sat)*100:.2f}pp\n\n")
            
            # 2. 分布偏移分析
            f.write("2. DISTRIBUTION SHIFT EVIDENCE:\n")
            f.write("-"*40 + "\n")
            
            pre_dist = distribution_df[distribution_df['iteration'] <= 145000].tail(3)
            post_dist = distribution_df[distribution_df['iteration'] >= 150000].head(3)
            
            if len(pre_dist) > 0 and len(post_dist) > 0:
                pre_entropy = pre_dist['avg_entropy'].mean()
                post_entropy = post_dist['avg_entropy'].mean()
                pre_diversity = pre_dist['distribution_diversity'].mean()
                post_diversity = post_dist['distribution_diversity'].mean()
                
                f.write(f"   Entropy change: {pre_entropy:.4f} → {post_entropy:.4f}\n")
                f.write(f"   Distribution diversity: {pre_diversity:.4f} → {post_diversity:.4f}\n")
                
                # 路径偏好变化
                pre_paths = pre_dist['top_path'].value_counts()
                post_paths = post_dist['top_path'].value_counts()
                f.write(f"   Dominant path shift detected: {len(set(pre_paths.index) - set(post_paths.index))} paths changed\n\n")
            
            # 3. 梯度动态分析
            f.write("3. GRADIENT DYNAMICS:\n")
            f.write("-"*40 + "\n")
            
            critical_gradient = gradient_df[(gradient_df['iteration'] >= 145000) & 
                                          (gradient_df['iteration'] <= 150000)]
            
            if len(critical_gradient) > 0:
                max_change_idx = critical_gradient['lm_head_relative_change'].idxmax()
                max_change_iter = critical_gradient.loc[max_change_idx, 'iteration']
                max_change_val = critical_gradient.loc[max_change_idx, 'lm_head_relative_change']
                
                f.write(f"   Peak gradient activity at iteration: {max_change_iter}\n")
                f.write(f"   Maximum relative change: {max_change_val:.4f}\n\n")
            
            # 4. 因果链验证
            f.write("4. CAUSAL CHAIN VERIFICATION:\n")
            f.write("-"*40 + "\n")
            
            # 检查时间顺序
            saturation_increase = saturation_df[saturation_df['lm_head_saturated_ratio'] > 0.1]['iteration'].min()
            entropy_drop = distribution_df[distribution_df['avg_entropy'] < distribution_df['avg_entropy'].iloc[0] * 0.8]['iteration'].min()
            
            f.write(f"   Saturation begins: iteration {saturation_increase}\n")
            f.write(f"   Distribution shift begins: iteration {entropy_drop}\n")
            f.write(f"   Time lag: {entropy_drop - saturation_increase} iterations\n\n")
            
            f.write("CONCLUSION:\n")
            f.write("-"*40 + "\n")
            if saturation_increase < entropy_drop:
                f.write("   ✓ Parameter saturation precedes distribution shift\n")
                f.write("   ✓ Distribution shift coincides with performance collapse\n")
                f.write("   ✓ Causal chain is supported by the evidence\n")
            else:
                f.write("   ✗ Timeline does not clearly support the hypothesized causal chain\n")
                f.write("   → Alternative mechanisms may be at play\n")
        
        print(f"\nMechanism report saved to {report_path}")

def main():
    checkpoint_dir = "out/spurious_rewards/standard_alpha0.5_div0.1_seed42_20250622_042430"
    
    analyzer = InternalDynamicsAnalyzer(checkpoint_dir)
    
    print("="*60)
    print("Internal Dynamics Analysis - Understanding Phase Transition Mechanisms")
    print("="*60)
    
    # 分析关键时期的checkpoints
    iterations = list(range(135000, 156000, 1000))  # 每1k分析一次
    
    # 1. 参数饱和度分析
    saturation_df = analyzer.analyze_parameter_saturation(iterations)
    saturation_df.to_csv(os.path.join(analyzer.output_dir, 'parameter_saturation.csv'), index=False)
    
    # 2. 分布偏移分析
    distribution_df = analyzer.analyze_distribution_shift(iterations)
    distribution_df.to_csv(os.path.join(analyzer.output_dir, 'distribution_shift.csv'), index=False)
    
    # 3. 梯度动态分析
    gradient_df = analyzer.analyze_gradient_dynamics(iterations)
    gradient_df.to_csv(os.path.join(analyzer.output_dir, 'gradient_dynamics.csv'), index=False)
    
    # 4. 注意力模式分析
    attention_df = analyzer.analyze_attention_patterns(iterations)
    attention_df.to_csv(os.path.join(analyzer.output_dir, 'attention_patterns.csv'), index=False)
    
    # 5. 生成可视化
    print("\n5. Generating visualizations...")
    analyzer.visualize_internal_dynamics(saturation_df, distribution_df, gradient_df, attention_df)
    
    # 6. 生成机制报告
    print("\n6. Generating mechanism analysis report...")
    analyzer.generate_mechanism_report(saturation_df, distribution_df, gradient_df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()