# investigate_data.py
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. 检查图的结构
G = nx.read_graphml('data/simple_graph/200/path_graph.graphml')
print(f"图中的节点数: {G.number_of_nodes()}")
print(f"图中的边数: {G.number_of_edges()}")

# 2. 检查每个节点的度
degrees = dict(G.degree())
print(f"\n节点度数统计:")
print(f"最小度: {min(degrees.values())}")
print(f"最大度: {max(degrees.values())}")
print(f"平均度: {sum(degrees.values())/len(degrees):.2f}")

# 3. 找出度为0的节点（孤立节点）
isolated_nodes = [node for node, degree in degrees.items() if degree == 0]
print(f"\n孤立节点数: {len(isolated_nodes)}")
if isolated_nodes[:10]:
    print(f"前10个孤立节点: {isolated_nodes[:10]}")

# 4. 检查高编号节点的连接情况
high_nodes_degrees = {node: degrees[node] for node in degrees if int(node) >= 100}
print(f"\n编号>=100的节点度数:")
for node, deg in sorted(high_nodes_degrees.items())[:10]:
    print(f"  节点{node}: 度={deg}")

# 5. 分析训练数据中实际使用的节点
train_data = np.memmap('data/simple_graph/200/train_20.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/simple_graph/200/val.bin', dtype=np.uint16, mode='r')

# 统计token分布
all_tokens = np.concatenate([train_data[:100000], val_data[:100000]])
unique_tokens, counts = np.unique(all_tokens, return_counts=True)

print(f"\n数据中的token统计:")
print(f"唯一token数: {len(unique_tokens)}")
print(f"Token范围: {unique_tokens.min()} - {unique_tokens.max()}")

# 节点token (>=2)
node_tokens = unique_tokens[unique_tokens >= 2]
node_ids = node_tokens - 2
print(f"\n实际使用的节点:")
print(f"节点数: {len(node_ids)}")
print(f"节点范围: {node_ids.min()} - {node_ids.max()}")

# 绘制节点度数分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(list(degrees.values()), bins=30, edgecolor='black')
plt.xlabel('Node Degree')
plt.ylabel('Count')
plt.title('Node Degree Distribution')

plt.subplot(1, 2, 2)
node_numbers = [int(node) for node in degrees.keys()]
node_degrees = [degrees[str(node)] for node in sorted(node_numbers)]
plt.scatter(sorted(node_numbers), node_degrees, alpha=0.5, s=10)
plt.xlabel('Node ID')
plt.ylabel('Degree')
plt.title('Node Degree vs Node ID')
plt.axvline(x=63, color='r', linestyle='--', label='Max used node')
plt.legend()

plt.tight_layout()
plt.savefig('node_analysis.png')
print("\n图形已保存到 node_analysis.png")