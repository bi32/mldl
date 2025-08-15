# 图神经网络理论：从基础到前沿 🕸️

深入理解图神经网络的核心理论，从图论基础到最新的图学习算法。

## 1. 图论基础 🌐

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class GraphTheoryBasics:
    """图论基础"""
    
    def __init__(self):
        self.graphs = {}
    
    def graph_definitions(self):
        """图的基本定义"""
        print("=== 图的基本定义 ===")
        
        print("图的定义:")
        print("G = (V, E)")
        print("- V: 节点集合 (Vertices/Nodes)")
        print("- E: 边集合 (Edges)")
        print("- |V| = n: 节点数量")
        print("- |E| = m: 边数量")
        print()
        
        graph_types = {
            "无向图": {
                "定义": "边没有方向",
                "表示": "E ⊆ {{u,v} : u,v ∈ V}",
                "特点": "邻接矩阵对称",
                "应用": "社交网络、分子结构"
            },
            "有向图": {
                "定义": "边有方向",
                "表示": "E ⊆ {(u,v) : u,v ∈ V}",
                "特点": "邻接矩阵可能不对称",
                "应用": "网页链接、引用网络"
            },
            "加权图": {
                "定义": "边带有权重",
                "表示": "w: E → ℝ",
                "特点": "权重矩阵W",
                "应用": "交通网络、知识图谱"
            },
            "多重图": {
                "定义": "节点间可有多条边",
                "表示": "E是多重集合",
                "特点": "边计数",
                "应用": "通信网络、生物网络"
            }
        }
        
        for graph_type, details in graph_types.items():
            print(f"{graph_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化不同类型的图
        self.visualize_graph_types()
        
        return graph_types
    
    def visualize_graph_types(self):
        """可视化不同类型的图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 无向图
        G_undirected = nx.Graph()
        G_undirected.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
        
        pos = nx.spring_layout(G_undirected, seed=42)
        nx.draw(G_undirected, pos, ax=axes[0, 0], with_labels=True, 
                node_color='lightblue', node_size=500, font_size=12)
        axes[0, 0].set_title('无向图')
        
        # 2. 有向图
        G_directed = nx.DiGraph()
        G_directed.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
        
        nx.draw(G_directed, pos, ax=axes[0, 1], with_labels=True,
                node_color='lightgreen', node_size=500, font_size=12,
                arrows=True, arrowsize=20)
        axes[0, 1].set_title('有向图')
        
        # 3. 加权图
        G_weighted = nx.Graph()
        edges_with_weights = [(0, 1, 0.5), (1, 2, 1.2), (2, 3, 0.8), (3, 0, 1.5), (1, 3, 0.3)]
        G_weighted.add_weighted_edges_from(edges_with_weights)
        
        edge_labels = nx.get_edge_attributes(G_weighted, 'weight')
        nx.draw(G_weighted, pos, ax=axes[1, 0], with_labels=True,
                node_color='lightcoral', node_size=500, font_size=12)
        nx.draw_networkx_edge_labels(G_weighted, pos, edge_labels, ax=axes[1, 0])
        axes[1, 0].set_title('加权图')
        
        # 4. 多重图
        G_multi = nx.MultiGraph()
        G_multi.add_edges_from([(0, 1), (0, 1), (1, 2), (2, 3), (3, 0)])
        
        nx.draw(G_multi, pos, ax=axes[1, 1], with_labels=True,
                node_color='lightyellow', node_size=500, font_size=12)
        axes[1, 1].set_title('多重图')
        
        plt.tight_layout()
        plt.show()
    
    def graph_representations(self):
        """图的表示方法"""
        print("=== 图的表示方法 ===")
        
        representations = {
            "邻接矩阵": {
                "定义": "A[i,j] = 1 if (i,j) ∈ E, else 0",
                "空间复杂度": "O(n²)",
                "优点": "操作简单，适合稠密图",
                "缺点": "稀疏图浪费空间"
            },
            "邻接表": {
                "定义": "每个节点维护邻居列表",
                "空间复杂度": "O(n + m)",
                "优点": "节省空间，适合稀疏图",
                "缺点": "查询边存在性较慢"
            },
            "边列表": {
                "定义": "所有边的列表",
                "空间复杂度": "O(m)",
                "优点": "简单直接",
                "缺点": "查询效率低"
            },
            "关联矩阵": {
                "定义": "B[i,j] = 1 if 节点i与边j相连",
                "空间复杂度": "O(nm)",
                "优点": "理论分析方便",
                "缺点": "空间开销大"
            }
        }
        
        for rep, details in representations.items():
            print(f"{rep}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 示例图的不同表示
        self.demonstrate_representations()
        
        return representations
    
    def demonstrate_representations(self):
        """演示图的不同表示"""
        print("=== 图表示示例 ===")
        
        # 创建示例图
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
        
        # 邻接矩阵
        adj_matrix = nx.adjacency_matrix(G).todense()
        print("邻接矩阵:")
        print(adj_matrix)
        print()
        
        # 邻接表
        print("邻接表:")
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            print(f"节点 {node}: {neighbors}")
        print()
        
        # 边列表
        print("边列表:")
        edge_list = list(G.edges())
        print(edge_list)
        print()
        
        # 可视化不同表示的性能比较
        self.visualize_representation_comparison()
        
        return adj_matrix, edge_list
    
    def visualize_representation_comparison(self):
        """可视化表示方法的性能比较"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 空间复杂度比较
        node_counts = np.array([10, 50, 100, 500, 1000])
        edge_ratios = [0.1, 0.5]  # 稀疏图和稠密图
        
        for i, ratio in enumerate(edge_ratios):
            edge_counts = ratio * node_counts * (node_counts - 1) / 2
            
            adj_matrix_space = node_counts ** 2
            adj_list_space = node_counts + edge_counts
            edge_list_space = edge_counts
            
            ax = axes[i]
            ax.loglog(node_counts, adj_matrix_space, 'o-', label='邻接矩阵', linewidth=2)
            ax.loglog(node_counts, adj_list_space, 's-', label='邻接表', linewidth=2)
            ax.loglog(node_counts, edge_list_space, '^-', label='边列表', linewidth=2)
            
            ax.set_xlabel('节点数量')
            ax.set_ylabel('空间复杂度')
            ax.set_title(f'空间复杂度比较 (边密度={ratio})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def graph_properties(self):
        """图的基本性质"""
        print("=== 图的基本性质 ===")
        
        properties = {
            "度 (Degree)": {
                "定义": "deg(v) = |{u : (u,v) ∈ E}|",
                "含义": "节点连接的边数",
                "性质": "Σ deg(v) = 2|E| (握手定理)",
                "类型": "入度、出度（有向图）"
            },
            "路径 (Path)": {
                "定义": "节点序列，相邻节点间有边",
                "简单路径": "不重复访问节点",
                "距离": "最短路径长度",
                "直径": "图中最大距离"
            },
            "连通性": {
                "连通图": "任意两节点间存在路径",
                "连通分量": "最大连通子图",
                "强连通": "有向图中任意两节点相互可达",
                "弱连通": "有向图忽略方向后连通"
            },
            "聚类系数": {
                "定义": "C(v) = 2E(N(v)) / (deg(v)(deg(v)-1))",
                "含义": "邻居间连接密度",
                "全局": "所有节点的平均聚类系数",
                "应用": "社交网络分析"
            }
        }
        
        for prop, details in properties.items():
            print(f"{prop}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {details}")
            print()
        
        # 计算示例图的性质
        self.compute_graph_properties()
        
        return properties
    
    def compute_graph_properties(self):
        """计算示例图的性质"""
        print("=== 图性质计算示例 ===")
        
        # 创建示例图
        G = nx.karate_club_graph()  # 著名的空手道俱乐部图
        
        # 基本统计
        print(f"节点数: {G.number_of_nodes()}")
        print(f"边数: {G.number_of_edges()}")
        print(f"平均度数: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
        print()
        
        # 度分布
        degrees = [G.degree(n) for n in G.nodes()]
        print(f"度数分布: 最小={min(degrees)}, 最大={max(degrees)}, 平均={np.mean(degrees):.2f}")
        print()
        
        # 路径和距离
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            print(f"平均最短路径长度: {avg_path_length:.2f}")
            print(f"直径: {diameter}")
        print()
        
        # 聚类系数
        avg_clustering = nx.average_clustering(G)
        print(f"平均聚类系数: {avg_clustering:.3f}")
        print()
        
        # 可视化图性质
        self.visualize_graph_properties(G, degrees)
        
        return G
    
    def visualize_graph_properties(self, G, degrees):
        """可视化图性质"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 图结构
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=axes[0, 0], with_labels=True, 
                node_color='lightblue', node_size=100, font_size=8)
        axes[0, 0].set_title('空手道俱乐部图')
        
        # 2. 度分布
        degree_counts = np.bincount(degrees)
        axes[0, 1].bar(range(len(degree_counts)), degree_counts, alpha=0.7)
        axes[0, 1].set_xlabel('度数')
        axes[0, 1].set_ylabel('节点数量')
        axes[0, 1].set_title('度分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 聚类系数分布
        clustering_coeffs = list(nx.clustering(G).values())
        axes[1, 0].hist(clustering_coeffs, bins=15, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('聚类系数')
        axes[1, 0].set_ylabel('节点数量')
        axes[1, 0].set_title('聚类系数分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 度-聚类系数关系
        node_degrees = [G.degree(n) for n in G.nodes()]
        axes[1, 1].scatter(node_degrees, clustering_coeffs, alpha=0.7, color='red')
        axes[1, 1].set_xlabel('度数')
        axes[1, 1].set_ylabel('聚类系数')
        axes[1, 1].set_title('度数 vs 聚类系数')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class SpectralGraphTheory:
    """谱图理论"""
    
    def __init__(self):
        pass
    
    def laplacian_matrices(self):
        """拉普拉斯矩阵"""
        print("=== 拉普拉斯矩阵 ===")
        
        print("度矩阵 D:")
        print("D[i,i] = deg(i), D[i,j] = 0 for i ≠ j")
        print()
        
        print("拉普拉斯矩阵变体:")
        laplacian_types = {
            "未归一化拉普拉斯": {
                "定义": "L = D - A",
                "性质": "半正定，最小特征值为0",
                "特征向量": "常向量是0特征值对应的特征向量",
                "应用": "连通性分析"
            },
            "对称归一化拉普拉斯": {
                "定义": "L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2)",
                "性质": "对称矩阵，特征值在[0,2]",
                "优点": "数值稳定性好",
                "应用": "谱聚类"
            },
            "随机游走拉普拉斯": {
                "定义": "L_rw = D^(-1) L = I - D^(-1) A",
                "性质": "非对称，但与L_sym有相同特征值",
                "解释": "随机游走转移矩阵",
                "应用": "图上的扩散过程"
            }
        }
        
        for lap_type, details in laplacian_types.items():
            print(f"{lap_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 计算和可视化拉普拉斯矩阵
        self.compute_laplacian_matrices()
        
        return laplacian_types
    
    def compute_laplacian_matrices(self):
        """计算拉普拉斯矩阵"""
        print("=== 拉普拉斯矩阵计算示例 ===")
        
        # 创建简单图
        G = nx.cycle_graph(6)  # 6节点环图
        
        # 邻接矩阵和度矩阵
        A = nx.adjacency_matrix(G).todense()
        degrees = np.array([G.degree(n) for n in G.nodes()])
        D = np.diag(degrees)
        
        # 不同类型的拉普拉斯矩阵
        L = D - A  # 未归一化
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt  # 对称归一化
        D_inv = np.diag(1.0 / degrees)
        L_rw = D_inv @ L  # 随机游走
        
        print("邻接矩阵 A:")
        print(A.astype(int))
        print("\n度矩阵 D:")
        print(D.astype(int))
        print("\n未归一化拉普拉斯 L:")
        print(L.astype(int))
        print("\n对称归一化拉普拉斯 L_sym:")
        print(np.round(L_sym, 3))
        
        # 计算特征值和特征向量
        eigenvals_L = np.real(eigvals(L))
        eigenvals_L_sym = np.real(eigvals(L_sym))
        
        print(f"\nL的特征值: {np.round(np.sort(eigenvals_L), 3)}")
        print(f"L_sym的特征值: {np.round(np.sort(eigenvals_L_sym), 3)}")
        
        # 可视化
        self.visualize_laplacian_spectrum(G, eigenvals_L, eigenvals_L_sym)
        
        return L, L_sym, L_rw
    
    def visualize_laplacian_spectrum(self, G, eigenvals_L, eigenvals_L_sym):
        """可视化拉普拉斯谱"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 图结构
        pos = nx.circular_layout(G)
        nx.draw(G, pos, ax=axes[0], with_labels=True,
                node_color='lightblue', node_size=500, font_size=12)
        axes[0].set_title('6节点环图')
        
        # 2. 未归一化拉普拉斯谱
        axes[1].bar(range(len(eigenvals_L)), np.sort(eigenvals_L), 
                   alpha=0.7, color='red')
        axes[1].set_xlabel('特征值索引')
        axes[1].set_ylabel('特征值')
        axes[1].set_title('未归一化拉普拉斯谱')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 对称归一化拉普拉斯谱
        axes[2].bar(range(len(eigenvals_L_sym)), np.sort(eigenvals_L_sym),
                   alpha=0.7, color='blue')
        axes[2].set_xlabel('特征值索引')
        axes[2].set_ylabel('特征值')
        axes[2].set_title('对称归一化拉普拉斯谱')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def spectral_properties(self):
        """谱性质"""
        print("=== 图的谱性质 ===")
        
        properties = {
            "代数连通度": {
                "定义": "拉普拉斯矩阵第二小特征值 λ₂",
                "含义": "图的连通性强度",
                "性质": "λ₂ > 0 当且仅当图连通",
                "应用": "图的分割、聚类"
            },
            "Fiedler向量": {
                "定义": "λ₂对应的特征向量",
                "性质": "节点按该向量值排序可实现二分",
                "应用": "谱聚类的基础",
                "意义": "图的天然二分结构"
            },
            "谱间隙": {
                "定义": "λ₂ - λ₁ 或 λₖ₊₁ - λₖ",
                "含义": "聚类结构的强度",
                "大间隙": "明显的聚类结构",
                "应用": "确定聚类数量"
            },
            "切比雪夫常数": {
                "定义": "与最大特征值相关",
                "应用": "随机游走收敛速度",
                "意义": "图的扩展性质",
                "计算": "扩散过程分析"
            }
        }
        
        for prop, details in properties.items():
            print(f"{prop}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 演示不同图的谱性质
        self.demonstrate_spectral_properties()
        
        return properties
    
    def demonstrate_spectral_properties(self):
        """演示不同图的谱性质"""
        # 创建不同类型的图
        graphs = {
            "路径图": nx.path_graph(10),
            "环图": nx.cycle_graph(10),
            "完全图": nx.complete_graph(10),
            "星图": nx.star_graph(9)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, G) in enumerate(graphs.items()):
            # 计算拉普拉斯矩阵
            L = nx.normalized_laplacian_matrix(G).todense()
            eigenvals = np.real(eigvals(L))
            eigenvals = np.sort(eigenvals)
            
            # 绘制谱
            axes[i].bar(range(len(eigenvals)), eigenvals, alpha=0.7)
            axes[i].set_title(f'{name}\nλ₂ = {eigenvals[1]:.3f}')
            axes[i].set_xlabel('特征值索引')
            axes[i].set_ylabel('特征值')
            axes[i].grid(True, alpha=0.3)
            
            print(f"{name}: 代数连通度 λ₂ = {eigenvals[1]:.3f}")
        
        plt.tight_layout()
        plt.show()

class GraphNeuralNetworkFoundations:
    """图神经网络基础"""
    
    def __init__(self):
        pass
    
    def message_passing_framework(self):
        """消息传递框架"""
        print("=== 消息传递框架 ===")
        
        print("基本思想:")
        print("- 节点通过边传递消息")
        print("- 聚合邻居信息更新节点表示")
        print("- 迭代多轮获得更大感受野")
        print()
        
        print("通用消息传递框架:")
        print("1. 消息计算: m_{ij}^{(l)} = M^{(l)}(h_i^{(l)}, h_j^{(l)}, e_{ij})")
        print("2. 消息聚合: m_i^{(l)} = AGG({m_{ij}^{(l)} : j ∈ N(i)})")
        print("3. 节点更新: h_i^{(l+1)} = U^{(l)}(h_i^{(l)}, m_i^{(l)})")
        print()
        
        print("关键组件:")
        components = {
            "消息函数 M": {
                "作用": "计算节点间传递的消息",
                "输入": "源节点、目标节点、边特征",
                "输出": "消息向量",
                "例子": "神经网络、线性变换"
            },
            "聚合函数 AGG": {
                "作用": "聚合邻居消息",
                "常见函数": "求和、均值、最大值、注意力",
                "性质": "置换不变性",
                "选择": "影响表达能力"
            },
            "更新函数 U": {
                "作用": "更新节点表示",
                "输入": "旧表示、聚合消息",
                "输出": "新节点表示",
                "实现": "RNN、MLP、门控机制"
            }
        }
        
        for comp, details in components.items():
            print(f"{comp}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化消息传递过程
        self.visualize_message_passing()
        
        return components
    
    def visualize_message_passing(self):
        """可视化消息传递过程"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 创建简单图
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)])
        pos = nx.spring_layout(G, seed=42)
        
        # 1. 初始状态
        nx.draw(G, pos, ax=axes[0], with_labels=True,
                node_color='lightblue', node_size=800, font_size=14)
        axes[0].set_title('第0层：初始节点特征')
        
        # 2. 消息传递第1层
        # 突出显示节点1的邻居
        node_colors = ['orange' if n in [0, 2, 3] else 'lightblue' for n in G.nodes()]
        nx.draw(G, pos, ax=axes[1], with_labels=True,
                node_color=node_colors, node_size=800, font_size=14)
        
        # 添加消息箭头
        for neighbor in [0, 2, 3]:
            axes[1].annotate('', xy=pos[1], xytext=pos[neighbor],
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        axes[1].set_title('第1层：节点1接收邻居消息')
        
        # 3. 更新后状态
        node_colors = ['lightgreen' if n == 1 else 'lightblue' for n in G.nodes()]
        nx.draw(G, pos, ax=axes[2], with_labels=True,
                node_color=node_colors, node_size=800, font_size=14)
        axes[2].set_title('第1层：节点1更新后的表示')
        
        plt.tight_layout()
        plt.show()
    
    def gnn_architectures(self):
        """GNN架构"""
        print("=== 主要GNN架构 ===")
        
        architectures = {
            "GCN (Graph Convolutional Network)": {
                "核心思想": "谱图卷积的一阶近似",
                "消息传递": "h_i^{(l+1)} = σ(Σ_{j∈N(i)∪{i}} (1/√(d_i d_j)) W^{(l)} h_j^{(l)})",
                "特点": "简单高效，归一化邻接矩阵",
                "限制": "只使用节点度信息归一化"
            },
            "GraphSAGE": {
                "核心思想": "采样和聚合邻居信息",
                "聚合函数": "Mean, LSTM, Pool, Attention",
                "特点": "支持大图，归纳学习",
                "优势": "可处理未见过的节点"
            },
            "GAT (Graph Attention Network)": {
                "核心思想": "基于注意力机制的消息传递",
                "注意力": "α_{ij} = softmax(LeakyReLU(a^T[W h_i || W h_j]))",
                "特点": "自适应权重，多头注意力",
                "优势": "不需要预先知道图结构重要性"
            },
            "GIN (Graph Isomorphism Network)": {
                "核心思想": "最大化表达能力",
                "更新规则": "h_i^{(l+1)} = MLP((1+ε)h_i^{(l)} + Σ_{j∈N(i)} h_j^{(l)})",
                "理论保证": "与WL测试等价",
                "特点": "理论上最强的表达能力"
            },
            "MPNN (Message Passing Neural Network)": {
                "核心思想": "统一的消息传递框架",
                "通用性": "包含大多数GNN作为特例",
                "组件": "可自定义消息、聚合、更新函数",
                "意义": "理论分析的统一框架"
            }
        }
        
        for arch, details in architectures.items():
            print(f"{arch}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 比较不同架构的特点
        self.compare_gnn_architectures()
        
        return architectures
    
    def compare_gnn_architectures(self):
        """比较GNN架构"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 表达能力比较
        models = ['GCN', 'GraphSAGE', 'GAT', 'GIN']
        expressiveness = [3, 4, 4, 5]  # 相对表达能力评分
        computational_cost = [2, 3, 4, 3]  # 计算复杂度
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, expressiveness, width, 
                              label='表达能力', alpha=0.7, color='blue')
        bars2 = axes[0, 0].bar(x + width/2, computational_cost, width,
                              label='计算复杂度', alpha=0.7, color='red')
        
        axes[0, 0].set_xlabel('GNN模型')
        axes[0, 0].set_ylabel('评分')
        axes[0, 0].set_title('表达能力 vs 计算复杂度')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 适用场景
        scenarios = ['节点分类', '图分类', '链接预测', '图生成']
        gcn_scores = [5, 3, 4, 2]
        gat_scores = [5, 4, 4, 3]
        gin_scores = [4, 5, 3, 4]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        axes[0, 1].bar(x - width, gcn_scores, width, label='GCN', alpha=0.7)
        axes[0, 1].bar(x, gat_scores, width, label='GAT', alpha=0.7)
        axes[0, 1].bar(x + width, gin_scores, width, label='GIN', alpha=0.7)
        
        axes[0, 1].set_xlabel('应用场景')
        axes[0, 1].set_ylabel('适用性评分')
        axes[0, 1].set_title('不同模型的适用场景')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 聚合函数比较
        aggregation_methods = ['求和', '均值', '最大值', '注意力', 'LSTM']
        permutation_invariant = [1, 1, 1, 1, 0]  # 是否置换不变
        differentiable = [1, 1, 1, 1, 1]  # 是否可微
        expressive_power = [3, 2, 2, 5, 4]  # 表达能力
        
        x = np.arange(len(aggregation_methods))
        axes[1, 0].bar(x, expressive_power, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('聚合方法')
        axes[1, 0].set_ylabel('表达能力')
        axes[1, 0].set_title('聚合函数表达能力比较')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(aggregation_methods, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 层数与性能关系
        layers = np.arange(1, 9)
        performance = [60, 75, 85, 88, 87, 85, 82, 78]  # 模拟性能曲线
        over_smoothing = [0, 5, 10, 15, 25, 40, 60, 80]  # 过平滑程度
        
        ax_perf = axes[1, 1]
        ax_smooth = ax_perf.twinx()
        
        line1 = ax_perf.plot(layers, performance, 'b-o', label='性能', linewidth=2)
        line2 = ax_smooth.plot(layers, over_smoothing, 'r-s', label='过平滑', linewidth=2)
        
        ax_perf.set_xlabel('网络层数')
        ax_perf.set_ylabel('性能 (%)', color='blue')
        ax_smooth.set_ylabel('过平滑程度', color='red')
        ax_perf.set_title('层数对性能的影响')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_perf.legend(lines, labels, loc='center right')
        
        ax_perf.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def comprehensive_gnn_summary():
    """图神经网络理论综合总结"""
    print("=== 图神经网络理论综合总结 ===")
    
    summary = {
        "基础理论": {
            "图论基础": "图的表示、性质、连通性、度分布",
            "谱图理论": "拉普拉斯矩阵、特征值分解、谱性质",
            "消息传递": "节点间信息传播的统一框架",
            "表达能力": "WL测试、图同构、理论限制"
        },
        
        "核心架构": {
            "GCN": "谱图卷积，局部平均聚合",
            "GraphSAGE": "采样聚合，支持大图",
            "GAT": "注意力机制，自适应权重",
            "GIN": "最大表达能力，理论保证",
            "MPNN": "统一框架，可定制组件"
        },
        
        "关键技术": {
            "聚合函数": "求和、均值、最大值、注意力",
            "归一化": "度归一化、批归一化、层归一化",
            "正则化": "Dropout、DropEdge、DropNode",
            "残差连接": "缓解过平滑、加深网络"
        },
        
        "应用任务": {
            "节点分类": "社交网络分析、蛋白质功能预测",
            "图分类": "分子性质预测、程序分析",
            "链接预测": "推荐系统、知识图谱补全",
            "图生成": "药物发现、分子设计"
        },
        
        "挑战与解决": {
            "过平滑": "随层数增加节点表示趋于相同",
            "表达能力": "无法区分某些图结构",
            "大图扩展": "内存和计算复杂度问题",
            "动态图": "时间演化的图结构建模"
        },
        
        "前沿发展": {
            "图Transformer": "全局注意力机制",
            "图预训练": "自监督学习、迁移学习",
            "可解释性": "注意力可视化、特征重要性",
            "几何深度学习": "统一框架、群不变性"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("图神经网络理论指南加载完成！")
```

## 参考文献 📚

- Hamilton (2020): "Graph Representation Learning"
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Veličković et al. (2018): "Graph Attention Networks"
- Xu et al. (2019): "How Powerful are Graph Neural Networks?"
- Gilmer et al. (2017): "Neural Message Passing for Quantum Chemistry"

## 下一步学习
- [图深度学习](../graph_deep_learning.md) - 高级图神经网络
- [知识图谱](../../nlp/knowledge_graphs.md) - 结构化知识表示
- [几何深度学习](../geometric_deep_learning.md) - 统一理论框架