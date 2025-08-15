# 因果推理理论：从相关到因果的跨越 🔗

深入理解因果推理的核心理论，从图模型到实验设计。

## 1. 因果推理基础 🎯

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class CausalInferenceBasics:
    """因果推理基础概念"""
    
    def __init__(self):
        self.concepts = {}
    
    def correlation_vs_causation(self):
        """相关性vs因果性"""
        print("=== 相关性 vs 因果性 ===")
        
        print("相关性 (Correlation):")
        print("- 定义: 两个变量间的统计关联")
        print("- 度量: 相关系数、互信息等")
        print("- 性质: 对称、易于测量")
        print("- 局限: 不能确定因果方向")
        print()
        
        print("因果性 (Causation):")
        print("- 定义: 一个变量对另一个变量的因果影响")
        print("- 特征: 非对称、有方向性")
        print("- 条件: 时间先后、机制可信")
        print("- 重要性: 预测干预效果")
        print()
        
        print("经典误区:")
        fallacies = {
            "虚假相关": {
                "描述": "第三变量导致的相关性",
                "例子": "冰淇淋销量与溺水死亡率",
                "真相": "温度是共同原因",
                "教训": "需要控制混杂变量"
            },
            "反向因果": {
                "描述": "因果方向判断错误",
                "例子": "财富与健康的关系",
                "问题": "是财富导致健康还是相反？",
                "解决": "工具变量、自然实验"
            },
            "选择偏差": {
                "描述": "样本选择导致的偏差",
                "例子": "大学教育与收入",
                "偏差": "能力等不可观测因素",
                "方法": "随机化、回归不连续"
            }
        }
        
        for fallacy, details in fallacies.items():
            print(f"{fallacy}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化相关性与因果性
        self.visualize_correlation_causation()
        
        return fallacies
    
    def visualize_correlation_causation(self):
        """可视化相关性与因果性的区别"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        np.random.seed(42)
        n = 1000
        
        # 1. 真实因果关系: X -> Y
        X1 = np.random.normal(0, 1, n)
        Y1 = 2 * X1 + np.random.normal(0, 0.5, n)
        
        axes[0, 0].scatter(X1, Y1, alpha=0.6, s=10)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title(f'真实因果: X→Y\n相关系数={np.corrcoef(X1, Y1)[0,1]:.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加回归线
        z = np.polyfit(X1, Y1, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(X1, p(X1), "r--", alpha=0.8)
        
        # 2. 虚假相关: Z -> X, Z -> Y
        Z = np.random.normal(0, 1, n)
        X2 = Z + np.random.normal(0, 0.3, n)
        Y2 = -Z + np.random.normal(0, 0.3, n)
        
        axes[0, 1].scatter(X2, Y2, alpha=0.6, s=10, color='orange')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].set_title(f'虚假相关: Z→X, Z→Y\n相关系数={np.corrcoef(X2, Y2)[0,1]:.3f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        z = np.polyfit(X2, Y2, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(X2, p(X2), "r--", alpha=0.8)
        
        # 3. 碰撞器偏差: X -> Z <- Y
        X3 = np.random.normal(0, 1, n)
        Y3 = np.random.normal(0, 1, n)  # X和Y独立
        Z3 = X3 + Y3 + np.random.normal(0, 0.1, n)
        
        # 条件于Z的子样本
        mask = Z3 > np.percentile(Z3, 80)  # 选择Z值较大的样本
        X3_cond = X3[mask]
        Y3_cond = Y3[mask]
        
        axes[0, 2].scatter(X3_cond, Y3_cond, alpha=0.6, s=10, color='green')
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        axes[0, 2].set_title(f'碰撞器偏差: 条件于Z\n相关系数={np.corrcoef(X3_cond, Y3_cond)[0,1]:.3f}')
        axes[0, 2].grid(True, alpha=0.3)
        
        z = np.polyfit(X3_cond, Y3_cond, 1)
        p = np.poly1d(z)
        axes[0, 2].plot(X3_cond, p(X3_cond), "r--", alpha=0.8)
        
        # 4-6. 对应的因果图
        causal_structures = [
            ("X → Y", [(0, 1)], ['X', 'Y']),
            ("Z → X, Z → Y", [(0, 1), (0, 2)], ['Z', 'X', 'Y']),
            ("X → Z ← Y", [(0, 2), (1, 2)], ['X', 'Y', 'Z'])
        ]
        
        for i, (title, edges, labels) in enumerate(causal_structures):
            ax = axes[1, i]
            G = nx.DiGraph()
            G.add_edges_from(edges)
            
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, with_labels=True, labels={i: labels[i] for i in range(len(labels))},
                   node_color='lightblue', node_size=1000, font_size=12, arrows=True, arrowsize=20)
            ax.set_title(f'因果结构: {title}')
        
        plt.tight_layout()
        plt.show()
    
    def causal_hierarchy(self):
        """因果层次结构"""
        print("=== 因果层次结构 (Ladder of Causation) ===")
        
        print("Judea Pearl的因果层次:")
        hierarchy = {
            "第一层: 关联 (Association)": {
                "问题": "观察到的相关性是什么？",
                "符号": "P(y|x) - 条件概率",
                "例子": "服药的病人康复率如何？",
                "数据": "观察数据",
                "方法": "统计关联、机器学习"
            },
            "第二层: 干预 (Intervention)": {
                "问题": "如果我们干预会怎样？",
                "符号": "P(y|do(x)) - 干预分布",
                "例子": "如果给病人服药会如何？",
                "数据": "实验数据或因果模型",
                "方法": "随机实验、因果图推断"
            },
            "第三层: 反事实 (Counterfactuals)": {
                "问题": "如果当时做了不同选择会怎样？",
                "符号": "P(y_x|x',y') - 反事实概率",
                "例子": "如果这个康复的病人没服药会如何？",
                "数据": "需要因果模型和个体信息",
                "方法": "结构因果模型、反事实推理"
            }
        }
        
        for level, details in hierarchy.items():
            print(f"{level}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化层次结构
        self.visualize_causal_hierarchy()
        
        return hierarchy
    
    def visualize_causal_hierarchy(self):
        """可视化因果层次结构"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 创建层次图
        levels = ['关联\n(Association)', '干预\n(Intervention)', '反事实\n(Counterfactuals)']
        questions = ['看到了什么？', '如果做了会怎样？', '如果当时不同会怎样？']
        examples = ['P(y|x)', 'P(y|do(x))', 'P(y_x|x\',y\')']
        
        # 绘制层次结构
        y_positions = [0.2, 0.5, 0.8]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        for i, (level, question, example, y_pos, color) in enumerate(zip(levels, questions, examples, y_positions, colors)):
            # 绘制层次框
            rect = plt.Rectangle((0.1, y_pos-0.08), 0.8, 0.16, 
                               facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # 添加文本
            ax.text(0.15, y_pos+0.03, level, fontsize=14, weight='bold')
            ax.text(0.15, y_pos-0.02, question, fontsize=12)
            ax.text(0.15, y_pos-0.06, example, fontsize=11, style='italic')
            
            # 添加层次标签
            ax.text(0.05, y_pos, f'第{i+1}层', fontsize=12, weight='bold', 
                   ha='right', va='center')
        
        # 添加箭头表示层次递进
        for i in range(len(y_positions)-1):
            ax.annotate('', xy=(0.5, y_positions[i+1]-0.08), xytext=(0.5, y_positions[i]+0.08),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('因果推理的层次结构', fontsize=16, weight='bold')
        ax.axis('off')
        
        # 添加说明
        ax.text(0.95, 0.1, '复杂度和洞察力递增', fontsize=12, 
               ha='right', style='italic', color='red')
        
        plt.tight_layout()
        plt.show()

class CausalGraphicalModels:
    """因果图模型"""
    
    def __init__(self):
        pass
    
    def directed_acyclic_graphs(self):
        """有向无环图(DAG)"""
        print("=== 有向无环图 (DAG) ===")
        
        print("基本概念:")
        print("- 节点: 变量")
        print("- 有向边: 直接因果关系")
        print("- 无环: 不存在因果循环")
        print("- 路径: 节点间的连接序列")
        print()
        
        concepts = {
            "父节点": "直接指向某节点的节点",
            "子节点": "被某节点直接指向的节点", 
            "祖先": "通过有向路径能到达某节点的节点",
            "后代": "从某节点通过有向路径能到达的节点",
            "根节点": "没有父节点的节点",
            "叶节点": "没有子节点的节点"
        }
        
        for concept, definition in concepts.items():
            print(f"{concept}: {definition}")
        print()
        
        print("路径类型:")
        path_types = {
            "有向路径": {
                "定义": "所有边都指向同一方向",
                "意义": "因果链",
                "例子": "X → Y → Z"
            },
            "后门路径": {
                "定义": "从X到Y且以指向X的边开始",
                "意义": "混杂路径",
                "例子": "X ← Z → Y"
            },
            "前门路径": {
                "定义": "从X到Y且以从X出发的边开始",
                "意义": "中介路径",
                "例子": "X → M → Y"
            }
        }
        
        for path_type, details in path_types.items():
            print(f"{path_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化DAG概念
        self.visualize_dag_concepts()
        
        return path_types
    
    def visualize_dag_concepts(self):
        """可视化DAG概念"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 基本DAG结构
        G1 = nx.DiGraph()
        G1.add_edges_from([('X', 'Y'), ('Z', 'X'), ('Z', 'Y'), ('W', 'Z')])
        
        pos1 = {'W': (0, 1), 'Z': (1, 1), 'X': (0, 0), 'Y': (2, 0)}
        nx.draw(G1, pos1, ax=axes[0, 0], with_labels=True, 
                node_color='lightblue', node_size=1000, font_size=12, 
                arrows=True, arrowsize=20)
        axes[0, 0].set_title('基本DAG结构')
        
        # 2. 三种基本结构
        structures = [
            ("链式: X→Y→Z", [('X', 'Y'), ('Y', 'Z')]),
            ("分叉: X←Y→Z", [('Y', 'X'), ('Y', 'Z')]),
            ("碰撞: X→Y←Z", [('X', 'Y'), ('Z', 'Y')])
        ]
        
        for i, (title, edges) in enumerate(structures):
            G = nx.DiGraph()
            G.add_edges_from(edges)
            
            if i == 0:  # 链式
                pos = {'X': (0, 0), 'Y': (1, 0), 'Z': (2, 0)}
            elif i == 1:  # 分叉
                pos = {'Y': (1, 1), 'X': (0, 0), 'Z': (2, 0)}
            else:  # 碰撞
                pos = {'X': (0, 1), 'Z': (2, 1), 'Y': (1, 0)}
            
            if i == 0:
                ax = axes[0, 1]
            else:
                ax = axes[1, i-1]
                
            nx.draw(G, pos, ax=ax, with_labels=True,
                   node_color='lightgreen', node_size=800, font_size=12,
                   arrows=True, arrowsize=20)
            ax.set_title(title)
        
        # 删除空的子图
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.show()
    
    def d_separation(self):
        """d-分离"""
        print("=== d-分离 (d-separation) ===")
        
        print("定义:")
        print("给定观察集合Z，路径P在以下情况被阻断:")
        print("1. 链式或分叉结构中，中间节点在Z中")
        print("2. 碰撞结构中，碰撞节点及其后代都不在Z中")
        print()
        
        print("d-分离规则:")
        rules = {
            "链式 X→Y→Z": {
                "结构": "X通过Y影响Z",
                "条件于Y": "阻断路径，X⊥Z|Y",
                "不条件": "路径开放，X和Z相关",
                "含义": "控制中介变量阻断因果链"
            },
            "分叉 X←Y→Z": {
                "结构": "Y是X和Z的共同原因",
                "条件于Y": "阻断路径，X⊥Z|Y", 
                "不条件": "路径开放，X和Z相关",
                "含义": "控制混杂变量消除虚假相关"
            },
            "碰撞 X→Y←Z": {
                "结构": "Y是X和Z的共同结果",
                "条件于Y": "开放路径，X和Z相关",
                "不条件": "路径阻断，X⊥Z",
                "含义": "控制碰撞变量产生虚假相关"
            }
        }
        
        for structure, details in rules.items():
            print(f"{structure}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 演示d-分离
        self.demonstrate_d_separation()
        
        return rules
    
    def demonstrate_d_separation(self):
        """演示d-分离概念"""
        np.random.seed(42)
        n = 1000
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 链式结构: X → Y → Z
        X1 = np.random.normal(0, 1, n)
        Y1 = X1 + np.random.normal(0, 0.3, n)
        Z1 = Y1 + np.random.normal(0, 0.3, n)
        
        # 不控制Y时的X-Z关系
        axes[0, 0].scatter(X1, Z1, alpha=0.6, s=10)
        corr_xz = np.corrcoef(X1, Z1)[0, 1]
        axes[0, 0].set_title(f'链式: X→Y→Z\n不控制Y: r(X,Z)={corr_xz:.3f}')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Z')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 控制Y后的残差关系
        from sklearn.linear_model import LinearRegression
        reg_xy = LinearRegression().fit(X1.reshape(-1, 1), Y1)
        reg_zy = LinearRegression().fit(Y1.reshape(-1, 1), Z1)
        
        X_resid = X1 - reg_xy.predict(X1.reshape(-1, 1))
        Z_resid = Z1 - reg_zy.predict(Y1.reshape(-1, 1))
        
        axes[1, 0].scatter(X_resid, Z_resid, alpha=0.6, s=10, color='orange')
        corr_resid = np.corrcoef(X_resid, Z_resid)[0, 1]
        axes[1, 0].set_title(f'控制Y后残差\nr(X,Z|Y)={corr_resid:.3f}')
        axes[1, 0].set_xlabel('X残差')
        axes[1, 0].set_ylabel('Z残差')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2. 分叉结构: X ← Y → Z
        Y2 = np.random.normal(0, 1, n)
        X2 = Y2 + np.random.normal(0, 0.3, n)
        Z2 = Y2 + np.random.normal(0, 0.3, n)
        
        axes[0, 1].scatter(X2, Z2, alpha=0.6, s=10, color='green')
        corr_xz2 = np.corrcoef(X2, Z2)[0, 1]
        axes[0, 1].set_title(f'分叉: X←Y→Z\n不控制Y: r(X,Z)={corr_xz2:.3f}')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 控制Y后
        reg_xy2 = LinearRegression().fit(Y2.reshape(-1, 1), X2)
        reg_zy2 = LinearRegression().fit(Y2.reshape(-1, 1), Z2)
        
        X_resid2 = X2 - reg_xy2.predict(Y2.reshape(-1, 1))
        Z_resid2 = Z2 - reg_zy2.predict(Y2.reshape(-1, 1))
        
        axes[1, 1].scatter(X_resid2, Z_resid2, alpha=0.6, s=10, color='green')
        corr_resid2 = np.corrcoef(X_resid2, Z_resid2)[0, 1]
        axes[1, 1].set_title(f'控制Y后残差\nr(X,Z|Y)={corr_resid2:.3f}')
        axes[1, 1].set_xlabel('X残差')
        axes[1, 1].set_ylabel('Z残差')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 3. 碰撞结构: X → Y ← Z
        X3 = np.random.normal(0, 1, n)
        Z3 = np.random.normal(0, 1, n)
        Y3 = X3 + Z3 + np.random.normal(0, 0.3, n)
        
        axes[0, 2].scatter(X3, Z3, alpha=0.6, s=10, color='red')
        corr_xz3 = np.corrcoef(X3, Z3)[0, 1]
        axes[0, 2].set_title(f'碰撞: X→Y←Z\n不控制Y: r(X,Z)={corr_xz3:.3f}')
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Z')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 控制Y后（选择Y值较大的样本）
        mask = Y3 > np.percentile(Y3, 70)
        X3_cond = X3[mask]
        Z3_cond = Z3[mask]
        
        axes[1, 2].scatter(X3_cond, Z3_cond, alpha=0.6, s=10, color='red')
        corr_cond = np.corrcoef(X3_cond, Z3_cond)[0, 1]
        axes[1, 2].set_title(f'条件于Y(>70%分位)\nr(X,Z|Y)={corr_cond:.3f}')
        axes[1, 2].set_xlabel('X')
        axes[1, 2].set_ylabel('Z')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class CausalIdentification:
    """因果识别"""
    
    def __init__(self):
        pass
    
    def backdoor_criterion(self):
        """后门准则"""
        print("=== 后门准则 (Backdoor Criterion) ===")
        
        print("目标: 识别因果效应 P(Y|do(X))")
        print()
        
        print("后门准则条件:")
        print("变量集合Z满足后门准则，当且仅当:")
        print("1. Z不包含X的后代")
        print("2. Z阻断X到Y的所有后门路径")
        print()
        
        print("应用:")
        print("如果Z满足后门准则，则:")
        print("P(Y|do(X)) = Σ_z P(Y|X,Z=z)P(Z=z)")
        print("即：干预分布 = 调整公式")
        print()
        
        # 实现后门调整
        self.implement_backdoor_adjustment()
        
        return self.frontdoor_criterion()
    
    def implement_backdoor_adjustment(self):
        """实现后门调整"""
        print("=== 后门调整示例 ===")
        
        # 生成数据：Z → X, Z → Y, X → Y
        np.random.seed(42)
        n = 5000
        
        # 混杂变量Z
        Z = np.random.normal(0, 1, n)
        
        # 治疗变量X（受Z影响）
        X_prob = 1 / (1 + np.exp(-(Z + np.random.normal(0, 0.5, n))))
        X = np.random.binomial(1, X_prob, n)
        
        # 结果变量Y（受X和Z影响）
        Y = 2 * X + 1.5 * Z + np.random.normal(0, 0.5, n)
        
        print("数据生成过程:")
        print("Z ~ N(0,1)")
        print("X ~ Bernoulli(sigmoid(Z + noise))")
        print("Y = 2*X + 1.5*Z + noise")
        print("真实因果效应: E[Y|do(X=1)] - E[Y|do(X=0)] = 2")
        print()
        
        # 1. 朴素估计（有偏差）
        naive_effect = np.mean(Y[X==1]) - np.mean(Y[X==0])
        print(f"朴素估计: {naive_effect:.3f}")
        
        # 2. 后门调整估计
        # 按Z分层计算条件期望
        z_bins = np.linspace(Z.min(), Z.max(), 10)
        backdoor_effect = 0
        
        for i in range(len(z_bins)-1):
            z_mask = (Z >= z_bins[i]) & (Z < z_bins[i+1])
            if np.sum(z_mask) > 0:
                y1_mean = np.mean(Y[(X==1) & z_mask]) if np.sum((X==1) & z_mask) > 0 else 0
                y0_mean = np.mean(Y[(X==0) & z_mask]) if np.sum((X==0) & z_mask) > 0 else 0
                bin_effect = (y1_mean - y0_mean) * np.mean(z_mask)
                backdoor_effect += bin_effect
        
        print(f"后门调整估计: {backdoor_effect:.3f}")
        
        # 3. 回归调整
        from sklearn.linear_model import LinearRegression
        
        # 拟合Y ~ X + Z
        reg = LinearRegression()
        features = np.column_stack([X, Z])
        reg.fit(features, Y)
        
        regression_effect = reg.coef_[0]
        print(f"回归调整估计: {regression_effect:.3f}")
        
        # 可视化混杂效应
        self.visualize_confounding_adjustment(X, Y, Z)
        
        return naive_effect, backdoor_effect, regression_effect
    
    def visualize_confounding_adjustment(self, X, Y, Z):
        """可视化混杂调整"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 原始数据中的X-Y关系
        treated = X == 1
        control = X == 0
        
        axes[0].scatter(X[control], Y[control], alpha=0.6, label='X=0', color='blue', s=10)
        axes[0].scatter(X[treated], Y[treated], alpha=0.6, label='X=1', color='red', s=10)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('原始X-Y关系')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Z的分布差异
        axes[1].hist(Z[control], alpha=0.7, label='X=0', bins=30, density=True, color='blue')
        axes[1].hist(Z[treated], alpha=0.7, label='X=1', bins=30, density=True, color='red')
        axes[1].set_xlabel('Z (混杂变量)')
        axes[1].set_ylabel('密度')
        axes[1].set_title('不同X组的Z分布')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 按Z分层的效应
        z_bins = np.linspace(Z.min(), Z.max(), 5)
        bin_centers = []
        bin_effects = []
        
        for i in range(len(z_bins)-1):
            z_mask = (Z >= z_bins[i]) & (Z < z_bins[i+1])
            if np.sum(z_mask & (X==1)) > 10 and np.sum(z_mask & (X==0)) > 10:
                y1_mean = np.mean(Y[z_mask & (X==1)])
                y0_mean = np.mean(Y[z_mask & (X==0)])
                bin_centers.append((z_bins[i] + z_bins[i+1]) / 2)
                bin_effects.append(y1_mean - y0_mean)
        
        axes[2].bar(bin_centers, bin_effects, width=0.5, alpha=0.7, color='green')
        axes[2].axhline(y=2, color='red', linestyle='--', linewidth=2, label='真实效应=2')
        axes[2].set_xlabel('Z区间')
        axes[2].set_ylabel('X的因果效应')
        axes[2].set_title('不同Z层的因果效应')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def frontdoor_criterion(self):
        """前门准则"""
        print("\n=== 前门准则 (Frontdoor Criterion) ===")
        
        print("应用场景:")
        print("- 存在不可观测的混杂变量")
        print("- 但有可观测的中介变量")
        print("- 中介变量满足特定条件")
        print()
        
        print("前门准则条件:")
        print("变量集合M满足前门准则，当且仅当:")
        print("1. M拦截所有X到Y的有向路径")
        print("2. X到M没有后门路径")
        print("3. M满足相对于Y的后门准则（条件于X）")
        print()
        
        print("前门公式:")
        print("P(Y|do(X)) = Σ_m P(M=m|X) Σ_x' P(Y|X=x',M=m)P(X=x')")
        print()
        
        criterion_info = {
            "优势": "可处理不可观测混杂",
            "限制": "需要满足严格条件的中介变量",
            "应用": "相对较少，条件难满足",
            "例子": "广告→态度→购买，有不可观测的偏好"
        }
        
        for key, value in criterion_info.items():
            print(f"{key}: {value}")
        
        return criterion_info

def comprehensive_causal_inference_summary():
    """因果推理综合总结"""
    print("=== 因果推理理论综合总结 ===")
    
    summary = {
        "核心概念": {
            "因果vs相关": "方向性、机制性、干预效应",
            "因果层次": "关联、干预、反事实三层",
            "图模型": "DAG表示因果结构",
            "d-分离": "图中的条件独立关系"
        },
        
        "识别方法": {
            "后门调整": "控制混杂变量",
            "前门调整": "利用中介变量",
            "工具变量": "外生变异来源",
            "回归不连续": "阈值附近的准实验"
        },
        
        "实验设计": {
            "随机化实验": "金标准，内部效度高",
            "自然实验": "利用外生变异",
            "准实验": "倾向得分、匹配",
            "A/B测试": "在线实验平台"
        },
        
        "估计方法": {
            "回归调整": "线性假设下的控制",
            "匹配方法": "寻找相似对照组",
            "双重差分": "时间和组别的交互",
            "合成控制": "构造反事实对照"
        },
        
        "应用领域": {
            "经济学": "政策评估、市场分析",
            "医学": "药物效果、治疗方案",
            "社会科学": "教育政策、社会干预",
            "机器学习": "公平性、可解释性"
        },
        
        "挑战与限制": {
            "假设严格": "不可验证的识别假设",
            "外部效度": "结果的泛化能力",
            "复杂性": "多重治疗、时变混杂",
            "数据需求": "高质量数据要求"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("因果推理理论指南加载完成！")
```

## 参考文献 📚

- Pearl (2009): "Causality: Models, Reasoning and Inference"
- Hernán & Robins (2020): "Causal Inference: What If"
- Imbens & Rubin (2015): "Causal Inference for Statistics, Social, and Biomedical Sciences"
- Peters, Janzing & Schölkopf (2017): "Elements of Causal Inference"

## 下一步学习
- [实验设计](experimental_design.md) - 随机化实验方法
- [观察性研究](observational_studies.md) - 准实验设计
- [因果发现](causal_discovery.md) - 从数据学习因果结构