# 信息理论：机器学习的信息论基础 📡

深入理解信息理论在机器学习中的应用，从熵到互信息。

## 1. 信息论基础概念 💡

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class InformationTheoryBasics:
    """信息论基础"""
    
    def __init__(self):
        self.concepts = {}
    
    def entropy_concept(self):
        """熵的概念"""
        print("=== 信息熵 (Information Entropy) ===")
        
        print("Shannon熵定义:")
        print("H(X) = -Σᵢ P(xᵢ) log₂ P(xᵢ)")
        print()
        print("物理意义:")
        print("- 测量随机变量的不确定性")
        print("- 编码该变量所需的最少比特数")
        print("- 系统的信息含量")
        print()
        
        entropy_properties = {
            "非负性": "H(X) ≥ 0，当且仅当X确定时等号成立",
            "最大值": "H(X) ≤ log₂|𝒳|，均匀分布时达到最大",
            "可加性": "独立变量：H(X,Y) = H(X) + H(Y)",
            "凹函数": "熵是概率分布的凹函数"
        }
        
        for prop, desc in entropy_properties.items():
            print(f"{prop}: {desc}")
        
        # 熵的计算示例
        self.demonstrate_entropy()
        
        return entropy_properties
    
    def demonstrate_entropy(self):
        """演示熵的计算"""
        print("\n=== 熵计算示例 ===")
        
        # 不同概率分布的熵
        distributions = {
            "确定性": [1.0, 0.0, 0.0, 0.0],
            "偏斜分布": [0.7, 0.2, 0.08, 0.02],
            "均匀分布": [0.25, 0.25, 0.25, 0.25]
        }
        
        def calculate_entropy(probs):
            """计算熵"""
            probs = np.array(probs)
            probs = probs[probs > 0]  # 避免log(0)
            return -np.sum(probs * np.log2(probs))
        
        entropies = {}
        for name, probs in distributions.items():
            entropy = calculate_entropy(probs)
            entropies[name] = entropy
            print(f"{name}: H = {entropy:.3f} bits")
        
        # 可视化不同分布的熵
        self.visualize_entropy_distributions(distributions, entropies)
        
        return entropies
    
    def visualize_entropy_distributions(self, distributions, entropies):
        """可视化不同分布的熵"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 绘制概率分布
        for i, (name, probs) in enumerate(distributions.items()):
            ax = axes[0, i] if i < 2 else axes[1, 0]
            
            categories = [f'X{j+1}' for j in range(len(probs))]
            bars = ax.bar(categories, probs, alpha=0.7, 
                         color=plt.cm.viridis(i/len(distributions)))
            
            # 添加数值标签
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.2f}', ha='center', va='bottom')
            
            ax.set_title(f'{name}\nH = {entropies[name]:.3f} bits')
            ax.set_ylabel('概率')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
        
        # 熵值比较
        ax = axes[1, 1]
        names = list(entropies.keys())
        values = list(entropies.values())
        
        bars = ax.bar(names, values, alpha=0.7, color='orange')
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title('熵值比较')
        ax.set_ylabel('熵 (bits)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def conditional_entropy(self):
        """条件熵"""
        print("=== 条件熵 (Conditional Entropy) ===")
        
        print("定义:")
        print("H(Y|X) = Σₓ P(x) H(Y|X=x)")
        print("     = -Σₓᵧ P(x,y) log₂ P(y|x)")
        print()
        
        print("性质:")
        print("- H(Y|X) ≤ H(Y)，等号成立当且仅当X,Y独立")
        print("- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)")
        print("- 链式法则：H(X₁,...,Xₙ) = Σᵢ H(Xᵢ|X₁,...,Xᵢ₋₁)")
        print()
        
        # 条件熵实例
        self.demonstrate_conditional_entropy()
        
        return self.visualize_entropy_relationships()
    
    def demonstrate_conditional_entropy(self):
        """演示条件熵"""
        print("=== 条件熵计算示例 ===")
        
        # 创建联合分布表
        # X: 天气 (晴天=0, 雨天=1)
        # Y: 心情 (好=0, 坏=1)
        joint_prob = np.array([
            [0.4, 0.1],  # P(X=0,Y=0), P(X=0,Y=1) - 晴天
            [0.2, 0.3]   # P(X=1,Y=0), P(X=1,Y=1) - 雨天
        ])
        
        # 边际概率
        p_x = joint_prob.sum(axis=1)  # P(X)
        p_y = joint_prob.sum(axis=0)  # P(Y)
        
        # 条件概率 P(Y|X)
        p_y_given_x = joint_prob / p_x[:, np.newaxis]
        
        # 计算各种熵
        def entropy(probs):
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        
        H_X = entropy(p_x)
        H_Y = entropy(p_y)
        H_XY = entropy(joint_prob.flatten())
        
        # 条件熵 H(Y|X)
        H_Y_given_X = np.sum(p_x * [entropy(p_y_given_x[i]) for i in range(len(p_x))])
        
        print(f"H(X) = {H_X:.3f} bits")
        print(f"H(Y) = {H_Y:.3f} bits")
        print(f"H(X,Y) = {H_XY:.3f} bits")
        print(f"H(Y|X) = {H_Y_given_X:.3f} bits")
        print(f"验证链式法则: H(X,Y) = H(X) + H(Y|X) = {H_X + H_Y_given_X:.3f}")
        
        return joint_prob, p_x, p_y
    
    def visualize_entropy_relationships(self):
        """可视化熵的关系"""
        print("\n=== 熵关系图 ===")
        
        # 创建维恩图风格的熵关系图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制概念图
        from matplotlib.patches import Circle, Rectangle
        
        # H(X,Y) 总框
        rect = Rectangle((0, 0), 4, 3, linewidth=2, edgecolor='black', 
                        facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)
        
        # H(X) 圆
        circle_x = Circle((1.2, 1.5), 0.8, linewidth=2, edgecolor='red', 
                         facecolor='pink', alpha=0.5)
        ax.add_patch(circle_x)
        
        # H(Y) 圆
        circle_y = Circle((2.8, 1.5), 0.8, linewidth=2, edgecolor='blue', 
                         facecolor='lightcyan', alpha=0.5)
        ax.add_patch(circle_y)
        
        # 标注
        ax.text(0.8, 1.5, 'H(X|Y)', fontsize=12, ha='center', va='center')
        ax.text(2, 1.5, 'I(X;Y)', fontsize=12, ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax.text(3.2, 1.5, 'H(Y|X)', fontsize=12, ha='center', va='center')
        ax.text(1.2, 2.5, 'H(X)', fontsize=12, ha='center', va='center', weight='bold')
        ax.text(2.8, 2.5, 'H(Y)', fontsize=12, ha='center', va='center', weight='bold')
        ax.text(2, 0.3, 'H(X,Y)', fontsize=14, ha='center', va='center', weight='bold')
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('信息论量之间的关系', fontsize=16)
        
        plt.tight_layout()
        plt.show()

class MutualInformation:
    """互信息"""
    
    def __init__(self):
        pass
    
    def mutual_information_concept(self):
        """互信息概念"""
        print("=== 互信息 (Mutual Information) ===")
        
        print("定义:")
        print("I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)")
        print("      = H(X) + H(Y) - H(X,Y)")
        print("      = Σₓᵧ P(x,y) log₂[P(x,y)/(P(x)P(y))]")
        print()
        
        print("物理意义:")
        print("- 测量两个随机变量之间的相互依赖程度")
        print("- X关于Y提供的信息量")
        print("- 减少的不确定性")
        print()
        
        print("性质:")
        print("- I(X;Y) ≥ 0，等号成立当且仅当X,Y独立")
        print("- I(X;Y) = I(Y;X) (对称性)")
        print("- I(X;X) = H(X)")
        print("- I(X;Y) ≤ min(H(X), H(Y))")
        
        # 互信息计算示例
        self.demonstrate_mutual_information()
        
        return self.feature_selection_example()
    
    def demonstrate_mutual_information(self):
        """演示互信息计算"""
        print("\n=== 互信息计算示例 ===")
        
        # 不同相关程度的变量对
        np.random.seed(42)
        n_samples = 1000
        
        # 情况1: 完全独立
        x1 = np.random.binomial(1, 0.5, n_samples)
        y1 = np.random.binomial(1, 0.5, n_samples)
        
        # 情况2: 部分相关
        x2 = np.random.binomial(1, 0.5, n_samples)
        y2 = np.where(np.random.random(n_samples) < 0.7, x2, 1-x2)
        
        # 情况3: 完全相关
        x3 = np.random.binomial(1, 0.5, n_samples)
        y3 = x3.copy()
        
        # 计算互信息
        cases = [
            ("独立变量", x1, y1),
            ("部分相关", x2, y2),
            ("完全相关", x3, y3)
        ]
        
        mutual_infos = []
        for name, x, y in cases:
            mi = mutual_info_score(x, y)
            mutual_infos.append(mi)
            print(f"{name}: I(X;Y) = {mi:.3f} bits")
        
        # 可视化互信息
        self.visualize_mutual_information(cases, mutual_infos)
        
        return mutual_infos
    
    def visualize_mutual_information(self, cases, mutual_infos):
        """可视化互信息"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, ((name, x, y), mi) in enumerate(zip(cases, mutual_infos)):
            # 散点图
            ax1 = axes[0, i]
            ax1.scatter(x + np.random.normal(0, 0.05, len(x)), 
                       y + np.random.normal(0, 0.05, len(y)), 
                       alpha=0.6, s=10)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title(f'{name}\nI(X;Y) = {mi:.3f}')
            ax1.grid(True, alpha=0.3)
            
            # 联合分布热图
            ax2 = axes[1, i]
            
            # 计算联合概率
            joint_counts = np.zeros((2, 2))
            for xi, yi in zip(x, y):
                joint_counts[int(xi), int(yi)] += 1
            joint_prob = joint_counts / len(x)
            
            im = ax2.imshow(joint_prob, cmap='Blues', interpolation='nearest')
            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xlabel('Y')
            ax2.set_ylabel('X')
            ax2.set_title(f'联合分布 P(X,Y)')
            
            # 添加数值标注
            for xi in range(2):
                for yi in range(2):
                    ax2.text(yi, xi, f'{joint_prob[xi, yi]:.2f}', 
                            ha='center', va='center', color='red')
        
        plt.tight_layout()
        plt.show()
    
    def feature_selection_example(self):
        """特征选择示例"""
        print("\n=== 基于互信息的特征选择 ===")
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=10, 
                                 n_informative=5, n_redundant=2, 
                                 random_state=42)
        
        # 计算每个特征与目标的互信息
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 创建特征名称
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # 按互信息排序
        mi_df = pd.DataFrame({
            'Feature': feature_names,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)
        
        print("特征的互信息得分:")
        print(mi_df)
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        bars = plt.bar(mi_df['Feature'], mi_df['Mutual_Information'], alpha=0.7)
        plt.xlabel('特征')
        plt.ylabel('互信息得分')
        plt.title('基于互信息的特征重要性')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, mi_df['Mutual_Information']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return mi_df

class KLDivergence:
    """KL散度"""
    
    def __init__(self):
        pass
    
    def kl_divergence_concept(self):
        """KL散度概念"""
        print("=== KL散度 (Kullback-Leibler Divergence) ===")
        
        print("定义:")
        print("D_KL(P||Q) = Σᵢ P(i) log[P(i)/Q(i)]")
        print("         = E_P[log P(X) - log Q(X)]")
        print()
        
        print("性质:")
        print("- D_KL(P||Q) ≥ 0，等号成立当且仅当P=Q")
        print("- 非对称：D_KL(P||Q) ≠ D_KL(Q||P)")
        print("- 不满足三角不等式，不是真正的距离")
        print()
        
        print("机器学习应用:")
        print("- 损失函数（交叉熵）")
        print("- 变分推断")
        print("- 生成模型训练")
        print("- 模型压缩")
        
        # KL散度计算示例
        self.demonstrate_kl_divergence()
        
        return self.cross_entropy_connection()
    
    def demonstrate_kl_divergence(self):
        """演示KL散度计算"""
        print("\n=== KL散度计算示例 ===")
        
        # 不同分布之间的KL散度
        x = np.linspace(0.01, 0.99, 100)
        
        # 分布P（真实分布）
        p = stats.beta.pdf(x, 2, 5)
        p = p / np.sum(p)  # 归一化
        
        # 不同的近似分布Q
        q1 = stats.beta.pdf(x, 2, 5)  # 相同分布
        q1 = q1 / np.sum(q1)
        
        q2 = stats.beta.pdf(x, 3, 4)  # 相似分布
        q2 = q2 / np.sum(q2)
        
        q3 = np.ones_like(x)  # 均匀分布
        q3 = q3 / np.sum(q3)
        
        # 计算KL散度
        def kl_divergence(p, q):
            # 避免log(0)
            epsilon = 1e-10
            q = np.maximum(q, epsilon)
            return np.sum(p * np.log(p / q))
        
        kl1 = kl_divergence(p, q1)
        kl2 = kl_divergence(p, q2)
        kl3 = kl_divergence(p, q3)
        
        print(f"D_KL(P||P) = {kl1:.4f}")
        print(f"D_KL(P||Q_similar) = {kl2:.4f}")
        print(f"D_KL(P||Q_uniform) = {kl3:.4f}")
        
        # 可视化KL散度
        self.visualize_kl_divergence(x, p, [q1, q2, q3], [kl1, kl2, kl3])
        
        return [kl1, kl2, kl3]
    
    def visualize_kl_divergence(self, x, p, q_list, kl_list):
        """可视化KL散度"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始分布
        axes[0, 0].plot(x, p, 'b-', linewidth=2, label='P (真实分布)')
        axes[0, 0].fill_between(x, p, alpha=0.3, color='blue')
        axes[0, 0].set_title('真实分布 P')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('概率密度')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 不同近似分布
        labels = ['相同分布', '相似分布', '均匀分布']
        colors = ['green', 'orange', 'red']
        
        for i, (q, kl, label, color) in enumerate(zip(q_list, kl_list, labels, colors)):
            ax = axes[0, 1] if i == 0 else axes[1, i-1]
            
            ax.plot(x, p, 'b-', linewidth=2, label='P', alpha=0.7)
            ax.plot(x, q, color=color, linewidth=2, label=f'Q ({label})')
            ax.fill_between(x, p, alpha=0.3, color='blue')
            ax.fill_between(x, q, alpha=0.3, color=color)
            
            ax.set_title(f'{label}\nD_KL(P||Q) = {kl:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('概率密度')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def cross_entropy_connection(self):
        """交叉熵连接"""
        print("\n=== KL散度与交叉熵的关系 ===")
        
        print("交叉熵定义:")
        print("H(P,Q) = -Σᵢ P(i) log Q(i)")
        print()
        print("关系:")
        print("D_KL(P||Q) = H(P,Q) - H(P)")
        print("因此：H(P,Q) = H(P) + D_KL(P||Q)")
        print()
        print("在机器学习中:")
        print("- P: 真实标签分布")
        print("- Q: 模型预测分布")
        print("- 最小化交叉熵 ≡ 最小化KL散度（因为H(P)是常数）")
        
        # 分类问题中的交叉熵示例
        self.classification_cross_entropy_example()
        
        return self.js_divergence()
    
    def classification_cross_entropy_example(self):
        """分类问题中的交叉熵示例"""
        print("\n=== 分类交叉熵示例 ===")
        
        # 真实标签（one-hot编码）
        y_true = np.array([
            [1, 0, 0],  # 类别0
            [0, 1, 0],  # 类别1
            [0, 0, 1],  # 类别2
        ])
        
        # 不同质量的预测
        predictions = {
            "完美预测": np.array([
                [0.99, 0.005, 0.005],
                [0.005, 0.99, 0.005],
                [0.005, 0.005, 0.99]
            ]),
            "好预测": np.array([
                [0.8, 0.15, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.15, 0.8]
            ]),
            "差预测": np.array([
                [0.4, 0.35, 0.25],
                [0.3, 0.4, 0.3],
                [0.25, 0.35, 0.4]
            ])
        }
        
        # 计算交叉熵
        def cross_entropy(y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        results = {}
        for name, y_pred in predictions.items():
            ce = cross_entropy(y_true, y_pred)
            results[name] = ce
            print(f"{name}: 交叉熵 = {ce:.4f}")
        
        # 可视化预测质量
        self.visualize_prediction_quality(y_true, predictions, results)
        
        return results
    
    def visualize_prediction_quality(self, y_true, predictions, results):
        """可视化预测质量"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 真实标签
        axes[0, 0].imshow(y_true, cmap='Blues', interpolation='nearest')
        axes[0, 0].set_title('真实标签')
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('样本')
        
        # 预测结果
        for i, (name, y_pred) in enumerate(predictions.items()):
            ax = axes[0, 1] if i == 0 else axes[1, i-1]
            
            im = ax.imshow(y_pred, cmap='Reds', interpolation='nearest')
            ax.set_title(f'{name}\n交叉熵: {results[name]:.4f}')
            ax.set_xlabel('类别')
            ax.set_ylabel('样本')
            
            # 添加数值标注
            for row in range(y_pred.shape[0]):
                for col in range(y_pred.shape[1]):
                    ax.text(col, row, f'{y_pred[row, col]:.2f}', 
                           ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.show()
    
    def js_divergence(self):
        """JS散度"""
        print("\n=== JS散度 (Jensen-Shannon Divergence) ===")
        
        print("定义:")
        print("JS(P,Q) = (1/2)D_KL(P||M) + (1/2)D_KL(Q||M)")
        print("其中 M = (1/2)(P + Q)")
        print()
        print("性质:")
        print("- 对称：JS(P,Q) = JS(Q,P)")
        print("- 有界：0 ≤ JS(P,Q) ≤ 1")
        print("- 平方根√JS(P,Q)是真正的距离度量")
        print()
        print("应用:")
        print("- GAN中的损失函数")
        print("- 模型比较")
        print("- 聚类分析")
        
        # JS散度计算示例
        return self.demonstrate_js_divergence()
    
    def demonstrate_js_divergence(self):
        """演示JS散度"""
        print("\n=== JS散度计算示例 ===")
        
        # 两个概率分布
        p = np.array([0.5, 0.3, 0.2])
        q1 = np.array([0.5, 0.3, 0.2])  # 相同分布
        q2 = np.array([0.3, 0.4, 0.3])  # 不同分布
        
        def js_divergence(p, q):
            """计算JS散度"""
            def kl_div(p, q):
                epsilon = 1e-10
                return np.sum(p * np.log((p + epsilon) / (q + epsilon)))
            
            m = 0.5 * (p + q)
            return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        
        js1 = js_divergence(p, q1)
        js2 = js_divergence(p, q2)
        
        print(f"JS(P,P) = {js1:.4f}")
        print(f"JS(P,Q) = {js2:.4f}")
        
        # 可视化JS散度
        distributions = [p, q1, q2]
        names = ['P', 'Q (相同)', 'Q (不同)']
        
        plt.figure(figsize=(12, 4))
        
        for i, (dist, name) in enumerate(zip(distributions, names)):
            plt.subplot(1, 3, i+1)
            plt.bar(['X₁', 'X₂', 'X₃'], dist, alpha=0.7)
            plt.title(name)
            plt.ylabel('概率')
            plt.ylim(0, 0.6)
            
            # 添加数值标签
            for j, val in enumerate(dist):
                plt.text(j, val + 0.02, f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return js1, js2

def comprehensive_information_theory_summary():
    """信息理论综合总结"""
    print("=== 信息理论综合总结 ===")
    
    summary = {
        "基本概念": {
            "信息量": "I(x) = -log₂ P(x)",
            "熵": "H(X) = E[-log₂ P(X)]",
            "条件熵": "H(Y|X) = E[H(Y|X=x)]",
            "联合熵": "H(X,Y) = E[-log₂ P(X,Y)]"
        },
        
        "信息度量": {
            "互信息": "I(X;Y) = H(X) - H(X|Y)",
            "KL散度": "D_KL(P||Q) = E_P[log P/Q]",
            "JS散度": "JS(P,Q) = (D_KL(P||M) + D_KL(Q||M))/2",
            "交叉熵": "H(P,Q) = -E_P[log Q]"
        },
        
        "重要性质": {
            "链式法则": "H(X,Y) = H(X) + H(Y|X)",
            "数据处理不等式": "I(X;Z) ≤ I(X;Y) if X-Y-Z",
            "Fano不等式": "关联错误概率与条件熵",
            "信息瓶颈": "最大化相关信息，最小化无关信息"
        },
        
        "ML应用": {
            "损失函数": "交叉熵损失、KL散度损失",
            "特征选择": "基于互信息的特征重要性",
            "模型压缩": "知识蒸馏中的KL散度",
            "生成模型": "VAE中的ELBO、GAN中的JS散度",
            "强化学习": "策略梯度中的熵正则化"
        },
        
        "实践技巧": {
            "数值稳定性": "log-sum-exp技巧",
            "离散化": "连续变量的信息估计",
            "采样估计": "大数据集的信息度量",
            "正则化": "熵作为正则化项"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("信息理论基础指南加载完成！")
```

## 参考文献 📚

- Cover & Thomas (2006): "Elements of Information Theory"
- MacKay (2003): "Information Theory, Inference and Learning Algorithms"
- Shannon (1948): "A Mathematical Theory of Communication"
- Kullback & Leibler (1951): "On Information and Sufficiency"

## 下一步学习
- [概率统计理论](probability_statistics_theory.md) - 概率论基础
- [优化算法理论](optimization_theory.md) - 优化方法
- [贝叶斯机器学习](bayesian_ml_theory.md) - 贝叶斯方法