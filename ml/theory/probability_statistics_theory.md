# 概率统计理论：机器学习的数学基础 📊

深入理解机器学习中的概率统计理论，从基础概念到高级应用。

## 1. 概率论基础 🎲

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class ProbabilityFoundations:
    """概率论基础"""
    
    def __init__(self):
        self.distributions = {}
    
    def probability_axioms(self):
        """概率公理"""
        print("=== 概率公理 (Kolmogorov Axioms) ===")
        
        axioms = {
            "非负性": {
                "公式": "P(A) ≥ 0",
                "含义": "任何事件的概率都非负",
                "ML应用": "概率预测、分类器输出"
            },
            "规范性": {
                "公式": "P(Ω) = 1",
                "含义": "样本空间的概率为1",
                "ML应用": "概率分布归一化"
            },
            "可加性": {
                "公式": "P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...",
                "含义": "互斥事件概率可加",
                "ML应用": "多类分类概率求和"
            }
        }
        
        for axiom, details in axioms.items():
            print(f"\n{axiom}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 概率可视化演示
        self.demonstrate_probability_axioms()
        
        return axioms
    
    def demonstrate_probability_axioms(self):
        """演示概率公理"""
        # 投掷骰子的概率分布
        outcomes = np.arange(1, 7)
        probabilities = np.ones(6) / 6
        
        plt.figure(figsize=(15, 5))
        
        # 公理1: 非负性
        plt.subplot(1, 3, 1)
        plt.bar(outcomes, probabilities, alpha=0.7, color='blue')
        plt.title('公理1: 非负性\nP(X) ≥ 0')
        plt.xlabel('骰子点数')
        plt.ylabel('概率')
        plt.ylim(0, 0.3)
        
        # 公理2: 规范性
        plt.subplot(1, 3, 2)
        plt.bar(outcomes, probabilities, alpha=0.7, color='green')
        plt.axhline(y=probabilities.sum(), color='red', linestyle='--', 
                   label=f'总概率 = {probabilities.sum():.1f}')
        plt.title('公理2: 规范性\nΣP(X) = 1')
        plt.xlabel('骰子点数')
        plt.ylabel('概率')
        plt.legend()
        
        # 公理3: 可加性
        plt.subplot(1, 3, 3)
        even_prob = probabilities[[1, 3, 5]].sum()  # 偶数概率
        odd_prob = probabilities[[0, 2, 4]].sum()   # 奇数概率
        
        plt.bar(['偶数', '奇数'], [even_prob, odd_prob], alpha=0.7, color='orange')
        plt.title('公理3: 可加性\nP(偶数) + P(奇数) = 1')
        plt.ylabel('概率')
        
        plt.tight_layout()
        plt.show()
    
    def conditional_probability(self):
        """条件概率与贝叶斯定理"""
        print("=== 条件概率与贝叶斯定理 ===")
        
        print("条件概率公式:")
        print("P(A|B) = P(A ∩ B) / P(B)")
        print()
        
        print("贝叶斯定理:")
        print("P(A|B) = P(B|A) × P(A) / P(B)")
        print("其中:")
        print("- P(A|B): 后验概率")
        print("- P(B|A): 似然函数")
        print("- P(A): 先验概率")
        print("- P(B): 边际概率")
        print()
        
        # 贝叶斯定理实际应用：垃圾邮件分类
        self.bayesian_spam_classification()
        
        return self.visualize_bayes_theorem()
    
    def bayesian_spam_classification(self):
        """贝叶斯垃圾邮件分类示例"""
        print("=== 贝叶斯垃圾邮件分类示例 ===")
        
        # 假设数据
        prob_spam = 0.3  # 先验概率：30%的邮件是垃圾邮件
        prob_ham = 0.7   # 先验概率：70%的邮件是正常邮件
        
        # 似然概率：包含"免费"这个词的概率
        prob_free_given_spam = 0.8   # 垃圾邮件中包含"免费"的概率
        prob_free_given_ham = 0.1    # 正常邮件中包含"免费"的概率
        
        # 边际概率：邮件包含"免费"的总概率
        prob_free = (prob_free_given_spam * prob_spam + 
                    prob_free_given_ham * prob_ham)
        
        # 后验概率：包含"免费"的邮件是垃圾邮件的概率
        prob_spam_given_free = (prob_free_given_spam * prob_spam) / prob_free
        
        print(f"先验概率 P(垃圾邮件) = {prob_spam}")
        print(f"似然概率 P('免费'|垃圾邮件) = {prob_free_given_spam}")
        print(f"边际概率 P('免费') = {prob_free:.3f}")
        print(f"后验概率 P(垃圾邮件|'免费') = {prob_spam_given_free:.3f}")
        
        # 可视化贝叶斯更新过程
        self.visualize_bayesian_update(prob_spam, prob_spam_given_free)
    
    def visualize_bayesian_update(self, prior, posterior):
        """可视化贝叶斯更新"""
        plt.figure(figsize=(10, 6))
        
        categories = ['先验概率\nP(垃圾邮件)', '后验概率\nP(垃圾邮件|"免费")']
        probabilities = [prior, posterior]
        colors = ['lightblue', 'orange']
        
        bars = plt.bar(categories, probabilities, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.ylabel('概率')
        plt.title('贝叶斯更新：观察到证据后概率的变化')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def visualize_bayes_theorem(self):
        """可视化贝叶斯定理"""
        print("\n=== 贝叶斯定理可视化 ===")
        
        # 创建二维概率分布
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 联合概率分布
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # 二元正态分布
        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        rv = stats.multivariate_normal(mean, cov)
        Z = rv.pdf(np.dstack((X, Y)))
        
        # 联合分布
        im1 = axes[0, 0].contourf(X, Y, Z, levels=20, cmap='Blues')
        axes[0, 0].set_title('联合概率分布 P(X,Y)')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        
        # 边际分布 P(X)
        marginal_x = np.sum(Z, axis=0)
        marginal_x = marginal_x / np.sum(marginal_x)
        axes[0, 1].plot(x, marginal_x, 'b-', linewidth=2)
        axes[0, 1].fill_between(x, marginal_x, alpha=0.3)
        axes[0, 1].set_title('边际分布 P(X)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('概率密度')
        
        # 条件分布 P(Y|X=0)
        x_idx = np.argmin(np.abs(x))
        conditional_y = Z[:, x_idx]
        conditional_y = conditional_y / np.sum(conditional_y)
        axes[1, 0].plot(y, conditional_y, 'r-', linewidth=2)
        axes[1, 0].fill_between(y, conditional_y, alpha=0.3, color='red')
        axes[1, 0].set_title('条件分布 P(Y|X=0)')
        axes[1, 0].set_xlabel('Y')
        axes[1, 0].set_ylabel('概率密度')
        
        # 贝叶斯定理示意图
        axes[1, 1].text(0.1, 0.8, 'P(A|B) = P(B|A) × P(A) / P(B)', 
                        fontsize=16, transform=axes[1, 1].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].text(0.1, 0.6, '后验 = 似然 × 先验 / 证据', 
                        fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('贝叶斯定理')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return mean, cov

class StatisticalDistributions:
    """统计分布"""
    
    def __init__(self):
        self.distributions = {}
    
    def common_distributions(self):
        """常见分布族"""
        print("=== 机器学习中的常见分布 ===")
        
        distributions = {
            "伯努利分布": {
                "参数": "p (成功概率)",
                "PMF": "P(X=k) = p^k (1-p)^(1-k)",
                "应用": "二分类、硬币投掷",
                "ML用途": "逻辑回归输出"
            },
            "二项分布": {
                "参数": "n (试验次数), p (成功概率)",
                "PMF": "P(X=k) = C(n,k) p^k (1-p)^(n-k)",
                "应用": "重复试验成功次数",
                "ML用途": "分类准确率分析"
            },
            "泊松分布": {
                "参数": "λ (平均发生率)",
                "PMF": "P(X=k) = λ^k e^(-λ) / k!",
                "应用": "单位时间内事件发生次数",
                "ML用途": "计数数据建模"
            },
            "正态分布": {
                "参数": "μ (均值), σ² (方差)",
                "PDF": "f(x) = 1/√(2πσ²) exp(-(x-μ)²/2σ²)",
                "应用": "连续数据建模",
                "ML用途": "线性回归误差、特征分布"
            },
            "指数分布": {
                "参数": "λ (率参数)",
                "PDF": "f(x) = λe^(-λx), x ≥ 0",
                "应用": "等待时间建模",
                "ML用途": "生存分析、可靠性"
            },
            "Beta分布": {
                "参数": "α, β (形状参数)",
                "PDF": "f(x) = x^(α-1)(1-x)^(β-1) / B(α,β)",
                "应用": "概率的概率",
                "ML用途": "贝叶斯推断先验"
            }
        }
        
        for dist, details in distributions.items():
            print(f"\n{dist}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 可视化常见分布
        self.visualize_distributions()
        
        return distributions
    
    def visualize_distributions(self):
        """可视化常见分布"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. 伯努利分布
        x_bern = [0, 1]
        p_bern = 0.3
        y_bern = [1-p_bern, p_bern]
        axes[0].bar(x_bern, y_bern, alpha=0.7, color='blue')
        axes[0].set_title(f'伯努利分布 (p={p_bern})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('P(X)')
        
        # 2. 二项分布
        n, p = 20, 0.3
        x_binom = np.arange(0, n+1)
        y_binom = stats.binom.pmf(x_binom, n, p)
        axes[1].bar(x_binom, y_binom, alpha=0.7, color='green')
        axes[1].set_title(f'二项分布 (n={n}, p={p})')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('P(X)')
        
        # 3. 泊松分布
        lam = 3
        x_poisson = np.arange(0, 15)
        y_poisson = stats.poisson.pmf(x_poisson, lam)
        axes[2].bar(x_poisson, y_poisson, alpha=0.7, color='red')
        axes[2].set_title(f'泊松分布 (λ={lam})')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('P(X)')
        
        # 4. 正态分布
        x_norm = np.linspace(-4, 4, 100)
        y_norm1 = stats.norm.pdf(x_norm, 0, 1)
        y_norm2 = stats.norm.pdf(x_norm, 0, 0.5)
        axes[3].plot(x_norm, y_norm1, label='σ=1', linewidth=2)
        axes[3].plot(x_norm, y_norm2, label='σ=0.5', linewidth=2)
        axes[3].fill_between(x_norm, y_norm1, alpha=0.3)
        axes[3].set_title('正态分布 (μ=0)')
        axes[3].set_xlabel('X')
        axes[3].set_ylabel('f(x)')
        axes[3].legend()
        
        # 5. 指数分布
        x_exp = np.linspace(0, 5, 100)
        y_exp1 = stats.expon.pdf(x_exp, scale=1)
        y_exp2 = stats.expon.pdf(x_exp, scale=0.5)
        axes[4].plot(x_exp, y_exp1, label='λ=1', linewidth=2)
        axes[4].plot(x_exp, y_exp2, label='λ=2', linewidth=2)
        axes[4].fill_between(x_exp, y_exp1, alpha=0.3)
        axes[4].set_title('指数分布')
        axes[4].set_xlabel('X')
        axes[4].set_ylabel('f(x)')
        axes[4].legend()
        
        # 6. Beta分布
        x_beta = np.linspace(0, 1, 100)
        y_beta1 = stats.beta.pdf(x_beta, 2, 2)
        y_beta2 = stats.beta.pdf(x_beta, 0.5, 0.5)
        axes[5].plot(x_beta, y_beta1, label='α=2, β=2', linewidth=2)
        axes[5].plot(x_beta, y_beta2, label='α=0.5, β=0.5', linewidth=2)
        axes[5].fill_between(x_beta, y_beta1, alpha=0.3)
        axes[5].set_title('Beta分布')
        axes[5].set_xlabel('X')
        axes[5].set_ylabel('f(x)')
        axes[5].legend()
        
        plt.tight_layout()
        plt.show()
    
    def central_limit_theorem(self):
        """中心极限定理"""
        print("=== 中心极限定理 ===")
        
        print("定理表述:")
        print("设X₁, X₂, ..., Xₙ是独立同分布的随机变量，")
        print("均值为μ，方差为σ²，则当n足够大时：")
        print("(X̄ - μ) / (σ/√n) → N(0,1)")
        print()
        
        # 演示中心极限定理
        self.demonstrate_clt()
        
        print("机器学习意义:")
        print("- 解释为什么很多统计量服从正态分布")
        print("- 为大样本统计推断提供理论基础")
        print("- 神经网络权重初始化的理论依据")
        print("- Bootstrap方法的理论基础")
    
    def demonstrate_clt(self):
        """演示中心极限定理"""
        # 原始分布：均匀分布
        np.random.seed(42)
        
        # 不同样本大小
        sample_sizes = [1, 5, 30, 100]
        n_experiments = 1000
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, n in enumerate(sample_sizes):
            # 进行多次实验，每次取n个样本的均值
            sample_means = []
            for _ in range(n_experiments):
                # 从均匀分布中抽取n个样本
                samples = np.random.uniform(0, 1, n)
                sample_means.append(np.mean(samples))
            
            # 绘制样本均值的分布
            axes[i].hist(sample_means, bins=50, density=True, alpha=0.7, 
                        color='skyblue', edgecolor='black')
            
            # 理论正态分布
            mu = 0.5  # 均匀分布[0,1]的均值
            sigma = np.sqrt(1/12 / n)  # 样本均值的标准差
            x_theory = np.linspace(min(sample_means), max(sample_means), 100)
            y_theory = stats.norm.pdf(x_theory, mu, sigma)
            axes[i].plot(x_theory, y_theory, 'r-', linewidth=2, 
                        label='理论正态分布')
            
            axes[i].set_title(f'样本大小 n = {n}')
            axes[i].set_xlabel('样本均值')
            axes[i].set_ylabel('密度')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('中心极限定理演示：样本均值趋向正态分布', fontsize=16)
        plt.tight_layout()
        plt.show()

class BayesianMachineLearning:
    """贝叶斯机器学习"""
    
    def __init__(self):
        pass
    
    def bayesian_inference(self):
        """贝叶斯推断"""
        print("=== 贝叶斯推断 ===")
        
        print("贝叶斯推断框架:")
        print("P(θ|D) = P(D|θ) × P(θ) / P(D)")
        print("其中:")
        print("- θ: 模型参数")
        print("- D: 观测数据")
        print("- P(θ): 先验分布")
        print("- P(D|θ): 似然函数")
        print("- P(θ|D): 后验分布")
        print("- P(D): 边际似然")
        print()
        
        # 贝叶斯线性回归示例
        self.bayesian_linear_regression()
        
        return self.demonstrate_conjugate_priors()
    
    def bayesian_linear_regression(self):
        """贝叶斯线性回归"""
        print("=== 贝叶斯线性回归 ===")
        
        # 生成数据
        np.random.seed(42)
        n_points = 20
        true_w = [2, -1]  # 真实权重
        true_sigma = 0.3   # 噪声标准差
        
        x = np.linspace(-1, 1, n_points)
        X = np.column_stack([np.ones(n_points), x])  # 添加偏置项
        y = X @ true_w + np.random.normal(0, true_sigma, n_points)
        
        # 贝叶斯推断
        # 先验：w ~ N(0, σ²I)
        prior_sigma = 1.0
        prior_precision = 1 / prior_sigma**2 * np.eye(2)
        prior_mean = np.zeros(2)
        
        # 似然：y = Xw + ε, ε ~ N(0, σ²I)
        noise_precision = 1 / true_sigma**2
        
        # 后验参数
        posterior_precision = prior_precision + noise_precision * X.T @ X
        posterior_covariance = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_covariance @ (noise_precision * X.T @ y)
        
        print(f"真实权重: {true_w}")
        print(f"后验均值: {posterior_mean}")
        print(f"后验协方差对角线: {np.diag(posterior_covariance)}")
        
        # 可视化
        self.visualize_bayesian_regression(x, y, X, posterior_mean, posterior_covariance)
    
    def visualize_bayesian_regression(self, x, y, X, posterior_mean, posterior_cov):
        """可视化贝叶斯回归"""
        plt.figure(figsize=(12, 8))
        
        # 数据点
        plt.subplot(2, 2, 1)
        plt.scatter(x, y, alpha=0.7, color='blue', label='观测数据')
        
        # 后验预测
        x_test = np.linspace(-1.5, 1.5, 100)
        X_test = np.column_stack([np.ones(len(x_test)), x_test])
        
        # 预测均值
        y_pred_mean = X_test @ posterior_mean
        
        # 预测不确定性
        y_pred_var = np.array([X_test[i] @ posterior_cov @ X_test[i] 
                              for i in range(len(X_test))])
        y_pred_std = np.sqrt(y_pred_var + 0.3**2)  # 加上噪声方差
        
        plt.plot(x_test, y_pred_mean, 'r-', linewidth=2, label='后验预测均值')
        plt.fill_between(x_test, 
                        y_pred_mean - 2*y_pred_std,
                        y_pred_mean + 2*y_pred_std,
                        alpha=0.3, color='red', label='95%置信区间')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('贝叶斯线性回归')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 权重的后验分布
        plt.subplot(2, 2, 2)
        # 采样权重
        n_samples = 1000
        weight_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, n_samples)
        
        plt.scatter(weight_samples[:, 0], weight_samples[:, 1], alpha=0.5, s=10)
        plt.scatter(posterior_mean[0], posterior_mean[1], color='red', s=100, 
                   marker='x', linewidth=3, label='后验均值')
        plt.xlabel('w₀ (偏置)')
        plt.ylabel('w₁ (斜率)')
        plt.title('权重参数的后验分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 先验vs后验对比
        plt.subplot(2, 2, 3)
        prior_samples = np.random.multivariate_normal([0, 0], np.eye(2), n_samples)
        plt.scatter(prior_samples[:, 0], prior_samples[:, 1], alpha=0.3, s=5, 
                   color='gray', label='先验样本')
        plt.scatter(weight_samples[:, 0], weight_samples[:, 1], alpha=0.5, s=5,
                   color='blue', label='后验样本')
        plt.xlabel('w₀')
        plt.ylabel('w₁')
        plt.title('先验 vs 后验分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 预测不确定性变化
        plt.subplot(2, 2, 4)
        plt.plot(x_test, y_pred_std, 'purple', linewidth=2)
        plt.axvline(x=x.min(), color='gray', linestyle='--', alpha=0.5, label='数据范围')
        plt.axvline(x=x.max(), color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('预测标准差')
        plt.title('预测不确定性')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_conjugate_priors(self):
        """演示共轭先验"""
        print("\n=== 共轭先验 ===")
        
        conjugate_pairs = {
            "伯努利-Beta": {
                "似然": "Bernoulli(p)",
                "先验": "Beta(α, β)",
                "后验": "Beta(α + Σxᵢ, β + n - Σxᵢ)",
                "应用": "点击率估计、A/B测试"
            },
            "正态-正态": {
                "似然": "N(μ, σ²) (σ²已知)",
                "先验": "N(μ₀, σ₀²)",
                "后验": "N(μₙ, σₙ²)",
                "应用": "线性回归、高斯过程"
            },
            "泊松-Gamma": {
                "似然": "Poisson(λ)",
                "先验": "Gamma(α, β)",
                "后验": "Gamma(α + Σxᵢ, β + n)",
                "应用": "计数数据建模"
            }
        }
        
        for pair, details in conjugate_pairs.items():
            print(f"\n{pair}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # Beta-Bernoulli共轭演示
        self.beta_bernoulli_demo()
        
        return conjugate_pairs
    
    def beta_bernoulli_demo(self):
        """Beta-Bernoulli共轭演示"""
        print("\n=== Beta-Bernoulli共轭演示 ===")
        
        # 参数设置
        true_p = 0.3  # 真实成功概率
        alpha_0, beta_0 = 1, 1  # 先验参数（均匀分布）
        
        # 模拟数据收集过程
        np.random.seed(42)
        n_experiments = [0, 5, 20, 100]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        x = np.linspace(0, 1, 1000)
        
        for i, n in enumerate(n_experiments):
            if n == 0:
                # 先验分布
                alpha_n, beta_n = alpha_0, beta_0
                title = "先验分布"
            else:
                # 生成数据
                data = np.random.binomial(1, true_p, n)
                successes = np.sum(data)
                
                # 更新后验参数
                alpha_n = alpha_0 + successes
                beta_n = beta_0 + n - successes
                title = f"观测 {n} 次后的后验分布"
            
            # 绘制分布
            y = stats.beta.pdf(x, alpha_n, beta_n)
            axes[i].plot(x, y, linewidth=2, color='blue')
            axes[i].fill_between(x, y, alpha=0.3, color='blue')
            axes[i].axvline(true_p, color='red', linestyle='--', 
                           linewidth=2, label=f'真实值 p={true_p}')
            axes[i].axvline(alpha_n/(alpha_n + beta_n), color='green', 
                           linestyle=':', linewidth=2, label='后验均值')
            
            axes[i].set_xlabel('p (成功概率)')
            axes[i].set_ylabel('密度')
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Beta-Bernoulli共轭：先验到后验的更新', fontsize=16)
        plt.tight_layout()
        plt.show()

def comprehensive_probability_summary():
    """概率统计理论综合总结"""
    print("=== 概率统计理论综合总结 ===")
    
    summary = {
        "基础概念": {
            "概率空间": "(Ω, F, P) - 样本空间、事件域、概率测度",
            "随机变量": "映射函数 X: Ω → ℝ",
            "分布函数": "CDF: F(x) = P(X ≤ x)",
            "概率质量/密度": "PMF/PDF: 描述概率分布"
        },
        
        "重要定理": {
            "贝叶斯定理": "P(A|B) = P(B|A)P(A)/P(B)",
            "全概率定理": "P(B) = Σᵢ P(B|Aᵢ)P(Aᵢ)",
            "中心极限定理": "样本均值渐近正态分布",
            "大数定律": "样本均值收敛到期望值"
        },
        
        "统计推断": {
            "点估计": "参数的最佳猜测值",
            "区间估计": "置信区间、credible interval",
            "假设检验": "p值、第一类/二类错误",
            "贝叶斯推断": "先验→似然→后验"
        },
        
        "ML应用": {
            "概率模型": "朴素贝叶斯、高斯混合模型",
            "不确定性量化": "预测区间、模型置信度",
            "贝叶斯深度学习": "权重不确定性、变分推断",
            "强化学习": "策略梯度、价值函数"
        },
        
        "实践技巧": {
            "数值稳定性": "log-sum-exp trick",
            "采样方法": "MCMC、变分推断",
            "模型选择": "AIC、BIC、交叉验证",
            "正则化": "先验作为正则化项"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("概率统计理论基础指南加载完成！")
```

## 参考文献 📚

- Ross (2014): "A First Course in Probability"
- Casella & Berger (2002): "Statistical Inference"
- Murphy (2012): "Machine Learning: A Probabilistic Perspective"
- Bishop (2006): "Pattern Recognition and Machine Learning"
- Gelman et al. (2013): "Bayesian Data Analysis"

## 下一步学习
- [贝叶斯机器学习](bayesian_ml_theory.md) - 贝叶斯方法深入
- [信息理论](information_theory.md) - 熵与信息测度
- [因果推理](causal_inference_theory.md) - 因果关系建模