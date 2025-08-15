# 贝叶斯机器学习理论：不确定性建模的艺术 🎯

深入理解贝叶斯机器学习的核心理论，从贝叶斯推断到变分方法。

## 1. 贝叶斯推断基础 🔍

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

class BayesianInference:
    """贝叶斯推断基础"""
    
    def __init__(self):
        self.models = {}
    
    def bayesian_framework(self):
        """贝叶斯框架"""
        print("=== 贝叶斯推断框架 ===")
        
        print("贝叶斯定理:")
        print("P(θ|D) = P(D|θ) × P(θ) / P(D)")
        print()
        
        components = {
            "后验分布 P(θ|D)": {
                "含义": "观测数据后对参数的信念",
                "重要性": "推断的最终目标",
                "获得方式": "通过贝叶斯定理计算",
                "用途": "预测、决策、不确定性量化"
            },
            "似然函数 P(D|θ)": {
                "含义": "给定参数下数据的概率",
                "作用": "数据对参数的约束",
                "性质": "不是概率密度函数",
                "计算": "基于数据生成模型"
            },
            "先验分布 P(θ)": {
                "含义": "观测数据前对参数的信念",
                "来源": "领域知识、历史数据、主观判断",
                "影响": "小样本时影响大，大样本时影响小",
                "选择": "共轭先验、无信息先验、正则化先验"
            },
            "边际似然 P(D)": {
                "含义": "数据的边际概率",
                "计算": "P(D) = ∫ P(D|θ)P(θ)dθ",
                "作用": "归一化常数、模型选择",
                "别名": "证据、模型证据"
            }
        }
        
        for comp, details in components.items():
            print(f"{comp}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化贝叶斯更新过程
        self.visualize_bayesian_update()
        
        return components
    
    def visualize_bayesian_update(self):
        """可视化贝叶斯更新过程"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 模拟贝叶斯更新：估计硬币正面概率
        true_p = 0.7  # 真实概率
        prior_alpha, prior_beta = 1, 1  # Beta先验参数
        
        # 不同数据量的观测
        observations = [0, 5, 20, 100]
        
        x = np.linspace(0, 1, 1000)
        
        for i, n_obs in enumerate(observations):
            if i >= 6:
                break
                
            if n_obs == 0:
                # 先验分布
                alpha, beta = prior_alpha, prior_beta
                title = "先验分布"
            else:
                # 生成观测数据
                np.random.seed(42)
                data = np.random.binomial(1, true_p, n_obs)
                successes = np.sum(data)
                failures = n_obs - successes
                
                # 更新后验参数
                alpha = prior_alpha + successes
                beta = prior_beta + failures
                title = f"观测{n_obs}次后的后验"
            
            # 绘制分布
            row, col = i // 3, i % 3
            y = stats.beta.pdf(x, alpha, beta)
            axes[row, col].plot(x, y, 'b-', linewidth=2, label=f'Beta({alpha}, {beta})')
            axes[row, col].fill_between(x, y, alpha=0.3)
            axes[row, col].axvline(true_p, color='red', linestyle='--', 
                                  linewidth=2, label=f'真实值={true_p}')
            axes[row, col].axvline(alpha/(alpha+beta), color='green', 
                                  linestyle=':', linewidth=2, label='后验均值')
            
            axes[row, col].set_xlabel('p (正面概率)')
            axes[row, col].set_ylabel('密度')
            axes[row, col].set_title(title)
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # 删除多余子图
        for i in range(4, 6):
            row, col = i // 3, i % 3
            axes[row, col].remove()
        
        # 添加概念图
        ax_concept = fig.add_subplot(2, 3, (5, 6))
        ax_concept.text(0.1, 0.8, "贝叶斯更新过程", fontsize=16, weight='bold',
                       transform=ax_concept.transAxes)
        ax_concept.text(0.1, 0.6, "先验 + 数据 → 后验", fontsize=14,
                       transform=ax_concept.transAxes)
        ax_concept.text(0.1, 0.4, "• 更多数据 → 更确定的后验", fontsize=12,
                       transform=ax_concept.transAxes)
        ax_concept.text(0.1, 0.3, "• 后验均值趋向真实值", fontsize=12,
                       transform=ax_concept.transAxes)
        ax_concept.text(0.1, 0.2, "• 后验方差逐渐减小", fontsize=12,
                       transform=ax_concept.transAxes)
        ax_concept.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def bayesian_vs_frequentist(self):
        """贝叶斯vs频率论比较"""
        print("=== 贝叶斯 vs 频率论 ===")
        
        comparison = {
            "概率解释": {
                "频率论": "长期频率，客观概率",
                "贝叶斯": "信念程度，主观概率",
                "例子": "硬币正面概率的含义"
            },
            "参数性质": {
                "频率论": "参数是固定未知常数",
                "贝叶斯": "参数是随机变量",
                "推断": "点估计 vs 分布估计"
            },
            "不确定性": {
                "频率论": "置信区间，抽样分布",
                "贝叶斯": "可信区间，后验分布",
                "解释": "程序vs信念"
            },
            "先验信息": {
                "频率论": "不使用先验信息",
                "贝叶斯": "自然融合先验信息",
                "优势": "小样本情况下的优势"
            },
            "计算复杂度": {
                "频率论": "通常较简单",
                "贝叶斯": "积分计算困难",
                "解决": "MCMC、变分推断"
            }
        }
        
        for aspect, details in comparison.items():
            print(f"{aspect}:")
            for approach, description in details.items():
                print(f"  {approach}: {description}")
            print()
        
        # 可视化比较
        self.visualize_frequentist_vs_bayesian()
        
        return comparison
    
    def visualize_frequentist_vs_bayesian(self):
        """可视化频率论vs贝叶斯方法"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 生成数据
        np.random.seed(42)
        true_mean = 2.5
        true_std = 1.0
        sample_sizes = [5, 10, 30, 100]
        
        for i, n in enumerate(sample_sizes):
            ax = axes[i//2, i%2]
            
            # 生成样本
            data = np.random.normal(true_mean, true_std, n)
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)
            
            # 频率论置信区间
            se = sample_std / np.sqrt(n)
            freq_ci_lower = sample_mean - 1.96 * se
            freq_ci_upper = sample_mean + 1.96 * se
            
            # 贝叶斯可信区间（假设已知方差）
            prior_mean = 0
            prior_var = 100
            posterior_var = 1 / (1/prior_var + n/true_std**2)
            posterior_mean = posterior_var * (prior_mean/prior_var + n*sample_mean/true_std**2)
            posterior_std = np.sqrt(posterior_var)
            bayes_ci_lower = posterior_mean - 1.96 * posterior_std
            bayes_ci_upper = posterior_mean + 1.96 * posterior_std
            
            # 绘制结果
            x_pos = [1, 2]
            estimates = [sample_mean, posterior_mean]
            ci_lowers = [freq_ci_lower, bayes_ci_lower]
            ci_uppers = [freq_ci_upper, bayes_ci_upper]
            colors = ['blue', 'red']
            labels = ['频率论', '贝叶斯']
            
            for j in range(2):
                ax.errorbar(x_pos[j], estimates[j], 
                           yerr=[[estimates[j] - ci_lowers[j]], [ci_uppers[j] - estimates[j]]],
                           fmt='o', color=colors[j], capsize=5, capthick=2, label=labels[j])
            
            ax.axhline(true_mean, color='green', linestyle='--', linewidth=2, label='真实值')
            ax.set_xlim(0.5, 2.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel('估计值')
            ax.set_title(f'样本大小 n={n}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class BayesianLinearRegression:
    """贝叶斯线性回归"""
    
    def __init__(self):
        pass
    
    def bayesian_regression_theory(self):
        """贝叶斯回归理论"""
        print("=== 贝叶斯线性回归理论 ===")
        
        print("模型假设:")
        print("y = Xw + ε")
        print("其中 ε ~ N(0, σ²I)")
        print()
        
        print("贝叶斯处理:")
        print("1. 权重先验: w ~ N(μ₀, Σ₀)")
        print("2. 似然函数: p(y|X,w,σ²) = N(Xw, σ²I)")
        print("3. 后验分布: p(w|X,y,σ²) = N(μₙ, Σₙ)")
        print()
        
        print("后验参数:")
        print("Σₙ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹")
        print("μₙ = Σₙ(Σ₀⁻¹μ₀ + σ⁻²Xᵀy)")
        print()
        
        print("预测分布:")
        print("p(y*|x*,X,y) = N(μₙᵀx*, x*ᵀΣₙx* + σ²)")
        print("包含参数不确定性和观测噪声")
        
        # 实现和可视化
        self.implement_bayesian_regression()
        
        return self.compare_with_frequentist()
    
    def implement_bayesian_regression(self):
        """实现贝叶斯线性回归"""
        print("\n=== 贝叶斯线性回归实现 ===")
        
        # 生成数据
        np.random.seed(42)
        n_samples = 50
        X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        true_w = [1.5, -0.3]  # 真实权重 [截距, 斜率]
        noise_std = 0.3
        
        # 添加偏置项
        X_design = np.column_stack([np.ones(n_samples), X.flatten()])
        y = X_design @ true_w + np.random.normal(0, noise_std, n_samples)
        
        # 贝叶斯线性回归
        # 先验参数
        prior_mean = np.zeros(2)
        prior_cov = np.eye(2) * 5  # 较宽松的先验
        noise_precision = 1 / noise_std**2
        
        # 后验参数
        posterior_precision = np.linalg.inv(prior_cov) + noise_precision * X_design.T @ X_design
        posterior_cov = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_cov @ (np.linalg.inv(prior_cov) @ prior_mean + 
                                        noise_precision * X_design.T @ y)
        
        print(f"真实权重: {true_w}")
        print(f"后验均值: {posterior_mean}")
        print(f"后验协方差对角线: {np.diag(posterior_cov)}")
        
        # 可视化
        self.visualize_bayesian_regression(X, y, X_design, posterior_mean, posterior_cov, noise_std)
        
        return posterior_mean, posterior_cov
    
    def visualize_bayesian_regression(self, X, y, X_design, posterior_mean, posterior_cov, noise_std):
        """可视化贝叶斯回归"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 数据和预测
        ax1 = axes[0, 0]
        ax1.scatter(X, y, alpha=0.7, color='blue', label='观测数据')
        
        # 预测
        X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
        X_test_design = np.column_stack([np.ones(len(X_test)), X_test.flatten()])
        
        # 预测均值
        y_pred_mean = X_test_design @ posterior_mean
        
        # 预测不确定性
        y_pred_var = []
        for i in range(len(X_test_design)):
            x_test = X_test_design[i]
            var = x_test @ posterior_cov @ x_test + noise_std**2
            y_pred_var.append(var)
        
        y_pred_std = np.sqrt(y_pred_var)
        
        ax1.plot(X_test, y_pred_mean, 'r-', linewidth=2, label='预测均值')
        ax1.fill_between(X_test.flatten(), 
                        y_pred_mean - 2*y_pred_std,
                        y_pred_mean + 2*y_pred_std,
                        alpha=0.3, color='red', label='95%预测区间')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('贝叶斯线性回归')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 权重后验分布
        ax2 = axes[0, 1]
        n_samples = 1000
        weight_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, n_samples)
        
        ax2.scatter(weight_samples[:, 0], weight_samples[:, 1], alpha=0.5, s=10, color='blue')
        ax2.scatter(posterior_mean[0], posterior_mean[1], color='red', s=100, 
                   marker='x', linewidth=3, label='后验均值')
        ax2.set_xlabel('w₀ (截距)')
        ax2.set_ylabel('w₁ (斜率)')
        ax2.set_title('权重参数后验分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 多条预测线（参数不确定性）
        ax3 = axes[1, 0]
        ax3.scatter(X, y, alpha=0.7, color='blue', label='观测数据')
        
        # 从后验采样多组权重并绘制预测线
        for i in range(20):
            w_sample = weight_samples[i]
            y_sample = X_test_design @ w_sample
            ax3.plot(X_test, y_sample, 'r-', alpha=0.3, linewidth=1)
        
        ax3.plot(X_test, y_pred_mean, 'black', linewidth=2, label='平均预测')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('参数不确定性可视化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 预测不确定性分解
        ax4 = axes[1, 1]
        
        # 分解不确定性
        epistemic_var = []  # 认知不确定性（参数不确定性）
        aleatoric_var = noise_std**2  # 偶然不确定性（观测噪声）
        
        for i in range(len(X_test_design)):
            x_test = X_test_design[i]
            epi_var = x_test @ posterior_cov @ x_test
            epistemic_var.append(epi_var)
        
        epistemic_std = np.sqrt(epistemic_var)
        aleatoric_std = np.sqrt(aleatoric_var)
        total_std = np.sqrt(np.array(epistemic_var) + aleatoric_var)
        
        ax4.plot(X_test, epistemic_std, label='认知不确定性', linewidth=2)
        ax4.axhline(aleatoric_std, color='red', linestyle='--', 
                   label=f'偶然不确定性={aleatoric_std:.2f}')
        ax4.plot(X_test, total_std, label='总不确定性', linewidth=2, color='black')
        
        ax4.set_xlabel('x')
        ax4.set_ylabel('预测标准差')
        ax4.set_title('不确定性分解')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_frequentist(self):
        """与频率论方法比较"""
        print("\n=== 贝叶斯 vs 频率论回归比较 ===")
        
        comparison = {
            "参数估计": {
                "频率论": "点估计 (MLE/OLS)",
                "贝叶斯": "分布估计 (后验分布)",
                "优势": "贝叶斯提供不确定性信息"
            },
            "正则化": {
                "频率论": "显式正则化 (Ridge, Lasso)",
                "贝叶斯": "隐式正则化 (先验分布)",
                "连接": "Ridge回归 ≈ 高斯先验"
            },
            "预测": {
                "频率论": "点预测 + 置信区间",
                "贝叶斯": "预测分布",
                "不确定性": "贝叶斯自然包含参数不确定性"
            },
            "模型选择": {
                "频率论": "交叉验证、信息准则",
                "贝叶斯": "模型证据、贝叶斯因子",
                "自动性": "贝叶斯自动进行模型平均"
            }
        }
        
        for aspect, details in comparison.items():
            print(f"{aspect}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        return comparison

class VariationalInference:
    """变分推断"""
    
    def __init__(self):
        pass
    
    def variational_theory(self):
        """变分推断理论"""
        print("=== 变分推断理论 ===")
        
        print("核心思想:")
        print("- 后验分布p(θ|D)通常难以计算")
        print("- 用简单分布q(θ)近似后验分布")
        print("- 最小化KL散度: KL(q(θ)||p(θ|D))")
        print()
        
        print("变分下界 (ELBO):")
        print("log p(D) = ELBO + KL(q(θ)||p(θ|D))")
        print("ELBO = E_q[log p(D|θ)] - KL(q(θ)||p(θ))")
        print("     = 重构项 - 正则化项")
        print()
        
        print("变分族选择:")
        variational_families = {
            "平均场": {
                "假设": "q(θ) = ∏ᵢ qᵢ(θᵢ)",
                "优点": "计算简单，易于实现",
                "缺点": "忽略参数间相关性",
                "应用": "大多数变分方法的基础"
            },
            "结构化变分": {
                "假设": "保留某些相关性结构",
                "例子": "分组独立、马尔科夫结构",
                "权衡": "表达能力 vs 计算复杂度",
                "应用": "时序模型、空间模型"
            },
            "神经变分": {
                "假设": "用神经网络参数化q(θ)",
                "优点": "表达能力强",
                "缺点": "计算复杂，局部最优",
                "应用": "深度学习、复杂模型"
            }
        }
        
        for family, details in variational_families.items():
            print(f"{family}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 实现简单的变分推断
        self.implement_variational_inference()
        
        return variational_families
    
    def implement_variational_inference(self):
        """实现变分推断示例"""
        print("=== 变分推断示例：贝叶斯线性回归 ===")
        
        # 生成数据
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        true_w = np.array([1.5, -0.8])
        noise_std = 0.2
        y = X @ true_w + np.random.normal(0, noise_std, n_samples)
        
        # 变分推断
        # 假设变分后验为高斯分布：q(w) = N(μ, Σ)
        
        def elbo(params):
            """计算ELBO"""
            mu = params[:2]
            log_sigma = params[2:4]
            sigma = np.exp(log_sigma)
            
            # 先验参数
            prior_mean = np.zeros(2)
            prior_std = 2.0
            
            # 重构项：E_q[log p(y|X,w)]
            reconstruction = 0
            n_mc_samples = 100
            for _ in range(n_mc_samples):
                w_sample = mu + sigma * np.random.randn(2)
                y_pred = X @ w_sample
                reconstruction += -0.5 * np.sum((y - y_pred)**2) / noise_std**2
            reconstruction /= n_mc_samples
            
            # KL项：KL(q(w)||p(w))
            kl_term = 0.5 * (np.sum(sigma**2 / prior_std**2) + 
                             np.sum((mu - prior_mean)**2 / prior_std**2) -
                             len(mu) - 2 * np.sum(log_sigma) + 
                             len(mu) * np.log(prior_std**2))
            
            return -(reconstruction - kl_term)  # 负号因为要最小化
        
        # 优化
        initial_params = np.array([0.0, 0.0, 0.0, 0.0])
        result = minimize(elbo, initial_params, method='L-BFGS-B')
        
        # 提取结果
        mu_opt = result.x[:2]
        sigma_opt = np.exp(result.x[2:4])
        
        print(f"真实权重: {true_w}")
        print(f"变分后验均值: {mu_opt}")
        print(f"变分后验标准差: {sigma_opt}")
        
        # 可视化结果
        self.visualize_variational_inference(true_w, mu_opt, sigma_opt)
        
        return mu_opt, sigma_opt
    
    def visualize_variational_inference(self, true_w, mu_opt, sigma_opt):
        """可视化变分推断结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 权重分布比较
        ax1 = axes[0]
        
        # 变分后验样本
        n_samples = 1000
        w_samples = mu_opt[:, np.newaxis] + sigma_opt[:, np.newaxis] * np.random.randn(2, n_samples)
        
        ax1.scatter(w_samples[0], w_samples[1], alpha=0.5, s=10, label='变分后验样本')
        ax1.scatter(mu_opt[0], mu_opt[1], color='red', s=100, marker='x', 
                   linewidth=3, label='变分后验均值')
        ax1.scatter(true_w[0], true_w[1], color='green', s=100, marker='o',
                   label='真实权重')
        
        ax1.set_xlabel('w₁')
        ax1.set_ylabel('w₂')
        ax1.set_title('权重分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ELBO收敛曲线（模拟）
        ax2 = axes[1]
        iterations = np.arange(1, 101)
        elbo_values = -1000 * np.exp(-iterations/20) + np.random.normal(0, 10, 100)
        
        ax2.plot(iterations, elbo_values, 'b-', linewidth=2)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('ELBO')
        ax2.set_title('ELBO收敛曲线')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class BayesianNeuralNetworks:
    """贝叶斯神经网络"""
    
    def __init__(self):
        pass
    
    def bnn_theory(self):
        """贝叶斯神经网络理论"""
        print("=== 贝叶斯神经网络理论 ===")
        
        print("核心思想:")
        print("- 将神经网络权重视为随机变量")
        print("- 学习权重的分布而非点估计")
        print("- 预测时考虑权重不确定性")
        print()
        
        print("挑战:")
        challenges = {
            "高维后验": {
                "问题": "神经网络参数数量巨大",
                "困难": "精确后验推断不可行",
                "解决": "变分推断、采样方法"
            },
            "计算复杂度": {
                "问题": "训练和推断成本高",
                "原因": "需要处理参数分布",
                "缓解": "近似方法、网络剪枝"
            },
            "先验选择": {
                "问题": "如何选择合适的权重先验",
                "影响": "影响正则化效果",
                "方法": "分层贝叶斯、经验贝叶斯"
            },
            "不确定性校准": {
                "问题": "预测不确定性是否准确",
                "评估": "校准图、可靠性分析",
                "重要性": "安全关键应用"
            }
        }
        
        for challenge, details in challenges.items():
            print(f"{challenge}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 介绍主要方法
        self.bnn_methods()
        
        return challenges
    
    def bnn_methods(self):
        """BNN主要方法"""
        print("=== BNN主要方法 ===")
        
        methods = {
            "变分贝叶斯": {
                "思想": "用简单分布近似权重后验",
                "实现": "Bayes by Backprop",
                "优点": "训练相对简单",
                "缺点": "后验近似质量有限"
            },
            "MC Dropout": {
                "思想": "Dropout近似贝叶斯推断",
                "原理": "随机失活等价于变分推断",
                "优点": "易于实现，计算高效",
                "局限": "近似质量有争议"
            },
            "深度集成": {
                "思想": "训练多个神经网络",
                "预测": "集成预测获得不确定性",
                "优点": "实用有效",
                "缺点": "计算成本高"
            },
            "拉普拉斯近似": {
                "思想": "后验分布的二阶近似",
                "计算": "基于Hessian矩阵",
                "优点": "理论基础强",
                "缺点": "Hessian计算困难"
            },
            "MCMC": {
                "思想": "采样获得后验样本",
                "方法": "HMC、SGLD等",
                "优点": "理论保证",
                "缺点": "计算成本极高"
            }
        }
        
        for method, details in methods.items():
            print(f"{method}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化方法比较
        self.visualize_bnn_methods()
        
        return methods
    
    def visualize_bnn_methods(self):
        """可视化BNN方法比较"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 方法性能比较
        methods = ['变分贝叶斯', 'MC Dropout', '深度集成', 'MCMC']
        accuracy = [85, 83, 87, 88]
        uncertainty_quality = [70, 60, 80, 95]
        computational_cost = [60, 20, 80, 100]
        
        x = np.arange(len(methods))
        width = 0.25
        
        bars1 = axes[0, 0].bar(x - width, accuracy, width, label='预测准确率', alpha=0.7)
        bars2 = axes[0, 0].bar(x, uncertainty_quality, width, label='不确定性质量', alpha=0.7)
        bars3 = axes[0, 0].bar(x + width, computational_cost, width, label='计算成本', alpha=0.7)
        
        axes[0, 0].set_xlabel('方法')
        axes[0, 0].set_ylabel('评分')
        axes[0, 0].set_title('BNN方法比较')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 不确定性vs准确性
        axes[1, 0].scatter(uncertainty_quality, accuracy, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            axes[1, 0].annotate(method, (uncertainty_quality[i], accuracy[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        axes[1, 0].set_xlabel('不确定性质量')
        axes[1, 0].set_ylabel('预测准确率')
        axes[1, 0].set_title('不确定性质量 vs 预测准确率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. 训练时间vs性能
        training_time = [100, 30, 300, 1000]  # 相对训练时间
        
        axes[0, 1].scatter(training_time, accuracy, s=100, alpha=0.7, color='red')
        for i, method in enumerate(methods):
            axes[0, 1].annotate(method, (training_time[i], accuracy[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        axes[0, 1].set_xlabel('训练时间 (相对)')
        axes[0, 1].set_ylabel('预测准确率')
        axes[0, 1].set_title('训练时间 vs 性能')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 4. 应用场景雷达图
        applications = ['计算机视觉', '自然语言处理', '医疗诊断', '自动驾驶', '金融预测']
        n_apps = len(applications)
        
        # 不同方法在各应用场景的适用性
        variational_scores = [3, 4, 3, 2, 4]
        dropout_scores = [4, 4, 2, 2, 3]
        ensemble_scores = [5, 4, 4, 4, 4]
        mcmc_scores = [2, 2, 5, 1, 3]
        
        angles = np.linspace(0, 2*np.pi, n_apps, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax_radar = plt.subplot(2, 2, 4, projection='polar')
        
        for scores, label, color in zip([variational_scores, dropout_scores, ensemble_scores, mcmc_scores],
                                       ['变分贝叶斯', 'MC Dropout', '深度集成', 'MCMC'],
                                       ['blue', 'green', 'red', 'orange']):
            scores += scores[:1]  # 闭合图形
            ax_radar.plot(angles, scores, 'o-', linewidth=2, label=label, color=color)
            ax_radar.fill(angles, scores, alpha=0.1, color=color)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(applications)
        ax_radar.set_ylim(0, 5)
        ax_radar.set_title('应用场景适用性')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        
        plt.tight_layout()
        plt.show()

def comprehensive_bayesian_ml_summary():
    """贝叶斯机器学习综合总结"""
    print("=== 贝叶斯机器学习综合总结 ===")
    
    summary = {
        "核心理念": {
            "概率建模": "一切皆概率分布",
            "不确定性": "参数和预测的不确定性量化",
            "先验知识": "自然融合领域知识",
            "一致性": "理论框架的数学一致性"
        },
        
        "主要方法": {
            "精确推断": "共轭先验、解析解",
            "近似推断": "变分推断、拉普拉斯近似",
            "采样方法": "MCMC、重要性采样",
            "深度集成": "多模型集成获得不确定性"
        },
        
        "应用领域": {
            "回归分析": "贝叶斯线性/非线性回归",
            "分类问题": "贝叶斯逻辑回归、朴素贝叶斯",
            "神经网络": "贝叶斯神经网络、不确定性量化",
            "高斯过程": "非参数贝叶斯、函数分布"
        },
        
        "优势特点": {
            "不确定性": "自然的不确定性量化",
            "正则化": "先验作为自然正则化",
            "小样本": "先验信息在小样本时有效",
            "模型选择": "贝叶斯因子、模型平均"
        },
        
        "计算挑战": {
            "积分困难": "高维积分通常无解析解",
            "计算复杂": "采样和变分推断成本高",
            "近似质量": "近似方法的准确性问题",
            "收敛判断": "MCMC收敛诊断"
        },
        
        "现代发展": {
            "变分自编码器": "VAE中的变分推断",
            "神经后验估计": "神经网络近似后验",
            "概率编程": "Stan、PyMC3等工具",
            "近似贝叶斯计算": "ABC方法"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("贝叶斯机器学习理论指南加载完成！")
```

## 参考文献 📚

- Murphy (2012): "Machine Learning: A Probabilistic Perspective"
- Bishop (2006): "Pattern Recognition and Machine Learning"
- Gelman et al. (2013): "Bayesian Data Analysis"
- Blei et al. (2017): "Variational Inference: A Review for Statisticians"
- MacKay (1992): "A Practical Bayesian Framework for Backpropagation Networks"

## 下一步学习
- [高斯过程](gaussian_processes.md) - 非参数贝叶斯方法
- [变分自编码器](../generative_models.md) - 深度生成模型中的变分推断
- [概率编程](probabilistic_programming.md) - 现代贝叶斯建模工具