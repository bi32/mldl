# 优化算法理论：从梯度下降到现代优化器 🎯

深入理解机器学习中的优化理论、算法和实践技巧。

## 1. 优化基础理论 📐

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim
from scipy.optimize import minimize
import seaborn as sns

class OptimizationFoundations:
    """优化基础理论"""
    
    def __init__(self):
        self.functions = {}
        self.optimizers = {}
    
    def convex_analysis(self):
        """凸优化分析"""
        print("=== 凸优化理论基础 ===")
        
        # 凸函数定义和性质
        print("凸函数定义:")
        print("f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y), ∀λ ∈ [0,1]")
        print()
        
        # 可视化凸函数和非凸函数
        x = np.linspace(-3, 3, 100)
        
        # 凸函数示例
        convex_funcs = {
            'x²': x**2,
            'e^x': np.exp(x),
            '|x|': np.abs(x),
            'max(0,x)': np.maximum(0, x)
        }
        
        # 非凸函数示例  
        nonconvex_funcs = {
            'x³': x**3,
            'sin(x)': np.sin(x),
            'x⁴ - 2x²': x**4 - 2*x**2
        }
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 绘制凸函数
        for i, (name, y) in enumerate(convex_funcs.items()):
            axes[0, i].plot(x, y, 'b-', linewidth=2)
            axes[0, i].set_title(f'凸函数: {name}')
            axes[0, i].grid(True, alpha=0.3)
        
        # 绘制非凸函数
        for i, (name, y) in enumerate(nonconvex_funcs.items()):
            if i < 3:
                axes[1, i].plot(x, y, 'r-', linewidth=2)
                axes[1, i].set_title(f'非凸函数: {name}')
                axes[1, i].grid(True, alpha=0.3)
        
        # 局部最优vs全局最优
        x_nonconvex = np.linspace(-2, 2, 100)
        y_nonconvex = x_nonconvex**4 - 2*x_nonconvex**2 + 0.5
        
        axes[1, 3].plot(x_nonconvex, y_nonconvex, 'r-', linewidth=2)
        
        # 标注局部最优点
        local_minima = [-1, 1]
        for xmin in local_minima:
            ymin = xmin**4 - 2*xmin**2 + 0.5
            axes[1, 3].plot(xmin, ymin, 'go', markersize=8, label='局部最优')
        
        # 标注全局最优点
        global_min_x = 0
        global_min_y = 0.5
        axes[1, 3].plot(global_min_x, global_min_y, 'ro', markersize=8, label='全局最优')
        axes[1, 3].set_title('局部vs全局最优')
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 凸优化的优势
        print("凸优化优势:")
        print("1. 局部最优即全局最优")
        print("2. 可以保证收敛到最优解")
        print("3. 有成熟的理论和算法")
        print("4. 计算复杂度通常较低")
        print()
        
        return convex_funcs, nonconvex_funcs
    
    def gradient_analysis(self):
        """梯度分析"""
        print("=== 梯度理论分析 ===")
        
        print("梯度的几何意义:")
        print("1. 梯度方向是函数增长最快的方向")
        print("2. 梯度大小表示变化率") 
        print("3. 负梯度方向是下降最快的方向")
        print()
        
        # 2D函数梯度可视化
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        
        # 示例函数: f(x,y) = x² + y²
        Z = X**2 + Y**2
        
        # 计算梯度: ∇f = (2x, 2y)
        grad_x = 2 * X
        grad_y = 2 * Y
        
        plt.figure(figsize=(12, 5))
        
        # 等高线和梯度场
        plt.subplot(1, 2, 1)
        contour = plt.contour(X, Y, Z, levels=15, alpha=0.6)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.quiver(X[::3, ::3], Y[::3, ::3], 
                  grad_x[::3, ::3], grad_y[::3, ::3], 
                  scale=50, color='red', alpha=0.7)
        plt.title('梯度场可视化')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        
        # 3D表面图
        ax = plt.subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        
        # 添加一些梯度向量
        for i in range(0, 20, 4):
            for j in range(0, 20, 4):
                ax.quiver(X[i,j], Y[i,j], Z[i,j], 
                         grad_x[i,j], grad_y[i,j], 0,
                         length=0.3, color='red', alpha=0.8)
        
        ax.set_title('3D函数及其梯度')
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        ax.set_zlabel('f(x,y)')
        
        plt.tight_layout()
        plt.show()
        
        return X, Y, Z, grad_x, grad_y
    
    def hessian_analysis(self):
        """Hessian矩阵分析"""
        print("=== Hessian矩阵分析 ===")
        
        print("Hessian矩阵定义:")
        print("H_ij = ∂²f / ∂x_i∂x_j")
        print()
        print("二阶条件:")
        print("- H > 0 (正定): 局部最小值")
        print("- H < 0 (负定): 局部最大值") 
        print("- H 不定: 鞍点")
        print("- H ≥ 0 (半正定): 可能的最小值")
        print()
        
        # 不同Hessian特征值对应的函数形状
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        # 1. 正定 Hessian (椭圆形)
        Z1 = X**2 + 4*Y**2
        im1 = axes[0, 0].contour(X, Y, Z1, levels=15)
        axes[0, 0].set_title('正定Hessian\n(局部最小值)')
        axes[0, 0].set_aspect('equal')
        
        # 2. 负定 Hessian (倒椭圆形)
        Z2 = -(X**2 + 4*Y**2) + 10
        axes[0, 1].contour(X, Y, Z2, levels=15)
        axes[0, 1].set_title('负定Hessian\n(局部最大值)')
        axes[0, 1].set_aspect('equal')
        
        # 3. 不定 Hessian (鞍点)
        Z3 = X**2 - Y**2
        axes[1, 0].contour(X, Y, Z3, levels=15)
        axes[1, 0].set_title('不定Hessian\n(鞍点)')
        axes[1, 0].set_aspect('equal')
        
        # 4. 半正定 Hessian
        Z4 = X**2
        axes[1, 1].contour(X, Y, Z4, levels=15)
        axes[1, 1].set_title('半正定Hessian\n(平坦方向)')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        # 计算具体的Hessian矩阵
        print("示例: f(x,y) = x² + 4y²")
        print("Hessian矩阵:")
        print("H = [[2, 0],")
        print("     [0, 8]]")
        print("特征值: λ₁ = 2, λ₂ = 8 (都为正，正定)")
        print()
        
        return Z1, Z2, Z3, Z4
```

## 2. 梯度下降算法族 🏃‍♂️

```python
class GradientDescentFamily:
    """梯度下降算法族"""
    
    def __init__(self):
        self.optimizers = {}
        self.history = {}
    
    def vanilla_gradient_descent(self, func, grad_func, x0, lr=0.01, max_iter=1000, tol=1e-6):
        """标准梯度下降"""
        print("=== 标准梯度下降 ===")
        
        x = x0.copy()
        history = {'x': [x.copy()], 'f': [func(x)]}
        
        for i in range(max_iter):
            grad = grad_func(x)
            x_new = x - lr * grad
            
            # 检查收敛
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def momentum_gradient_descent(self, func, grad_func, x0, lr=0.01, momentum=0.9, 
                                 max_iter=1000, tol=1e-6):
        """动量梯度下降"""
        print("=== 动量梯度下降 ===")
        
        x = x0.copy()
        v = np.zeros_like(x)  # 速度项
        history = {'x': [x.copy()], 'f': [func(x)]}
        
        for i in range(max_iter):
            grad = grad_func(x)
            v = momentum * v + lr * grad  # 更新速度
            x_new = x - v  # 更新位置
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def nesterov_momentum(self, func, grad_func, x0, lr=0.01, momentum=0.9,
                         max_iter=1000, tol=1e-6):
        """Nesterov加速梯度下降"""
        print("=== Nesterov加速梯度下降 ===")
        
        x = x0.copy()
        v = np.zeros_like(x)
        history = {'x': [x.copy()], 'f': [func(x)]}
        
        for i in range(max_iter):
            # 向前看一步
            x_lookahead = x - momentum * v
            grad = grad_func(x_lookahead)  # 在lookahead位置计算梯度
            
            v = momentum * v + lr * grad
            x_new = x - v
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def compare_gradient_methods(self):
        """比较不同梯度下降方法"""
        print("=== 梯度下降方法比较 ===")
        
        # 定义测试函数: Rosenbrock函数
        def rosenbrock(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        def rosenbrock_grad(x):
            grad_x0 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            grad_x1 = 200 * (x[1] - x[0]**2)
            return np.array([grad_x0, grad_x1])
        
        # 初始点
        x0 = np.array([-1.2, 1.0])
        
        # 运行不同算法
        methods = {
            'Vanilla GD': lambda: self.vanilla_gradient_descent(
                rosenbrock, rosenbrock_grad, x0, lr=0.001, max_iter=5000),
            'Momentum': lambda: self.momentum_gradient_descent(
                rosenbrock, rosenbrock_grad, x0, lr=0.001, momentum=0.9, max_iter=5000),
            'Nesterov': lambda: self.nesterov_momentum(
                rosenbrock, rosenbrock_grad, x0, lr=0.001, momentum=0.9, max_iter=5000)
        }
        
        results = {}
        for name, method in methods.items():
            print(f"\n运行 {name}:")
            x_final, history = method()
            results[name] = history
            print(f"最终点: [{x_final[0]:.4f}, {x_final[1]:.4f}]")
            print(f"最终函数值: {rosenbrock(x_final):.6f}")
            print(f"迭代次数: {len(history['x'])}")
        
        # 可视化比较
        self.visualize_optimization_paths(rosenbrock, results)
        
        return results
    
    def visualize_optimization_paths(self, func, results):
        """可视化优化路径"""
        # 创建函数的等高线图
        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(-0.5, 1.5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func(np.array([xi, yi])) for xi in x] for yi in y])
        
        plt.figure(figsize=(12, 8))
        
        # 等高线图
        levels = np.logspace(0, 3, 20)
        contour = plt.contour(X, Y, Z, levels=levels, alpha=0.6)
        plt.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
        
        # 绘制优化路径
        colors = ['red', 'blue', 'green']
        markers = ['o-', 's-', '^-']
        
        for i, (name, history) in enumerate(results.items()):
            path = np.array(history['x'])
            plt.plot(path[:, 0], path[:, 1], markers[i], 
                    color=colors[i], label=name, markersize=3, alpha=0.7)
            
            # 标注起点和终点
            plt.plot(path[0, 0], path[0, 1], 'ko', markersize=8, label='起点' if i == 0 else '')
            plt.plot(path[-1, 0], path[-1, 1], colors[i], marker='*', 
                    markersize=12, label=f'{name} 终点')
        
        # 标注全局最优点
        plt.plot(1, 1, 'gold', marker='*', markersize=15, label='全局最优点')
        
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title('优化路径比较 (Rosenbrock函数)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 收敛曲线
        plt.figure(figsize=(10, 6))
        
        for name, history in results.items():
            plt.semilogy(history['f'], label=name, linewidth=2)
        
        plt.xlabel('迭代次数')
        plt.ylabel('函数值 (log scale)')
        plt.title('收敛曲线比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
```

## 3. 自适应学习率算法 🎛️

```python
class AdaptiveLearningRateOptimizers:
    """自适应学习率优化器"""
    
    def __init__(self):
        pass
    
    def adagrad_implementation(self, func, grad_func, x0, lr=0.1, eps=1e-8, 
                              max_iter=1000, tol=1e-6):
        """AdaGrad实现"""
        print("=== AdaGrad优化器 ===")
        print("特点: 累积历史梯度的平方，自动调节学习率")
        print("公式: x_t = x_{t-1} - lr / √(G_t + ε) * g_t")
        print("其中 G_t = G_{t-1} + g_t²")
        print()
        
        x = x0.copy()
        G = np.zeros_like(x)  # 梯度平方和累积
        history = {'x': [x.copy()], 'f': [func(x)], 'lr': []}
        
        for i in range(max_iter):
            grad = grad_func(x)
            G += grad**2  # 累积梯度平方
            
            # 自适应学习率
            adapted_lr = lr / np.sqrt(G + eps)
            x_new = x - adapted_lr * grad
            
            history['lr'].append(adapted_lr.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def rmsprop_implementation(self, func, grad_func, x0, lr=0.01, decay_rate=0.9,
                              eps=1e-8, max_iter=1000, tol=1e-6):
        """RMSprop实现"""
        print("=== RMSprop优化器 ===")
        print("特点: 指数移动平均的梯度平方，解决AdaGrad学习率衰减过快问题")
        print("公式: x_t = x_{t-1} - lr / √(v_t + ε) * g_t")
        print("其中 v_t = β * v_{t-1} + (1-β) * g_t²")
        print()
        
        x = x0.copy()
        v = np.zeros_like(x)  # 梯度平方的移动平均
        history = {'x': [x.copy()], 'f': [func(x)], 'lr': []}
        
        for i in range(max_iter):
            grad = grad_func(x)
            v = decay_rate * v + (1 - decay_rate) * grad**2
            
            adapted_lr = lr / np.sqrt(v + eps)
            x_new = x - adapted_lr * grad
            
            history['lr'].append(adapted_lr.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def adam_implementation(self, func, grad_func, x0, lr=0.001, beta1=0.9, beta2=0.999,
                           eps=1e-8, max_iter=1000, tol=1e-6):
        """Adam优化器实现"""
        print("=== Adam优化器 ===")
        print("特点: 结合动量和自适应学习率")
        print("公式: m_t = β₁*m_{t-1} + (1-β₁)*g_t")
        print("      v_t = β₂*v_{t-1} + (1-β₂)*g_t²")
        print("      m̂_t = m_t / (1-β₁ᵗ)")
        print("      v̂_t = v_t / (1-β₂ᵗ)")
        print("      x_t = x_{t-1} - lr * m̂_t / (√v̂_t + ε)")
        print()
        
        x = x0.copy()
        m = np.zeros_like(x)  # 一阶矩估计
        v = np.zeros_like(x)  # 二阶矩估计
        history = {'x': [x.copy()], 'f': [func(x)], 'lr': []}
        
        for t in range(1, max_iter + 1):
            grad = grad_func(x)
            
            # 更新有偏一阶和二阶矩估计
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            # 偏差修正
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # 更新参数
            adapted_lr = lr / (np.sqrt(v_hat) + eps)
            x_new = x - adapted_lr * m_hat
            
            history['lr'].append(adapted_lr.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {t} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def adamw_implementation(self, func, grad_func, x0, lr=0.001, beta1=0.9, beta2=0.999,
                            eps=1e-8, weight_decay=0.01, max_iter=1000, tol=1e-6):
        """AdamW优化器实现"""
        print("=== AdamW优化器 ===")
        print("特点: Adam + 权重衰减（L2正则化的正确实现）")
        print("区别: 权重衰减直接应用于参数，不经过动量")
        print()
        
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = {'x': [x.copy()], 'f': [func(x)], 'lr': []}
        
        for t in range(1, max_iter + 1):
            grad = grad_func(x)
            
            # 权重衰减
            grad = grad + weight_decay * x
            
            # Adam步骤
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            adapted_lr = lr / (np.sqrt(v_hat) + eps)
            x_new = x - adapted_lr * m_hat
            
            history['lr'].append(adapted_lr.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {t} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def compare_adaptive_optimizers(self):
        """比较自适应优化器"""
        print("=== 自适应优化器比较 ===")
        
        # 测试函数：Beale函数 (多峰函数)
        def beale(x):
            return (1.5 - x[0] + x[0]*x[1])**2 + \
                   (2.25 - x[0] + x[0]*x[1]**2)**2 + \
                   (2.625 - x[0] + x[0]*x[1]**3)**2
        
        def beale_grad(x):
            t1 = 1.5 - x[0] + x[0]*x[1]
            t2 = 2.25 - x[0] + x[0]*x[1]**2  
            t3 = 2.625 - x[0] + x[0]*x[1]**3
            
            grad_x0 = 2*t1*(-1 + x[1]) + 2*t2*(-1 + x[1]**2) + 2*t3*(-1 + x[1]**3)
            grad_x1 = 2*t1*x[0] + 2*t2*x[0]*2*x[1] + 2*t3*x[0]*3*x[1]**2
            
            return np.array([grad_x0, grad_x1])
        
        x0 = np.array([1.0, 1.0])
        
        # 运行不同优化器
        optimizers = {
            'AdaGrad': lambda: self.adagrad_implementation(beale, beale_grad, x0, lr=0.1),
            'RMSprop': lambda: self.rmsprop_implementation(beale, beale_grad, x0, lr=0.01),
            'Adam': lambda: self.adam_implementation(beale, beale_grad, x0, lr=0.01),
            'AdamW': lambda: self.adamw_implementation(beale, beale_grad, x0, lr=0.01)
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            print(f"\n运行 {name}:")
            try:
                x_final, history = optimizer()
                results[name] = history
                print(f"最终点: [{x_final[0]:.4f}, {x_final[1]:.4f}]")
                print(f"最终函数值: {beale(x_final):.6f}")
                print(f"迭代次数: {len(history['x'])}")
            except Exception as e:
                print(f"优化失败: {e}")
        
        # 可视化学习率变化
        self.visualize_adaptive_learning_rates(results)
        
        return results
    
    def visualize_adaptive_learning_rates(self, results):
        """可视化自适应学习率变化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 函数值收敛
        axes[0, 0].set_title('收敛曲线')
        for name, history in results.items():
            if 'f' in history:
                axes[0, 0].semilogy(history['f'], label=name, linewidth=2)
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('函数值 (log)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率变化（第一个维度）
        axes[0, 1].set_title('学习率变化 (第一维度)')
        for name, history in results.items():
            if 'lr' in history and len(history['lr']) > 0:
                lr_dim0 = [lr[0] if isinstance(lr, np.ndarray) else lr for lr in history['lr']]
                axes[0, 1].plot(lr_dim0, label=name, linewidth=2)
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('学习率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 参数轨迹
        axes[1, 0].set_title('参数空间轨迹')
        for name, history in results.items():
            if 'x' in history:
                path = np.array(history['x'])
                axes[1, 0].plot(path[:, 0], path[:, 1], 'o-', 
                               label=name, markersize=3, alpha=0.7)
        axes[1, 0].set_xlabel('x₁')
        axes[1, 0].set_ylabel('x₂')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 梯度范数
        axes[1, 1].set_title('收敛速度比较')
        for name, history in results.items():
            if 'f' in history:
                # 计算收敛速度（函数值变化率）
                f_values = np.array(history['f'])
                if len(f_values) > 1:
                    convergence_rate = np.abs(np.diff(f_values))
                    axes[1, 1].semilogy(convergence_rate, label=name, linewidth=2)
        axes[1, 1].set_xlabel('迭代次数')
        axes[1, 1].set_ylabel('函数值变化 (log)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

## 4. 二阶优化方法 📈

```python
class SecondOrderOptimizers:
    """二阶优化方法"""
    
    def __init__(self):
        pass
    
    def newton_method(self, func, grad_func, hess_func, x0, max_iter=100, tol=1e-6):
        """牛顿法"""
        print("=== 牛顿法 ===")
        print("原理: 利用二阶信息快速收敛")
        print("公式: x_{k+1} = x_k - H⁻¹(x_k) * ∇f(x_k)")
        print("优点: 二次收敛速度")
        print("缺点: 需要计算和求逆Hessian矩阵")
        print()
        
        x = x0.copy()
        history = {'x': [x.copy()], 'f': [func(x)]}
        
        for i in range(max_iter):
            grad = grad_func(x)
            hess = hess_func(x)
            
            # 检查Hessian是否正定
            try:
                # 牛顿步长
                delta_x = np.linalg.solve(hess, grad)
                x_new = x - delta_x
            except np.linalg.LinAlgError:
                print(f"Hessian矩阵奇异，在第 {i} 步停止")
                break
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def quasi_newton_bfgs(self, func, grad_func, x0, max_iter=1000, tol=1e-6):
        """BFGS拟牛顿法"""
        print("=== BFGS拟牛顿法 ===")
        print("原理: 用正定矩阵近似Hessian逆矩阵")
        print("优点: 超线性收敛，不需要计算真实Hessian")
        print("缺点: 需要存储n×n矩阵")
        print()
        
        n = len(x0)
        x = x0.copy()
        H = np.eye(n)  # Hessian逆矩阵的近似
        grad = grad_func(x)
        
        history = {'x': [x.copy()], 'f': [func(x)]}
        
        for i in range(max_iter):
            # 计算搜索方向
            p = -H @ grad
            
            # 线搜索（简化版本，使用固定步长）
            alpha = self.line_search(func, grad_func, x, p)
            
            # 更新位置
            x_new = x + alpha * p
            grad_new = grad_func(x_new)
            
            # BFGS更新公式
            s = x_new - x
            y = grad_new - grad
            
            if np.dot(s, y) > 1e-10:  # 确保数值稳定性
                rho = 1.0 / np.dot(y, s)
                I = np.eye(n)
                H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + \
                    rho * np.outer(s, s)
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            grad = grad_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def limited_memory_bfgs(self, func, grad_func, x0, m=10, max_iter=1000, tol=1e-6):
        """L-BFGS有限内存BFGS"""
        print("=== L-BFGS有限内存BFGS ===")
        print(f"原理: 只保存最近{m}步的信息近似Hessian")
        print("优点: 内存需求低，适合大规模问题")
        print("缺点: 收敛速度略慢于BFGS")
        print()
        
        x = x0.copy()
        grad = grad_func(x)
        
        # 历史信息存储
        s_list = []  # x的变化
        y_list = []  # 梯度的变化
        rho_list = []  # 1/(s^T y)
        
        history = {'x': [x.copy()], 'f': [func(x)]}
        
        for i in range(max_iter):
            # 两循环递归计算搜索方向
            q = grad.copy()
            alpha_list = []
            
            # 第一个循环（反向）
            for j in range(len(s_list)-1, -1, -1):
                alpha = rho_list[j] * np.dot(s_list[j], q)
                q = q - alpha * y_list[j]
                alpha_list.insert(0, alpha)
            
            # 初始Hessian近似
            if len(s_list) > 0:
                gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
                r = gamma * q
            else:
                r = q
            
            # 第二个循环（正向）
            for j in range(len(s_list)):
                beta = rho_list[j] * np.dot(y_list[j], r)
                r = r + (alpha_list[j] - beta) * s_list[j]
            
            p = -r  # 搜索方向
            
            # 线搜索
            alpha = self.line_search(func, grad_func, x, p)
            
            # 更新
            x_new = x + alpha * p
            grad_new = grad_func(x_new)
            
            s = x_new - x
            y = grad_new - grad
            
            if np.dot(s, y) > 1e-10:
                # 更新历史信息
                if len(s_list) >= m:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)
                
                s_list.append(s)
                y_list.append(y)
                rho_list.append(1.0 / np.dot(s, y))
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"收敛于第 {i+1} 步")
                break
            
            x = x_new
            grad = grad_new
            history['x'].append(x.copy())
            history['f'].append(func(x))
        
        return x, history
    
    def line_search(self, func, grad_func, x, p, alpha0=1.0, c1=1e-4, c2=0.9):
        """Wolfe条件线搜索"""
        alpha = alpha0
        phi0 = func(x)
        dphi0 = np.dot(grad_func(x), p)
        
        # 简化的回退线搜索
        for _ in range(20):
            phi_alpha = func(x + alpha * p)
            
            # Armijo条件
            if phi_alpha <= phi0 + c1 * alpha * dphi0:
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def compare_second_order_methods(self):
        """比较二阶优化方法"""
        print("=== 二阶优化方法比较 ===")
        
        # 测试函数：二次函数 f(x) = 1/2 * x^T A x - b^T x
        A = np.array([[4, 1], [1, 2]])
        b = np.array([1, 2])
        
        def quadratic(x):
            return 0.5 * x.T @ A @ x - b.T @ x
        
        def quadratic_grad(x):
            return A @ x - b
        
        def quadratic_hess(x):
            return A
        
        x0 = np.array([2.0, 2.0])
        
        # 运行不同方法
        methods = {
            'Newton': lambda: self.newton_method(quadratic, quadratic_grad, quadratic_hess, x0),
            'BFGS': lambda: self.quasi_newton_bfgs(quadratic, quadratic_grad, x0),
            'L-BFGS': lambda: self.limited_memory_bfgs(quadratic, quadratic_grad, x0, m=5)
        }
        
        results = {}
        for name, method in methods.items():
            print(f"\n运行 {name}:")
            x_final, history = method()
            results[name] = history
            print(f"最终点: [{x_final[0]:.6f}, {x_final[1]:.6f}]")
            print(f"最终函数值: {quadratic(x_final):.8f}")
            print(f"迭代次数: {len(history['x'])}")
        
        # 可视化比较
        self.visualize_second_order_comparison(quadratic, results)
        
        return results
    
    def visualize_second_order_comparison(self, func, results):
        """可视化二阶方法比较"""
        plt.figure(figsize=(15, 5))
        
        # 优化路径
        plt.subplot(1, 3, 1)
        
        # 创建等高线
        x = np.linspace(-0.5, 2.5, 50)
        y = np.linspace(-0.5, 2.5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func(np.array([xi, yi])) for xi in x] for yi in y])
        
        contour = plt.contour(X, Y, Z, levels=15, alpha=0.6)
        plt.clabel(contour, inline=True, fontsize=8)
        
        colors = ['red', 'blue', 'green']
        markers = ['o-', 's-', '^-']
        
        for i, (name, history) in enumerate(results.items()):
            path = np.array(history['x'])
            plt.plot(path[:, 0], path[:, 1], markers[i], 
                    color=colors[i], label=name, markersize=4, alpha=0.7)
        
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title('优化路径比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 收敛曲线
        plt.subplot(1, 3, 2)
        for name, history in results.items():
            plt.semilogy(history['f'], label=name, linewidth=2)
        
        plt.xlabel('迭代次数')
        plt.ylabel('函数值 (log)')
        plt.title('收敛曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 收敛速度
        plt.subplot(1, 3, 3)
        for name, history in results.items():
            errors = [np.linalg.norm(x - np.array([0.2, 0.8])) 
                     for x in history['x']]  # 距离真实解的误差
            if len(errors) > 1:
                plt.semilogy(errors, label=name, linewidth=2)
        
        plt.xlabel('迭代次数')
        plt.ylabel('误差 (log)')
        plt.title('收敛到最优解的误差')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

## 5. 学习率调度策略 📊

```python
class LearningRateScheduling:
    """学习率调度策略"""
    
    def __init__(self):
        self.schedules = {}
    
    def step_decay_schedule(self, initial_lr=0.1, drop_rate=0.5, epochs_drop=10, epochs=100):
        """步阶衰减"""
        schedule = []
        for epoch in range(epochs):
            lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
            schedule.append(lr)
        return schedule
    
    def exponential_decay_schedule(self, initial_lr=0.1, decay_rate=0.95, epochs=100):
        """指数衰减"""
        schedule = []
        for epoch in range(epochs):
            lr = initial_lr * (decay_rate ** epoch)
            schedule.append(lr)
        return schedule
    
    def cosine_annealing_schedule(self, initial_lr=0.1, min_lr=1e-5, epochs=100):
        """余弦退火"""
        schedule = []
        for epoch in range(epochs):
            lr = min_lr + (initial_lr - min_lr) * \
                (1 + np.cos(np.pi * epoch / epochs)) / 2
            schedule.append(lr)
        return schedule
    
    def warmup_cosine_schedule(self, initial_lr=0.1, warmup_epochs=10, 
                              min_lr=1e-5, epochs=100):
        """预热+余弦退火"""
        schedule = []
        for epoch in range(epochs):
            if epoch < warmup_epochs:
                # 线性预热
                lr = initial_lr * epoch / warmup_epochs
            else:
                # 余弦退火
                lr = min_lr + (initial_lr - min_lr) * \
                    (1 + np.cos(np.pi * (epoch - warmup_epochs) / 
                               (epochs - warmup_epochs))) / 2
            schedule.append(lr)
        return schedule
    
    def cyclic_lr_schedule(self, base_lr=0.001, max_lr=0.1, step_size=20, epochs=100):
        """循环学习率"""
        schedule = []
        for epoch in range(epochs):
            cycle = np.floor(1 + epoch / (2 * step_size))
            x = np.abs(epoch / step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
            schedule.append(lr)
        return schedule
    
    def one_cycle_schedule(self, max_lr=0.1, epochs=100, pct_start=0.3):
        """One Cycle学习率"""
        schedule = []
        step_up = int(epochs * pct_start)
        step_down = epochs - step_up
        
        for epoch in range(epochs):
            if epoch <= step_up:
                # 上升阶段
                lr = max_lr * epoch / step_up
            else:
                # 下降阶段
                lr = max_lr * (1 - (epoch - step_up) / step_down)
            schedule.append(lr)
        return schedule
    
    def visualize_lr_schedules(self):
        """可视化学习率调度策略"""
        epochs = 100
        
        schedules = {
            'Step Decay': self.step_decay_schedule(epochs=epochs),
            'Exponential': self.exponential_decay_schedule(epochs=epochs),
            'Cosine Annealing': self.cosine_annealing_schedule(epochs=epochs),
            'Warmup+Cosine': self.warmup_cosine_schedule(epochs=epochs),
            'Cyclic LR': self.cyclic_lr_schedule(epochs=epochs),
            'One Cycle': self.one_cycle_schedule(epochs=epochs)
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, schedule) in enumerate(schedules.items()):
            axes[i].plot(range(epochs), schedule, linewidth=2)
            axes[i].set_title(name)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Learning Rate')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 所有策略的比较
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(schedules)))
        for (name, schedule), color in zip(schedules.items(), colors):
            plt.plot(range(epochs), schedule, label=name, linewidth=2, color=color)
        
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('学习率调度策略比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return schedules
    
    def analyze_schedule_effects(self):
        """分析不同调度策略的效果"""
        print("=== 学习率调度策略分析 ===")
        
        strategies = {
            "Step Decay": {
                "优点": ["简单易实现", "阶段性调整"],
                "缺点": ["需要手动设定衰减点", "可能过早衰减"],
                "适用": ["传统CNN训练", "固定训练周期"]
            },
            "Exponential Decay": {
                "优点": ["平滑衰减", "自动调整"],
                "缺点": ["可能衰减过快", "后期学习率过小"],
                "适用": ["长期训练", "稳定收敛场景"]
            },
            "Cosine Annealing": {
                "优点": ["平滑变化", "避免局部最优"],
                "缺点": ["需要预知总训练轮数"],
                "适用": ["现代深度学习", "图像分类任务"]
            },
            "Warmup + Cosine": {
                "优点": ["稳定的初始化", "平滑收敛"],
                "缺点": ["参数较多", "调参复杂"],
                "适用": ["Transformer训练", "大模型训练"]
            },
            "Cyclic LR": {
                "优点": ["跳出局部最优", "加速收敛"],
                "缺点": ["可能不稳定", "需要调参"],
                "适用": ["探索性训练", "快速原型"]
            },
            "One Cycle": {
                "优点": ["快速收敛", "高准确率"],
                "缺点": ["需要预设最大学习率"],
                "适用": ["快速训练", "资源受限场景"]
            }
        }
        
        for strategy, details in strategies.items():
            print(f"\n{strategy}:")
            for key, values in details.items():
                print(f"  {key}: {', '.join(values)}")
        
        return strategies
```

## 6. 优化理论深入 🔬

```python
class OptimizationTheoryDeep:
    """优化理论深入分析"""
    
    @staticmethod
    def convergence_analysis():
        """收敛性分析"""
        print("=== 收敛性理论分析 ===")
        
        convergence_types = {
            "线性收敛": {
                "定义": "||x_{k+1} - x*|| ≤ c||x_k - x*||, 0 < c < 1",
                "收敛率": "指数级",
                "代表算法": ["梯度下降(强凸)", "坐标下降"],
                "收敛速度": "较慢"
            },
            "超线性收敛": {
                "定义": "lim(k→∞) ||x_{k+1} - x*|| / ||x_k - x*|| = 0", 
                "收敛率": "超过线性",
                "代表算法": ["BFGS", "SR1"],
                "收敛速度": "快"
            },
            "二次收敛": {
                "定义": "||x_{k+1} - x*|| ≤ c||x_k - x*||²",
                "收敛率": "平方级",
                "代表算法": ["牛顿法"],
                "收敛速度": "非常快"
            },
            "次线性收敛": {
                "定义": "||x_{k+1} - x*|| / ||x_k - x*|| → 1",
                "收敛率": "慢于线性",
                "代表算法": ["梯度下降(非强凸)"],
                "收敛速度": "慢"
            }
        }
        
        for conv_type, details in convergence_types.items():
            print(f"\n{conv_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 可视化不同收敛速度
        iterations = np.arange(0, 20)
        
        plt.figure(figsize=(12, 6))
        
        # 线性收敛
        linear_conv = 0.8 ** iterations
        plt.subplot(1, 2, 1)
        plt.semilogy(iterations, linear_conv, 'b-', linewidth=2, label='线性收敛 (c=0.8)')
        
        # 超线性收敛（近似）
        superlinear_conv = 0.8 ** (iterations ** 1.2)
        plt.semilogy(iterations, superlinear_conv, 'g-', linewidth=2, label='超线性收敛')
        
        # 二次收敛
        quadratic_conv = [1.0]
        for i in range(1, len(iterations)):
            quadratic_conv.append(min(quadratic_conv[-1]**2, 1e-15))
        plt.semilogy(iterations, quadratic_conv, 'r-', linewidth=2, label='二次收敛')
        
        # 次线性收敛
        sublinear_conv = 1.0 / (iterations + 1)
        plt.semilogy(iterations, sublinear_conv, 'm-', linewidth=2, label='次线性收敛')
        
        plt.xlabel('迭代次数')
        plt.ylabel('误差 (log scale)')
        plt.title('不同收敛速度比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 实际迭代次数比较
        plt.subplot(1, 2, 2)
        target_error = 1e-6
        
        # 计算达到目标误差需要的迭代次数
        linear_iters = np.log(target_error) / np.log(0.8)
        sublinear_iters = 1.0 / target_error - 1
        
        methods = ['线性\n(强凸)', '超线性\n(BFGS)', '二次\n(牛顿)', '次线性\n(非强凸)']
        iterations_needed = [linear_iters, linear_iters*0.3, 10, min(sublinear_iters, 10000)]
        
        bars = plt.bar(methods, iterations_needed, 
                      color=['blue', 'green', 'red', 'magenta'], alpha=0.7)
        plt.ylabel('达到1e-6误差的迭代次数')
        plt.title('收敛速度实际比较')
        plt.yscale('log')
        
        # 添加数值标签
        for bar, iters in zip(bars, iterations_needed):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(iters)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return convergence_types
    
    @staticmethod
    def lipschitz_smoothness():
        """Lipschitz平滑性分析"""
        print("=== Lipschitz平滑性 ===")
        
        print("定义:")
        print("函数f是L-smooth的，如果:")
        print("||∇f(x) - ∇f(y)|| ≤ L||x - y||")
        print()
        
        print("意义:")
        print("- L是梯度的Lipschitz常数")
        print("- 梯度变化不会太剧烈")
        print("- 允许使用固定步长")
        print("- 步长上界: α ≤ 2/L")
        print()
        
        # 可视化Lipschitz平滑函数
        x = np.linspace(-2, 2, 1000)
        
        plt.figure(figsize=(15, 5))
        
        # L-smooth函数示例
        plt.subplot(1, 3, 1)
        f1 = x**2  # L = 2
        plt.plot(x, f1, 'b-', linewidth=2, label='f(x) = x² (L=2)')
        plt.title('L-smooth函数')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 非smooth函数示例
        plt.subplot(1, 3, 2)
        f2 = np.abs(x)**1.5  # 不可微
        plt.plot(x, f2, 'r-', linewidth=2, label='f(x) = |x|^1.5')
        plt.title('非smooth函数')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 梯度比较
        plt.subplot(1, 3, 3)
        grad1 = 2 * x  # smooth函数的梯度
        grad2 = 1.5 * np.sign(x) * np.abs(x)**0.5  # 非smooth函数的梯度
        
        plt.plot(x, grad1, 'b-', linewidth=2, label="∇f(x) = 2x")
        plt.plot(x, grad2, 'r-', linewidth=2, label="∇f(x) = 1.5|x|^0.5·sign(x)")
        plt.title('梯度比较')
        plt.xlabel('x')
        plt.ylabel("∇f(x)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return None
    
    @staticmethod
    def strong_convexity():
        """强凸性分析"""
        print("=== 强凸性 ===")
        
        print("定义:")
        print("函数f是μ-强凸的，如果:")
        print("f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + μ/2||y-x||²")
        print()
        
        print("等价条件:")
        print("- Hessian矩阵: H ⪰ μI (所有特征值 ≥ μ)")
        print("- 梯度条件: (∇f(x)-∇f(y))ᵀ(x-y) ≥ μ||x-y||²")
        print()
        
        print("收敛保证:")
        print("- 梯度下降线性收敛")
        print("- 收敛率: (1 - μ/L)^k")
        print("- 条件数: κ = L/μ")
        print()
        
        # 可视化强凸函数
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        # 强凸函数：f(x,y) = 2x² + 3y²
        Z1 = 2*X**2 + 3*Y**2
        
        # 凸但非强凸：f(x,y) = x² + 0.1y²  
        Z2 = X**2 + 0.1*Y**2
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 强凸函数
        contour1 = axes[0].contour(X, Y, Z1, levels=20)
        axes[0].clabel(contour1, inline=True, fontsize=8)
        axes[0].set_title('强凸函数\n(μ=2, L=3, κ=1.5)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_aspect('equal')
        
        # 凸但非强凸函数
        contour2 = axes[1].contour(X, Y, Z2, levels=20)
        axes[1].clabel(contour2, inline=True, fontsize=8)
        axes[1].set_title('凸但非强凸\n(μ≈0, L=1, κ→∞)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        return None
    
    @staticmethod
    def condition_number_analysis():
        """条件数分析"""
        print("=== 条件数分析 ===")
        
        print("定义:")
        print("κ = L/μ (最大特征值/最小特征值)")
        print()
        
        print("影响:")
        print("- κ = 1: 球形函数，收敛最快") 
        print("- κ >> 1: 椭圆形函数，收敛慢")
        print("- κ → ∞: 病态问题，难以优化")
        print()
        
        # 可视化不同条件数的影响
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        condition_numbers = [1, 5, 20]
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        for i, kappa in enumerate(condition_numbers):
            # 创建具有指定条件数的函数
            Z = X**2 + kappa * Y**2
            
            # 等高线图
            axes[0, i].contour(X, Y, Z, levels=15, alpha=0.7)
            axes[0, i].set_title(f'条件数 κ = {kappa}')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('y')
            axes[0, i].set_aspect('equal')
            
            # 模拟梯度下降轨迹
            def grad_func(pos):
                return np.array([2*pos[0], 2*kappa*pos[1]])
            
            # 梯度下降
            pos = np.array([1.5, 1.5])
            trajectory = [pos.copy()]
            lr = 0.1 / kappa  # 调整学习率避免发散
            
            for _ in range(50):
                grad = grad_func(pos)
                pos = pos - lr * grad
                trajectory.append(pos.copy())
                
                if np.linalg.norm(grad) < 1e-3:
                    break
            
            trajectory = np.array(trajectory)
            axes[0, i].plot(trajectory[:, 0], trajectory[:, 1], 'r-o', 
                          markersize=3, alpha=0.7, label='梯度下降路径')
            axes[0, i].legend()
            
            # 收敛曲线
            distances = [np.linalg.norm(p) for p in trajectory]
            axes[1, i].semilogy(distances, 'b-', linewidth=2)
            axes[1, i].set_xlabel('迭代次数')
            axes[1, i].set_ylabel('距离原点 (log)')
            axes[1, i].set_title(f'收敛曲线 (κ = {kappa})')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return condition_numbers

def comprehensive_optimization_summary():
    """综合优化理论总结"""
    print("=== 优化算法综合总结 ===")
    
    summary = {
        "算法分类": {
            "零阶方法": "不使用梯度信息 (遗传算法、模拟退火)",
            "一阶方法": "使用梯度信息 (SGD、Adam等)",
            "二阶方法": "使用Hessian信息 (牛顿法、BFGS)"
        },
        
        "收敛速度": {
            "次线性": "O(1/k) - 梯度下降(非强凸)",
            "线性": "O(ρᵏ) - 梯度下降(强凸)", 
            "超线性": "比线性快 - BFGS",
            "二次": "O(ρᵏ²) - 牛顿法"
        },
        
        "适用场景": {
            "SGD": "大规模、随机优化",
            "Adam": "深度学习默认选择",
            "L-BFGS": "小规模、精确优化",
            "牛顿法": "小规模、二次函数"
        },
        
        "调参建议": {
            "学习率": "最重要的超参数",
            "批次大小": "影响噪声和并行性",
            "动量系数": "加速收敛，避免震荡",
            "正则化": "防止过拟合"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("优化算法理论指南加载完成！")
```

## 参考文献 📚

- Nocedal & Wright (2006): "Numerical Optimization"
- Boyd & Vandenberghe (2004): "Convex Optimization"
- Ruder (2016): "An overview of gradient descent optimization algorithms"
- Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
- Robbins & Monro (1951): "A Stochastic Approximation Method"

## 下一步学习
- [深度学习基础](../dl/basics.md) - 神经网络优化应用
- [模型训练技巧](../dl/training_tricks.md) - 实践优化技术
- [超参数调优](hyperparameter_tuning.md) - 自动化参数搜索