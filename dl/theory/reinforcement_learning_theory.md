# 强化学习理论：智能决策的数学基础 🎮

深入理解强化学习的核心理论、算法和应用。

## 1. 强化学习基础框架 🏗️

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from collections import defaultdict, deque
import random
import warnings
warnings.filterwarnings('ignore')

class ReinforcementLearningFramework:
    """强化学习基础框架"""
    
    def __init__(self):
        self.components = {}
    
    def mdp_framework(self):
        """马尔可夫决策过程框架"""
        print("=== 马尔可夫决策过程 (MDP) ===")
        
        print("MDP定义: M = (S, A, P, R, γ)")
        print("其中:")
        print("- S: 状态空间")
        print("- A: 动作空间") 
        print("- P: 状态转移概率 P(s'|s,a)")
        print("- R: 奖励函数 R(s,a,s')")
        print("- γ: 折扣因子 [0,1]")
        print()
        
        print("马尔可夫性质:")
        print("P(S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, ...) = P(S_{t+1}|S_t, A_t)")
        print("即：未来只依赖于当前状态和动作，与历史无关")
        print()
        
        # MDP组件分析
        components = {
            "状态空间": {
                "类型": ["离散", "连续", "混合"],
                "特点": "完全可观测、部分可观测",
                "示例": "棋盘位置、机器人坐标、股价"
            },
            "动作空间": {
                "类型": ["离散", "连续"],
                "约束": "动作可行性、物理限制",
                "示例": "象棋走法、方向盘转角、投资比例"
            },
            "奖励函数": {
                "设计": "稠密奖励 vs 稀疏奖励",
                "问题": "奖励塑形、延迟奖励",
                "示例": "游戏得分、交易收益、任务完成"
            },
            "转移概率": {
                "确定性": "确定性 vs 随机性",
                "建模": "已知模型 vs 免模型",
                "示例": "规则游戏、现实物理、市场动态"
            }
        }
        
        for component, details in components.items():
            print(f"{component}:")
            for key, value in details.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(value)}")
                else:
                    print(f"  {key}: {value}")
            print()
        
        # 可视化MDP
        self.visualize_mdp_concept()
        
        return components
    
    def visualize_mdp_concept(self):
        """可视化MDP概念"""
        print("=== MDP可视化 ===")
        
        # 创建简单的网格世界MDP
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 状态空间（网格世界）
        grid = np.zeros((4, 4))
        grid[0, 3] = 1  # 目标状态
        grid[1, 1] = -1  # 陷阱状态
        grid[2, 2] = -1  # 陷阱状态
        
        im1 = axes[0, 0].imshow(grid, cmap='RdYlGn', interpolation='nearest')
        axes[0, 0].set_title('状态空间\n(绿色=目标, 红色=陷阱)')
        axes[0, 0].set_xticks(range(4))
        axes[0, 0].set_yticks(range(4))
        
        # 标注状态
        for i in range(4):
            for j in range(4):
                if grid[i, j] == 1:
                    axes[0, 0].text(j, i, 'G', ha='center', va='center', fontsize=16, color='white')
                elif grid[i, j] == -1:
                    axes[0, 0].text(j, i, 'T', ha='center', va='center', fontsize=16, color='white')
                else:
                    axes[0, 0].text(j, i, f'({i},{j})', ha='center', va='center', fontsize=8)
        
        # 2. 动作空间（箭头）
        actions = ['↑', '→', '↓', '←']
        action_effects = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        axes[0, 1].set_xlim(-0.5, 3.5)
        axes[0, 1].set_ylim(-0.5, 3.5)
        
        # 在中心位置显示可能的动作
        center = (1.5, 1.5)
        for i, (action, (di, dj)) in enumerate(zip(actions, action_effects)):
            start_x, start_y = center
            end_x, end_y = start_x + dj * 0.3, start_y + di * 0.3
            axes[0, 1].arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                            head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            axes[0, 1].text(end_x + dj * 0.2, end_y + di * 0.2, action,
                           ha='center', va='center', fontsize=12)
        
        axes[0, 1].set_title('动作空间\n(四个方向移动)')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 奖励函数
        rewards = np.array([
            [-0.1, -0.1, -0.1, +10],
            [-0.1, -10, -0.1, -0.1],
            [-0.1, -0.1, -10, -0.1],
            [-0.1, -0.1, -0.1, -0.1]
        ])
        
        im3 = axes[1, 0].imshow(rewards, cmap='RdYlGn', interpolation='nearest')
        axes[1, 0].set_title('奖励函数')
        
        for i in range(4):
            for j in range(4):
                axes[1, 0].text(j, i, f'{rewards[i, j]:.1f}', 
                               ha='center', va='center', 
                               color='white' if abs(rewards[i, j]) > 5 else 'black')
        
        # 4. 价值函数示例
        # 简化的价值函数（手工设定用于演示）
        values = np.array([
            [6.8, 7.7, 8.8, 10.0],
            [6.1, 0.0, 7.9, 8.8],
            [5.5, 6.1, 0.0, 7.9],
            [4.9, 5.5, 6.1, 7.0]
        ])
        
        im4 = axes[1, 1].imshow(values, cmap='Blues', interpolation='nearest')
        axes[1, 1].set_title('状态价值函数 V(s)')
        
        for i in range(4):
            for j in range(4):
                axes[1, 1].text(j, i, f'{values[i, j]:.1f}', 
                               ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.show()
    
    def value_functions(self):
        """价值函数理论"""
        print("=== 价值函数理论 ===")
        
        print("状态价值函数:")
        print("V^π(s) = E_π[G_t | S_t = s]")
        print("      = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]")
        print()
        
        print("动作价值函数:")
        print("Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]")
        print("         = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]")
        print()
        
        print("Bellman方程:")
        print("V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]")
        print("Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]")
        print()
        
        print("最优性方程:")
        print("V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]")
        print("Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]")
        
        # 价值函数计算示例
        self.demonstrate_value_iteration()
        
        return self.policy_types()
    
    def demonstrate_value_iteration(self):
        """演示价值迭代算法"""
        print("\n=== 价值迭代算法演示 ===")
        
        # 简单的网格世界
        # 状态: (row, col), 动作: 0=上, 1=右, 2=下, 3=左
        rows, cols = 3, 4
        n_states = rows * cols
        n_actions = 4
        
        # 状态编码
        def state_to_idx(r, c):
            return r * cols + c
        
        def idx_to_state(idx):
            return idx // cols, idx % cols
        
        # 转移概率和奖励
        def get_transitions(state, action):
            r, c = state
            # 动作效果
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
            dr, dc = moves[action]
            
            # 新位置
            new_r, new_c = r + dr, c + dc
            
            # 边界检查
            if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
                new_r, new_c = r, c  # 撞墙不动
            
            # 奖励
            if (new_r, new_c) == (0, 3):  # 目标位置
                reward = 10
            elif (new_r, new_c) == (1, 1):  # 陷阱
                reward = -10
            else:
                reward = -0.1  # 生存惩罚
            
            return (new_r, new_c), reward
        
        # 价值迭代
        gamma = 0.9
        theta = 1e-6
        V = np.zeros(n_states)
        
        iteration = 0
        delta_history = []
        
        while True:
            delta = 0
            V_new = np.zeros(n_states)
            
            for s in range(n_states):
                state = idx_to_state(s)
                
                # 跳过终止状态
                if state == (0, 3) or state == (1, 1):
                    V_new[s] = V[s]
                    continue
                
                # 计算所有动作的价值
                action_values = []
                for a in range(n_actions):
                    next_state, reward = get_transitions(state, a)
                    next_s = state_to_idx(*next_state)
                    action_value = reward + gamma * V[next_s]
                    action_values.append(action_value)
                
                V_new[s] = max(action_values)
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            delta_history.append(delta)
            iteration += 1
            
            if delta < theta:
                break
        
        print(f"价值迭代收敛，迭代次数: {iteration}")
        
        # 可视化结果
        self.visualize_value_iteration_results(V, rows, cols, delta_history)
        
        return V, iteration
    
    def visualize_value_iteration_results(self, V, rows, cols, delta_history):
        """可视化价值迭代结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 价值函数可视化
        V_grid = V.reshape(rows, cols)
        im = axes[0].imshow(V_grid, cmap='coolwarm', interpolation='nearest')
        
        # 添加数值标注
        for i in range(rows):
            for j in range(cols):
                axes[0].text(j, i, f'{V_grid[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='white' if abs(V_grid[i, j]) > 3 else 'black')
        
        axes[0].set_title('最优状态价值函数 V*(s)')
        axes[0].set_xticks(range(cols))
        axes[0].set_yticks(range(rows))
        plt.colorbar(im, ax=axes[0])
        
        # 收敛过程
        axes[1].plot(delta_history, 'b-', linewidth=2)
        axes[1].set_xlabel('迭代次数')
        axes[1].set_ylabel('最大价值变化 δ')
        axes[1].set_title('价值迭代收敛过程')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def policy_types(self):
        """策略类型"""
        print("\n=== 策略类型 ===")
        
        policy_types = {
            "确定性策略": {
                "定义": "π(s) = a",
                "特点": "每个状态对应唯一动作",
                "优点": "简单、执行效率高",
                "缺点": "缺乏探索能力"
            },
            "随机策略": {
                "定义": "π(a|s) = P(A_t=a | S_t=s)",
                "特点": "给出动作概率分布",
                "优点": "自然的探索机制",
                "缺点": "执行有随机性"
            },
            "ε-贪心策略": {
                "定义": "π(a|s) = 1-ε+ε/|A| if a=a*, ε/|A| otherwise",
                "特点": "平衡利用和探索",
                "优点": "简单有效的探索",
                "缺点": "探索是均匀的"
            },
            "Softmax策略": {
                "定义": "π(a|s) = exp(Q(s,a)/τ) / Σ_b exp(Q(s,b)/τ)",
                "特点": "温度参数控制随机性",
                "优点": "偏向高价值动作",
                "缺点": "需要调节温度参数"
            }
        }
        
        for policy_type, details in policy_types.items():
            print(f"{policy_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        return policy_types

class DynamicProgramming:
    """动态规划方法"""
    
    def __init__(self):
        pass
    
    def policy_evaluation(self):
        """策略评估"""
        print("=== 策略评估 (Policy Evaluation) ===")
        
        print("目标: 给定策略π，计算状态价值函数V^π(s)")
        print()
        print("迭代更新公式:")
        print("V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]")
        print()
        print("收敛性: 在γ < 1时，V_k收敛到V^π")
        
        # 策略评估实现示例
        return self.implement_policy_evaluation()
    
    def implement_policy_evaluation(self):
        """实现策略评估"""
        print("\n=== 策略评估实现 ===")
        
        # 使用简单的2x2网格世界
        # 状态: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # 状态3是终止状态（目标）
        
        n_states = 4
        n_actions = 4  # 上右下左
        gamma = 0.9
        
        # 定义一个简单的随机策略
        policy = np.ones((n_states, n_actions)) / n_actions
        
        # 转移概率矩阵 P[s][a][s'] 和奖励矩阵 R[s][a][s']
        P = np.zeros((n_states, n_actions, n_states))
        R = np.zeros((n_states, n_actions, n_states))
        
        # 定义转移和奖励（简化版本）
        # 动作: 0=上, 1=右, 2=下, 3=左
        transitions = {
            0: {0: 0, 1: 1, 2: 2, 3: 0},  # 从状态0
            1: {0: 1, 1: 1, 2: 3, 3: 0},  # 从状态1
            2: {0: 0, 1: 3, 2: 2, 3: 2},  # 从状态2
            3: {0: 3, 1: 3, 2: 3, 3: 3}   # 从状态3（终止状态）
        }
        
        # 设置转移概率
        for s in range(n_states):
            for a in range(n_actions):
                next_s = transitions[s][a]
                P[s, a, next_s] = 1.0
                
                # 奖励设置
                if next_s == 3:  # 到达目标
                    R[s, a, next_s] = 10.0
                else:
                    R[s, a, next_s] = -0.1
        
        # 策略评估迭代
        V = np.zeros(n_states)
        theta = 1e-6
        
        iteration = 0
        delta_history = []
        V_history = [V.copy()]
        
        while True:
            delta = 0
            V_new = np.zeros(n_states)
            
            for s in range(n_states):
                if s == 3:  # 终止状态
                    V_new[s] = 0
                    continue
                
                v = 0
                for a in range(n_actions):
                    for s_prime in range(n_states):
                        v += policy[s, a] * P[s, a, s_prime] * \
                             (R[s, a, s_prime] + gamma * V[s_prime])
                
                V_new[s] = v
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            delta_history.append(delta)
            V_history.append(V.copy())
            iteration += 1
            
            if delta < theta:
                break
        
        print(f"策略评估收敛，迭代次数: {iteration}")
        print(f"最终价值函数: {V}")
        
        # 可视化收敛过程
        self.visualize_policy_evaluation(V_history, delta_history)
        
        return V, iteration
    
    def visualize_policy_evaluation(self, V_history, delta_history):
        """可视化策略评估过程"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 价值函数收敛过程
        V_array = np.array(V_history)
        for s in range(V_array.shape[1]):
            if s < 3:  # 非终止状态
                axes[0].plot(V_array[:, s], label=f'State {s}', linewidth=2)
        
        axes[0].set_xlabel('迭代次数')
        axes[0].set_ylabel('状态价值')
        axes[0].set_title('策略评估收敛过程')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 最大变化量
        axes[1].plot(delta_history, 'r-', linewidth=2)
        axes[1].set_xlabel('迭代次数')
        axes[1].set_ylabel('最大价值变化 δ')
        axes[1].set_title('收敛速度')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def policy_improvement(self):
        """策略改进"""
        print("=== 策略改进 (Policy Improvement) ===")
        
        print("策略改进定理:")
        print("对于所有状态s，如果Q^π(s,π'(s)) ≥ V^π(s)，")
        print("则π' ≥ π（π'不差于π）")
        print()
        print("贪心策略改进:")
        print("π'(s) = argmax_a Q^π(s,a)")
        print("      = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]")
        
        return self.policy_iteration_algorithm()
    
    def policy_iteration_algorithm(self):
        """策略迭代算法"""
        print("\n=== 策略迭代算法 ===")
        
        print("算法流程:")
        print("1. 初始化: 任意策略π_0")
        print("2. 策略评估: 计算V^π_k")
        print("3. 策略改进: π_{k+1}(s) = argmax_a Q^π_k(s,a)")
        print("4. 重复2-3直到策略收敛")
        print()
        
        # 实现策略迭代
        return self.implement_policy_iteration()
    
    def implement_policy_iteration(self):
        """实现策略迭代算法"""
        print("=== 策略迭代实现 ===")
        
        # 使用之前的网格世界设置
        n_states = 4
        n_actions = 4
        gamma = 0.9
        
        # 转移概率和奖励（复用之前的定义）
        P = np.zeros((n_states, n_actions, n_states))
        R = np.zeros((n_states, n_actions, n_states))
        
        transitions = {
            0: {0: 0, 1: 1, 2: 2, 3: 0},
            1: {0: 1, 1: 1, 2: 3, 3: 0},
            2: {0: 0, 1: 3, 2: 2, 3: 2},
            3: {0: 3, 1: 3, 2: 3, 3: 3}
        }
        
        for s in range(n_states):
            for a in range(n_actions):
                next_s = transitions[s][a]
                P[s, a, next_s] = 1.0
                R[s, a, next_s] = 10.0 if next_s == 3 else -0.1
        
        # 初始化随机策略
        policy = np.random.randint(0, n_actions, n_states)
        policy[3] = 0  # 终止状态动作无关紧要
        
        policy_stable = False
        iteration = 0
        policy_history = [policy.copy()]
        value_history = []
        
        while not policy_stable:
            iteration += 1
            
            # 策略评估
            V = np.zeros(n_states)
            theta = 1e-6
            
            while True:
                delta = 0
                V_new = np.zeros(n_states)
                
                for s in range(n_states):
                    if s == 3:
                        V_new[s] = 0
                        continue
                    
                    a = policy[s]
                    v = 0
                    for s_prime in range(n_states):
                        v += P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime])
                    
                    V_new[s] = v
                    delta = max(delta, abs(V_new[s] - V[s]))
                
                V = V_new
                if delta < theta:
                    break
            
            value_history.append(V.copy())
            
            # 策略改进
            policy_stable = True
            new_policy = np.zeros(n_states, dtype=int)
            
            for s in range(n_states):
                if s == 3:
                    new_policy[s] = policy[s]
                    continue
                
                # 计算所有动作的价值
                action_values = np.zeros(n_actions)
                for a in range(n_actions):
                    for s_prime in range(n_states):
                        action_values[a] += P[s, a, s_prime] * \
                                          (R[s, a, s_prime] + gamma * V[s_prime])
                
                new_policy[s] = np.argmax(action_values)
                
                if new_policy[s] != policy[s]:
                    policy_stable = False
            
            policy = new_policy
            policy_history.append(policy.copy())
        
        print(f"策略迭代收敛，迭代次数: {iteration}")
        print(f"最优策略: {policy}")
        print(f"最优价值函数: {V}")
        
        # 可视化策略迭代过程
        self.visualize_policy_iteration(policy_history, value_history)
        
        return policy, V, iteration
    
    def visualize_policy_iteration(self, policy_history, value_history):
        """可视化策略迭代过程"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 策略变化
        policy_array = np.array(policy_history)
        action_names = ['↑', '→', '↓', '←']
        
        axes[0, 0].set_title('策略演化过程')
        for s in range(3):  # 只显示非终止状态
            policy_s = policy_array[:, s]
            axes[0, 0].plot(policy_s, 'o-', label=f'State {s}', linewidth=2, markersize=8)
        
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('选择的动作')
        axes[0, 0].set_yticks(range(4))
        axes[0, 0].set_yticklabels(action_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 价值函数变化
        if value_history:
            value_array = np.array(value_history)
            axes[0, 1].set_title('价值函数演化')
            for s in range(3):
                axes[0, 1].plot(value_array[:, s], 'o-', label=f'State {s}', linewidth=2)
            
            axes[0, 1].set_xlabel('迭代次数')
            axes[0, 1].set_ylabel('状态价值')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 最终策略可视化
        final_policy = policy_history[-1]
        policy_grid = np.array([final_policy[0], final_policy[1], 
                               final_policy[2], final_policy[3]]).reshape(2, 2)
        
        axes[1, 0].set_title('最终策略')
        for i in range(2):
            for j in range(2):
                state_idx = i * 2 + j
                if state_idx == 3:
                    axes[1, 0].text(j, i, 'GOAL', ha='center', va='center', 
                                   fontsize=12, color='red')
                else:
                    action = final_policy[state_idx]
                    axes[1, 0].text(j, i, action_names[action], ha='center', va='center', 
                                   fontsize=16, color='blue')
        
        axes[1, 0].set_xlim(-0.5, 1.5)
        axes[1, 0].set_ylim(-0.5, 1.5)
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].grid(True)
        
        # 最终价值函数
        if value_history:
            final_values = value_history[-1]
            value_grid = final_values.reshape(2, 2)
            
            im = axes[1, 1].imshow(value_grid, cmap='Blues', interpolation='nearest')
            axes[1, 1].set_title('最终价值函数')
            
            for i in range(2):
                for j in range(2):
                    axes[1, 1].text(j, i, f'{value_grid[i, j]:.2f}', 
                                   ha='center', va='center', color='white')
            
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()

class TemporalDifferenceLearning:
    """时序差分学习"""
    
    def __init__(self):
        pass
    
    def td_learning_concept(self):
        """TD学习概念"""
        print("=== 时序差分学习 (Temporal Difference Learning) ===")
        
        print("核心思想:")
        print("- 结合蒙特卡洛方法和动态规划的优点")
        print("- 无需等待episode结束即可更新")
        print("- 利用bootstrap（自举）更新价值估计")
        print()
        
        print("TD误差:")
        print("δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)")
        print("其中:")
        print("- R_{t+1} + γV(S_{t+1}): TD目标")
        print("- V(S_t): 当前估计")
        print("- δ_t: TD误差（时序差分误差）")
        print()
        
        print("TD(0)更新规则:")
        print("V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]")
        print("V(S_t) ← V(S_t) + αδ_t")
        
        # TD学习vs其他方法比较
        self.compare_learning_methods()
        
        return self.implement_td_zero()
    
    def compare_learning_methods(self):
        """比较不同学习方法"""
        print("\n=== 学习方法比较 ===")
        
        methods = {
            "蒙特卡洛": {
                "更新时机": "episode结束后",
                "目标": "实际回报G_t",
                "方差": "高（真实回报有噪声）",
                "偏差": "无偏",
                "收敛": "保证收敛到V^π"
            },
            "TD(0)": {
                "更新时机": "每步更新",
                "目标": "R_{t+1} + γV(S_{t+1})",
                "方差": "低（只有一步噪声）",
                "偏差": "有偏（bootstrap）",
                "收敛": "收敛到V^π"
            },
            "动态规划": {
                "更新时机": "全状态扫描",
                "目标": "期望回报（需要模型）",
                "方差": "无",
                "偏差": "无偏（有模型）",
                "收敛": "快速收敛"
            }
        }
        
        for method, details in methods.items():
            print(f"{method}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 可视化比较
        self.visualize_method_comparison()
        
        return methods
    
    def visualize_method_comparison(self):
        """可视化方法比较"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 模拟学习曲线
        episodes = np.arange(1, 101)
        
        # 蒙特卡洛：高方差，无偏
        mc_curve = 1 - 0.5 * np.exp(-episodes/30) + np.random.normal(0, 0.1, 100)
        
        # TD学习：低方差，初期有偏
        td_curve = 1 - 0.7 * np.exp(-episodes/20) + np.random.normal(0, 0.05, 100)
        
        # DP：快速收敛（如果有模型）
        dp_curve = 1 - 0.9 * np.exp(-episodes/10) + np.random.normal(0, 0.02, 100)
        
        # 学习曲线
        axes[0].plot(episodes, mc_curve, label='Monte Carlo', alpha=0.7, linewidth=2)
        axes[0].plot(episodes, td_curve, label='TD(0)', alpha=0.7, linewidth=2)
        axes[0].plot(episodes, dp_curve, label='Dynamic Programming', alpha=0.7, linewidth=2)
        axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='True Value')
        
        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel('Value Estimate')
        axes[0].set_title('Learning Curves Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 方差对比
        methods = ['Monte Carlo', 'TD(0)', 'DP']
        variances = [0.1, 0.05, 0.02]
        
        bars = axes[1].bar(methods, variances, color=['orange', 'blue', 'green'], alpha=0.7)
        axes[1].set_ylabel('Variance')
        axes[1].set_title('Variance Comparison')
        
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{var:.3f}', ha='center', va='bottom')
        
        # 收敛速度对比
        convergence_rates = [30, 20, 10]  # 时间常数
        
        bars = axes[2].bar(methods, convergence_rates, color=['orange', 'blue', 'green'], alpha=0.7)
        axes[2].set_ylabel('Convergence Time Constant')
        axes[2].set_title('Convergence Speed')
        
        for bar, rate in zip(bars, convergence_rates):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def implement_td_zero(self):
        """实现TD(0)算法"""
        print("\n=== TD(0)算法实现 ===")
        
        # 随机游走问题
        # 状态: 0, 1, 2, 3, 4, 5, 6
        # 起始状态: 3, 终止状态: 0, 6
        # 动作: 左(-1), 右(+1)
        # 奖励: 到达状态6得+1，到达状态0得0，其他为0
        
        n_states = 7
        start_state = 3
        alpha = 0.1  # 学习率
        gamma = 1.0  # 无折扣
        
        # 初始化价值函数
        V = np.zeros(n_states)
        V[0] = 0  # 终止状态
        V[6] = 0  # 终止状态
        
        # 真实价值函数（解析解）
        true_V = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 0])
        
        n_episodes = 1000
        V_history = []
        
        for episode in range(n_episodes):
            state = start_state
            
            while state != 0 and state != 6:
                # 随机选择动作（左或右）
                action = np.random.choice([-1, 1])
                next_state = state + action
                
                # 计算奖励
                if next_state == 6:
                    reward = 1
                else:
                    reward = 0
                
                # TD更新
                td_error = reward + gamma * V[next_state] - V[state]
                V[state] += alpha * td_error
                
                state = next_state
            
            # 记录价值函数
            if episode % 100 == 0:
                V_history.append(V.copy())
        
        print(f"TD(0)学习完成，episodes: {n_episodes}")
        print(f"真实价值函数: {true_V}")
        print(f"学习价值函数: {V}")
        print(f"均方误差: {np.mean((V - true_V)**2):.4f}")
        
        # 可视化TD学习过程
        self.visualize_td_learning(V_history, true_V)
        
        return V, V_history
    
    def visualize_td_learning(self, V_history, true_V):
        """可视化TD学习过程"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 价值函数学习过程
        states = range(len(true_V))
        
        for i, V in enumerate(V_history):
            alpha_val = 0.3 + 0.7 * i / len(V_history)
            axes[0].plot(states, V, 'b-', alpha=alpha_val, linewidth=1)
        
        axes[0].plot(states, true_V, 'r-', linewidth=3, label='True Value Function')
        axes[0].plot(states, V_history[-1], 'g--', linewidth=2, label='Learned Value Function')
        
        axes[0].set_xlabel('State')
        axes[0].set_ylabel('Value')
        axes[0].set_title('TD(0) Learning Process')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 误差收敛
        errors = [np.mean((V - true_V)**2) for V in V_history]
        episodes = np.arange(0, len(V_history) * 100, 100)
        
        axes[1].plot(episodes, errors, 'b-', linewidth=2)
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel('Mean Squared Error')
        axes[1].set_title('Learning Error')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def comprehensive_rl_theory_summary():
    """强化学习理论综合总结"""
    print("=== 强化学习理论综合总结 ===")
    
    summary = {
        "理论基础": {
            "MDP框架": "状态、动作、转移、奖励、折扣",
            "价值函数": "状态价值V(s)、动作价值Q(s,a)",
            "Bellman方程": "递归关系、最优性方程",
            "策略": "确定性、随机性、最优策略"
        },
        
        "求解方法": {
            "动态规划": "需要模型、精确解、计算复杂",
            "蒙特卡洛": "免模型、高方差、需要完整episode",
            "时序差分": "免模型、低方差、在线学习",
            "函数近似": "处理大状态空间、神经网络"
        },
        
        "核心算法": {
            "价值迭代": "V_{k+1} = max_a E[R + γV_k]",
            "策略迭代": "评估+改进、保证收敛",
            "TD(0)": "V(s) ← V(s) + α[R + γV(s') - V(s)]",
            "Q-Learning": "Q(s,a) ← Q(s,a) + α[R + γmax Q(s',a') - Q(s,a)]"
        },
        
        "探索策略": {
            "ε-贪心": "平衡利用和探索",
            "Softmax": "基于价值的概率选择",
            "UCB": "置信度上界、理论保证",
            "Thompson采样": "贝叶斯方法、自然探索"
        },
        
        "现代发展": {
            "深度强化学习": "DQN、Policy Gradient、Actor-Critic",
            "多智能体": "合作、竞争、通信",
            "元学习": "学会学习、快速适应",
            "安全强化学习": "约束、风险感知"
        },
        
        "应用领域": {
            "游戏AI": "Atari、围棋、星际争霸",
            "机器人控制": "导航、操作、步态",
            "推荐系统": "个性化、长期优化",
            "金融交易": "算法交易、风险管理",
            "自动驾驶": "路径规划、决策制定"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("强化学习理论基础指南加载完成！")
```

## 参考文献 📚

- Sutton & Barto (2018): "Reinforcement Learning: An Introduction"
- Bellman (1957): "Dynamic Programming"
- Watkins (1989): "Learning from Delayed Rewards"
- Kaelbling et al. (1996): "Reinforcement Learning: A Survey"
- Mnih et al. (2015): "Human-level control through deep reinforcement learning"

## 下一步学习
- [深度强化学习](deep_rl_theory.md) - DQN、Policy Gradient
- [多智能体理论](multi_agent_theory.md) - 博弈论与合作
- [元学习理论](meta_learning_theory.md) - 学会学习