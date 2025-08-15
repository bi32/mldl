# 自动化机器学习 (AutoML) 🤖

全面掌握AutoML的原理、方法和实践，实现机器学习的自动化。

## 1. AutoML概述 🌟

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AutoMLOverview:
    """AutoML概述"""
    
    def __init__(self):
        self.automl_components = {
            "数据预处理": ["特征选择", "数据清洗", "特征工程", "数据增强"],
            "模型选择": ["算法选择", "超参数优化", "架构搜索", "集成方法"],
            "评估优化": ["交叉验证", "早停策略", "模型压缩", "性能评估"],
            "部署监控": ["模型部署", "性能监控", "模型更新", "A/B测试"]
        }
    
    def automl_motivation(self):
        """AutoML的动机和价值"""
        print("=== AutoML的动机 ===")
        
        challenges = {
            "人工成本": {
                "问题": "需要大量专业知识和经验",
                "解决": "自动化流程，降低技术门槛",
                "价值": "让非专家也能使用ML"
            },
            "时间成本": {
                "问题": "模型开发周期长",
                "解决": "并行搜索和快速迭代",
                "价值": "加速模型部署上线"
            },
            "搜索空间": {
                "问题": "超参数空间巨大",
                "解决": "智能搜索策略",
                "价值": "找到更好的模型配置"
            },
            "一致性": {
                "问题": "不同人员结果差异大",
                "解决": "标准化流程和评估",
                "价值": "保证结果的可重复性"
            }
        }
        
        for challenge, details in challenges.items():
            print(f"\n{challenge}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        return challenges
    
    def automl_taxonomy(self):
        """AutoML分类体系"""
        print("=== AutoML分类体系 ===")
        
        # 按自动化程度分类
        automation_levels = {
            "部分自动化": {
                "特点": "自动化特定环节",
                "示例": "超参数调优、特征选择",
                "适用": "有一定ML经验的用户",
                "工具": "Optuna, Hyperopt, scikit-optimize"
            },
            "全自动化": {
                "特点": "端到端自动化",
                "示例": "从数据到模型的完整流程",
                "适用": "ML新手或快速原型",
                "工具": "AutoML platforms"
            },
            "交互式自动化": {
                "特点": "人机协作",
                "示例": "用户提供约束和偏好",
                "适用": "需要领域知识的场景",
                "工具": "Interactive ML systems"
            }
        }
        
        for level, details in automation_levels.items():
            print(f"\n{level}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 按应用领域分类
        domain_types = {
            "表格数据AutoML": ["H2O.ai", "AutoGluon", "TPOT"],
            "计算机视觉AutoML": ["AutoKeras", "NAS", "EfficientNet"],
            "自然语言处理AutoML": ["AutoNLP", "Neural Architecture Search"],
            "时间序列AutoML": ["Prophet", "Auto-ARIMA", "NeuralProphet"],
            "推荐系统AutoML": ["AutoRec", "AutoCTR", "AutoFM"]
        }
        
        print("\n=== 按应用领域分类 ===")
        for domain, tools in domain_types.items():
            print(f"{domain}: {', '.join(tools)}")
        
        return automation_levels, domain_types

class HyperparameterOptimization:
    """超参数优化"""
    
    def __init__(self):
        self.optimization_methods = {}
    
    def grid_search_implementation(self):
        """网格搜索实现"""
        print("=== 网格搜索 (Grid Search) ===")
        
        print("原理:")
        print("- 穷举搜索预定义的参数组合")
        print("- 保证找到最优解(在搜索空间内)")
        print("- 计算复杂度: O(n^d), d为参数维度")
        print()
        
        # 实现简单的网格搜索
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import make_classification
        
        # 生成示例数据
        X, y = make_classification(n_samples=1000, n_features=10, 
                                 n_informative=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # 网格搜索
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        print("执行网格搜索...")
        grid_search.fit(X_train, y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳得分: {grid_search.best_score_:.4f}")
        
        # 可视化搜索结果
        self.visualize_grid_search_results(grid_search)
        
        return grid_search
    
    def visualize_grid_search_results(self, grid_search):
        """可视化网格搜索结果"""
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # 选择主要参数进行可视化
        pivot_table = results_df.pivot_table(
            values='mean_test_score',
            index='param_n_estimators',
            columns='param_max_depth',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Grid Search Results: Accuracy by Parameters')
        plt.xlabel('max_depth')
        plt.ylabel('n_estimators')
        plt.tight_layout()
        plt.show()
    
    def random_search_implementation(self):
        """随机搜索实现"""
        print("=== 随机搜索 (Random Search) ===")
        
        print("原理:")
        print("- 从参数分布中随机采样")
        print("- 在固定预算下通常优于网格搜索")
        print("- 适合高维参数空间")
        print()
        
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, uniform
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=10, 
                                 n_informative=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 定义参数分布
        param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
        
        # 随机搜索
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=50, cv=5,
            scoring='accuracy', random_state=42, n_jobs=-1
        )
        
        print("执行随机搜索...")
        random_search.fit(X_train, y_train)
        
        print(f"最佳参数: {random_search.best_params_}")
        print(f"最佳得分: {random_search.best_score_:.4f}")
        
        # 比较搜索效率
        self.compare_search_methods(X_train, y_train)
        
        return random_search
    
    def compare_search_methods(self, X_train, y_train):
        """比较不同搜索方法的效率"""
        print("\n=== 搜索方法效率比较 ===")
        
        import time
        
        # 网格搜索（小规模）
        param_grid_small = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        # 网格搜索时间测试
        start_time = time.time()
        grid_search = GridSearchCV(rf, param_grid_small, cv=3)
        grid_search.fit(X_train, y_train)
        grid_time = time.time() - start_time
        
        # 随机搜索时间测试
        from scipy.stats import randint
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 10)
        }
        
        start_time = time.time()
        random_search = RandomizedSearchCV(
            rf, param_dist, n_iter=6, cv=3, random_state=42)
        random_search.fit(X_train, y_train)
        random_time = time.time() - start_time
        
        print(f"网格搜索时间: {grid_time:.2f}秒")
        print(f"随机搜索时间: {random_time:.2f}秒")
        print(f"网格搜索最佳得分: {grid_search.best_score_:.4f}")
        print(f"随机搜索最佳得分: {random_search.best_score_:.4f}")
    
    def bayesian_optimization(self):
        """贝叶斯优化"""
        print("=== 贝叶斯优化 ===")
        
        print("原理:")
        print("- 使用概率模型(如高斯过程)建模目标函数")
        print("- 通过获取函数(acquisition function)指导搜索")
        print("- 在探索(exploration)和利用(exploitation)间平衡")
        print()
        
        # 简化的贝叶斯优化示例
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
            
            # 定义搜索空间
            dimensions = [
                Integer(low=10, high=300, name='n_estimators'),
                Integer(low=1, high=20, name='max_depth'),
                Real(low=0.01, high=0.5, name='min_samples_split', prior='log-uniform'),
            ]
            
            # 生成数据
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 定义目标函数
            @use_named_args(dimensions)
            def objective(**params):
                rf = RandomForestClassifier(random_state=42, **params)
                score = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy').mean()
                return -score  # minimize负数 = maximize正数
            
            print("执行贝叶斯优化...")
            result = gp_minimize(func=objective, dimensions=dimensions, 
                               n_calls=20, random_state=42, verbose=False)
            
            print(f"最佳参数: {dict(zip([d.name for d in dimensions], result.x))}")
            print(f"最佳得分: {-result.fun:.4f}")
            
            # 可视化优化过程
            self.plot_bayesian_optimization(result)
            
        except ImportError:
            print("需要安装scikit-optimize: pip install scikit-optimize")
            self.manual_bayesian_example()
        
    def manual_bayesian_example(self):
        """手动贝叶斯优化示例"""
        print("\n手动贝叶斯优化概念演示:")
        
        # 模拟目标函数
        def objective_function(x):
            return -(x - 0.5)**2 + 0.8 + 0.1 * np.sin(10*x)
        
        x = np.linspace(0, 1, 100)
        y = [objective_function(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='真实目标函数')
        
        # 模拟已评估的点
        evaluated_x = [0.2, 0.7, 0.9]
        evaluated_y = [objective_function(xi) for xi in evaluated_x]
        plt.scatter(evaluated_x, evaluated_y, c='red', s=100, label='已评估点')
        
        plt.xlabel('参数值')
        plt.ylabel('目标函数值')
        plt.title('贝叶斯优化概念图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_bayesian_optimization(self, result):
        """绘制贝叶斯优化收敛曲线"""
        plt.figure(figsize=(10, 6))
        
        # 收敛曲线
        plt.subplot(1, 2, 1)
        plt.plot([-y for y in result.func_vals], 'b-o')
        plt.xlabel('迭代次数')
        plt.ylabel('最佳得分')
        plt.title('贝叶斯优化收敛曲线')
        plt.grid(True, alpha=0.3)
        
        # 评估历史
        plt.subplot(1, 2, 2)
        cumulative_best = np.maximum.accumulate([-y for y in result.func_vals])
        plt.plot(cumulative_best, 'g-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('累积最佳得分')
        plt.title('累积最佳性能')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class AutoMLPipelines:
    """AutoML管道"""
    
    def __init__(self):
        self.pipeline_components = {}
    
    def automated_feature_engineering(self):
        """自动化特征工程"""
        print("=== 自动化特征工程 ===")
        
        feature_engineering_techniques = {
            "特征选择": {
                "方法": ["相关性分析", "互信息", "递归特征消除"],
                "目标": "去除冗余和无关特征",
                "实现": "sklearn.feature_selection"
            },
            "特征转换": {
                "方法": ["标准化", "归一化", "多项式特征"],
                "目标": "改善数据分布和模型性能",
                "实现": "sklearn.preprocessing"
            },
            "特征构造": {
                "方法": ["交互特征", "时间特征", "聚合特征"],
                "目标": "创造新的有用特征",
                "实现": "featuretools, tsfresh"
            }
        }
        
        for technique, details in feature_engineering_techniques.items():
            print(f"\n{technique}:")
            for key, value in details.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(value)}")
                else:
                    print(f"  {key}: {value}")
        
        # 实现自动特征选择
        self.implement_auto_feature_selection()
        
        return feature_engineering_techniques
    
    def implement_auto_feature_selection(self):
        """实现自动特征选择"""
        print("\n=== 自动特征选择示例 ===")
        
        from sklearn.feature_selection import SelectKBest, f_classif, RFE
        from sklearn.ensemble import RandomForestClassifier
        
        # 生成高维数据
        X, y = make_classification(n_samples=1000, n_features=50, 
                                 n_informative=10, n_redundant=10,
                                 random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        print(f"原始特征数: {X_train.shape[1]}")
        
        # 方法1: 单变量特征选择
        selector_univariate = SelectKBest(score_func=f_classif, k=20)
        X_train_uni = selector_univariate.fit_transform(X_train, y_train)
        X_test_uni = selector_univariate.transform(X_test)
        
        print(f"单变量选择后特征数: {X_train_uni.shape[1]}")
        
        # 方法2: 递归特征消除
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        selector_rfe = RFE(estimator=rf, n_features_to_select=20)
        X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
        X_test_rfe = selector_rfe.transform(X_test)
        
        print(f"RFE选择后特征数: {X_train_rfe.shape[1]}")
        
        # 比较不同特征选择方法的效果
        methods = {
            'Original': (X_train, X_test),
            'Univariate': (X_train_uni, X_test_uni),
            'RFE': (X_train_rfe, X_test_rfe)
        }
        
        results = {}
        for method, (X_tr, X_te) in methods.items():
            rf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_eval.fit(X_tr, y_train)
            score = rf_eval.score(X_te, y_test)
            results[method] = score
        
        print("\n特征选择效果比较:")
        for method, score in results.items():
            print(f"{method}: {score:.4f}")
        
        # 可视化特征重要性
        self.visualize_feature_importance(rf, X_train.shape[1], selector_rfe)
        
        return results
    
    def visualize_feature_importance(self, model, n_features, selector_rfe):
        """可视化特征重要性"""
        # 获取特征重要性
        feature_importance = model.feature_importances_
        selected_features = selector_rfe.support_
        
        plt.figure(figsize=(12, 6))
        
        # 原始特征重要性
        plt.subplot(1, 2, 1)
        plt.bar(range(n_features), feature_importance)
        plt.xlabel('特征索引')
        plt.ylabel('重要性')
        plt.title('所有特征的重要性')
        
        # 选中特征的重要性
        plt.subplot(1, 2, 2)
        selected_importance = feature_importance[selected_features]
        selected_indices = np.where(selected_features)[0]
        plt.bar(range(len(selected_importance)), selected_importance)
        plt.xlabel('选中特征索引')
        plt.ylabel('重要性')
        plt.title('RFE选中特征的重要性')
        
        plt.tight_layout()
        plt.show()
    
    def automated_model_selection(self):
        """自动模型选择"""
        print("=== 自动模型选择 ===")
        
        # 定义候选模型
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, 
                                 n_informative=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 数据预处理
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 评估所有模型
        results = {}
        cv_results = {}
        
        print("评估各个模型:")
        for name, model in models.items():
            # 选择是否需要缩放的数据
            if name in ['LogisticRegression', 'SVM']:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5)
            cv_results[name] = cv_scores
            
            # 训练和测试
            model.fit(X_tr, y_train)
            test_score = model.score(X_te, y_test)
            results[name] = test_score
            
            print(f"{name}: CV={cv_scores.mean():.4f}(±{cv_scores.std():.4f}), "
                  f"Test={test_score:.4f}")
        
        # 选择最佳模型
        best_model = max(results, key=results.get)
        print(f"\n最佳模型: {best_model} (Test Score: {results[best_model]:.4f})")
        
        # 可视化模型比较
        self.visualize_model_comparison(cv_results, results)
        
        return results, best_model
    
    def visualize_model_comparison(self, cv_results, test_results):
        """可视化模型比较"""
        plt.figure(figsize=(15, 5))
        
        # 交叉验证结果箱线图
        plt.subplot(1, 3, 1)
        cv_data = [scores for scores in cv_results.values()]
        plt.boxplot(cv_data, labels=cv_results.keys())
        plt.ylabel('交叉验证得分')
        plt.title('模型交叉验证比较')
        plt.xticks(rotation=45)
        
        # 测试得分柱状图
        plt.subplot(1, 3, 2)
        models = list(test_results.keys())
        scores = list(test_results.values())
        plt.bar(models, scores)
        plt.ylabel('测试得分')
        plt.title('模型测试性能比较')
        plt.xticks(rotation=45)
        
        # CV均值vs测试得分散点图
        plt.subplot(1, 3, 3)
        cv_means = [np.mean(scores) for scores in cv_results.values()]
        test_scores = list(test_results.values())
        plt.scatter(cv_means, test_scores)
        
        for i, model in enumerate(models):
            plt.annotate(model, (cv_means[i], test_scores[i]))
        
        plt.xlabel('交叉验证均值')
        plt.ylabel('测试得分')
        plt.title('CV vs 测试性能')
        plt.plot([min(cv_means), max(cv_means)], 
                [min(cv_means), max(cv_means)], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

class AutoMLFrameworks:
    """AutoML框架"""
    
    def __init__(self):
        self.frameworks = {}
    
    def popular_automl_tools(self):
        """流行的AutoML工具"""
        print("=== 流行的AutoML工具 ===")
        
        tools = {
            "商业工具": {
                "H2O.ai": {
                    "特点": "开源+企业版，支持多种算法",
                    "优势": "易用性强，可解释性好",
                    "适用": "企业级应用",
                    "语言": "Python, R, Java, Scala"
                },
                "DataRobot": {
                    "特点": "全自动化平台",
                    "优势": "无需编程，MLOps集成",
                    "适用": "商业用户",
                    "语言": "Web界面"
                },
                "Google AutoML": {
                    "特点": "云原生，支持多模态",
                    "优势": "Google基础设施",
                    "适用": "云用户",
                    "语言": "API调用"
                }
            },
            
            "开源工具": {
                "Auto-sklearn": {
                    "特点": "基于scikit-learn",
                    "优势": "元学习，集成方法",
                    "适用": "Python用户",
                    "语言": "Python"
                },
                "TPOT": {
                    "特点": "遗传编程优化",
                    "优势": "搜索管道空间",
                    "适用": "研究和实验",
                    "语言": "Python"
                },
                "AutoGluon": {
                    "特点": "Amazon开发，多模态",
                    "优势": "易用，性能好",
                    "适用": "快速原型",
                    "语言": "Python"
                }
            }
        }
        
        for category, tools_dict in tools.items():
            print(f"\n{category}:")
            for tool, details in tools_dict.items():
                print(f"  {tool}:")
                for key, value in details.items():
                    print(f"    {key}: {value}")
        
        return tools
    
    def implement_simple_automl(self):
        """实现简单的AutoML系统"""
        print("=== 简单AutoML系统实现 ===")
        
        class SimpleAutoML:
            def __init__(self, time_budget=60):
                self.time_budget = time_budget
                self.best_model = None
                self.best_score = -float('inf')
                self.best_params = None
                self.preprocessing_steps = []
                
                # 定义候选模型和参数空间
                self.models_space = {
                    'rf': {
                        'model': RandomForestClassifier,
                        'params': {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5, 10]
                        }
                    },
                    'gb': {
                        'model': GradientBoostingClassifier,
                        'params': {
                            'n_estimators': [50, 100],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'max_depth': [3, 6, 9]
                        }
                    },
                    'lr': {
                        'model': LogisticRegression,
                        'params': {
                            'C': [0.1, 1, 10],
                            'solver': ['liblinear', 'lbfgs']
                        }
                    }
                }
            
            def fit(self, X, y):
                import time
                import itertools
                from sklearn.model_selection import cross_val_score
                
                start_time = time.time()
                
                print(f"开始AutoML训练，时间预算: {self.time_budget}秒")
                print(f"数据维度: {X.shape}")
                
                # 数据预处理
                X_processed = self._preprocess_data(X, y)
                
                # 模型搜索
                for model_name, model_config in self.models_space.items():
                    if time.time() - start_time > self.time_budget:
                        print(f"时间预算用完，停止搜索")
                        break
                    
                    model_class = model_config['model']
                    param_space = model_config['params']
                    
                    # 生成参数组合
                    param_names = list(param_space.keys())
                    param_values = list(param_space.values())
                    
                    for param_combination in itertools.product(*param_values):
                        if time.time() - start_time > self.time_budget:
                            break
                        
                        params = dict(zip(param_names, param_combination))
                        
                        try:
                            # 创建和评估模型
                            if model_name == 'lr':
                                model = model_class(random_state=42, max_iter=1000, **params)
                            else:
                                model = model_class(random_state=42, **params)
                            
                            # 交叉验证
                            scores = cross_val_score(model, X_processed, y, cv=3, 
                                                   scoring='accuracy')
                            avg_score = scores.mean()
                            
                            if avg_score > self.best_score:
                                self.best_score = avg_score
                                self.best_model = model
                                self.best_params = params
                                
                                print(f"新的最佳模型: {model_name} "
                                      f"(score: {avg_score:.4f}, params: {params})")
                        
                        except Exception as e:
                            continue
                
                # 训练最终模型
                if self.best_model is not None:
                    self.best_model.fit(X_processed, y)
                    
                elapsed_time = time.time() - start_time
                print(f"AutoML完成，用时: {elapsed_time:.2f}秒")
                print(f"最佳模型得分: {self.best_score:.4f}")
                
                return self
            
            def _preprocess_data(self, X, y):
                """数据预处理"""
                from sklearn.preprocessing import StandardScaler
                from sklearn.impute import SimpleImputer
                
                X_processed = X.copy()
                
                # 处理缺失值
                if np.isnan(X_processed).any():
                    imputer = SimpleImputer(strategy='mean')
                    X_processed = imputer.fit_transform(X_processed)
                    self.preprocessing_steps.append(('imputer', imputer))
                
                # 标准化
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X_processed)
                self.preprocessing_steps.append(('scaler', scaler))
                
                return X_processed
            
            def predict(self, X):
                """预测"""
                if self.best_model is None:
                    raise ValueError("模型未训练，请先调用fit方法")
                
                X_processed = X.copy()
                
                # 应用预处理步骤
                for step_name, step_obj in self.preprocessing_steps:
                    X_processed = step_obj.transform(X_processed)
                
                return self.best_model.predict(X_processed)
            
            def predict_proba(self, X):
                """预测概率"""
                if self.best_model is None:
                    raise ValueError("模型未训练，请先调用fit方法")
                
                X_processed = X.copy()
                
                for step_name, step_obj in self.preprocessing_steps:
                    X_processed = step_obj.transform(X_processed)
                
                return self.best_model.predict_proba(X_processed)
        
        # 测试SimpleAutoML
        print("\n测试SimpleAutoML:")
        X, y = make_classification(n_samples=1000, n_features=10, 
                                 n_informative=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 训练AutoML
        automl = SimpleAutoML(time_budget=30)
        automl.fit(X_train, y_train)
        
        # 评估
        y_pred = automl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"测试集准确率: {accuracy:.4f}")
        
        return SimpleAutoML
    
    def neural_architecture_search(self):
        """神经架构搜索 (NAS)"""
        print("=== 神经架构搜索 (NAS) ===")
        
        print("NAS概念:")
        print("- 自动设计神经网络架构")
        print("- 搜索空间包括层类型、连接方式、超参数")
        print("- 目标是找到最优的网络结构")
        print()
        
        nas_methods = {
            "强化学习NAS": {
                "代表": "NASNet, ENAS",
                "原理": "用RL agent生成架构",
                "优点": "可以发现新颖架构",
                "缺点": "计算成本极高"
            },
            "可微分NAS": {
                "代表": "DARTS, PC-DARTS",
                "原理": "将架构搜索转为连续优化",
                "优点": "效率高，梯度优化",
                "缺点": "搜索空间受限"
            },
            "进化算法NAS": {
                "代表": "AmoebaNet, AmobaNet",
                "原理": "进化算法搜索架构",
                "优点": "无梯度要求",
                "缺点": "需要大量计算资源"
            },
            "权重共享NAS": {
                "代表": "Once-for-All, BigNAS",
                "原理": "预训练超网络，子网继承权重",
                "优点": "大幅降低搜索成本",
                "缺点": "权重继承可能不optimal"
            }
        }
        
        for method, details in nas_methods.items():
            print(f"{method}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 简化的架构搜索演示
        self.demo_simple_nas()
        
        return nas_methods
    
    def demo_simple_nas(self):
        """演示简单的架构搜索"""
        print("=== 简化架构搜索演示 ===")
        
        # 定义搜索空间（不同的网络宽度和深度）
        architecture_space = {
            'n_layers': [1, 2, 3],
            'layer_sizes': [[32], [64], [128], [32, 16], [64, 32], [128, 64, 32]]
        }
        
        from sklearn.neural_network import MLPClassifier
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, 
                                 n_informative=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        best_architecture = None
        best_score = -float('inf')
        results = []
        
        print("搜索最佳网络架构:")
        
        for layer_sizes in architecture_space['layer_sizes']:
            try:
                # 创建MLP
                mlp = MLPClassifier(
                    hidden_layer_sizes=tuple(layer_sizes),
                    max_iter=500,
                    random_state=42
                )
                
                # 训练和评估
                cv_scores = cross_val_score(mlp, X_train_scaled, y_train, 
                                          cv=3, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                results.append({
                    'architecture': layer_sizes,
                    'score': avg_score,
                    'std': cv_scores.std()
                })
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_architecture = layer_sizes
                
                print(f"架构 {layer_sizes}: {avg_score:.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"架构 {layer_sizes} 失败: {e}")
                continue
        
        print(f"\n最佳架构: {best_architecture}")
        print(f"最佳得分: {best_score:.4f}")
        
        # 可视化架构搜索结果
        self.visualize_architecture_search(results)
        
        return best_architecture, results
    
    def visualize_architecture_search(self, results):
        """可视化架构搜索结果"""
        if not results:
            return
        
        # 提取数据
        architectures = [str(r['architecture']) for r in results]
        scores = [r['score'] for r in results]
        stds = [r['std'] for r in results]
        
        # 绘制结果
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        x = range(len(architectures))
        plt.errorbar(x, scores, yerr=stds, fmt='o-', capsize=5)
        plt.xlabel('架构配置')
        plt.ylabel('交叉验证得分')
        plt.title('不同架构的性能比较')
        plt.xticks(x, architectures, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 架构复杂度vs性能
        plt.subplot(1, 2, 2)
        complexities = [sum(r['architecture']) for r in results]  # 总神经元数作为复杂度
        plt.scatter(complexities, scores, s=100, alpha=0.7)
        
        for i, arch in enumerate(architectures):
            plt.annotate(arch, (complexities[i], scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('架构复杂度 (总神经元数)')
        plt.ylabel('性能得分')
        plt.title('架构复杂度 vs 性能')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def comprehensive_automl_summary():
    """AutoML综合总结"""
    print("=== AutoML综合总结 ===")
    
    summary = {
        "核心组件": {
            "数据预处理": "自动清洗、特征工程、数据增强",
            "模型选择": "算法选择、架构设计、超参数优化", 
            "模型评估": "交叉验证、性能评估、模型解释",
            "部署优化": "模型压缩、推理优化、监控更新"
        },
        
        "关键技术": {
            "搜索策略": "网格搜索、随机搜索、贝叶斯优化、进化算法",
            "元学习": "学习学习、迁移学习、few-shot学习",
            "多目标优化": "准确率-效率权衡、帕累托前沿",
            "早停策略": "资源预算、收敛检测、性能阈值"
        },
        
        "应用场景": {
            "企业应用": "降低AI门槛、加速部署、标准化流程",
            "科研探索": "快速原型、基准比较、新方法验证",
            "教育培训": "学习工具、概念理解、最佳实践",
            "个人项目": "快速建模、参数调优、性能提升"
        },
        
        "发展趋势": {
            "大模型AutoML": "预训练模型选择、提示工程自动化",
            "多模态AutoML": "视觉-语言-语音联合优化",
            "联邦AutoML": "分布式数据上的自动学习",
            "可解释AutoML": "自动生成模型解释、决策透明化",
            "绿色AutoML": "能耗优化、碳足迹考虑"
        },
        
        "挑战与限制": {
            "计算成本": "搜索空间大、评估耗时",
            "数据质量": "垃圾进垃圾出、数据偏差",
            "领域知识": "通用方法vs专业知识",
            "可解释性": "黑盒优化、结果可信度"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("自动化机器学习 (AutoML) 指南加载完成！")
```

## 最佳实践指南 📋

```python
def automl_best_practices():
    """AutoML最佳实践"""
    
    practices = {
        "数据准备": [
            "确保数据质量和完整性",
            "理解业务问题和评估指标", 
            "适当的数据分割和验证策略",
            "考虑数据不平衡问题"
        ],
        
        "搜索策略": [
            "合理设置时间和计算预算",
            "选择适合的搜索算法",
            "定义合适的搜索空间",
            "使用多种评估指标"
        ],
        
        "模型选择": [
            "不要忽视简单模型",
            "考虑模型的可解释性",
            "评估模型的稳定性",
            "测试模型的泛化能力"
        ],
        
        "部署考虑": [
            "评估推理延迟和资源需求",
            "考虑模型更新和维护",
            "设置监控和告警机制",
            "准备模型回滚策略"
        ]
    }
    
    return practices

# 常见陷阱
common_pitfalls = """
AutoML常见陷阱：

1. 过度依赖自动化
   - 忽视领域知识
   - 不理解模型原理
   - 缺乏结果验证

2. 数据泄露问题
   - 时间序列数据的未来泄露
   - 测试集信息泄露到训练中
   - 特征工程中使用全局统计

3. 评估偏差
   - 过拟合验证集
   - 评估指标不匹配业务目标
   - 忽视样本不平衡

4. 资源浪费
   - 搜索空间设置不当
   - 没有使用早停策略
   - 重复计算相同配置

5. 生产部署问题
   - 训练环境与生产环境不一致
   - 模型性能在生产中下降
   - 缺乏模型监控机制
"""

print("AutoML最佳实践和常见陷阱加载完成！")
```

## 参考文献 📚

- Hutter et al. (2019): "Automated Machine Learning: Methods, Systems, Challenges"
- Feurer & Hutter (2019): "Hyperparameter Optimization"
- Elsken et al. (2019): "Neural Architecture Search: A Survey"
- Zoph & Le (2017): "Neural Architecture Search with Reinforcement Learning"
- Real et al. (2019): "Regularized Evolution for Image Classifier Architecture Search"

## 下一步学习
- [超参数调优](hyperparameter_tuning.md) - 深入超参数优化
- [模型部署](../deployment/pytorch_deployment.md) - 生产环境部署
- [MLOps实践](mlops_practices.md) - 机器学习工程化