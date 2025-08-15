# 自然语言处理理论基础 📝

深入理解自然语言处理的核心理论、方法和最新发展。

## 1. 语言学基础 🔤

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import math
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from wordcloud import WordCloud

class LinguisticFoundations:
    """语言学基础理论"""
    
    def __init__(self):
        self.linguistic_levels = {
            "语音学/音韵学": "研究语言的声音系统",
            "形态学": "研究词汇的内部结构",
            "句法学": "研究句子的结构规则",
            "语义学": "研究意义和含义",
            "语用学": "研究语言的使用方式",
            "话语分析": "研究文本和对话的结构"
        }
    
    def morphology_analysis(self):
        """形态学分析"""
        print("=== 形态学分析 ===")
        
        morphology_concepts = {
            "词素(Morpheme)": {
                "定义": "最小的有意义语言单位",
                "类型": ["自由词素", "绑定词素"],
                "示例": "un-happy-ness (3个词素)"
            },
            "词根(Root)": {
                "定义": "词汇的核心部分",
                "功能": "承载主要意义",
                "示例": "walk in walking, walked"
            },
            "词缀(Affix)": {
                "定义": "附加到词根的绑定词素",
                "类型": ["前缀", "后缀", "中缀"],
                "示例": "un-(前缀), -ing(后缀)"
            }
        }
        
        for concept, details in morphology_concepts.items():
            print(f"\n{concept}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 实际分析示例
        words = ["unhappiness", "reconstruction", "uncomfortable"]
        print("\n形态学分解示例:")
        
        morpheme_analysis = {
            "unhappiness": ["un-", "happy", "-ness"],
            "reconstruction": ["re-", "construct", "-ion"], 
            "uncomfortable": ["un-", "comfort", "-able"]
        }
        
        for word, morphemes in morpheme_analysis.items():
            print(f"{word}: {' + '.join(morphemes)}")
        
        return morphology_concepts
    
    def syntax_analysis(self):
        """句法分析"""
        print("=== 句法学理论 ===")
        
        syntax_theories = {
            "短语结构语法": {
                "核心思想": "句子由嵌套的短语构成",
                "规则形式": "S → NP VP",
                "优点": "形式化程度高",
                "缺点": "难以处理复杂现象"
            },
            "依存语法": {
                "核心思想": "词汇之间存在依存关系",
                "表示方式": "有向图结构",
                "优点": "跨语言适用性强",
                "缺点": "缺乏层次结构信息"
            },
            "转换生成语法": {
                "核心思想": "深层结构通过转换得到表层结构",
                "关键概念": "深层结构、表层结构、转换规则",
                "优点": "解释语言创造性",
                "缺点": "规则复杂，难以计算"
            }
        }
        
        for theory, details in syntax_theories.items():
            print(f"\n{theory}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 可视化句法树
        self.visualize_syntax_tree()
        
        return syntax_theories
    
    def visualize_syntax_tree(self):
        """可视化句法树"""
        print("\n句法树示例: 'The cat sat on the mat'")
        
        # 简单的句法树可视化
        tree_structure = """
                    S
                   / \\
                  NP  VP
                  |   /|\\
                 Det V PP
                  |  |  /|\\
                The cat P NP
                       |  /|\\
                      on Det N
                          |   |
                         the mat
        """
        print(tree_structure)
        
        # 依存关系示例
        print("\n依存关系示例:")
        dependencies = [
            ("sat", "root", "ROOT"),
            ("cat", "nsubj", "sat"),
            ("The", "det", "cat"),
            ("on", "prep", "sat"),
            ("mat", "pobj", "on"),
            ("the", "det", "mat")
        ]
        
        for word, relation, head in dependencies:
            print(f"{word} --{relation}--> {head}")
    
    def semantics_analysis(self):
        """语义学分析"""
        print("=== 语义学理论 ===")
        
        semantic_theories = {
            "形式语义学": {
                "方法": "逻辑和数学工具",
                "核心概念": "真值条件、可能世界",
                "应用": "逻辑推理、问答系统"
            },
            "分布式语义学": {
                "方法": "统计学习",
                "核心假设": "相似上下文的词汇有相似含义",
                "应用": "词向量、语言模型"
            },
            "认知语义学": {
                "方法": "认知科学",
                "核心概念": "概念隐喻、意象图式",
                "应用": "常识推理、概念表示"
            }
        }
        
        for theory, details in semantic_theories.items():
            print(f"\n{theory}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        return semantic_theories

class LanguageModeling:
    """语言建模理论"""
    
    def __init__(self):
        self.models = {}
    
    def ngram_theory(self):
        """N-gram语言模型理论"""
        print("=== N-gram语言模型 ===")
        
        print("基本思想:")
        print("P(w₁, w₂, ..., wₙ) ≈ ∏ᵢ P(wᵢ|wᵢ₋ₙ₊₁, ..., wᵢ₋₁)")
        print()
        
        # N-gram模型比较
        ngram_types = {
            "Unigram": {
                "公式": "P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ)",
                "假设": "词汇独立",
                "优点": "简单，计算快",
                "缺点": "忽略上下文"
            },
            "Bigram": {
                "公式": "P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ|wᵢ₋₁)",
                "假设": "马尔科夫假设(n=1)",
                "优点": "考虑局部上下文",
                "缺点": "上下文窗口小"
            },
            "Trigram": {
                "公式": "P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ|wᵢ₋₂, wᵢ₋₁)",
                "假设": "马尔科夫假设(n=2)",
                "优点": "更多上下文信息",
                "缺点": "数据稀疏性"
            }
        }
        
        for model_type, details in ngram_types.items():
            print(f"\n{model_type}模型:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 实现简单的bigram模型
        self.implement_bigram_model()
        
        return ngram_types
    
    def implement_bigram_model(self):
        """实现bigram模型"""
        print("\n=== Bigram模型实现示例 ===")
        
        # 示例语料
        corpus = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "a cat and a dog played together"
        ]
        
        # 构建bigram计数
        bigram_counts = defaultdict(lambda: defaultdict(int))
        unigram_counts = defaultdict(int)
        
        for sentence in corpus:
            words = ["<s>"] + sentence.lower().split() + ["</s>"]
            
            for i in range(len(words)):
                unigram_counts[words[i]] += 1
                if i > 0:
                    bigram_counts[words[i-1]][words[i]] += 1
        
        # 计算概率
        def bigram_probability(w1, w2):
            if unigram_counts[w1] == 0:
                return 0
            return bigram_counts[w1][w2] / unigram_counts[w1]
        
        # 示例计算
        print("Bigram概率示例:")
        test_pairs = [("the", "cat"), ("cat", "sat"), ("<s>", "the")]
        for w1, w2 in test_pairs:
            prob = bigram_probability(w1, w2)
            print(f"P({w2}|{w1}) = {prob:.3f}")
        
        # 句子概率计算
        def sentence_probability(sentence):
            words = ["<s>"] + sentence.lower().split() + ["</s>"]
            prob = 1.0
            
            for i in range(1, len(words)):
                prob *= bigram_probability(words[i-1], words[i])
            
            return prob
        
        test_sentences = ["the cat sat", "a dog ran"]
        print("\n句子概率:")
        for sentence in test_sentences:
            prob = sentence_probability(sentence)
            print(f"P('{sentence}') = {prob:.6f}")
        
        return bigram_counts, unigram_counts
    
    def smoothing_techniques(self):
        """平滑技术"""
        print("=== 平滑技术 ===")
        
        print("数据稀疏性问题:")
        print("- 训练语料中未出现的n-gram概率为0")
        print("- 导致整个句子概率为0")
        print("- 需要平滑技术分配概率质量")
        print()
        
        smoothing_methods = {
            "加法平滑 (Add-k)": {
                "公式": "P(wᵢ|wᵢ₋₁) = (C(wᵢ₋₁,wᵢ) + k) / (C(wᵢ₋₁) + k·|V|)",
                "参数": "k (通常k=1, 称为拉普拉斯平滑)",
                "优点": "简单直观",
                "缺点": "对低频事件过度平滑"
            },
            "Good-Turing平滑": {
                "思想": "用频率r+1的事件数估计频率r的概率",
                "适用": "处理零频和低频事件",
                "优点": "理论基础强",
                "缺点": "实现复杂"
            },
            "Kneser-Ney平滑": {
                "思想": "基于continuation count而非频率",
                "特点": "考虑词汇在不同上下文中的分布",
                "优点": "效果最好",
                "缺点": "计算复杂"
            },
            "回退 (Back-off)": {
                "思想": "高阶n-gram不可靠时回退到低阶",
                "实现": "Katz回退、插值平滑",
                "优点": "充分利用各阶信息",
                "缺点": "需要多个模型"
            }
        }
        
        for method, details in smoothing_methods.items():
            print(f"\n{method}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # 实现拉普拉斯平滑
        self.implement_laplace_smoothing()
        
        return smoothing_methods
    
    def implement_laplace_smoothing(self):
        """实现拉普拉斯平滑"""
        print("\n=== 拉普拉斯平滑示例 ===")
        
        # 使用前面的bigram计数
        corpus = ["the cat sat", "the dog ran"]
        
        bigram_counts = defaultdict(lambda: defaultdict(int))
        unigram_counts = defaultdict(int)
        vocabulary = set()
        
        for sentence in corpus:
            words = ["<s>"] + sentence.split() + ["</s>"]
            vocabulary.update(words)
            
            for i in range(len(words)):
                unigram_counts[words[i]] += 1
                if i > 0:
                    bigram_counts[words[i-1]][words[i]] += 1
        
        V = len(vocabulary)
        
        def laplace_probability(w1, w2, k=1):
            """拉普拉斯平滑概率"""
            return (bigram_counts[w1][w2] + k) / (unigram_counts[w1] + k * V)
        
        def unsmoothed_probability(w1, w2):
            """未平滑概率"""
            if unigram_counts[w1] == 0:
                return 0
            return bigram_counts[w1][w2] / unigram_counts[w1]
        
        # 比较平滑前后的概率
        test_pairs = [("the", "cat"), ("cat", "dog"), ("dog", "elephant")]
        
        print("平滑前后概率比较:")
        print("Bigram\t\tUnsmoothed\tLaplace")
        print("-" * 40)
        
        for w1, w2 in test_pairs:
            unsmooth = unsmoothed_probability(w1, w2)
            laplace = laplace_probability(w1, w2)
            print(f"{w1},{w2}\t\t{unsmooth:.4f}\t\t{laplace:.4f}")
        
        return vocabulary, bigram_counts
```

## 2. 词汇表示理论 📊

```python
class WordRepresentationTheory:
    """词汇表示理论"""
    
    def __init__(self):
        self.representation_methods = {}
    
    def distributional_hypothesis(self):
        """分布假设理论"""
        print("=== 分布假设 ===")
        
        print("Firth (1957): 'You shall know a word by the company it keeps'")
        print()
        print("核心思想:")
        print("- 词汇的含义由其上下文决定")
        print("- 相似上下文中出现的词汇有相似含义")
        print("- 可以通过统计上下文信息学习词汇表示")
        print()
        
        # 上下文类型
        context_types = {
            "词袋上下文": {
                "定义": "固定窗口内的词汇集合",
                "特点": "忽略词序",
                "应用": "传统词向量模型"
            },
            "句法上下文": {
                "定义": "通过句法关系定义的上下文",
                "特点": "考虑语法结构",
                "应用": "句法词向量"
            },
            "文档上下文": {
                "定义": "整个文档作为上下文",
                "特点": "长距离依赖",
                "应用": "文档表示学习"
            }
        }
        
        for context_type, details in context_types.items():
            print(f"{context_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        return context_types
    
    def vector_space_models(self):
        """向量空间模型"""
        print("=== 向量空间模型 ===")
        
        # TF-IDF原理
        print("TF-IDF (Term Frequency - Inverse Document Frequency):")
        print("TF-IDF(t,d) = TF(t,d) × IDF(t)")
        print("其中:")
        print("- TF(t,d) = count(t,d) / |d|")
        print("- IDF(t) = log(|D| / |{d: t ∈ d}|)")
        print()
        
        # 实现简单的TF-IDF
        documents = [
            "the cat sat on the mat",
            "the dog ran in the park", 
            "cats and dogs are pets"
        ]
        
        # 计算TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        print("TF-IDF矩阵示例:")
        print("词汇:", feature_names[:10])  # 显示前10个词
        print("文档0向量:", tfidf_matrix[0].toarray().flatten()[:10])
        print()
        
        # 词汇相似度计算
        def cosine_similarity_manual(vec1, vec2):
            """手动计算余弦相似度"""
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
        
        # 计算文档间相似度
        print("文档相似度矩阵:")
        n_docs = tfidf_matrix.shape[0]
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        for i in range(n_docs):
            for j in range(n_docs):
                vec1 = tfidf_matrix[i].toarray().flatten()
                vec2 = tfidf_matrix[j].toarray().flatten()
                similarity_matrix[i, j] = cosine_similarity_manual(vec1, vec2)
        
        print(similarity_matrix)
        
        # 可视化TF-IDF
        self.visualize_tfidf(tfidf_matrix, feature_names, documents)
        
        return tfidf_matrix, feature_names
    
    def visualize_tfidf(self, tfidf_matrix, feature_names, documents):
        """可视化TF-IDF"""
        # 创建热图
        plt.figure(figsize=(12, 8))
        
        # 选择前15个特征进行可视化
        n_features = min(15, len(feature_names))
        tfidf_dense = tfidf_matrix.toarray()[:, :n_features]
        
        plt.subplot(2, 2, 1)
        sns.heatmap(tfidf_dense, 
                   xticklabels=feature_names[:n_features],
                   yticklabels=[f'Doc {i}' for i in range(len(documents))],
                   annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('TF-IDF 热图')
        plt.xlabel('词汇')
        plt.ylabel('文档')
        
        # PCA降维可视化
        if tfidf_matrix.shape[1] > 2:
            pca = PCA(n_components=2)
            tfidf_2d = pca.fit_transform(tfidf_matrix.toarray())
            
            plt.subplot(2, 2, 2)
            plt.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], s=100, alpha=0.7)
            for i, doc in enumerate(documents):
                plt.annotate(f'Doc {i}', (tfidf_2d[i, 0], tfidf_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points')
            plt.title('文档在TF-IDF空间的分布')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # 词汇频率分布
        plt.subplot(2, 2, 3)
        word_freqs = np.sum(tfidf_matrix.toarray(), axis=0)
        top_words_idx = np.argsort(word_freqs)[-10:]
        top_words = [feature_names[i] for i in top_words_idx]
        top_freqs = word_freqs[top_words_idx]
        
        plt.barh(range(len(top_words)), top_freqs)
        plt.yticks(range(len(top_words)), top_words)
        plt.title('Top 10 词汇 TF-IDF 权重')
        plt.xlabel('TF-IDF Score')
        
        # 词云
        plt.subplot(2, 2, 4)
        all_text = ' '.join(documents)
        if all_text.strip():  # 检查文本不为空
            try:
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white').generate(all_text)
                plt.imshow(wordcloud, interpolation='bilinear')
            except:
                plt.text(0.5, 0.5, 'WordCloud\nNot Available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('词云')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def word_embeddings_theory(self):
        """词向量理论"""
        print("=== 词向量理论 ===")
        
        embedding_models = {
            "Word2Vec": {
                "架构": ["Skip-gram", "CBOW"],
                "目标": "预测上下文词汇",
                "优点": ["高效训练", "捕获语义相似性"],
                "缺点": ["静态表示", "多义词问题"]
            },
            "GloVe": {
                "架构": "全局向量",
                "目标": "分解词汇共现矩阵",
                "优点": ["结合全局和局部信息"],
                "缺点": ["计算复杂", "内存需求大"]
            },
            "FastText": {
                "架构": "子词嵌入",
                "目标": "考虑词汇内部结构",
                "优点": ["处理OOV词汇", "形态信息"],
                "缺点": ["向量维度高", "噪声敏感"]
            }
        }
        
        for model, details in embedding_models.items():
            print(f"\n{model}:")
            for key, value in details.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(value)}")
                else:
                    print(f"  {key}: {value}")
        
        # Word2Vec数学原理
        print("\n=== Word2Vec数学原理 ===")
        print("Skip-gram目标函数:")
        print("L = Σ_{w∈C} Σ_{c∈context(w)} log P(c|w)")
        print("其中:")
        print("P(c|w) = exp(u_c^T v_w) / Σ_{w'} exp(u_{w'}^T v_w)")
        print()
        
        print("负采样优化:")
        print("L = log σ(u_c^T v_w) + Σ_{i=1}^k E_{w_i~P_n(w)} [log σ(-u_{w_i}^T v_w)]")
        print("其中 σ(x) = 1/(1+exp(-x))")
        
        return embedding_models
    
    def contextual_embeddings(self):
        """上下文词向量"""
        print("=== 上下文词向量 ===")
        
        print("传统词向量问题:")
        print("- 一词一向量，无法处理多义词")
        print("- 静态表示，忽略上下文变化")
        print("- 例如：'bank'在不同语境下含义不同")
        print()
        
        contextual_models = {
            "ELMo": {
                "架构": "双向LSTM",
                "特点": "层次化表示",
                "创新": "字符级输入",
                "表示": "加权平均各层输出"
            },
            "GPT": {
                "架构": "Transformer解码器",
                "特点": "单向语言模型",
                "创新": "大规模预训练",
                "表示": "最后一层隐藏状态"
            },
            "BERT": {
                "架构": "Transformer编码器",
                "特点": "双向上下文",
                "创新": "掩码语言模型",
                "表示": "所有层的加权组合"
            }
        }
        
        for model, details in contextual_models.items():
            print(f"{model}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 上下文表示的数学描述
        print("上下文表示公式:")
        print("h_i^(l) = Transformer_layer^(l)(h_i^(l-1), context)")
        print("其中 context 包含整个序列的信息")
        
        return contextual_models

class SemanticComposition:
    """语义组合理论"""
    
    def __init__(self):
        pass
    
    def compositionality_principle(self):
        """组合性原理"""
        print("=== 语义组合性原理 ===")
        
        print("Frege原理:")
        print("'句子的含义是其组成部分含义的函数'")
        print()
        
        composition_methods = {
            "加法组合": {
                "公式": "v(A B) = v(A) + v(B)",
                "优点": "简单高效",
                "缺点": "忽略语法结构",
                "适用": "词袋模型"
            },
            "乘法组合": {
                "公式": "v(A B) = v(A) ⊙ v(B) (逐元素相乘)",
                "优点": "保留特征交互",
                "缺点": "信息可能丢失",
                "适用": "简单组合"
            },
            "张量积": {
                "公式": "v(A B) = v(A) ⊗ v(B)",
                "优点": "保留完整信息",
                "缺点": "维度爆炸",
                "适用": "理论分析"
            },
            "循环卷积": {
                "公式": "v(A B) = v(A) ⊛ v(B)",
                "优点": "维度不变",
                "缺点": "不可交换",
                "适用": "序列建模"
            }
        }
        
        for method, details in composition_methods.items():
            print(f"{method}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 实现简单的组合示例
        self.demonstrate_composition()
        
        return composition_methods
    
    def demonstrate_composition(self):
        """演示组合方法"""
        print("=== 组合方法示例 ===")
        
        # 假设词向量
        np.random.seed(42)
        word_vectors = {
            "red": np.random.randn(5),
            "car": np.random.randn(5),
            "fast": np.random.randn(5),
            "blue": np.random.randn(5)
        }
        
        # 不同组合方法
        def additive_composition(v1, v2):
            return v1 + v2
        
        def multiplicative_composition(v1, v2):
            return v1 * v2
        
        def weighted_composition(v1, v2, alpha=0.5):
            return alpha * v1 + (1 - alpha) * v2
        
        # 计算短语表示
        phrases = [("red", "car"), ("fast", "car"), ("blue", "car")]
        
        print("短语表示比较:")
        print("Phrase\t\tAdditive\tMultiplicative\tWeighted")
        print("-" * 60)
        
        for w1, w2 in phrases:
            v1, v2 = word_vectors[w1], word_vectors[w2]
            
            add_comp = additive_composition(v1, v2)
            mult_comp = multiplicative_composition(v1, v2)
            weighted_comp = weighted_composition(v1, v2)
            
            print(f"{w1} {w2}\t{add_comp[:2]}\t{mult_comp[:2]}\t{weighted_comp[:2]}")
        
        # 计算组合后的相似度
        red_car = additive_composition(word_vectors["red"], word_vectors["car"])
        blue_car = additive_composition(word_vectors["blue"], word_vectors["car"])
        fast_car = additive_composition(word_vectors["fast"], word_vectors["car"])
        
        def cosine_sim(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        print(f"\n相似度分析:")
        print(f"sim(red car, blue car) = {cosine_sim(red_car, blue_car):.3f}")
        print(f"sim(red car, fast car) = {cosine_sim(red_car, fast_car):.3f}")
        
        return word_vectors

class DiscourseAnalysis:
    """话语分析理论"""
    
    def __init__(self):
        pass
    
    def coherence_and_cohesion(self):
        """连贯性和凝聚性"""
        print("=== 连贯性和凝聚性 ===")
        
        print("连贯性 (Coherence):")
        print("- 文本的整体意义统一性")
        print("- 读者能够理解文本的逻辑关系")
        print("- 基于语义和语用层面")
        print()
        
        print("凝聚性 (Cohesion):")
        print("- 文本表面的语言连接")
        print("- 通过词汇和语法手段实现")
        print("- 包括指代、替换、省略等")
        print()
        
        cohesion_devices = {
            "词汇凝聚": {
                "重复": "同一词汇的重复使用",
                "同义": "使用同义词或近义词", 
                "上下义": "使用上位词或下位词",
                "搭配": "词汇之间的习惯搭配"
            },
            "语法凝聚": {
                "指代": "人称代词、指示代词",
                "替代": "用其他词汇替代前文内容",
                "省略": "省略可从上下文推断的内容",
                "连接": "连接词表示逻辑关系"
            }
        }
        
        for device_type, devices in cohesion_devices.items():
            print(f"{device_type}:")
            for device, description in devices.items():
                print(f"  {device}: {description}")
            print()
        
        return cohesion_devices
    
    def discourse_relations(self):
        """话语关系"""
        print("=== 话语关系理论 ===")
        
        discourse_relations = {
            "因果关系": {
                "标记词": ["因为", "所以", "因此", "由于"],
                "示例": "因为下雨，所以我带了伞",
                "逻辑": "P → Q"
            },
            "对比关系": {
                "标记词": ["但是", "然而", "相反", "不过"],
                "示例": "他很聪明，但是不够努力",
                "逻辑": "P ∧ ¬Q"
            },
            "递进关系": {
                "标记词": ["而且", "并且", "不仅...还", "更"],
                "示例": "他不仅聪明，而且努力",
                "逻辑": "P ∧ Q (Q > P)"
            },
            "条件关系": {
                "标记词": ["如果", "假如", "只要", "除非"],
                "示例": "如果明天晴天，我们就去郊游",
                "逻辑": "P → Q"
            }
        }
        
        for relation, details in discourse_relations.items():
            print(f"{relation}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        return discourse_relations
```

## 3. 语言理解与生成 💭

```python
class LanguageUnderstandingGeneration:
    """语言理解与生成理论"""
    
    def __init__(self):
        pass
    
    def parsing_theories(self):
        """句法解析理论"""
        print("=== 句法解析理论 ===")
        
        parsing_approaches = {
            "自顶向下解析": {
                "策略": "从根节点开始，逐步展开",
                "优点": "目标导向，易于实现",
                "缺点": "可能产生无用分支",
                "算法": "递归下降、预测解析"
            },
            "自底向上解析": {
                "策略": "从词汇开始，逐步构建树",
                "优点": "只构建有用的结构",
                "缺点": "可能产生歧义",
                "算法": "移位-归约、CKY算法"
            },
            "Chart解析": {
                "策略": "动态规划，避免重复计算",
                "优点": "高效处理歧义",
                "缺点": "内存需求大",
                "算法": "Earley算法、CKY算法"
            },
            "依存解析": {
                "策略": "构建词汇间的依存关系",
                "优点": "跨语言适用性强",
                "缺点": "缺乏层次结构",
                "算法": "基于转移、基于图"
            }
        }
        
        for approach, details in parsing_approaches.items():
            print(f"{approach}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # CKY算法示例
        self.demonstrate_cky_parsing()
        
        return parsing_approaches
    
    def demonstrate_cky_parsing(self):
        """演示CKY解析算法"""
        print("=== CKY算法示例 ===")
        
        # CNF文法规则
        grammar_rules = {
            # 终结符规则 A -> a
            "Det": ["the", "a"],
            "N": ["cat", "dog", "mat"],
            "V": ["sat", "ran"],
            "P": ["on", "in"],
            # 非终结符规则 A -> BC
            "NP": [("Det", "N")],
            "PP": [("P", "NP")],
            "VP": [("V", "PP")],
            "S": [("NP", "VP")]
        }
        
        sentence = ["the", "cat", "sat", "on", "the", "mat"]
        n = len(sentence)
        
        # 初始化CKY表
        table = [[set() for _ in range(n)] for _ in range(n)]
        
        # 填充对角线（长度为1的子串）
        for i in range(n):
            word = sentence[i]
            for lhs, rhs_list in grammar_rules.items():
                if word in rhs_list:
                    table[i][i].add(lhs)
        
        # 填充表格（长度从2到n）
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                for k in range(i, j):
                    # 检查所有可能的分割点
                    left_set = table[i][k]
                    right_set = table[k+1][j]
                    
                    for lhs, rhs_list in grammar_rules.items():
                        for rhs in rhs_list:
                            if isinstance(rhs, tuple) and len(rhs) == 2:
                                if rhs[0] in left_set and rhs[1] in right_set:
                                    table[i][j].add(lhs)
        
        # 输出解析表
        print("CKY解析表:")
        print("句子:", ' '.join(sentence))
        print()
        
        for i in range(n):
            for j in range(i, n):
                if table[i][j]:
                    span = ' '.join(sentence[i:j+1])
                    categories = ', '.join(table[i][j])
                    print(f"[{i},{j}] '{span}': {categories}")
        
        # 检查是否可以解析为完整句子
        if "S" in table[0][n-1]:
            print(f"\n✓ 句子可以解析为S (sentence)")
        else:
            print(f"\n✗ 句子无法解析")
        
        return table
    
    def semantic_parsing(self):
        """语义解析"""
        print("=== 语义解析 ===")
        
        print("语义解析目标:")
        print("将自然语言转换为形式化的语义表示")
        print()
        
        semantic_representations = {
            "一阶逻辑": {
                "形式": "∃x (cat(x) ∧ sat(x, mat))",
                "优点": "表达能力强，推理能力强",
                "缺点": "复杂，难以学习",
                "应用": "问答系统、推理"
            },
            "Lambda演算": {
                "形式": "λx.cat(x) ∧ sat(x, mat)",
                "优点": "组合性强",
                "缺点": "抽象程度高",
                "应用": "组合语义学"
            },
            "SQL查询": {
                "形式": "SELECT * FROM table WHERE cat='true'",
                "优点": "直接可执行",
                "缺点": "表达能力有限",
                "应用": "数据库查询"
            },
            "抽象语法树": {
                "形式": "树状结构表示",
                "优点": "结构清晰",
                "缺点": "领域特定",
                "应用": "代码生成、API调用"
            }
        }
        
        for repr_type, details in semantic_representations.items():
            print(f"{repr_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        return semantic_representations
    
    def text_generation_theories(self):
        """文本生成理论"""
        print("=== 文本生成理论 ===")
        
        generation_approaches = {
            "基于规则": {
                "方法": "预定义模板和规则",
                "优点": "可控性强，质量稳定",
                "缺点": "缺乏多样性，扩展性差",
                "应用": "报告生成、数据到文本"
            },
            "基于检索": {
                "方法": "从语料库中检索相似文本",
                "优点": "语法正确性高",
                "缺点": "创新性不足",
                "应用": "聊天机器人、问答系统"
            },
            "基于统计": {
                "方法": "n-gram语言模型",
                "优点": "可以生成新内容",
                "缺点": "长距离一致性差",
                "应用": "文本补全、摘要"
            },
            "基于神经网络": {
                "方法": "RNN、Transformer等",
                "优点": "流畅度高，多样性好",
                "缺点": "可能产生幻觉",
                "应用": "对话、创作、翻译"
            }
        }
        
        for approach, details in generation_approaches.items():
            print(f"{approach}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        # 生成策略
        print("=== 生成策略 ===")
        
        generation_strategies = {
            "贪心解码": {
                "策略": "每步选择概率最高的词",
                "优点": "简单快速",
                "缺点": "可能陷入局部最优",
                "数学": "w_t = argmax P(w|w_<t)"
            },
            "束搜索": {
                "策略": "保持k个最优候选序列",
                "优点": "质量较高",
                "缺点": "计算复杂度高",
                "数学": "保持top-k序列"
            },
            "随机采样": {
                "策略": "按概率分布随机采样",
                "优点": "多样性好",
                "缺点": "质量不稳定",
                "数学": "w_t ~ P(w|w_<t)"
            },
            "Top-k采样": {
                "策略": "在top-k词汇中随机采样",
                "优点": "平衡质量和多样性",
                "缺点": "k值需要调优",
                "数学": "重新归一化top-k概率"
            },
            "Top-p采样": {
                "策略": "累积概率达到p时截断",
                "优点": "自适应词汇集合大小",
                "缺点": "p值需要调优",
                "数学": "P_cumsum ≤ p"
            }
        }
        
        for strategy, details in generation_strategies.items():
            print(f"{strategy}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
        
        return generation_approaches, generation_strategies

def comprehensive_nlp_theory_summary():
    """NLP理论综合总结"""
    print("=== NLP理论综合总结 ===")
    
    summary = {
        "理论基础": {
            "语言学基础": "形态学、句法学、语义学、语用学",
            "数学基础": "概率论、信息论、线性代数",
            "认知基础": "心理语言学、认知科学"
        },
        
        "核心概念": {
            "分布假设": "词汇含义由上下文决定",
            "组合性原理": "整体含义由部分含义组合",
            "稀疏性问题": "自然语言的高维稀疏特性"
        },
        
        "表示学习": {
            "符号表示": "离散符号、语法树、逻辑形式",
            "分布表示": "词向量、句向量、文档向量",
            "上下文表示": "动态表示、多义词消歧"
        },
        
        "模型演进": {
            "统计方法": "n-gram、HMM、CRF",
            "神经方法": "RNN、CNN、Attention",
            "预训练模型": "BERT、GPT、T5"
        },
        
        "应用任务": {
            "理解任务": "分类、标注、解析、问答",
            "生成任务": "翻译、摘要、对话、创作",
            "多模态": "图文匹配、视频描述"
        },
        
        "未来方向": {
            "大语言模型": "规模化、涌现能力、对齐",
            "多模态融合": "视觉-语言、语音-语言",
            "知识增强": "常识推理、知识图谱",
            "可解释性": "注意力可视化、探测实验"
        }
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    return summary

print("自然语言处理理论指南加载完成！")
```

## 参考文献 📚

- Manning & Schütze (1999): "Foundations of Statistical Natural Language Processing"
- Jurafsky & Martin (2020): "Speech and Language Processing"
- Goldberg (2017): "Neural Network Methods for Natural Language Processing"
- Koehn (2020): "Neural Machine Translation"
- Rogers et al. (2020): "A Primer on Neural Network Models for Natural Language Processing"

## 下一步学习
- [NLP模型实现](../nlp_models.md) - 实际模型构建
- [Transformer架构](transformer_architecture.md) - 深入理解Transformer
- [大语言模型理论](llm_architecture_theory.md) - LLM架构分析