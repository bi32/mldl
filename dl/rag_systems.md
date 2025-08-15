# RAG系统实战：检索增强生成 🔍

从原理到实践，构建高质量的检索增强生成系统。

## 1. RAG核心概念 🎯

```python
import numpy as np
from typing import List, Dict, Any
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    """RAG系统基础架构"""
    
    def __init__(self, retriever_model="BAAI/bge-large-zh-v1.5", 
                 generator_model="meta-llama/Llama-2-7b-chat-hf"):
        # 检索模型
        self.retriever = SentenceTransformer(retriever_model)
        
        # 生成模型
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModel.from_pretrained(generator_model)
        
        # 向量数据库
        self.index = None
        self.documents = []
    
    def build_index(self, documents: List[str]):
        """构建向量索引"""
        # 编码文档
        embeddings = self.retriever.encode(documents)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.documents = documents
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """检索相关文档"""
        # 编码查询
        query_embedding = self.retriever.encode([query])
        
        # 搜索最近邻
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        # 返回文档
        return [self.documents[idx] for idx in indices[0]]
    
    def generate(self, query: str, context: List[str]) -> str:
        """基于上下文生成回答"""
        # 构建提示
        prompt = self._build_prompt(query, context)
        
        # 生成回答
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.generator.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _build_prompt(self, query: str, context: List[str]) -> str:
        """构建提示模板"""
        context_str = "\n".join(context)
        prompt = f"""基于以下上下文回答问题：

上下文：
{context_str}

问题：{query}

回答："""
        return prompt
    
    def query(self, question: str, k: int = 5) -> str:
        """完整的RAG流程"""
        # 1. 检索
        relevant_docs = self.retrieve(question, k)
        
        # 2. 生成
        answer = self.generate(question, relevant_docs)
        
        return answer

# 使用示例
rag = RAGPipeline()
documents = [
    "机器学习是人工智能的一个分支。",
    "深度学习使用神经网络进行学习。",
    "Transformer是一种基于注意力机制的架构。"
]
rag.build_index(documents)
answer = rag.query("什么是机器学习？")
print(answer)
```

## 2. 文档处理与切分 📄

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
import tiktoken

class DocumentProcessor:
    """文档处理和切分"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_by_characters(self, text: str) -> List[str]:
        """按字符切分"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        return splitter.split_text(text)
    
    def split_by_tokens(self, text: str, model="gpt-3.5-turbo") -> List[str]:
        """按token切分"""
        encoding = tiktoken.encoding_for_model(model)
        
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=encoding.name
        )
        return splitter.split_text(text)
    
    def split_by_sentences(self, text: str) -> List[str]:
        """按句子切分"""
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0
        )
        return splitter.split_text(text)
    
    def sliding_window(self, text: str, window_size: int, stride: int) -> List[str]:
        """滑动窗口切分"""
        chunks = []
        sentences = text.split('。')
        
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = '。'.join(sentences[i:i + window_size])
            chunks.append(chunk)
        
        return chunks
    
    def hierarchical_split(self, text: str) -> Dict[str, List[str]]:
        """层次化切分"""
        # 大块
        large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        large_chunks = large_splitter.split_text(text)
        
        # 小块
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        
        result = {
            "large": large_chunks,
            "small": []
        }
        
        for chunk in large_chunks:
            small_chunks = small_splitter.split_text(chunk)
            result["small"].extend(small_chunks)
        
        return result

# 智能切分策略
class SmartChunker:
    """智能文档切分"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
    
    def chunk_by_document_type(self, text: str, doc_type: str) -> List[str]:
        """根据文档类型选择切分策略"""
        
        strategies = {
            "code": {"size": 300, "overlap": 50},
            "paper": {"size": 500, "overlap": 100},
            "dialogue": {"size": 200, "overlap": 20},
            "narrative": {"size": 800, "overlap": 150}
        }
        
        if doc_type in strategies:
            config = strategies[doc_type]
            self.processor.chunk_size = config["size"]
            self.processor.chunk_overlap = config["overlap"]
        
        return self.processor.split_by_characters(text)
    
    def semantic_chunking(self, text: str, threshold: float = 0.7) -> List[str]:
        """语义切分：基于语义相似度"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = text.split('。')
        
        # 计算句子嵌入
        embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # 计算相似度
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            if similarity < threshold:
                # 语义转折，创建新块
                chunks.append('。'.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append('。'.join(current_chunk))
        
        return chunks
```

## 3. 向量数据库实战 🗄️

```python
# Chroma向量数据库
import chromadb
from chromadb.utils import embedding_functions

class ChromaVectorStore:
    """Chroma向量数据库封装"""
    
    def __init__(self, collection_name="documents"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # 使用OpenAI嵌入
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="your-api-key",
            model_name="text-embedding-ada-002"
        )
        
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档"""
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )
    
    def search(self, query: str, k: int = 5) -> Dict:
        """搜索文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def hybrid_search(self, query: str, k: int = 5) -> List[str]:
        """混合搜索：向量 + 关键词"""
        # 向量搜索
        vector_results = self.search(query, k * 2)
        
        # 关键词搜索（简化版）
        keyword_results = self.collection.query(
            where_document={"$contains": query.split()[0]},
            n_results=k
        )
        
        # 合并结果
        all_docs = set()
        for docs in vector_results["documents"]:
            all_docs.update(docs)
        for docs in keyword_results["documents"]:
            all_docs.update(docs)
        
        return list(all_docs)[:k]

# Pinecone向量数据库
import pinecone

class PineconeVectorStore:
    """Pinecone向量数据库"""
    
    def __init__(self, index_name="rag-index"):
        pinecone.init(
            api_key="your-api-key",
            environment="us-west1-gcp"
        )
        
        # 创建索引
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=768,
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def upsert(self, documents: List[str], ids: List[str] = None):
        """插入或更新文档"""
        embeddings = self.encoder.encode(documents)
        
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        
        vectors = [
            (ids[i], embeddings[i].tolist(), {"text": documents[i]})
            for i in range(len(documents))
        ]
        
        self.index.upsert(vectors)
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """查询相似文档"""
        query_embedding = self.encoder.encode([query_text])[0]
        
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches

# Weaviate向量数据库
import weaviate

class WeaviateVectorStore:
    """Weaviate向量数据库"""
    
    def __init__(self):
        self.client = weaviate.Client("http://localhost:8080")
        self._create_schema()
    
    def _create_schema(self):
        """创建schema"""
        schema = {
            "class": "Document",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"]
                },
                {
                    "name": "source",
                    "dataType": ["string"]
                }
            ]
        }
        
        try:
            self.client.schema.create_class(schema)
        except:
            pass  # Schema already exists
    
    def add_document(self, content: str, source: str = ""):
        """添加文档"""
        self.client.data_object.create(
            {
                "content": content,
                "source": source
            },
            "Document"
        )
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """向量搜索"""
        result = (
            self.client.query
            .get("Document", ["content", "source"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .do()
        )
        
        return result["data"]["Get"]["Document"]
```

## 4. 检索优化技术 🚀

```python
class RetrievalOptimizer:
    """检索优化技术"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """重排序：使用交叉编码器"""
        from sentence_transformers import CrossEncoder
        
        # 交叉编码器评分
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        pairs = [[query, doc] for doc in documents]
        scores = reranker.predict(pairs)
        
        # 排序并返回
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in sorted_indices]
    
    def mmr_selection(self, query: str, documents: List[str], 
                     lambda_param: float = 0.5, k: int = 5) -> List[str]:
        """最大边际相关性(MMR)选择"""
        # 编码
        query_embedding = self.encoder.encode([query])[0]
        doc_embeddings = self.encoder.encode(documents)
        
        # 计算相似度
        similarities = np.dot(doc_embeddings, query_embedding)
        
        selected = []
        selected_indices = []
        
        for _ in range(min(k, len(documents))):
            if not selected_indices:
                # 选择最相似的
                idx = np.argmax(similarities)
            else:
                # MMR分数
                mmr_scores = []
                for i, doc_emb in enumerate(doc_embeddings):
                    if i in selected_indices:
                        mmr_scores.append(-np.inf)
                        continue
                    
                    # 与查询的相似度
                    query_sim = similarities[i]
                    
                    # 与已选文档的最大相似度
                    selected_embs = doc_embeddings[selected_indices]
                    max_doc_sim = np.max(np.dot(selected_embs, doc_emb))
                    
                    # MMR分数
                    mmr = lambda_param * query_sim - (1 - lambda_param) * max_doc_sim
                    mmr_scores.append(mmr)
                
                idx = np.argmax(mmr_scores)
            
            selected.append(documents[idx])
            selected_indices.append(idx)
        
        return selected
    
    def query_expansion(self, query: str) -> List[str]:
        """查询扩展"""
        expanded_queries = [query]
        
        # 1. 同义词扩展
        from nltk.corpus import wordnet
        words = query.split()
        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                synonyms = set()
                for syn in synsets[:2]:  # 取前两个含义
                    for lemma in syn.lemmas()[:3]:  # 每个含义取3个同义词
                        synonyms.add(lemma.name())
                
                for synonym in synonyms:
                    expanded = query.replace(word, synonym)
                    if expanded != query:
                        expanded_queries.append(expanded)
        
        # 2. 回译扩展
        # 这里简化处理，实际需要调用翻译API
        
        return expanded_queries[:5]  # 限制数量
    
    def hypothetical_document_embedding(self, query: str, llm_model) -> np.ndarray:
        """HyDE：生成假设文档进行检索"""
        # 让LLM生成假设答案
        prompt = f"""请为以下问题生成一个详细的答案：
        
        问题：{query}
        
        答案："""
        
        hypothetical_answer = llm_model.generate(prompt)
        
        # 编码假设答案
        embedding = self.encoder.encode([hypothetical_answer])[0]
        
        return embedding

# 多路检索
class MultiPathRetrieval:
    """多路径检索策略"""
    
    def __init__(self):
        self.dense_retriever = SentenceTransformer('all-MiniLM-L6-v2')
        self.sparse_retriever = None  # BM25等
    
    def ensemble_retrieval(self, query: str, documents: List[str], 
                          alpha: float = 0.5) -> List[str]:
        """集成检索：密集+稀疏"""
        # 密集检索
        dense_scores = self._dense_search(query, documents)
        
        # 稀疏检索(BM25)
        sparse_scores = self._bm25_search(query, documents)
        
        # 分数融合
        final_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        # 排序返回
        sorted_indices = np.argsort(final_scores)[::-1]
        return [documents[i] for i in sorted_indices]
    
    def _dense_search(self, query: str, documents: List[str]) -> np.ndarray:
        """密集向量搜索"""
        query_emb = self.dense_retriever.encode([query])[0]
        doc_embs = self.dense_retriever.encode(documents)
        
        scores = np.dot(doc_embs, query_emb)
        return scores
    
    def _bm25_search(self, query: str, documents: List[str]) -> np.ndarray:
        """BM25稀疏检索"""
        from rank_bm25 import BM25Okapi
        
        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        
        return scores
```

## 5. 提示工程与上下文管理 📝

```python
class PromptManager:
    """提示工程管理"""
    
    def __init__(self):
        self.templates = {
            "qa": """根据提供的上下文回答问题。如果上下文中没有相关信息，请说"我无法根据提供的信息回答这个问题"。

上下文：
{context}

问题：{question}

答案：""",
            
            "summary": """请总结以下文档的主要内容：

{context}

总结：""",
            
            "extraction": """从以下文本中提取{entity_type}：

文本：{context}

提取结果：""",
            
            "comparison": """比较以下两个观点：

观点1：{context1}

观点2：{context2}

比较分析："""
        }
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """格式化提示"""
        template = self.templates.get(template_name, self.templates["qa"])
        return template.format(**kwargs)
    
    def create_few_shot_prompt(self, examples: List[Dict], query: str) -> str:
        """少样本提示"""
        prompt = "以下是一些问答示例：\n\n"
        
        for example in examples:
            prompt += f"问题：{example['question']}\n"
            prompt += f"答案：{example['answer']}\n\n"
        
        prompt += f"现在请回答：\n问题：{query}\n答案："
        
        return prompt
    
    def chain_of_thought_prompt(self, question: str, context: str) -> str:
        """思维链提示"""
        return f"""请一步一步思考并回答问题。

上下文：{context}

问题：{question}

让我们一步步思考：
1. 首先，我需要理解问题在问什么
2. 然后，我需要在上下文中找到相关信息
3. 最后，基于找到的信息形成答案

思考过程："""

class ContextManager:
    """上下文管理"""
    
    def __init__(self, max_context_length: int = 2048):
        self.max_context_length = max_context_length
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def compress_context(self, documents: List[str]) -> str:
        """压缩上下文"""
        # 计算token数
        full_context = "\n".join(documents)
        tokens = self.tokenizer.encode(full_context)
        
        if len(tokens) <= self.max_context_length:
            return full_context
        
        # 压缩策略
        compressed_docs = []
        current_tokens = 0
        
        for doc in documents:
            doc_tokens = self.tokenizer.encode(doc)
            if current_tokens + len(doc_tokens) > self.max_context_length:
                # 截断文档
                remaining = self.max_context_length - current_tokens
                truncated = self.tokenizer.decode(doc_tokens[:remaining])
                compressed_docs.append(truncated + "...")
                break
            else:
                compressed_docs.append(doc)
                current_tokens += len(doc_tokens)
        
        return "\n".join(compressed_docs)
    
    def extract_key_sentences(self, text: str, num_sentences: int = 5) -> str:
        """提取关键句子"""
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        
        summary = summarizer(parser.document, num_sentences)
        
        return " ".join([str(sentence) for sentence in summary])
```

## 6. 高级RAG技术 🎓

```python
class AdvancedRAG:
    """高级RAG技术"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def iterative_retrieval(self, query: str, documents: List[str], 
                           max_iterations: int = 3) -> List[str]:
        """迭代检索"""
        retrieved_docs = []
        current_query = query
        
        for iteration in range(max_iterations):
            # 检索
            new_docs = self._retrieve(current_query, documents, k=3)
            retrieved_docs.extend(new_docs)
            
            # 基于已检索文档生成新查询
            current_query = self._generate_followup_query(query, retrieved_docs)
            
            # 去重
            documents = [d for d in documents if d not in retrieved_docs]
            
            if not documents:
                break
        
        return retrieved_docs
    
    def _generate_followup_query(self, original_query: str, 
                                 retrieved_docs: List[str]) -> str:
        """生成后续查询"""
        # 这里简化处理，实际需要LLM
        # 提取关键词并扩展查询
        keywords = set()
        for doc in retrieved_docs:
            # 简单提取名词
            words = doc.split()
            keywords.update(words[:5])  # 取前5个词
        
        expanded_query = original_query + " " + " ".join(keywords)
        return expanded_query
    
    def graph_based_retrieval(self, query: str, 
                            knowledge_graph: Dict) -> List[str]:
        """基于图的检索"""
        import networkx as nx
        
        # 构建图
        G = nx.Graph()
        for entity, relations in knowledge_graph.items():
            for relation, target in relations:
                G.add_edge(entity, target, relation=relation)
        
        # 找到查询中的实体
        query_entities = self._extract_entities(query)
        
        # 图遍历获取相关节点
        relevant_nodes = set()
        for entity in query_entities:
            if entity in G:
                # 获取邻居节点
                neighbors = list(G.neighbors(entity))
                relevant_nodes.update(neighbors[:5])
                
                # 获取2跳邻居
                for neighbor in neighbors[:3]:
                    second_neighbors = list(G.neighbors(neighbor))
                    relevant_nodes.update(second_neighbors[:2])
        
        # 构建文档
        documents = []
        for node in relevant_nodes:
            # 获取节点相关的文本
            doc = self._get_node_text(node, G)
            documents.append(doc)
        
        return documents
    
    def _extract_entities(self, text: str) -> List[str]:
        """实体提取（简化版）"""
        # 实际应使用NER模型
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    def _get_node_text(self, node: str, graph) -> str:
        """获取节点文本"""
        # 构建节点描述
        edges = graph.edges(node, data=True)
        text = f"{node}: "
        relations = []
        for _, target, data in edges:
            relation = data.get('relation', 'related to')
            relations.append(f"{relation} {target}")
        text += ", ".join(relations)
        return text

# 自适应RAG
class AdaptiveRAG:
    """自适应RAG系统"""
    
    def __init__(self):
        self.query_classifier = self._init_classifier()
        self.strategies = {
            "factual": self._factual_strategy,
            "analytical": self._analytical_strategy,
            "creative": self._creative_strategy
        }
    
    def _init_classifier(self):
        """初始化查询分类器"""
        # 简化版，实际需要训练分类器
        return lambda x: "factual"  # 默认返回factual
    
    def query(self, question: str) -> str:
        """自适应查询"""
        # 分类查询类型
        query_type = self.query_classifier(question)
        
        # 选择策略
        strategy = self.strategies.get(query_type, self._factual_strategy)
        
        # 执行策略
        return strategy(question)
    
    def _factual_strategy(self, question: str) -> str:
        """事实型问题策略"""
        # 密集检索 + 精确匹配
        # 较少的上下文
        # 温度参数低
        return "Factual answer"
    
    def _analytical_strategy(self, question: str) -> str:
        """分析型问题策略"""
        # 多路检索
        # 更多上下文
        # 思维链提示
        return "Analytical answer"
    
    def _creative_strategy(self, question: str) -> str:
        """创造型问题策略"""
        # 少量检索
        # 更高温度
        # 更多生成
        return "Creative answer"
```

## 7. 评估与监控 📊

```python
class RAGEvaluator:
    """RAG系统评估"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval(self, queries: List[str], 
                          ground_truth: List[List[str]], 
                          retrieved: List[List[str]]) -> Dict:
        """评估检索质量"""
        from sklearn.metrics import precision_recall_fscore_support
        
        metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": []  # Mean Reciprocal Rank
        }
        
        for i, (gt, ret) in enumerate(zip(ground_truth, retrieved)):
            # Precision & Recall
            gt_set = set(gt)
            ret_set = set(ret)
            
            if ret_set:
                precision = len(gt_set & ret_set) / len(ret_set)
                metrics["precision"].append(precision)
            
            if gt_set:
                recall = len(gt_set & ret_set) / len(gt_set)
                metrics["recall"].append(recall)
            
            # F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                metrics["f1"].append(f1)
            
            # MRR
            for rank, doc in enumerate(ret, 1):
                if doc in gt_set:
                    metrics["mrr"].append(1 / rank)
                    break
            else:
                metrics["mrr"].append(0)
        
        # 计算平均值
        return {
            metric: np.mean(values) 
            for metric, values in metrics.items()
        }
    
    def evaluate_generation(self, questions: List[str], 
                          generated_answers: List[str], 
                          reference_answers: List[str]) -> Dict:
        """评估生成质量"""
        from rouge import Rouge
        from bert_score import score
        
        rouge = Rouge()
        
        # ROUGE分数
        rouge_scores = rouge.get_scores(generated_answers, reference_answers, avg=True)
        
        # BERT Score
        P, R, F1 = score(generated_answers, reference_answers, lang="en")
        
        # BLEU分数
        from nltk.translate.bleu_score import sentence_bleu
        bleu_scores = []
        for gen, ref in zip(generated_answers, reference_answers):
            score = sentence_bleu([ref.split()], gen.split())
            bleu_scores.append(score)
        
        return {
            "rouge-1": rouge_scores["rouge-1"]["f"],
            "rouge-2": rouge_scores["rouge-2"]["f"],
            "rouge-l": rouge_scores["rouge-l"]["f"],
            "bert_score": F1.mean().item(),
            "bleu": np.mean(bleu_scores)
        }
    
    def faithfulness_score(self, answer: str, context: str) -> float:
        """忠实度评分：答案是否忠于上下文"""
        # 使用NLI模型判断
        from transformers import pipeline
        
        nli = pipeline("text-classification", 
                      model="roberta-large-mnli")
        
        result = nli(f"{context} {answer}")
        
        # 返回entailment分数
        for item in result:
            if item['label'] == 'ENTAILMENT':
                return item['score']
        return 0.0
    
    def relevance_score(self, answer: str, question: str) -> float:
        """相关性评分"""
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        q_emb = encoder.encode([question])[0]
        a_emb = encoder.encode([answer])[0]
        
        similarity = np.dot(q_emb, a_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(a_emb)
        )
        
        return similarity

# 监控系统
class RAGMonitor:
    """RAG系统监控"""
    
    def __init__(self):
        self.metrics_history = []
    
    def log_query(self, query_data: Dict):
        """记录查询"""
        import time
        
        query_data['timestamp'] = time.time()
        self.metrics_history.append(query_data)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.metrics_history:
            return {}
        
        latencies = [m['latency'] for m in self.metrics_history]
        retrieval_counts = [m['num_retrieved'] for m in self.metrics_history]
        
        return {
            "avg_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "avg_docs_retrieved": np.mean(retrieval_counts),
            "total_queries": len(self.metrics_history)
        }
    
    def detect_anomalies(self) -> List[Dict]:
        """检测异常"""
        anomalies = []
        
        for metric in self.metrics_history[-100:]:  # 最近100个查询
            # 高延迟
            if metric['latency'] > 5.0:
                anomalies.append({
                    "type": "high_latency",
                    "query": metric['query'],
                    "latency": metric['latency']
                })
            
            # 无结果
            if metric['num_retrieved'] == 0:
                anomalies.append({
                    "type": "no_results",
                    "query": metric['query']
                })
        
        return anomalies
```

## 8. 生产部署 🚢

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from typing import Optional

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.7

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class ProductionRAG:
    """生产级RAG系统"""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.cache = {}
        self.monitor = RAGMonitor()
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """处理查询"""
        import time
        start_time = time.time()
        
        # 缓存检查
        cache_key = f"{request.question}_{request.top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 检索
            sources = await self._async_retrieve(
                request.question, 
                request.top_k
            )
            
            # 生成
            answer = await self._async_generate(
                request.question,
                sources,
                request.temperature
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(answer, sources)
            
            response = QueryResponse(
                answer=answer,
                sources=sources,
                confidence=confidence
            )
            
            # 缓存结果
            self.cache[cache_key] = response
            
            # 记录指标
            self.monitor.log_query({
                "query": request.question,
                "latency": time.time() - start_time,
                "num_retrieved": len(sources)
            })
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _async_retrieve(self, query: str, k: int) -> List[str]:
        """异步检索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.rag_pipeline.retrieve, 
            query, 
            k
        )
    
    async def _async_generate(self, query: str, 
                             context: List[str], 
                             temperature: float) -> str:
        """异步生成"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rag_pipeline.generate,
            query,
            context
        )
    
    def _calculate_confidence(self, answer: str, sources: List[str]) -> float:
        """计算置信度分数"""
        # 简化版：基于答案长度和源文档数量
        if not answer or not sources:
            return 0.0
        
        length_score = min(len(answer) / 500, 1.0)
        source_score = min(len(sources) / 5, 1.0)
        
        return (length_score + source_score) / 2

# 初始化系统
rag_system = ProductionRAG()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询端点"""
    return await rag_system.process_query(request)

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """获取指标"""
    return rag_system.monitor.get_statistics()

# Docker部署
dockerfile = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Kubernetes部署
k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag
  template:
    metadata:
      labels:
        app: rag
    spec:
      containers:
      - name: rag-container
        image: rag-system:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
spec:
  selector:
    app: rag
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
```

## 最佳实践总结 📋

```python
def rag_best_practices():
    """RAG最佳实践"""
    
    practices = {
        "文档处理": [
            "选择合适的chunk大小（通常200-800 tokens）",
            "保留chunk之间的重叠（10-20%）",
            "保存元数据（来源、日期、章节等）",
            "考虑文档结构（标题、段落、列表）"
        ],
        
        "检索优化": [
            "使用混合检索（密集+稀疏）",
            "实施重排序机制",
            "应用MMR减少冗余",
            "缓存常见查询结果"
        ],
        
        "生成质量": [
            "设计清晰的提示模板",
            "实施答案验证机制",
            "控制生成长度和温度",
            "添加引用来源"
        ],
        
        "系统设计": [
            "异步处理提高吞吐量",
            "实施熔断机制",
            "监控关键指标",
            "定期更新知识库"
        ],
        
        "安全考虑": [
            "输入验证和清洗",
            "防止提示注入",
            "限制API调用频率",
            "敏感信息过滤"
        ]
    }
    
    return practices

# 常见问题解决
troubleshooting = """
问题1：检索结果不相关
- 检查embedding模型是否适合领域
- 调整chunk大小
- 尝试查询扩展
- 使用重排序

问题2：生成幻觉
- 降低temperature
- 增加上下文验证
- 使用更严格的提示
- 实施事实检查

问题3：延迟过高
- 使用缓存
- 异步处理
- 优化向量索引
- 减少检索数量

问题4：成本过高
- 使用更小的模型
- 实施智能缓存
- 批处理请求
- 优化token使用
"""

print("RAG系统实战指南完成！")
```

## 下一步学习
- [模型微调](finetuning.md) - 定制化你的模型
- [NLP模型](nlp_models.md) - 深入理解语言模型
- [LLM部署](llm_deployment.md) - 大模型生产部署