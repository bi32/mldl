# 大语言模型(LLM)部署实战 🤖

从Transformers到vLLM、Ollama，掌握大模型高效部署的所有技术栈。

## 1. Transformers库基础 📚

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)
import torch

# 基础使用
def basic_transformers_usage():
    """Transformers库基础使用"""
    
    # 1. 使用pipeline（最简单）
    generator = pipeline('text-generation', model='gpt2')
    result = generator("Hello, I'm a language model", 
                      max_length=50, 
                      num_return_sequences=2)
    print(result)
    
    # 2. 手动加载模型和分词器
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 对话示例
    def chat(input_text, chat_history_ids=None):
        # 编码输入
        new_input_ids = tokenizer.encode(
            input_text + tokenizer.eos_token, 
            return_tensors='pt'
        )
        
        # 拼接历史
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids
        
        # 生成回复
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
        
        # 解码输出
        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        
        return response, chat_history_ids
    
    # 测试对话
    response, history = chat("How are you?")
    print(f"Bot: {response}")

# 加载大模型（量化）
def load_large_model_quantized():
    """加载量化的大模型"""
    
    # 4bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf"
    )
    
    return model, tokenizer

# 流式生成
class StreamingLLM:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def generate_stream(self, prompt, max_new_tokens=100):
        """流式生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                
                # 生成下一个token
                next_token = self.tokenizer.decode(next_token_id[0])
                yield next_token
                
                # 更新输入
                inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_id], dim=1)
                
                # 检查是否结束
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

# 使用示例
streamer = StreamingLLM()
for token in streamer.generate_stream("Once upon a time"):
    print(token, end='', flush=True)
```

## 2. vLLM - 高性能推理引擎 ⚡

```python
# 安装: pip install vllm

from vllm import LLM, SamplingParams

class vLLMDeployment:
    """vLLM部署类"""
    
    def __init__(self, model_name="facebook/opt-125m", gpu_memory_utilization=0.9):
        """
        初始化vLLM
        gpu_memory_utilization: GPU内存使用率
        """
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,  # 张量并行
            dtype="half",  # 使用FP16
            max_model_len=2048
        )
    
    def generate(self, prompts, **kwargs):
        """批量生成"""
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.8),
            top_p=kwargs.get('top_p', 0.95),
            max_tokens=kwargs.get('max_tokens', 100),
            presence_penalty=kwargs.get('presence_penalty', 0.0),
            frequency_penalty=kwargs.get('frequency_penalty', 0.0)
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                'prompt': output.prompt,
                'generated_text': output.outputs[0].text,
                'finish_reason': output.outputs[0].finish_reason
            })
        
        return results
    
    def benchmark(self, num_prompts=100, prompt_len=128, output_len=128):
        """性能测试"""
        import time
        
        # 生成测试prompts
        prompts = ["Write a story about a robot"] * num_prompts
        
        start_time = time.time()
        outputs = self.generate(prompts, max_tokens=output_len)
        total_time = time.time() - start_time
        
        # 计算吞吐量
        total_tokens = sum(len(o['generated_text'].split()) for o in outputs)
        throughput = total_tokens / total_time
        
        print(f"处理 {num_prompts} 个请求")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"吞吐量: {throughput:.2f} tokens/秒")
        print(f"平均延迟: {total_time/num_prompts:.3f} 秒/请求")
        
        return throughput

# vLLM服务器部署
def start_vllm_server():
    """启动vLLM API服务器"""
    import subprocess
    
    # 启动命令
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "facebook/opt-125m",
        "--port", "8000",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.9"
    ]
    
    subprocess.run(cmd)

# 客户端调用
def vllm_client_example():
    """vLLM客户端示例"""
    import openai
    
    # 设置API端点
    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "dummy"
    
    # 调用
    response = openai.Completion.create(
        model="facebook/opt-125m",
        prompt="Hello, my name is",
        max_tokens=100,
        temperature=0.8
    )
    
    print(response.choices[0].text)
```

## 3. Ollama - 本地LLM运行 🦙

```python
# 安装: curl https://ollama.ai/install.sh | sh

import requests
import json

class OllamaClient:
    """Ollama客户端"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def pull_model(self, model_name="llama2"):
        """拉取模型"""
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name}
        )
        return response.json()
    
    def generate(self, model, prompt, stream=False):
        """生成文本"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=stream
        )
        
        if stream:
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            return response.json()
    
    def chat(self, model, messages):
        """对话接口"""
        payload = {
            "model": model,
            "messages": messages
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        
        return response.json()
    
    def list_models(self):
        """列出已安装的模型"""
        response = requests.get(f"{self.base_url}/api/tags")
        return response.json()

# 使用示例
ollama = OllamaClient()

# 拉取模型
# ollama.pull_model("llama2:7b")

# 生成文本
result = ollama.generate(
    model="llama2",
    prompt="What is machine learning?"
)
print(result['response'])

# 流式生成
for chunk in ollama.generate("llama2", "Tell me a joke", stream=True):
    print(chunk['response'], end='', flush=True)

# 对话
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like?"}
]
response = ollama.chat("llama2", messages)
print(response['message']['content'])

# Ollama模型文件
def create_modelfile():
    """创建自定义Ollama模型"""
    modelfile = """
    FROM llama2
    
    # 设置参数
    PARAMETER temperature 0.8
    PARAMETER top_p 0.9
    
    # 设置系统提示
    SYSTEM You are a helpful AI assistant specialized in coding.
    
    # 设置模板
    TEMPLATE \"\"\"
    {{ .System }}
    User: {{ .Prompt }}
    Assistant: \"\"\"
    """
    
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    
    # 创建模型
    # ollama create mymodel -f Modelfile
```

## 4. 优化技术 🚀

```python
# Flash Attention
def setup_flash_attention():
    """配置Flash Attention加速"""
    from flash_attn import flash_attn_func
    
    class FlashAttentionLLM(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.flash_attn = flash_attn_func
            # ... 其他初始化
        
        def forward(self, hidden_states):
            # 使用Flash Attention
            attn_output = self.flash_attn(
                hidden_states,
                hidden_states,
                hidden_states,
                causal=True
            )
            return attn_output

# PagedAttention (vLLM核心技术)
class PagedAttention:
    """分页注意力机制"""
    def __init__(self, block_size=16, num_blocks=256):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.kv_cache = self._init_cache()
    
    def _init_cache(self):
        """初始化KV缓存"""
        return {
            'keys': torch.zeros(self.num_blocks, self.block_size, 768),
            'values': torch.zeros(self.num_blocks, self.block_size, 768),
            'block_table': {}  # 虚拟到物理块的映射
        }
    
    def allocate_blocks(self, seq_id, num_tokens):
        """为序列分配内存块"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        # 分配逻辑...

# Continuous Batching
class ContinuousBatchingServer:
    """连续批处理服务器"""
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.request_queue = []
        self.active_sequences = {}
    
    def add_request(self, prompt, request_id):
        """添加请求到队列"""
        self.request_queue.append({
            'id': request_id,
            'prompt': prompt,
            'generated_tokens': [],
            'finished': False
        })
    
    def process_batch(self):
        """处理一批请求"""
        # 构建批次
        batch = self._build_batch()
        
        # 模型推理
        outputs = self.model.generate(batch)
        
        # 更新序列状态
        self._update_sequences(outputs)
        
        # 返回完成的序列
        return self._get_finished_sequences()
    
    def _build_batch(self):
        """构建批次（填充到相同长度）"""
        # 实现细节...
        pass
```

## 5. 量化和压缩 📦

```python
# GPTQ量化
def gptq_quantization(model_name):
    """GPTQ量化"""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False
    )
    
    # 加载和量化
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config
    )
    
    # 保存量化模型
    model.save_quantized("model-gptq-4bit")
    
    return model

# AWQ量化
def awq_quantization(model_name):
    """AWQ量化"""
    from awq import AutoAWQForCausalLM
    
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    
    # 量化配置
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
    
    # 量化
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_data
    )
    
    # 保存
    model.save_quantized("model-awq-4bit")
    
    return model

# SmoothQuant
def smooth_quant(model):
    """SmoothQuant量化"""
    from smoothquant import smooth_quantize
    
    # 平滑量化
    smooth_model = smooth_quantize(
        model,
        alpha=0.5,  # 平滑因子
        quant_mode="int8"
    )
    
    return smooth_model
```

## 6. 分布式部署 🌐

```python
# Ray Serve部署
import ray
from ray import serve

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1}
)
class LLMDeployment:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    async def __call__(self, request):
        prompt = request.query_params["prompt"]
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100)
        
        response = self.tokenizer.decode(outputs[0])
        return {"generated_text": response}

# 启动Ray Serve
ray.init()
serve.start()
LLMDeployment.deploy()

# Kubernetes部署
k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-server
  template:
    metadata:
      labels:
        app: llm-server
    spec:
      containers:
      - name: llm-container
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-hf"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
```

## 7. 监控和优化 📊

```python
class LLMMonitor:
    """LLM监控系统"""
    
    def __init__(self):
        self.metrics = {
            'latency': [],
            'throughput': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
    
    def log_request(self, request_id, prompt_len, output_len, latency):
        """记录请求指标"""
        self.metrics['latency'].append(latency)
        throughput = (prompt_len + output_len) / latency
        self.metrics['throughput'].append(throughput)
    
    def get_gpu_metrics(self):
        """获取GPU指标"""
        import pynvml
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # GPU利用率
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # 内存使用
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            'gpu_util': util.gpu,
            'memory_used': mem_info.used / 1024**3,  # GB
            'memory_total': mem_info.total / 1024**3
        }
    
    def optimize_batch_size(self, model, test_prompts):
        """自动优化批大小"""
        best_throughput = 0
        best_batch_size = 1
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            try:
                throughput = self._test_batch_size(model, test_prompts, batch_size)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
            except RuntimeError:  # OOM
                break
        
        return best_batch_size

# Prometheus指标导出
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter('llm_requests_total', 'Total LLM requests')
request_latency = Histogram('llm_request_latency_seconds', 'LLM request latency')
gpu_memory = Gauge('llm_gpu_memory_bytes', 'GPU memory usage')

@request_latency.time()
def serve_request(prompt):
    """处理请求并记录指标"""
    request_count.inc()
    # 处理逻辑...
```

## 最佳实践总结 📝

```python
def deployment_recommendations():
    """部署建议"""
    
    recommendations = {
        "小模型(<7B)": {
            "框架": "Transformers + FastAPI",
            "量化": "INT8 动态量化",
            "部署": "单GPU或CPU"
        },
        "中模型(7B-13B)": {
            "框架": "vLLM或TGI",
            "量化": "GPTQ 4-bit",
            "部署": "单GPU (24GB+)"
        },
        "大模型(30B+)": {
            "框架": "vLLM + Ray",
            "量化": "AWQ 4-bit",
            "部署": "多GPU并行"
        },
        "生产环境": {
            "服务": "Triton Inference Server",
            "编排": "Kubernetes",
            "监控": "Prometheus + Grafana"
        }
    }
    
    return recommendations

# 性能优化清单
optimization_checklist = """
✅ 使用适当的量化方法（GPTQ/AWQ/SmoothQuant）
✅ 启用Flash Attention或类似优化
✅ 使用连续批处理（Continuous Batching）
✅ 实现KV缓存复用
✅ 优化采样参数（温度、top_p等）
✅ 使用模型并行或张量并行（大模型）
✅ 监控GPU内存和利用率
✅ 实现请求队列和负载均衡
✅ 使用CDN缓存常见响应
✅ 实现模型A/B测试
"""

print(optimization_checklist)
```

## 下一步学习
- [NLP模型](nlp_models.md) - BERT、GPT、T5详解
- [模型微调](finetuning.md) - LoRA、QLoRA等高效微调
- [RAG系统](rag_systems.md) - 检索增强生成