# 模型微调完全指南：从PEFT到QLoRA 🎯

掌握现代模型微调技术，用最少的资源实现最好的效果。

## 1. 微调基础概念 📚

```python
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

class BasicFineTuning:
    """基础微调示例"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def prepare_dataset(self, texts, labels=None):
        """准备数据集"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        # 创建数据集
        dataset = load_dataset("text", data_files={"train": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def full_finetune(self, train_dataset, eval_dataset=None):
        """全量微调"""
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        trainer.train()
        
        return trainer
    
    def selective_finetune(self, layers_to_freeze=None):
        """选择性微调：冻结部分层"""
        if layers_to_freeze is None:
            # 冻结前6层
            layers_to_freeze = list(range(6))
        
        # 冻结指定层
        for i, layer in enumerate(self.model.transformer.h):
            if i in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return self.model

# 学习率调度策略
def get_lr_scheduler(optimizer, num_training_steps, warmup_steps=0):
    """获取学习率调度器"""
    from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    
    # 线性衰减
    linear_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 余弦衰减
    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return cosine_scheduler
```

## 2. LoRA：低秩适应 🎨

```python
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class LoRAFineTuning:
    """LoRA微调实现"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def apply_lora(self, r=8, lora_alpha=32, lora_dropout=0.1, 
                   target_modules=None):
        """应用LoRA配置"""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # LoRA配置
        peft_config = LoraConfig(
            r=r,  # 低秩矩阵的秩
            lora_alpha=lora_alpha,  # LoRA缩放参数
            target_modules=target_modules,  # 要应用LoRA的模块
            lora_dropout=lora_dropout,  # LoRA dropout
            bias="none",  # 偏置处理方式
            task_type=TaskType.CAUSAL_LM,  # 任务类型
        )
        
        # 获取PEFT模型
        self.model = get_peft_model(self.base_model, peft_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
        
        return self.model
    
    def train(self, dataset, output_dir="./lora_model"):
        """训练LoRA模型"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        
        # 保存LoRA权重
        self.model.save_pretrained(output_dir)
        
    def merge_and_save(self, output_dir="./merged_model"):
        """合并LoRA权重到基础模型"""
        # 合并权重
        merged_model = self.model.merge_and_unload()
        
        # 保存合并后的模型
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return merged_model
    
    def load_lora_model(self, lora_path, base_model_name):
        """加载LoRA模型"""
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        return model

# 多LoRA合并
class MultiLoRAMerger:
    """多个LoRA模型合并"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        
    def weighted_merge(self, lora_models, weights):
        """加权合并多个LoRA"""
        assert len(lora_models) == len(weights)
        assert abs(sum(weights) - 1.0) < 1e-6
        
        # 初始化合并参数
        merged_params = {}
        
        for lora_model, weight in zip(lora_models, weights):
            for name, param in lora_model.named_parameters():
                if "lora" in name:
                    if name not in merged_params:
                        merged_params[name] = weight * param.data
                    else:
                        merged_params[name] += weight * param.data
        
        # 应用合并的参数
        for name, param in merged_params.items():
            self.base_model.state_dict()[name].copy_(param)
        
        return self.base_model
    
    def task_arithmetic_merge(self, lora_models, lambda_param=0.5):
        """任务算术合并：用于多任务"""
        # 获取任务向量
        task_vectors = []
        
        for lora_model in lora_models:
            task_vector = {}
            for name, param in lora_model.named_parameters():
                if "lora" in name:
                    # 计算与基础模型的差异
                    base_param = self.base_model.state_dict()[name.replace("lora_", "")]
                    task_vector[name] = param.data - base_param
            task_vectors.append(task_vector)
        
        # 合并任务向量
        merged_vector = {}
        for name in task_vectors[0].keys():
            merged_vector[name] = sum(tv[name] for tv in task_vectors) / len(task_vectors)
        
        # 应用到基础模型
        for name, delta in merged_vector.items():
            base_name = name.replace("lora_", "")
            self.base_model.state_dict()[base_name] += lambda_param * delta
        
        return self.base_model
```

## 3. QLoRA：量化LoRA 📦

```python
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

class QLoRAFineTuning:
    """QLoRA：4bit量化 + LoRA"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        
    def load_quantized_model(self):
        """加载4bit量化模型"""
        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # 双重量化
            bnb_4bit_quant_type="nf4",  # NormalFloat4量化
            bnb_4bit_compute_dtype=torch.bfloat16  # 计算数据类型
        )
        
        # 加载量化模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 准备模型用于k-bit训练
        model = self.prepare_model_for_kbit_training(model)
        
        return model
    
    def prepare_model_for_kbit_training(self, model):
        """准备模型进行k-bit训练"""
        model.gradient_checkpointing_enable()
        
        # 将某些层转为fp32以提高稳定性
        for param in model.parameters():
            param.requires_grad = False  # 冻结所有参数
            
        # 启用输入梯度
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        return model
    
    def apply_qlora(self, model, r=4, lora_alpha=16):
        """应用QLoRA配置"""
        # 找到所有Linear层
        target_modules = self.find_all_linear_names(model)
        
        # LoRA配置
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA
        model = get_peft_model(model, peft_config)
        
        return model
    
    def find_all_linear_names(self, model):
        """找到所有Linear层的名称"""
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[-1])
        
        # 移除一些不应该适配的层
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        
        return list(lora_module_names)
    
    def train_qlora(self, model, dataset, output_dir="./qlora_model"):
        """训练QLoRA模型"""
        from transformers import TrainingArguments
        from trl import SFTTrainer
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",  # 分页优化器
            learning_rate=2e-4,
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            warmup_steps=5,
            group_by_length=True,
            lr_scheduler_type="constant",
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=512,
            packing=False,
        )
        
        trainer.train()
        
        return trainer

# 内存优化技术
class MemoryEfficientTraining:
    """内存高效训练技术"""
    
    @staticmethod
    def gradient_checkpointing(model):
        """梯度检查点：用计算换内存"""
        model.gradient_checkpointing_enable()
        return model
    
    @staticmethod
    def mixed_precision_training():
        """混合精度训练"""
        from torch.cuda.amp import autocast, GradScaler
        
        scaler = GradScaler()
        
        def train_step(model, inputs, optimizer):
            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            return loss
        
        return train_step
    
    @staticmethod
    def gradient_accumulation(batch_size=32, micro_batch_size=4):
        """梯度累积"""
        accumulation_steps = batch_size // micro_batch_size
        
        def train_with_accumulation(model, dataloader, optimizer):
            model.train()
            optimizer.zero_grad()
            
            for i, batch in enumerate(dataloader):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        
        return train_with_accumulation
```

## 4. Adapter微调 🔌

```python
class AdapterFineTuning:
    """Adapter微调实现"""
    
    def __init__(self, model, adapter_size=64):
        self.model = model
        self.adapter_size = adapter_size
        self.adapters = {}
        
    def add_adapter_layer(self, name, input_size, adapter_size=None):
        """添加Adapter层"""
        if adapter_size is None:
            adapter_size = self.adapter_size
        
        adapter = nn.Sequential(
            nn.Linear(input_size, adapter_size),
            nn.ReLU(),
            nn.Linear(adapter_size, input_size)
        )
        
        # 初始化为恒等映射
        nn.init.zeros_(adapter[0].weight)
        nn.init.zeros_(adapter[0].bias)
        nn.init.zeros_(adapter[2].weight)
        nn.init.zeros_(adapter[2].bias)
        
        self.adapters[name] = adapter
        return adapter
    
    def forward_with_adapter(self, x, layer, adapter):
        """带Adapter的前向传播"""
        # 原始层输出
        output = layer(x)
        
        # Adapter路径
        adapter_output = adapter(x)
        
        # 残差连接
        return output + adapter_output
    
    def freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只训练adapter
        for adapter in self.adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True

# Prefix Tuning
class PrefixTuning:
    """前缀微调"""
    
    def __init__(self, model, prefix_length=10, hidden_size=768):
        self.model = model
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        
        # 前缀嵌入
        self.prefix_embeddings = nn.Embedding(prefix_length, hidden_size)
        
        # 前缀投影
        self.prefix_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size * model.config.num_hidden_layers * 2)
        )
        
    def get_prefix(self, batch_size):
        """获取前缀"""
        prefix_tokens = torch.arange(self.prefix_length).unsqueeze(0).expand(batch_size, -1)
        prefix_emb = self.prefix_embeddings(prefix_tokens)
        
        # 投影到所有层
        past_key_values = self.prefix_projection(prefix_emb)
        
        # 重塑为正确的形状
        past_key_values = past_key_values.view(
            batch_size, 
            self.prefix_length, 
            self.model.config.num_hidden_layers * 2,
            self.model.config.num_attention_heads,
            self.hidden_size // self.model.config.num_attention_heads
        )
        
        # 分割为key和value
        past_key_values = torch.split(past_key_values, 2, dim=2)
        
        return past_key_values
    
    def forward(self, input_ids, **kwargs):
        """前向传播"""
        batch_size = input_ids.shape[0]
        
        # 获取前缀
        past_key_values = self.get_prefix(batch_size)
        
        # 调用模型
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            **kwargs
        )
        
        return outputs

# Prompt Tuning
class PromptTuning:
    """提示微调"""
    
    def __init__(self, model, num_virtual_tokens=20):
        self.model = model
        self.num_virtual_tokens = num_virtual_tokens
        
        # 软提示嵌入
        self.soft_prompt = nn.Parameter(
            torch.randn(num_virtual_tokens, model.config.hidden_size)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """添加软提示的前向传播"""
        batch_size = input_ids.shape[0]
        
        # 获取输入嵌入
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 扩展软提示到批次大小
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 连接软提示和输入嵌入
        inputs_embeds = torch.cat([soft_prompt_expanded, inputs_embeds], dim=1)
        
        # 更新attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 前向传播
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        return outputs
```

## 5. 指令微调 📖

```python
from datasets import Dataset
import json

class InstructionTuning:
    """指令微调"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 设置pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def format_instruction(self, instruction, input_text="", output=""):
        """格式化指令"""
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""
        
        return prompt
    
    def prepare_dataset(self, data_path):
        """准备指令数据集"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            formatted_text = self.format_instruction(
                item.get("instruction", ""),
                item.get("input", ""),
                item.get("output", "")
            )
            formatted_data.append({"text": formatted_text})
        
        dataset = Dataset.from_list(formatted_data)
        
        # 分词
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train_with_sft(self, dataset, output_dir="./sft_model"):
        """使用SFT训练器进行指令微调"""
        from trl import SFTTrainer
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
        )
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=512,
            dataset_text_field="text",
        )
        
        trainer.train()
        
        return trainer

# 多任务指令微调
class MultiTaskInstructionTuning:
    """多任务指令微调"""
    
    def __init__(self, model):
        self.model = model
        self.task_prompts = {
            "translation": "Translate the following text from {src_lang} to {tgt_lang}:",
            "summarization": "Summarize the following text:",
            "qa": "Answer the following question based on the context:",
            "classification": "Classify the following text into one of these categories: {categories}",
        }
    
    def create_multitask_dataset(self, datasets_dict):
        """创建多任务数据集"""
        all_examples = []
        
        for task_name, dataset in datasets_dict.items():
            task_prompt = self.task_prompts.get(task_name, "")
            
            for example in dataset:
                formatted_example = {
                    "task": task_name,
                    "instruction": task_prompt.format(**example.get("metadata", {})),
                    "input": example.get("input", ""),
                    "output": example.get("output", "")
                }
                all_examples.append(formatted_example)
        
        # 打乱数据
        import random
        random.shuffle(all_examples)
        
        return Dataset.from_list(all_examples)
    
    def balanced_sampling(self, datasets_dict, samples_per_task=1000):
        """平衡采样多任务数据"""
        balanced_data = []
        
        for task_name, dataset in datasets_dict.items():
            # 采样固定数量
            task_samples = dataset.shuffle().select(range(min(samples_per_task, len(dataset))))
            balanced_data.extend(task_samples)
        
        return balanced_data
```

## 6. RLHF与DPO 🎮

```python
from trl import PPOTrainer, PPOConfig, DPOTrainer

class RLHFTraining:
    """基于人类反馈的强化学习"""
    
    def __init__(self, model, reward_model):
        self.model = model
        self.reward_model = reward_model
        
    def setup_ppo(self):
        """设置PPO训练"""
        ppo_config = PPOConfig(
            model_name="gpt2",
            learning_rate=1.41e-5,
            log_with="wandb",
            batch_size=128,
            mini_batch_size=4,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
        )
        
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
        )
        
        return ppo_trainer
    
    def train_with_ppo(self, ppo_trainer, prompts):
        """PPO训练循环"""
        for epoch in range(10):
            for batch in prompts:
                # 生成响应
                query_tensors = batch["input_ids"]
                response_tensors = ppo_trainer.generate(query_tensors)
                
                # 计算奖励
                rewards = self.compute_rewards(query_tensors, response_tensors)
                
                # PPO步骤
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # 记录统计
                ppo_trainer.log_stats(stats, batch, rewards)
    
    def compute_rewards(self, queries, responses):
        """计算奖励"""
        # 使用奖励模型
        rewards = []
        for query, response in zip(queries, responses):
            reward = self.reward_model(query, response)
            rewards.append(reward)
        
        return torch.tensor(rewards)

class DPOTraining:
    """直接偏好优化"""
    
    def __init__(self, model_name="gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def prepare_preference_dataset(self, data):
        """准备偏好数据集"""
        # 数据格式: {"prompt": str, "chosen": str, "rejected": str}
        formatted_data = []
        
        for item in data:
            formatted_item = {
                "prompt": item["prompt"],
                "chosen": item["prompt"] + item["chosen"],
                "rejected": item["prompt"] + item["rejected"]
            }
            formatted_data.append(formatted_item)
        
        return Dataset.from_list(formatted_data)
    
    def train_dpo(self, dataset, output_dir="./dpo_model"):
        """DPO训练"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
        )
        
        dpo_trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            beta=0.1,  # DPO温度参数
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        dpo_trainer.train()
        
        return dpo_trainer
```

## 7. 高效微调技巧 💡

```python
class EfficientFineTuningTricks:
    """高效微调技巧集合"""
    
    @staticmethod
    def layer_wise_lr_decay(model, base_lr=1e-4, decay_rate=0.9):
        """层级学习率衰减"""
        parameters = []
        
        # 从后向前遍历层
        num_layers = len(list(model.children()))
        for i, (name, param) in enumerate(model.named_parameters()):
            layer_id = num_layers - 1 - i // 2  # 简化的层ID计算
            lr = base_lr * (decay_rate ** layer_id)
            
            parameters.append({
                "params": param,
                "lr": lr
            })
        
        return parameters
    
    @staticmethod
    def smart_initialization(model, init_method="xavier"):
        """智能初始化新增层"""
        for name, param in model.named_parameters():
            if "lora" in name or "adapter" in name:
                if init_method == "xavier":
                    nn.init.xavier_uniform_(param)
                elif init_method == "kaiming":
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif init_method == "normal":
                    nn.init.normal_(param, mean=0.0, std=0.02)
    
    @staticmethod
    def progressive_unfreezing(model, current_epoch, total_epochs):
        """渐进式解冻"""
        num_layers = len(list(model.children()))
        layers_to_unfreeze = int(num_layers * (current_epoch / total_epochs))
        
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻后面的层
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= num_layers - layers_to_unfreeze:
                param.requires_grad = True
    
    @staticmethod
    def mixout_regularization(model, mixout_prob=0.1):
        """Mixout正则化"""
        class MixoutWrapper(nn.Module):
            def __init__(self, module, p):
                super().__init__()
                self.module = module
                self.p = p
                self.initial_weights = module.weight.data.clone()
            
            def forward(self, x):
                if self.training:
                    # 随机混合初始权重和当前权重
                    mask = torch.bernoulli(torch.ones_like(self.module.weight) * (1 - self.p))
                    mixed_weight = mask * self.module.weight + (1 - mask) * self.initial_weights
                    return F.linear(x, mixed_weight, self.module.bias)
                else:
                    return self.module(x)
        
        # 包装所有Linear层
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                wrapped = MixoutWrapper(module, mixout_prob)
                setattr(model, name, wrapped)
        
        return model

# 数据高效微调
class DataEfficientFineTuning:
    """数据高效的微调方法"""
    
    @staticmethod
    def few_shot_learning(model, support_set, query_set, k_shot=5):
        """少样本学习"""
        # 构建提示
        prompt = "Here are some examples:\n\n"
        
        for i in range(k_shot):
            example = support_set[i]
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        
        prompt += f"Now, given the input: {query_set['input']}\nOutput:"
        
        # 生成输出
        output = model.generate(prompt)
        
        return output
    
    @staticmethod
    def active_learning_selection(model, unlabeled_data, n_samples=100):
        """主动学习样本选择"""
        uncertainties = []
        
        for data in unlabeled_data:
            # 计算模型不确定性
            with torch.no_grad():
                outputs = model(data['input'])
                # 使用熵作为不确定性度量
                probs = torch.softmax(outputs.logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                uncertainties.append(entropy.mean().item())
        
        # 选择不确定性最高的样本
        selected_indices = np.argsort(uncertainties)[-n_samples:]
        
        return [unlabeled_data[i] for i in selected_indices]
    
    @staticmethod
    def curriculum_learning(dataset, difficulty_scores):
        """课程学习：从易到难"""
        # 按难度排序
        sorted_indices = np.argsort(difficulty_scores)
        
        # 分阶段
        easy_samples = sorted_indices[:len(sorted_indices)//3]
        medium_samples = sorted_indices[len(sorted_indices)//3:2*len(sorted_indices)//3]
        hard_samples = sorted_indices[2*len(sorted_indices)//3:]
        
        # 构建课程
        curriculum = {
            "phase1": [dataset[i] for i in easy_samples],
            "phase2": [dataset[i] for i in medium_samples],
            "phase3": [dataset[i] for i in hard_samples]
        }
        
        return curriculum
```

## 8. 评估与部署 📈

```python
class FineTuningEvaluation:
    """微调模型评估"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_perplexity(self, test_dataset):
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataset:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                total_tokens += batch["input_ids"].numel()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def evaluate_generation_quality(self, prompts, max_length=100):
        """评估生成质量"""
        from nltk.translate.bleu_score import sentence_bleu
        from rouge import Rouge
        
        rouge = Rouge()
        bleu_scores = []
        rouge_scores = []
        
        for prompt in prompts:
            # 生成
            generated = self.model.generate(
                prompt["input"],
                max_length=max_length,
                temperature=0.7,
                do_sample=True
            )
            
            # 计算BLEU
            reference = prompt["target"].split()
            hypothesis = self.tokenizer.decode(generated[0]).split()
            bleu = sentence_bleu([reference], hypothesis)
            bleu_scores.append(bleu)
            
            # 计算ROUGE
            rouge_score = rouge.get_scores(
                self.tokenizer.decode(generated[0]),
                prompt["target"]
            )[0]
            rouge_scores.append(rouge_score)
        
        return {
            "bleu": np.mean(bleu_scores),
            "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
            "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores])
        }
    
    def a_b_testing(self, model_a, model_b, test_prompts, num_evaluators=5):
        """A/B测试"""
        preferences = {"model_a": 0, "model_b": 0, "tie": 0}
        
        for prompt in test_prompts:
            # 两个模型生成
            output_a = model_a.generate(prompt)
            output_b = model_b.generate(prompt)
            
            # 人工评估（这里简化为自动评估）
            score_a = len(output_a)  # 简化：用长度代替质量
            score_b = len(output_b)
            
            if score_a > score_b:
                preferences["model_a"] += 1
            elif score_b > score_a:
                preferences["model_b"] += 1
            else:
                preferences["tie"] += 1
        
        return preferences

# 部署优化
class DeploymentOptimization:
    """部署优化"""
    
    @staticmethod
    def merge_lora_weights(base_model, lora_model):
        """合并LoRA权重用于部署"""
        merged_model = lora_model.merge_and_unload()
        return merged_model
    
    @staticmethod
    def quantize_for_deployment(model):
        """部署前量化"""
        from transformers import AutoModelForCausalLM
        import torch.quantization as quantization
        
        # 动态量化
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def optimize_for_inference(model):
        """推理优化"""
        model.eval()
        
        # 禁用dropout
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
        
        # 融合操作
        if hasattr(torch.quantization, 'fuse_modules'):
            torch.quantization.fuse_modules(
                model,
                [['conv', 'bn', 'relu']],
                inplace=True
            )
        
        return model
```

## 最佳实践总结 📝

```python
def finetuning_best_practices():
    """微调最佳实践"""
    
    practices = {
        "选择微调方法": {
            "全量微调": "小模型(<1B)，充足计算资源",
            "LoRA": "大模型(>1B)，中等资源",
            "QLoRA": "超大模型(>7B)，有限资源",
            "Prefix/Prompt": "极少参数调整需求"
        },
        
        "超参数建议": {
            "学习率": "比预训练小10-100倍",
            "批次大小": "尽可能大（使用梯度累积）",
            "训练轮数": "3-5轮（避免过拟合）",
            "Warmup": "总步数的5-10%"
        },
        
        "数据准备": {
            "质量>数量": "高质量的少量数据胜过大量低质量数据",
            "格式统一": "保持指令格式一致",
            "多样性": "覆盖目标任务的各种情况",
            "清洗": "去除重复和低质量样本"
        },
        
        "训练技巧": {
            "早停": "监控验证集性能",
            "检查点": "定期保存，支持恢复",
            "混合精度": "FP16/BF16加速训练",
            "梯度裁剪": "防止梯度爆炸"
        },
        
        "评估策略": {
            "多维度": "自动指标+人工评估",
            "领域相关": "使用领域特定的评估集",
            "对比基线": "与原始模型和其他方法对比",
            "在线测试": "小流量A/B测试"
        }
    }
    
    return practices

# 常见问题解决
troubleshooting = """
问题1：显存不足
- 使用QLoRA或更小的r值
- 减小批次大小
- 启用梯度检查点
- 使用梯度累积

问题2：训练不稳定
- 降低学习率
- 增加warmup步数
- 使用梯度裁剪
- 检查数据质量

问题3：过拟合
- 增加dropout
- 减少训练轮数
- 使用更多数据
- 增加正则化

问题4：效果不佳
- 检查数据标注质量
- 调整LoRA的r值
- 尝试不同的目标模块
- 增加训练数据
"""

print("微调指南完成！")
```

## 下一步学习
- [RAG系统](rag_systems.md) - 检索增强生成
- [LLM部署](llm_deployment.md) - 大模型部署
- [NLP模型](nlp_models.md) - NLP架构详解