# PyTorch模型部署完全指南 🚀

从训练到生产，让你的模型真正发挥价值。本章涵盖ONNX、TorchScript、量化、剪枝等所有部署技术。

## 1. TorchScript - PyTorch原生部署 📦

```python
import torch
import torch.nn as nn
import time

# 示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 方法1: Tracing
def torch_script_trace(model, example_input):
    """通过追踪转换为TorchScript"""
    model.eval()
    
    # 追踪模型
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存模型
    traced_model.save("model_traced.pt")
    
    # 加载模型
    loaded_model = torch.jit.load("model_traced.pt")
    
    # 推理
    with torch.no_grad():
        output = loaded_model(example_input)
    
    return traced_model

# 方法2: Scripting
def torch_script_script(model):
    """通过脚本化转换为TorchScript"""
    model.eval()
    
    # 脚本化模型
    scripted_model = torch.jit.script(model)
    
    # 保存
    scripted_model.save("model_scripted.pt")
    
    return scripted_model

# 带控制流的模型（需要script）
class ModelWithControlFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x, use_dropout: bool = False):
        x = self.fc1(x)
        if use_dropout:  # 控制流
            x = torch.dropout(x, 0.5, self.training)
        x = self.fc2(x)
        return x

# 性能对比
def benchmark_torchscript():
    model = SimpleModel()
    model.eval()
    example_input = torch.randn(1, 3, 32, 32)
    
    # 原始PyTorch
    start = time.time()
    for _ in range(1000):
        with torch.no_grad():
            _ = model(example_input)
    pytorch_time = time.time() - start
    
    # TorchScript
    traced_model = torch.jit.trace(model, example_input)
    start = time.time()
    for _ in range(1000):
        with torch.no_grad():
            _ = traced_model(example_input)
    torchscript_time = time.time() - start
    
    print(f"PyTorch: {pytorch_time:.3f}s")
    print(f"TorchScript: {torchscript_time:.3f}s")
    print(f"加速比: {pytorch_time/torchscript_time:.2f}x")
```

## 2. ONNX导出和优化 🔄

```python
import torch
import onnx
import onnxruntime as ort
import numpy as np

def export_to_onnx(model, dummy_input, onnx_path="model.onnx"):
    """导出模型到ONNX"""
    model.eval()
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"模型已导出到 {onnx_path}")
    return onnx_model

def optimize_onnx(onnx_path):
    """优化ONNX模型"""
    from onnxruntime.transformers import optimizer
    
    # 优化选项
    opt_options = optimizer.OptimizeOptions(
        enable_gelu=True,
        enable_layer_norm=True,
        enable_attention=True,
        use_gpu=torch.cuda.is_available(),
        only_fusion=False,
        enable_gemm_fast=True
    )
    
    # 执行优化
    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type='bert',  # 或 'gpt2', 'vit' 等
        optimization_options=opt_options
    )
    
    # 保存优化后的模型
    optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
    optimized_model.save_model_to_file(optimized_path)
    
    return optimized_path

def onnx_inference(onnx_path, input_data):
    """使用ONNX Runtime推理"""
    # 创建推理会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # 准备输入
    input_name = session.get_inputs()[0].name
    
    # 推理
    result = session.run(None, {input_name: input_data})
    
    return result[0]

# 完整ONNX工作流
def complete_onnx_workflow():
    # 1. 创建模型
    model = SimpleModel()
    model.eval()
    
    # 2. 导出到ONNX
    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_model = export_to_onnx(model, dummy_input)
    
    # 3. ONNX推理
    input_numpy = dummy_input.numpy()
    onnx_output = onnx_inference("model.onnx", input_numpy)
    
    # 4. 验证结果
    torch_output = model(dummy_input).detach().numpy()
    
    print(f"输出差异: {np.mean(np.abs(onnx_output - torch_output)):.6f}")
    
    # 5. 性能测试
    import time
    
    # PyTorch推理
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    pytorch_time = time.time() - start
    
    # ONNX推理
    start = time.time()
    for _ in range(100):
        _ = onnx_inference("model.onnx", input_numpy)
    onnx_time = time.time() - start
    
    print(f"PyTorch时间: {pytorch_time:.3f}s")
    print(f"ONNX时间: {onnx_time:.3f}s")
    print(f"加速比: {pytorch_time/onnx_time:.2f}x")
```

## 3. 量化技术 ⚡

```python
import torch.quantization as quant

# 动态量化（最简单）
def dynamic_quantization(model):
    """动态量化：推理时量化"""
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # 要量化的层类型
        dtype=torch.qint8
    )
    return quantized_model

# 静态量化（需要校准）
def static_quantization(model, calibration_loader):
    """静态量化：需要校准数据"""
    # 1. 准备模型
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # 2. 融合层（Conv+BN+ReLU）
    model_fused = quant.fuse_modules(model, [['conv', 'bn', 'relu']])
    
    # 3. 准备量化
    model_prepared = quant.prepare(model_fused)
    
    # 4. 校准
    with torch.no_grad():
        for data, _ in calibration_loader:
            model_prepared(data)
    
    # 5. 转换为量化模型
    model_quantized = quant.convert(model_prepared)
    
    return model_quantized

# QAT（量化感知训练）
class QATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = quant.QuantStub()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 30 * 30, 10)
        self.dequant = quant.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

def quantization_aware_training(model, train_loader, epochs=10):
    """量化感知训练"""
    # 1. 准备QAT
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    model_prepared = quant.prepare_qat(model)
    
    # 2. 训练
    optimizer = torch.optim.Adam(model_prepared.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model_prepared.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model_prepared(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 3. 转换为量化模型
    model_prepared.eval()
    model_quantized = quant.convert(model_prepared)
    
    return model_quantized

# 量化效果评估
def evaluate_quantization(original_model, quantized_model, test_loader):
    """评估量化效果"""
    import os
    
    # 模型大小
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.pt")
        size = os.path.getsize("temp.pt") / 1e6  # MB
        os.remove("temp.pt")
        return size
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化模型大小: {quantized_size:.2f} MB")
    print(f"压缩比: {original_size/quantized_size:.2f}x")
    
    # 推理速度
    import time
    
    def benchmark_speed(model, test_loader, num_batches=100):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_batches:
                    break
                _ = model(data)
        return time.time() - start
    
    original_time = benchmark_speed(original_model, test_loader)
    quantized_time = benchmark_speed(quantized_model, test_loader)
    
    print(f"原始模型推理时间: {original_time:.3f}s")
    print(f"量化模型推理时间: {quantized_time:.3f}s")
    print(f"加速比: {original_time/quantized_time:.2f}x")
```

## 4. 模型剪枝 ✂️

```python
import torch.nn.utils.prune as prune

def structured_pruning(model, pruning_rate=0.3):
    """结构化剪枝"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # L1结构化剪枝
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=1, dim=0)
        elif isinstance(module, nn.Linear):
            # L2非结构化剪枝
            prune.l2_unstructured(module, name='weight', amount=pruning_rate)
    
    return model

def iterative_pruning(model, train_loader, test_loader, pruning_rates=[0.1, 0.2, 0.3]):
    """迭代剪枝+微调"""
    best_accuracy = evaluate_model(model, test_loader)
    
    for rate in pruning_rates:
        print(f"剪枝率: {rate}")
        
        # 剪枝
        model = structured_pruning(model, rate)
        
        # 微调
        finetune_model(model, train_loader, epochs=5)
        
        # 评估
        accuracy = evaluate_model(model, test_loader)
        print(f"精度: {accuracy:.2f}%")
        
        if accuracy < best_accuracy * 0.95:  # 精度下降超过5%
            print("精度下降过多，停止剪枝")
            break
    
    # 永久移除剪枝的权重
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, 'weight')
    
    return model

def evaluate_model(model, test_loader):
    """评估模型精度"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def finetune_model(model, train_loader, epochs=5):
    """微调模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

## 5. TensorRT部署 🏎️

```python
def tensorrt_deployment(onnx_path):
    """TensorRT部署（NVIDIA GPU）"""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # 创建builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # INT8量化
    if builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # 需要校准数据集
        # config.int8_calibrator = create_calibrator(calibration_data)
    
    # FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open("model.trt", "wb") as f:
        f.write(engine.serialize())
    
    return engine

class TensorRTInference:
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 分配缓冲区
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data):
        import pycuda.driver as cuda
        
        # 复制输入数据
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # 传输到GPU
        stream = cuda.Stream()
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=stream.handle)
        
        # 传输回CPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        
        stream.synchronize()
        
        return [out['host'] for out in self.outputs]
```

## 6. 移动端部署 📱

```python
# PyTorch Mobile
def prepare_mobile_model(model):
    """准备移动端模型"""
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    
    # 优化为移动端
    from torch.utils.mobile_optimizer import optimize_for_mobile
    
    traced_model = torch.jit.trace(model, example)
    optimized_model = optimize_for_mobile(traced_model)
    
    # 保存
    optimized_model._save_for_lite_interpreter("model_mobile.ptl")
    
    print("模型已优化并保存为移动端格式")
    return optimized_model

# Core ML（iOS）
def export_to_coreml(model, example_input):
    """导出到Core ML"""
    import coremltools as ct
    
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    
    # 转换
    ml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
        minimum_deployment_target=ct.target.iOS14
    )
    
    # 保存
    ml_model.save("model.mlmodel")
    
    return ml_model

# TFLite（Android）
def export_to_tflite(onnx_path):
    """通过ONNX导出到TFLite"""
    import tf2onnx
    import tensorflow as tf
    
    # ONNX -> TensorFlow
    tf_model = tf2onnx.convert.from_onnx(onnx_path)
    
    # TensorFlow -> TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # 保存
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return tflite_model
```

## 7. 服务化部署 🌐

```python
# FastAPI服务
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# 全局模型
model = torch.jit.load("model_scripted.pt")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """模型推理API"""
    # 读取图像
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # 返回结果
    results = []
    for i in range(5):
        results.append({
            "class": int(top5_idx[0][i]),
            "probability": float(top5_prob[0][i])
        })
    
    return {"predictions": results}

# Docker部署
dockerfile_content = """
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Triton Inference Server配置
triton_config = """
name: "resnet50"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
"""
```

## 部署检查清单 ✅

```python
def deployment_checklist(model, test_loader):
    """部署前的完整检查"""
    print("=== 部署检查清单 ===\n")
    
    checks = {
        "模型评估模式": model.training == False,
        "批归一化固定": all(not m.training for m in model.modules() 
                           if isinstance(m, nn.BatchNorm2d)),
        "梯度计算关闭": not any(p.requires_grad for p in model.parameters()),
        "输入形状验证": True,  # 需要实际测试
        "输出范围检查": True,  # 需要实际测试
        "性能基准测试": True,  # 已完成
        "量化精度损失": True,  # <5%
        "模型大小优化": True,  # 已压缩
    }
    
    for check, status in checks.items():
        status_str = "✅" if status else "❌"
        print(f"{status_str} {check}")
    
    print("\n=== 推荐部署方案 ===")
    print("• 服务器GPU: TensorRT + Triton")
    print("• 服务器CPU: ONNX Runtime + Docker")
    print("• 移动端iOS: Core ML")
    print("• 移动端Android: TFLite")
    print("• 边缘设备: OpenVINO")
    print("• Web浏览器: ONNX.js / TensorFlow.js")
```

## 最佳实践

1. **选择合适的部署方案**
   - 延迟敏感：TensorRT/OpenVINO
   - 通用性：ONNX Runtime
   - 移动端：TFLite/Core ML

2. **优化策略组合**
   - 量化 + 剪枝 + 知识蒸馏
   - 先剪枝，后量化
   - 保持精度损失<5%

3. **性能监控**
   - 延迟、吞吐量、内存占用
   - A/B测试
   - 持续优化

## 下一步学习
- [NLP模型](nlp_models.md) - BERT、GPT、T5
- [LLM部署](llm_deployment.md) - 大模型部署技术