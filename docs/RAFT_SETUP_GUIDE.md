# RAFT光流模型使用指南

## 概述

本项目支持**两种方式**使用RAFT光流估计：

1. **OpenCV光流** (推荐用于快速测试) - 无需下载大模型
2. **RAFT官方预训练模型** (推荐用于生产) - 精度更高

---

## 方式1: 使用OpenCV光流 (推荐)

### 优点
- ? 无需下载大模型文件
- ? 安装简单，依赖少
- ? 速度快
- ? 适合快速原型和测试

### 使用方法

```python
from raft_model_simple import RAFTPredictor

# 创建预测器 (自动使用OpenCV)
predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)

# 预测光流
flow = predictor.predict_flow(image1, image2)
```

### 当前项目使用情况

当前项目在以下模块中使用了光流：
- `static_object_analyzer.py` - 静态物体动态度分析
- `dynamic_motion_compensation/` - 动态运动补偿

这些模块**已经使用了OpenCV光流**（通过 `simple_raft.py`），运行良好！

---

## 方式2: 使用RAFT官方预训练模型

### 优点
- ? 光流精度更高
- ? 对复杂场景鲁棒性更好
- ? 学术/生产级质量

### 缺点
- ?? 需要下载大模型文件 (~150MB)
- ?? 需要更多GPU内存
- ?? 推理速度较慢

---

## 下载RAFT官方预训练权重

### 方法1: 从官方仓库下载

访问 RAFT 官方 GitHub:
```
https://github.com/princeton-vl/RAFT
```

下载以下任一模型:

| 模型名称 | 训练数据集 | 适用场景 | 大小 |
|---------|----------|---------|------|
| **raft-things.pth** | Things3D | 通用场景 | ~150MB |
| **raft-sintel.pth** | Sintel | 电影级场景 | ~150MB |
| **raft-chairs.pth** | FlyingChairs | 简单场景 | ~150MB |
| **raft-kitti.pth** | KITTI | 自动驾驶 | ~150MB |

**推荐**: `raft-things.pth` (泛化性能最好)

### 方法2: 直接下载链接

```bash
# raft-things.pth
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth

# raft-sintel.pth  
wget https://dl.dropboxusercontent.com/s/kqdjpd17kkb8syk/raft-sintel.pth
```

### 方法3: 使用 gdown (Google Drive)

```bash
pip install gdown

# raft-things
gdown 1M5QHhdMI6oWF3Bv8Y1oW8vvVGMxM8Gru -O raft-things.pth

# raft-sintel
gdown 1Sxb0RDsJ7JBz9NJ6wj4QzHXlZ7PdYJKd -O raft-sintel.pth
```

---

## 使用RAFT官方模型

### 步骤1: 下载模型文件

将下载的 `.pth` 文件放到以下任一位置:
- `AIGC_detector/raft-things.pth`
- `AIGC_detector/pretrained_models/raft-things.pth`

### 步骤2: 使用模型

```python
from raft_model_simple import RAFTPredictor

# 指定模型路径
predictor = RAFTPredictor(
    model_path='raft-things.pth',
    device='cuda'  # 或 'cpu'
)

# 预测光流
flow = predictor.predict_flow(image1, image2)
```

### 步骤3: (可选) 更新现有代码

如果想在现有模块中使用RAFT官方模型:

```python
# 在 static_object_analyzer.py 中
from raft_model_simple import RAFTPredictor

# 替换原来的 SimpleRAFTPredictor
self.flow_predictor = RAFTPredictor(
    model_path='raft-things.pth',
    device='cuda'
)
```

---

## 代码对比

### 当前实现 (raft_model.py)
- **390行代码** - 完整实现了RAFT架构
- 包含: ResidualBlock, FeatureEncoder, CorrBlock, UpdateBlock等
- ? 复杂度高，难维护
- ? 需要自己训练或转换权重格式

### 简化版 (raft_model_simple.py)
- **~200行代码** - 只是加载器和预处理
- 直接使用官方预训练权重
- ? 简单易懂
- ? 官方权重开箱即用
- ? 支持OpenCV后备方案

---

## 性能对比

| 方法 | 精度 | 速度 | GPU内存 | 模型大小 |
|-----|------|------|---------|---------|
| OpenCV Farneback | 中等 | 快 (~50ms) | 无需GPU | 0 MB |
| RAFT官方模型 | 高 | 中等 (~100ms) | ~2GB | ~150 MB |
| 自己训练RAFT | 高 | 中等 | ~2GB | ~150 MB |

---

## 建议

### 用于开发/测试
```python
# 使用 OpenCV - 快速、简单
predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)
```

### 用于生产/发布
```python
# 使用 RAFT官方模型 - 高精度
predictor = RAFTPredictor(model_path='raft-things.pth')
```

---

## 常见问题

### Q1: raft_model.py 已删除

**? 已简化！** 项目已删除390行的完整RAFT实现：
- ~~`raft_model.py`~~ - 已删除（过于复杂）
- `raft_model_simple.py` - 简化版（推荐使用）
- `simple_raft.py` - OpenCV实现（当前使用中）

### Q2: 官方权重文件格式?

RAFT官方权重是标准的 PyTorch `.pth` 文件:
```python
torch.load('raft-things.pth')
```

### Q3: 模型加载失败?

如果加载失败，会**自动回退到OpenCV光流**:
```python
predictor = RAFTPredictor(
    model_path='raft-things.pth',
    use_opencv_fallback=True  # 加载失败自动使用OpenCV
)
```

### Q4: GPU内存不足?

使用CPU模式:
```python
predictor = RAFTPredictor(model_path='raft-things.pth', device='cpu')
```

或使用OpenCV:
```python
predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)
```

---

## 总结

? **推荐做法**: 
- 保持现有代码使用 OpenCV 光流
- 如需更高精度，下载 `raft-things.pth` 并使用 `raft_model_simple.py`
- 不需要自己实现 RAFT 架构代码

? **避免**: 
- 从零实现 RAFT (390行代码)
- 自己训练 RAFT 模型
- 复杂的权重格式转换

