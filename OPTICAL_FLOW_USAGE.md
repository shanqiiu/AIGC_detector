# 光流算法使用指南

## 快速开始

现在 `simple_raft.py` 支持三种光流算法，通过统一接口使用：

### 方法1: Farneback（推荐开始使用）

```python
from simple_raft import SimpleRAFTPredictor

# 最简单 - 无需任何额外依赖
predictor = SimpleRAFTPredictor(method='farneback')
flow = predictor.predict_flow(image1, image2)
```

**特点**:
- ? 速度快 (~50ms/frame)
- ? OpenCV内置，无需额外安装
- ? 能检测明显的质量问题
- ?? 精度中等

---

### 方法2: TV-L1（生产环境推荐）

```python
from simple_raft import SimpleRAFTPredictor

# 需要安装: pip install opencv-contrib-python
predictor = SimpleRAFTPredictor(method='tvl1')
flow = predictor.predict_flow(image1, image2)
```

**特点**:
- ? 精度高，边界清晰
- ? CPU友好
- ? 能检测微小的质量问题
- ?? 速度较慢 (~200ms/frame)
- ?? 需要安装 opencv-contrib-python

---

### 方法3: RAFT官方（最高精度）

```python
from simple_raft import SimpleRAFTPredictor

# 需要: 
# 1. third_party/RAFT 目录（官方代码）
# 2. pretrained_models/raft-things.pth（预训练模型）
predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth',
    device='cuda'  # 或 'cpu'
)
flow = predictor.predict_flow(image1, image2)
```

**特点**:
- ? 精度最高
- ? 小运动检测能力强
- ? 边界最清晰
- ?? 需要GPU（CPU也可用但很慢）
- ?? 需要下载模型文件 (~150MB)
- ?? 速度中等 (~100ms/frame on GPU)

---

## 完整示例

```python
from simple_raft import SimpleRAFTPredictor
import cv2
import numpy as np

# 加载图像
image1 = cv2.imread('frame1.png')
image2 = cv2.imread('frame2.png')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# 选择一种方法
predictor = SimpleRAFTPredictor(method='farneback')  # 或 'tvl1' 或 'raft'

# 预测光流
flow = predictor.predict_flow(image1, image2)
print(f"光流形状: {flow.shape}")  # (2, H, W)

# 预测序列
images = [image1, image2, image3, ...]
flows = predictor.predict_flow_sequence(images)
```

---

## 在现有代码中使用

### video_processor.py 中切换

```python
# 当前默认（Farneback）
self.raft_predictor = SimpleRAFTPredictor()

# 切换到TV-L1
self.raft_predictor = SimpleRAFTPredictor(method='tvl1')

# 切换到RAFT官方
self.raft_predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth'
)
```

### demo.py 中切换

只需修改一行：

```python
# 原来
flow_predictor = SimpleRAFTPredictor()

# 改为
flow_predictor = SimpleRAFTPredictor(method='tvl1')  # 或 'raft'
```

---

## 性能对比

| 方法 | 速度 | 精度 | GPU需求 | 额外依赖 |
|------|------|------|---------|---------|
| **Farneback** | ???? | ??? | 否 | 无 |
| **TV-L1** | ?? | ???? | 否 | opencv-contrib-python |
| **RAFT** | ??? (GPU) | ????? | 推荐 | 模型文件 + 官方代码 |

---

## 选择建议

### 场景1: 快速开发/演示
```python
predictor = SimpleRAFTPredictor(method='farneback')
```
- 无需配置，开箱即用
- 能检测明显的质量问题

### 场景2: 生产部署
```python
# 安装: pip install opencv-contrib-python
predictor = SimpleRAFTPredictor(method='tvl1')
```
- 精度提升明显
- CPU友好，部署简单

### 场景3: 研究/极致精度
```python
predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth',
    device='cuda'
)
```
- 学术标准精度
- 能检测最微小的问题

---

## RAFT 设置指南

如果选择使用 RAFT 官方模型，需要：

### 1. 下载官方代码
```bash
# 已经放置在 AIGC_detector/third_party/RAFT/
```

### 2. 下载预训练模型
下载 `raft-things.pth` 并放置到 `pretrained_models/` 目录

- 官方地址: https://github.com/princeton-vl/RAFT
- 模型大小: ~150MB

### 3. 使用
```python
predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth'
)
```

---

## 故障排除

### TV-L1 无法使用
```bash
# 错误: opencv-contrib-python未安装
pip install opencv-contrib-python
```

### RAFT 加载失败
检查：
1. `third_party/RAFT/core/` 目录存在
2. `pretrained_models/raft-things.pth` 文件存在
3. 会自动回退到 Farneback

### NumPy 版本问题
```bash
# 如果遇到 NumPy 2.x 兼容性问题
pip install "numpy<2.0"
```

---

## 总结

- **开始**: 使用 `method='farneback'`（无需配置）
- **升级**: 切换到 `method='tvl1'`（更高精度）
- **极致**: 使用 `method='raft'`（最高精度，需要配置）

所有方法使用相同的接口，可以轻松切换！

