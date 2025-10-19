# 环境配置指南

## 当前问题

您遇到了 **NumPy 2.x 与 OpenCV 不兼容**的问题：
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2
```

## 解决方案

### 方法1: 降级 NumPy（推荐）

```bash
pip install "numpy<2.0"
```

### 方法2: 重装完整环境

```bash
# 卸载冲突的包
pip uninstall numpy opencv-python opencv-contrib-python -y

# 按正确顺序重装
pip install "numpy<2.0"
pip install opencv-python opencv-contrib-python

# 或直接使用requirements.txt
pip install -r requirements.txt
```

### 方法3: 创建新的虚拟环境

```bash
# 创建新环境
conda create -n aigc_detector python=3.9 -y
conda activate aigc_detector

# 安装依赖
pip install -r requirements.txt
```

---

## 已修复的代码问题

除了 NumPy 版本问题，我还修复了以下代码问题：

### 1. ? 光流数组索引问题
```python
# 之前（错误）
static_flow = flow[static_mask]
magnitude = np.sqrt(static_flow[:, 0]**2 + static_flow[:, 1]**2)

# 现在（正确）
static_flow_x = flow[:, :, 0][static_mask]
static_flow_y = flow[:, :, 1][static_mask]
magnitude = np.sqrt(static_flow_x**2 + static_flow_y**2)
```

### 2. ? 批量处理文件去重
Windows下文件名大小写不敏感导致重复

### 3. ? 所有文件添加UTF-8编码声明

---

## 安装步骤

1. **降级 NumPy**
```bash
pip install "numpy<2.0"
```

2. **安装缺失的依赖**
```bash
pip install opencv-contrib-python scikit-learn
```

3. **验证安装**
```bash
python -c "import cv2; import numpy as np; print('OpenCV:', cv2.__version__); print('NumPy:', np.__version__)"
```

期望输出：
```
OpenCV: 4.x.x
NumPy: 1.x.x (< 2.0)
```

---

## 运行测试

安装完成后，运行：

```bash
# 单个视频测试
python video_processor.py -i videos/test.mp4 -o output_single/ --no-visualize

# 批量处理
python video_processor.py -i videos/ -o results/ --batch --no-visualize --max_frames 50 --frame_skip 3
```

---

## 依赖清单

已更新的 `requirements.txt`:
- `numpy>=1.21.0,<2.0.0` ← **限制版本避免兼容性问题**
- `opencv-python>=4.5.0`
- `opencv-contrib-python>=4.5.0` ← **新增，支持TV-L1**
- `scikit-learn>=0.24.0` ← **新增，RANSAC需要**
- torch, scipy, matplotlib 等

---

## 快速开始

```bash
# 1. 修复环境
pip install "numpy<2.0" opencv-contrib-python scikit-learn

# 2. 测试单个视频
python video_processor.py -i videos/test.mp4 -o test_output/ --no-visualize

# 3. 批量处理（快速模式）
python video_processor.py -i videos/ -o batch_results/ --batch --no-visualize --max_frames 30 --frame_skip 3
```

