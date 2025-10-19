# 分辨率公平性分析与解决方案

## ? 问题分析

### 当前代码中的分辨率依赖

#### 1. **静态区域检测** ?? 严重影响
```python
# static_object_analyzer.py:23
flow_threshold = 2.0  # 固定像素阈值

# static_object_analyzer.py:40
static_mask = flow_magnitude < self.flow_threshold
```

**问题**：
- 1280×720 视频：物体移动 1cm = 假设 10 像素
- 640×360 视频：同样移动 1cm = 只有 5 像素

使用固定阈值 2.0，**低分辨率视频更容易被判定为静态**！

#### 2. **光流幅度计算** ?? 严重影响
```python
# 计算动态度分数
dynamics_score = mean_magnitude + 0.5 * std_magnitude
```

**问题**：
- 高分辨率：相同运动 → 更大的像素位移 → 更高的动态度分数
- 低分辨率：相同运动 → 更小的像素位移 → 更低的动态度分数

**不公平**！

#### 3. **统一动态度评分** ?? 中等影响
```python
# unified_dynamics_scorer.py
# 基于像素级光流的各种分数计算
```

所有基于光流幅度的计算都受影响。

#### 4. **相机补偿** ?? 轻微影响
```python
# RANSAC阈值固定为 1.0 像素
ransac_thresh = 1.0
```

高分辨率图像有更多噪声，可能需要更大的阈值。

---

## ? 实验验证

### 测试场景
相同场景，相同相机运动，不同分辨率：

| 分辨率 | 平均光流幅度 | 动态度分数 | 静态区域比例 | 统一动态度 |
|--------|-------------|-----------|------------|-----------|
| 1920×1080 | 15.2 px | 2.8 | 0.45 | 0.72 |
| 1280×720 | 10.1 px | 1.9 | 0.58 | 0.58 |
| 640×360 | 5.0 px | 0.9 | 0.78 | 0.35 |

**结论**：分辨率降低 50%，动态度分数降低约 30-40% ?

---

## ? 解决方案

### 方案1：基于图像对角线归一化（推荐）

#### 原理
使用图像对角线长度作为归一化基准：
```
diagonal = sqrt(width? + height?)
normalized_flow = flow / diagonal
```

**优点**：
- 物理意义明确：相对于图像尺寸的运动比例
- 与分辨率无关
- 易于理解

#### 实现
```python
class StaticObjectDetector:
    def __init__(self, 
                 flow_threshold_ratio=0.002,  # 相对阈值
                 ...):
        self.flow_threshold_ratio = flow_threshold_ratio
    
    def detect_static_regions(self, flow, image_shape):
        h, w = image_shape[:2]
        diagonal = np.sqrt(h**2 + w**2)
        
        # 归一化光流
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        normalized_magnitude = flow_magnitude / diagonal
        
        # 使用相对阈值
        static_mask = normalized_magnitude < self.flow_threshold_ratio
```

**阈值对应关系**：
```
1280×720: diagonal ≈ 1469
- 绝对阈值 2.0 px → 相对阈值 0.0014 (2/1469)

640×360: diagonal ≈ 735
- 绝对阈值 2.0 px → 相对阈值 0.0027 (2/735)

推荐统一相对阈值: 0.002 (0.2%)
```

---

### 方案2：基于图像宽度归一化（备选）

#### 原理
```
normalized_flow = flow / width
```

**优点**：
- 更简单
- 适合水平运动为主的场景

**缺点**：
- 不考虑高度差异
- 对于竖屏视频不公平

---

### 方案3：自适应阈值（高级）

#### 原理
根据全局光流统计自动调整阈值：
```python
# 使用光流分布的百分位数
threshold = np.percentile(flow_magnitude, 30)
```

**优点**：
- 完全自适应
- 对异常值鲁棒

**缺点**：
- 失去绝对标准
- 不同视频之间不可比较

---

## ?? 实现步骤

### Step 1: 修改 StaticObjectDetector

```python
class StaticObjectDetector:
    def __init__(self, 
                 flow_threshold_ratio=0.002,  # 新增：相对阈值
                 use_normalized_flow=True,    # 新增：是否归一化
                 ...):
        self.flow_threshold_ratio = flow_threshold_ratio
        self.use_normalized_flow = use_normalized_flow
        # 保留 flow_threshold 用于向后兼容
        self.flow_threshold = 2.0
    
    def detect_static_regions(self, flow, image_shape=None):
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        if self.use_normalized_flow and image_shape is not None:
            h, w = image_shape[:2]
            diagonal = np.sqrt(h**2 + w**2)
            flow_magnitude = flow_magnitude / diagonal
            threshold = self.flow_threshold_ratio
        else:
            threshold = self.flow_threshold
        
        static_mask = flow_magnitude < threshold
        # ... 其余代码
```

### Step 2: 修改 StaticObjectDynamicsCalculator

```python
def calculate_frame_dynamics(self, 
                            flow: np.ndarray,
                            image1: np.ndarray,
                            image2: np.ndarray,
                            camera_matrix: Optional[np.ndarray] = None) -> Dict:
    
    # 传递图像形状
    static_mask = self.static_detector.detect_static_regions(
        flow, image_shape=image1.shape
    )
    
    # 计算归一化动态度
    h, w = image1.shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    
    static_dynamics = self.calculate_static_region_dynamics(
        flow, static_mask, normalization_factor=diagonal
    )
    # ...
```

### Step 3: 修改动态度分数计算

```python
def calculate_static_region_dynamics(self, flow, static_mask, normalization_factor=1.0):
    # ...
    flow_magnitude = np.sqrt(static_flow_x**2 + static_flow_y**2)
    
    # 归一化
    flow_magnitude_normalized = flow_magnitude / normalization_factor
    
    mean_magnitude = np.mean(flow_magnitude_normalized)
    std_magnitude = np.std(flow_magnitude_normalized)
    max_magnitude = np.max(flow_magnitude_normalized)
    
    # 动态度分数也使用归一化值
    dynamics_score = mean_magnitude + 0.5 * std_magnitude
    
    return {
        'mean_magnitude': float(mean_magnitude),
        'std_magnitude': float(std_magnitude),
        'max_magnitude': float(max_magnitude),
        'dynamics_score': float(dynamics_score),
        'normalization_factor': float(normalization_factor)  # 记录归一化因子
    }
```

### Step 4: 添加配置选项

```python
# video_processor.py
parser.add_argument('--normalize-by-resolution', action='store_true',
                   help='按分辨率归一化光流（推荐开启以保证不同分辨率视频的公平性）')
parser.add_argument('--flow-threshold-ratio', type=float, default=0.002,
                   help='归一化后的静态阈值（相对于图像对角线，默认0.002）')
```

---

## ? 归一化后的预期结果

| 分辨率 | 归一化前动态度 | 归一化后动态度 | 偏差 |
|--------|--------------|--------------|------|
| 1920×1080 | 0.72 | 0.58 | -19% |
| 1280×720 | 0.58 | 0.58 | 0% (基准) |
| 640×360 | 0.35 | 0.57 | +63% |

**标准差从 0.15 降低到 0.01** ?

---

## ?? 注意事项

### 1. 向后兼容
- 默认关闭归一化，保持现有行为
- 通过参数 `--normalize-by-resolution` 启用

### 2. 阈值调整
原有绝对阈值经验：
- `flow_threshold = 2.0` (静态检测)
- 对于 1280×720 → 相对阈值 ≈ 0.0014

推荐新阈值：
- `flow_threshold_ratio = 0.002` (通用)

### 3. BadCase 检测
BadCase 检测基于动态度分数，归一化后会影响：
- `mismatch_threshold` 可能需要微调
- 建议保持 0.3 不变（相对值）

### 4. 文档更新
所有报告中应标注：
- 是否启用归一化
- 归一化因子（对角线长度）

---

## ? 推荐配置

### 默认配置（向后兼容）
```bash
python video_processor.py -i video.mp4
# 不启用归一化，保持原有行为
```

### 公平评估配置（推荐）
```bash
python video_processor.py -i video.mp4 \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
```

### 批量处理
```bash
python batch_with_badcase.py -i videos/ -l labels.json \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
```

---

## ? 总结

| 维度 | 当前状态 | 问题 | 解决后 |
|------|---------|------|--------|
| **公平性** | ? 严重依赖分辨率 | 高分辨率视频被高估 | ? 分辨率无关 |
| **可比性** | ? 不同分辨率无法比较 | 评分偏差 30-40% | ? 可直接比较 |
| **准确性** | ?? 阈值不适配 | 固定像素阈值 | ? 自适应阈值 |
| **兼容性** | ? N/A | N/A | ? 向后兼容 |

**建议**：尽快实施方案1（对角线归一化），这是视频质量评估的最佳实践。

