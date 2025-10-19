# 统一动态度评分系统 - 使用指南

## ? 概述

统一动态度评分系统将所有动态度相关指标整合为一个**0-1标准化分数**，适用于所有类型的视频。

### 分数含义

```
0.0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0
 ↑                    ↑                    ↑        ↑
纯静态              低动态              中等动态    高动态
(建筑)            (飘动旗帜)           (行人)    (跳舞)
```

**标准化映射**：
- **0.0 - 0.2**: ?? 纯静态（建筑物、雕塑）
- **0.2 - 0.4**: ? 低动态（轻微运动，如飘动的旗帜）
- **0.4 - 0.6**: ? 中等动态（人物行走、日常活动）
- **0.6 - 0.8**: ? 高动态（跑步、跳舞）
- **0.8 - 1.0**: ? 极高动态（激烈运动、快速舞蹈）

---

## ? 快速使用

### 方法1：自动集成（推荐）

统一动态度评分已自动集成到 `video_processor.py` 中：

```bash
# 正常运行即可，无需额外参数
python video_processor.py -i your_video.mp4 -o output/
```

### 方法2：Python API

```python
from video_processor import VideoProcessor

# 创建处理器
processor = VideoProcessor(device='cuda')

# 处理视频
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir="output")

# 获取统一动态度分数
unified_score = result['unified_dynamics']['unified_dynamics_score']
category = result['dynamics_classification']['category']

print(f"动态度分数: {unified_score:.3f}")
print(f"分类: {category}")
```

---

## ? 输出结果

### JSON结果

```json
{
  "unified_dynamics_score": 0.652,
  "scene_type": "dynamic",
  "dynamics_category": "high_dynamic",
  "dynamics_category_id": 3,
  
  "unified_dynamics": {
    "unified_dynamics_score": 0.652,
    "scene_type": "dynamic",
    "confidence": 0.85,
    "component_scores": {
      "flow_magnitude": 0.68,
      "spatial_coverage": 0.72,
      "temporal_variation": 0.45,
      "spatial_consistency": 0.55,
      "camera_factor": 0.40
    },
    "interpretation": "...",
    "normalization_params": {
      "mode": "auto",
      "detected_scene": "dynamic"
    }
  },
  
  "dynamics_classification": {
    "category": "high_dynamic",
    "category_id": 3,
    "description": "高动态场景",
    "typical_examples": ["跑步", "跳舞", "体育运动"]
  }
}
```

### 文本报告

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
? 统一动态度评估 (Unified Dynamics Score)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

综合动态度分数: 0.652 / 1.000
场景类型: dynamic
置信度: 85.0%

分类结果: 高动态场景
典型例子: 跑步, 跳舞, 体育运动

? 动态度: 0.652 (高动态)
场景类型: 动态场景（物体运动）
主要贡献: 运动覆盖 (0.720)

分数解释:
- 0.0-0.2: 纯静态物体（如建筑、雕塑）
- 0.2-0.4: 轻微运动（如飘动的旗帜）
- 0.4-0.6: 中等运动（如行走的人）
- 0.6-0.8: 活跃运动（如跑步、舞蹈）
- 0.8-1.0: 剧烈运动（如快速舞蹈、体育运动）
```

---

## ? 技术原理

### 多维度指标融合

统一动态度分数由5个维度融合而成：

#### 1. 光流幅度 (35% 权重)

**含义**：运动的强度

```python
# 静态场景：使用残差光流（补偿后）
# 动态场景：使用原始光流
flow_score = sigmoid(mean_flow_magnitude, threshold=5.0)
```

#### 2. 空间覆盖 (25% 权重)

**含义**：运动区域占比

```python
spatial_score = dynamic_ratio = 1.0 - static_ratio
```

#### 3. 时序变化 (20% 权重)

**含义**：运动的时间变化丰富度

```python
temporal_score = sigmoid(std_dynamics_score, threshold=1.0)
```

#### 4. 空间一致性 (10% 权重)

**含义**：运动的空间均匀性（反向指标）

```python
consistency_score = 1.0 - mean_consistency_score
```

#### 5. 相机因子 (10% 权重)

**含义**：相机补偿效果（如果启用）

```python
camera_score = 1.0 - camera_success_rate
```

### 自适应权重

系统会根据场景类型自动调整权重：

**静态场景**（有相机运动）：
```python
weights = {
    'flow_magnitude': 0.40,      # 更关注残差
    'spatial_coverage': 0.20,
    'temporal_variation': 0.15,
    'spatial_consistency': 0.15,
    'camera_factor': 0.10
}
```

**动态场景**（物体运动）：
```python
weights = {
    'flow_magnitude': 0.45,      # 更关注原始光流
    'spatial_coverage': 0.30,    # 更关注覆盖
    'temporal_variation': 0.15,
    'spatial_consistency': 0.05,
    'camera_factor': 0.05
}
```

### Sigmoid归一化

使用Sigmoid函数将任意范围映射到0-1：

$$
\text{score} = \frac{1}{1 + e^{-k(x - x_0)}}
$$

其中：
- $x$: 原始指标值
- $x_0$: 阈值（中点）
- $k$: 陡峭度

---

## ?? 高级配置

### 自定义权重

```python
from unified_dynamics_scorer import UnifiedDynamicsScorer

custom_weights = {
    'flow_magnitude': 0.5,
    'spatial_coverage': 0.3,
    'temporal_variation': 0.1,
    'spatial_consistency': 0.05,
    'camera_factor': 0.05
}

scorer = UnifiedDynamicsScorer(weights=custom_weights)
```

### 自定义阈值

```python
custom_thresholds = {
    'flow_low': 0.5,      # 低动态阈值
    'flow_mid': 3.0,      # 中等动态阈值
    'flow_high': 10.0,    # 高动态阈值
    'static_ratio': 0.6   # 静态判断阈值
}

scorer = UnifiedDynamicsScorer(thresholds=custom_thresholds)
```

### 指定场景模式

```python
# 静态场景模式（强制使用残差光流）
scorer = UnifiedDynamicsScorer(mode='static_scene')

# 动态场景模式（强制使用原始光流）
scorer = UnifiedDynamicsScorer(mode='dynamic_scene')

# 自动检测模式（默认）
scorer = UnifiedDynamicsScorer(mode='auto')
```

---

## ? 应用场景

### 场景1：视频分类

```python
from unified_dynamics_scorer import DynamicsClassifier

classifier = DynamicsClassifier()

# 处理视频
result = processor.process_video(frames, output_dir="output")
unified_score = result['unified_dynamics']['unified_dynamics_score']

# 分类
classification = classifier.classify(unified_score)
print(classification['category'])  # 'high_dynamic'
```

### 场景2：质量筛选

```python
# 筛选纯静态视频（建筑）
if unified_score < 0.2:
    print("纯静态建筑视频")

# 筛选高动态视频（舞蹈）
if unified_score > 0.7:
    print("高动态人物视频")
```

### 场景3：批量评估

```python
# 批量处理
results = []
for video in video_list:
    frames = processor.load_video(video)
    result = processor.process_video(frames)
    results.append(result)

# 批量统计
batch_stats = scorer.batch_calculate(
    [r for r in results],
    camera_comp_enabled=True
)

print(f"平均动态度: {batch_stats['mean_score']:.3f}")
print(f"标准差: {batch_stats['std_score']:.3f}")
```

---

## ? 诊断与调试

### 查看各维度贡献

```python
component_scores = result['unified_dynamics']['component_scores']

for name, score in component_scores.items():
    print(f"{name}: {score:.3f}")

# 输出:
# flow_magnitude: 0.680
# spatial_coverage: 0.720
# temporal_variation: 0.450
# spatial_consistency: 0.550
# camera_factor: 0.400
```

### 分析置信度

```python
confidence = result['unified_dynamics']['confidence']

if confidence < 0.6:
    print("?? 置信度较低，结果可能不稳定")
    print("建议检查：")
    print("- 视频质量")
    print("- 光流计算准确性")
    print("- 相机补偿效果")
```

### 场景类型检测

```python
scene_type = result['unified_dynamics']['scene_type']

if scene_type == 'static':
    print("检测为静态场景（相机运动）")
    print("使用残差光流计算动态度")
else:
    print("检测为动态场景（物体运动）")
    print("使用原始光流计算动态度")
```

---

## ? 最佳实践

### 1. 视频预处理

? **推荐**：
- 稳定的帧率
- 清晰的画质
- 适当的分辨率

?? **避免**：
- 严重运动模糊
- 极低分辨率
- 编码损坏

### 2. 相机补偿选择

**启用相机补偿** (默认):
- 适合：相机转动拍摄静态场景
- 优点：更准确的静态物体评估

**禁用相机补偿**:
- 适合：固定机位拍摄动态场景
- 优点：直接评估物体运动

### 3. 结果解释

**动态度 < 0.3**:
- 可能是：固定机位拍摄静态物体
- 建议：检查是否真的是静态场景

**动态度 0.3-0.7**:
- 可能是：正常的人物活动
- 适合：大多数日常视频

**动态度 > 0.7**:
- 可能是：激烈运动
- 建议：确认是否符合预期

---

## ? 常见问题

### Q1: 分数偏高/偏低？

**原因**：
- 阈值设置不当
- 场景类型检测错误

**解决**：
```python
# 调整归一化阈值
scorer = UnifiedDynamicsScorer(
    thresholds={'flow_mid': 8.0}  # 提高阈值降低分数
)

# 或指定场景模式
scorer = UnifiedDynamicsScorer(mode='static_scene')
```

### Q2: 置信度过低？

**原因**：
- 时序不稳定
- 各维度分数差异大

**解决**：
- 检查视频质量
- 增加处理帧数
- 调整光流计算参数

### Q3: 如何与ground truth标签对齐？

**方法1：线性映射**
```python
# 如果你的标签是0-1
# 统一分数已经是0-1，可直接使用

# 如果需要调整范围
def map_to_label(score):
    # 例如：将0.3-0.7映射到0-1
    return (score - 0.3) / 0.4
```

**方法2：调整分类阈值**
```python
from unified_dynamics_scorer import DynamicsClassifier

# 自定义阈值
custom_thresholds = {
    'pure_static': 0.10,
    'low_dynamic': 0.30,
    'medium_dynamic': 0.60,
    'high_dynamic': 0.80
}

classifier = DynamicsClassifier(thresholds=custom_thresholds)
```

---

## ? 相关文档

- [静态物体分析原理](STATIC_OBJECT_ANALYSIS_EXPLAINED.md)
- [相机补偿使用指南](CAMERA_COMPENSATION_GUIDE.md)
- [集成总结](INTEGRATION_SUMMARY.md)

---

## ? 总结

统一动态度评分系统提供了：

? **通用性**：适用于所有类型视频  
? **标准化**：0-1统一分数，便于比较  
? **可解释**：多维度分解，清晰理解  
? **自适应**：自动检测场景类型  
? **可配置**：灵活调整权重和阈值

**核心优势**：
```
多个指标 → 统一融合 → 单一分数 → 简单易用
```

---

**文档版本**: 1.0  
**最后更新**: 2025-10-19

