# 归一化后的阈值完整调整指南

## ? 您的问题：归一化后阈值需要变化吗？

**答案：是的！** 所有基于绝对像素值的阈值都需要相应调整。

---

## ? 阈值影响分析

### 数值范围变化

| 指标 | 归一化前（像素） | 归一化后（比例） | 缩放因子 |
|------|----------------|----------------|---------|
| mean_dynamics_score | 0.5 ~ 10.0 | 0.0003 ~ 0.007 | ÷ 1469 |
| std_dynamics_score | 0.1 ~ 3.0 | 0.00007 ~ 0.002 | ÷ 1469 |
| flow_magnitude | 1.0 ~ 30.0 | 0.0007 ~ 0.02 | ÷ 1469 |

*基于 1280×720 分辨率，diagonal ≈ 1469*

---

## ? 已自动调整的阈值

### 1. StaticObjectDetector （? 已实现）

```python
# 归一化前
flow_threshold = 2.0  # 像素

# 归一化后（自动）
flow_threshold_ratio = 0.002  # 相对值
实际阈值 = 0.002 × diagonal
  - 1280×720: 0.002 × 1469 = 2.94 px
  - 640×360:  0.002 × 734 = 1.47 px  # 自适应！
```

### 2. UnifiedDynamicsScorer （? 刚实现）

```python
# 归一化前
thresholds = {
    'flow_low': 1.0,      # 像素
    'flow_mid': 5.0,      # 像素
    'flow_high': 15.0,    # 像素
    'temporal_std': 1.0   # 像素
}

# 归一化后（自动，baseline=1469）
thresholds = {
    'flow_low': 0.00068,     # 1.0/1469
    'flow_mid': 0.0034,      # 5.0/1469
    'flow_high': 0.0102,     # 15.0/1469
    'temporal_std': 0.00068  # 1.0/1469
}
```

### 3. StaticObjectDynamicsCalculator （? 已实现）

所有动态度计算都会除以归一化因子。

---

## ? 不需要调整的阈值

### 1. DynamicsClassifier

```python
thresholds = {
    'pure_static': 0.15,      # unified_score阈值（0-1范围）
    'low_dynamic': 0.35,
    'medium_dynamic': 0.60,
    'high_dynamic': 0.85,
}
```

**原因**：这些是 `unified_dynamics_score` 的阈值，该分数始终在 0-1 范围，不受底层归一化影响。

### 2. BadCaseDetector

```python
mismatch_threshold = 0.3  # unified_score的差值（0-1范围）
```

**原因**：比较的是 `unified_score`，始终 0-1 范围。

**建议**：保持 0.3 不变，但可以根据实际效果微调（0.25 ~ 0.35）。

---

## ? 自动适配机制

### 完整的参数传递链

```
CLI参数: --normalize-by-resolution
    ↓
VideoProcessor(use_normalized_flow=True)
    ↓
├─ StaticObjectDynamicsCalculator(use_normalized_flow=True)
│   └─ StaticObjectDetector(use_normalized_flow=True)
│       └─ 阈值: flow_threshold_ratio = 0.002
│
└─ UnifiedDynamicsScorer(use_normalized_flow=True)
    └─ 阈值: flow_mid = 0.0034 (自动转换)
```

**关键**：一个参数控制整个链路的阈值调整！

---

## ? 阈值转换参考表

### 常用阈值转换（baseline: 1280×720, diagonal=1469）

| 用途 | 绝对值（px） | 归一化值 | 代码变量 |
|------|------------|---------|---------|
| 静态检测 | 2.0 | 0.0014 | flow_threshold_ratio |
| 低动态 | 1.0 | 0.00068 | thresholds['flow_low'] |
| 中等动态 | 5.0 | 0.0034 | thresholds['flow_mid'] |
| 高动态 | 15.0 | 0.0102 | thresholds['flow_high'] |
| 时序变化 | 1.0 | 0.00068 | thresholds['temporal_std'] |

### 不同基准分辨率的转换

| 基准分辨率 | 对角线 | flow_mid | 转换后 |
|-----------|-------|----------|--------|
| 1920×1080 | 2203 | 5.0 px | 0.00227 |
| **1280×720** | **1469** | **5.0 px** | **0.0034** ? |
| 960×540 | 1101 | 5.0 px | 0.00454 |
| 640×360 | 734 | 5.0 px | 0.00681 |

**推荐**：使用 1280×720 作为基准（中等分辨率，覆盖面广）

---

## ? 验证方法

### 1. 检查 unified_score 分布

```bash
python batch_with_badcase.py -i videos/ -l labels.json --normalize-by-resolution

# 查看输出
cat output/badcase_summary.json | grep unified_score
```

**正常情况**：
```json
"unified_score": 0.15,  // 静态场景
"unified_score": 0.42,  // 低动态
"unified_score": 0.68,  // 中等动态
```

**异常情况**（阈值未调整）：
```json
"unified_score": 0.0001,  // ? 所有都接近0
"unified_score": 0.0002,
```

### 2. 检查 component_scores

```json
{
  "component_scores": {
    "flow_magnitude": 0.45,      // ? 应该在 0-1 范围
    "spatial_coverage": 0.68,    // ?
    "temporal_variation": 0.32   // ?
  }
}
```

### 3. 对比测试

```bash
# 同一视频，两种模式
python video_processor.py -i test.mp4 -o output_no_norm/
python video_processor.py -i test.mp4 -o output_norm/ --normalize-by-resolution

# 比较结果
# unified_score 应该相近（±0.1），不应该差异巨大
```

---

## ?? 手动微调阈值（高级）

如果自动阈值不适合您的场景，可以手动调整：

### 方法1：修改代码

```python
# 创建处理器时指定自定义阈值
custom_thresholds = {
    'flow_low': 0.0005,    # 更严格的静态判断
    'flow_mid': 0.0030,    # 略微降低中等动态阈值
    'flow_high': 0.0120,   # 略微提高高动态阈值
    'static_ratio': 0.5,
    'temporal_std': 0.0008
}

processor = VideoProcessor(
    use_normalized_flow=True,
    ...
)

processor.unified_scorer.thresholds = custom_thresholds
```

### 方法2：场景特定配置

```python
# 静态场景（建筑）- 更敏感
static_scene_thresholds = {
    'flow_mid': 0.0025,   # 降低阈值，更敏感
}

# 动态场景（人物）- 更宽松
dynamic_scene_thresholds = {
    'flow_mid': 0.0045,   # 提高阈值，更宽松
}
```

---

## ? 完整阈值清单（归一化后）

### 模块：StaticObjectDetector
| 阈值 | 原值 | 归一化值 | 状态 |
|------|------|---------|------|
| flow_threshold | 2.0 px | 0.002 (ratio) | ? 已调整 |
| min_region_size | 100 px? | 100 px? | ? 不变（绝对像素数） |

### 模块：UnifiedDynamicsScorer
| 阈值 | 原值 | 归一化值 | 状态 |
|------|------|---------|------|
| flow_low | 1.0 px | 0.00068 | ? 自动调整 |
| flow_mid | 5.0 px | 0.0034 | ? 自动调整 |
| flow_high | 15.0 px | 0.0102 | ? 自动调整 |
| temporal_std | 1.0 px | 0.00068 | ? 自动调整 |
| static_ratio | 0.5 | 0.5 | ? 不变 |

### 模块：DynamicsClassifier
| 阈值 | 值 | 状态 |
|------|---|------|
| pure_static | 0.15 | ? 不变 |
| low_dynamic | 0.35 | ? 不变 |
| medium_dynamic | 0.60 | ? 不变 |
| high_dynamic | 0.85 | ? 不变 |

### 模块：BadCaseDetector
| 阈值 | 值 | 状态 |
|------|---|------|
| mismatch_threshold | 0.3 | ? 不变 |
| confidence_threshold | 0.6 | ? 不变 |

---

## ? 结论

### ? 已完成的自动调整

1. **StaticObjectDetector**：flow_threshold → flow_threshold_ratio
2. **UnifiedDynamicsScorer**：所有 flow_* 阈值根据归一化状态自动调整
3. **参数传递**：`use_normalized_flow` 自动传递到所有需要的模块

### ? 无需调整的阈值

1. **DynamicsClassifier**：基于 unified_score（0-1范围）
2. **BadCaseDetector**：基于 unified_score 差值

### ? 使用方法

**只需一个参数**：
```bash
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution
```

**系统会自动**：
- ? 调整 StaticObjectDetector 阈值
- ? 调整 UnifiedDynamicsScorer 阈值
- ? 保持 DynamicsClassifier 阈值
- ? 保持 BadCaseDetector 阈值

**无需手动修改任何阈值！**

---

## ?? 注意事项

### 如果遇到评分异常

症状：所有视频 unified_score ≈ 0 或 ≈ 1

可能原因：
1. 阈值未自动调整（检查代码更新）
2. 基准对角线设置不当
3. 输入数据异常

解决：
```python
# 临时手动指定阈值
processor.unified_scorer.thresholds = {
    'flow_mid': 0.0034,  # 根据实际调整
    ...
}
```

---

## ? 相关文档

- [阈值调整指南](./THRESHOLD_ADJUSTMENT_GUIDE.md) - 详细理论
- [归一化实现总结](./NORMALIZATION_IMPLEMENTATION_SUMMARY.md) - 代码实现
- [快速开始](./QUICK_START_NORMALIZATION.md) - 使用示例

---

**总结**：所有阈值已实现自动调整，您只需添加 `--normalize-by-resolution` 参数即可！?

