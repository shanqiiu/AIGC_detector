# 归一化后的阈值调整指南

## ? 核心问题

**归一化改变了数值范围**：

| 指标 | 归一化前 | 归一化后 | 变化 |
|------|---------|---------|------|
| mean_dynamics_score | 0.5 ~ 10.0 像素 | 0.0003 ~ 0.007 | 缩小 1000+ 倍 |
| flow_magnitude | 1 ~ 30 像素 | 0.0007 ~ 0.02 | 缩小 1000+ 倍 |

**影响**：所有基于绝对值的阈值都失效了！

---

## ? 需要调整的阈值

### 1. ? StaticObjectDetector - 已调整

```python
# 原阈值（绝对值）
flow_threshold = 2.0  # 像素

# 新阈值（相对值）
flow_threshold_ratio = 0.002  # 相对于对角线
```

**状态**：? 已实现

---

### 2. ?? UnifiedDynamicsScorer - 需要调整

#### 当前代码（基于像素值）

```python
self.default_thresholds = {
    'flow_low': 1.0,      # 低动态阈值（像素/帧）
    'flow_mid': 5.0,      # 中等动态阈值
    'flow_high': 15.0,    # 高动态阈值
    'static_ratio': 0.5,  # 静态区域判断阈值
}

# 使用示例
score = self._sigmoid_normalize(
    raw_value,  # mean_dynamics_score，如果归一化会是 0.003
    threshold=self.thresholds['flow_mid'],  # 5.0 - 太大！
    steepness=0.5
)
```

**问题**：
- 如果 `mean_dynamics_score = 0.003`（归一化值）
- 但阈值仍是 `5.0`（绝对值）
- sigmoid 输出 ≈ 0.0（错误！）

#### 解决方案

```python
class UnifiedDynamicsScorer:
    def __init__(self, 
                 mode: str = 'auto',
                 use_normalized_flow: bool = False,  # 新增
                 ...):
        
        if use_normalized_flow:
            # 归一化阈值（相对值，对于1280×720 diagonal≈1469）
            self.default_thresholds = {
                'flow_low': 0.0007,   # 1.0 / 1469
                'flow_mid': 0.0034,   # 5.0 / 1469
                'flow_high': 0.0102,  # 15.0 / 1469
                'static_ratio': 0.5,  # 不变
            }
        else:
            # 绝对阈值（像素值）
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
```

---

### 3. ? DynamicsClassifier - 不需要调整

```python
self.default_thresholds = {
    'pure_static': 0.15,    # unified_score的阈值
    'low_dynamic': 0.35,
    'medium_dynamic': 0.60,
    'high_dynamic': 0.85,
}
```

**原因**：这些是 `unified_score` 的阈值（始终0-1范围），不受底层归一化影响。

**状态**：? 无需修改

---

### 4. ?? BadCaseDetector - 理论不变，实际可能需要微调

```python
mismatch_threshold = 0.3  # 动态度分数差异
```

**理论**：
- 比较的是 `unified_score`（0-1范围）
- 理论上不受底层归一化影响

**实际**：
- 如果 `unified_score` 的计算方式改变
- 可能需要根据实际效果微调

**建议**：保持 0.3，观察效果后再调整

---

## ? 阈值转换公式

### 基准分辨率法（推荐）

选择一个基准分辨率（如 1280×720），计算转换因子：

```python
# 基准分辨率
baseline_w, baseline_h = 1280, 720
baseline_diagonal = sqrt(1280? + 720?) ≈ 1469

# 转换公式
normalized_threshold = absolute_threshold / baseline_diagonal

# 示例
flow_low:  1.0 / 1469 ≈ 0.00068
flow_mid:  5.0 / 1469 ≈ 0.0034
flow_high: 15.0 / 1469 ≈ 0.0102
```

### 实际分辨率范围法

基于您的实际视频分辨率范围计算：

```python
# 您的分辨率范围
resolutions = [(1280, 720), (750, 960), (1080, 1920), ...]

# 计算平均对角线
diagonals = [sqrt(w? + h?) for w, h in resolutions]
avg_diagonal = mean(diagonals)

# 转换阈值
normalized_threshold = absolute_threshold / avg_diagonal
```

---

## ? 推荐阈值配置

### 方案A：自动适配（推荐）?

在 `UnifiedDynamicsScorer` 中自动检测归一化状态：

```python
class UnifiedDynamicsScorer:
    def __init__(self,
                 mode: str = 'auto',
                 use_normalized_flow: bool = False):
        
        # 根据归一化状态选择阈值
        if use_normalized_flow:
            baseline_diagonal = 1469.0  # 1280×720基准
            self.default_thresholds = {
                'flow_low': 1.0 / baseline_diagonal,   # ≈ 0.00068
                'flow_mid': 5.0 / baseline_diagonal,   # ≈ 0.0034
                'flow_high': 15.0 / baseline_diagonal, # ≈ 0.0102
                'static_ratio': 0.5,
            }
        else:
            # 原有阈值
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
```

### 方案B：用户指定

添加命令行参数：

```bash
--unified-flow-mid <float>     # UnifiedScorer的中等动态阈值
--unified-flow-high <float>    # UnifiedScorer的高动态阈值
```

---

## ?? 实施步骤

### Step 1: 修改 UnifiedDynamicsScorer

```python
# unified_dynamics_scorer.py

class UnifiedDynamicsScorer:
    def __init__(self,
                 mode: str = 'auto',
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 use_normalized_flow: bool = False):  # 新增参数
        
        self.mode = mode
        self.use_normalized_flow = use_normalized_flow
        
        # 根据归一化状态设置默认阈值
        if use_normalized_flow:
            # 归一化阈值（基于1280×720，diagonal≈1469）
            baseline_diagonal = 1469.0
            self.default_thresholds = {
                'flow_low': 1.0 / baseline_diagonal,    # 0.00068
                'flow_mid': 5.0 / baseline_diagonal,    # 0.0034
                'flow_high': 15.0 / baseline_diagonal,  # 0.0102
                'static_ratio': 0.5,
            }
        else:
            # 绝对阈值（像素值）
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
        
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds
```

### Step 2: 修改 VideoProcessor

```python
# video_processor.py

self.unified_scorer = UnifiedDynamicsScorer(
    mode='static_scene',
    use_normalized_flow=use_normalized_flow  # 传递归一化状态
)
```

### Step 3: 添加命令行参数（可选）

```python
# 高级用户可手动指定阈值
parser.add_argument('--unified-thresholds', type=str,
                   help='统一评分器阈值（JSON格式）')

# 使用
if args.unified_thresholds:
    thresholds = json.loads(args.unified_thresholds)
    processor.unified_scorer.thresholds = thresholds
```

---

## ? 阈值转换表

### 基于1280×720（diagonal = 1469）

| 阈值名称 | 绝对值（像素） | 归一化值（比例） | 说明 |
|---------|--------------|----------------|------|
| `flow_low` | 1.0 | 0.00068 | 低动态阈值 |
| `flow_mid` | 5.0 | 0.0034 | 中等动态阈值 |
| `flow_high` | 15.0 | 0.0102 | 高动态阈值 |
| `static_ratio` | 0.5 | 0.5 | 不变（已是比例） |

### 不同基准分辨率的转换

| 基准分辨率 | 对角线 | flow_mid绝对值 | flow_mid归一化值 |
|-----------|-------|--------------|----------------|
| 1920×1080 | 2203 | 5.0 | 0.00227 |
| 1280×720 | 1469 | 5.0 | 0.00340 |
| 640×360 | 734 | 5.0 | 0.00681 |

**建议**：统一使用 1280×720 作为基准（中等分辨率）

---

## ? 快速修复方案

### 最小改动方案（推荐）

只需修改 `video_processor.py` 初始化部分：

```python
# 初始化统一动态度评分器（传递归一化状态）
self.unified_scorer = UnifiedDynamicsScorer(
    mode='static_scene',
    use_normalized_flow=use_normalized_flow  # 新增
)
```

然后在 `UnifiedDynamicsScorer.__init__` 中根据 `use_normalized_flow` 调整阈值。

---

## ? 验证方法

### 检查阈值是否正确

```python
# 处理一个视频
python video_processor.py -i video.mp4 --normalize-by-resolution

# 查看输出
{
  "unified_dynamics": {
    "unified_dynamics_score": 0.58,  # 应该在 0-1 范围内
    "component_scores": {
      "flow_magnitude": 0.45  # 应该在 0-1 范围内
    }
  }
}
```

**正常情况**：
- ? `unified_dynamics_score` 在 0-1 范围
- ? `component_scores` 在 0-1 范围
- ? 分类结果合理（静态/动态）

**异常情况**（阈值未调整）：
- ? 所有视频 `unified_score` ≈ 0.0（阈值太大）
- ? 或 `unified_score` ≈ 1.0（阈值太小）

---

## ? 完整阈值清单

### 受归一化影响（需要调整）??

| 模块 | 阈值 | 原值 | 归一化值 | 状态 |
|------|------|------|---------|------|
| StaticObjectDetector | flow_threshold | 2.0 px | 0.0014 | ? 已调整 |
| UnifiedDynamicsScorer | flow_low | 1.0 px | 0.00068 | ?? 待调整 |
| UnifiedDynamicsScorer | flow_mid | 5.0 px | 0.0034 | ?? 待调整 |
| UnifiedDynamicsScorer | flow_high | 15.0 px | 0.0102 | ?? 待调整 |
| UnifiedDynamicsScorer | temporal_std | 1.0 px | 0.00068 | ?? 待调整 |

### 不受影响（无需调整）?

| 模块 | 阈值 | 值 | 说明 |
|------|------|----|----|
| UnifiedDynamicsScorer | static_ratio | 0.5 | 比例值 |
| DynamicsClassifier | pure_static | 0.15 | unified_score阈值 |
| DynamicsClassifier | low_dynamic | 0.35 | unified_score阈值 |
| DynamicsClassifier | medium_dynamic | 0.60 | unified_score阈值 |
| DynamicsClassifier | high_dynamic | 0.85 | unified_score阈值 |
| BadCaseDetector | mismatch_threshold | 0.3 | unified_score差值 |

---

## ? 实施优先级

### P0 - 必须修改（立即）

? **StaticObjectDetector** - 已完成
- flow_threshold → flow_threshold_ratio

?? **UnifiedDynamicsScorer** - 待实施
- 添加 `use_normalized_flow` 参数
- 根据归一化状态选择阈值

### P1 - 建议修改（短期）

? **文档更新**
- 说明归一化对阈值的影响
- 提供阈值转换工具/表格

? **验证测试**
- 测试归一化前后的 unified_score 分布
- 确认分类结果合理性

### P2 - 可选优化（长期）

? **自适应阈值**
- 根据视频分辨率自动调整
- 提供可视化阈值调优工具

---

## ? 推荐配置

### 临时解决方案（在修改 UnifiedDynamicsScorer 前）

```bash
# 方案1：暂时不启用归一化，保持原有阈值
python batch_with_badcase.py -i videos/ -l labels.json
# 缺点：分辨率不公平问题仍存在

# 方案2：启用归一化 + 手动调整统一评分器阈值
# （需要先实施下面的修改）
```

### 完整解决方案（修改后）

```bash
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
# 所有阈值自动适配
```

---

## ? 如何判断阈值是否需要调整

### 检查 unified_dynamics_score 分布

```bash
# 处理一批视频
python batch_with_badcase.py -i videos/ -l labels.json --normalize-by-resolution

# 查看结果
cat output/badcase_summary.json
```

**正常分布**（阈值正确）：
```json
[
  {"unified_score": 0.15, ...},  // 静态
  {"unified_score": 0.42, ...},  // 低动态
  {"unified_score": 0.68, ...},  // 中等动态
  {"unified_score": 0.88, ...},  // 高动态
]
```

**异常分布**（阈值错误）：
```json
// 所有视频都是 0.0 或 1.0
[
  {"unified_score": 0.0001, ...},  // ? 阈值太大
  {"unified_score": 0.0002, ...},
  {"unified_score": 0.0001, ...},
]
```

---

## ? 立即行动建议

### 1. 修改 unified_dynamics_scorer.py

添加归一化感知：

```python
class UnifiedDynamicsScorer:
    def __init__(self,
                 mode: str = 'auto',
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 use_normalized_flow: bool = False,
                 baseline_diagonal: float = 1469.0):
        
        self.use_normalized_flow = use_normalized_flow
        
        # 自动调整阈值
        if use_normalized_flow and thresholds is None:
            self.default_thresholds = {
                'flow_low': 1.0 / baseline_diagonal,
                'flow_mid': 5.0 / baseline_diagonal,
                'flow_high': 15.0 / baseline_diagonal,
                'static_ratio': 0.5,
            }
        else:
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
        
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds
```

### 2. 修改 video_processor.py

传递归一化状态：

```python
self.unified_scorer = UnifiedDynamicsScorer(
    mode='static_scene',
    use_normalized_flow=use_normalized_flow  # 传递状态
)
```

### 3. 测试验证

```bash
# 测试相同视频，比较归一化前后
python video_processor.py -i test.mp4 -o output1/
python video_processor.py -i test.mp4 -o output2/ --normalize-by-resolution

# 比较 unified_dynamics_score 是否合理
```

---

## ? 总结

| 阈值类型 | 是否需要调整 | 状态 | 优先级 |
|---------|------------|------|--------|
| StaticObjectDetector.flow_threshold | ? 是 | ? 已完成 | P0 |
| UnifiedDynamicsScorer.flow_* | ? 是 | ?? 待实施 | **P0** |
| DynamicsClassifier.* | ? 否 | ? 无需修改 | - |
| BadCaseDetector.mismatch | ?? 可能 | ? 待观察 | P1 |

**关键**：`UnifiedDynamicsScorer` 的阈值调整是启用归一化的**前提条件**，否则评分会失效。

**建议**：我立即帮您实施 UnifiedDynamicsScorer 的阈值自适应？

