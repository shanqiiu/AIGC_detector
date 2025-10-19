# BadCase检测使用指南

## 功能说明

BadCase检测器用于筛选劣质AIGC生成视频，检测两类质量问题：

### 类型A：期望静态→实际高动态
**场景举例**：
- 建筑物视频出现抖动、飘移
- 静物产品视频有异常运动
- 风景视频中静态元素在晃动

**检测逻辑**：
```
期望标签: static (0.0)
实际动态度: > 0.5
判定: BadCase - static_to_dynamic
```

### 类型B：期望动态→实际低动态
**场景举例**：
- 演唱会大屏幕应该动态但保持静止
- 人物舞蹈视频中人物僵硬不动
- 体育运动视频帧率过低或卡顿

**检测逻辑**：
```
期望标签: dynamic (1.0)
实际动态度: < 0.4
判定: BadCase - dynamic_to_static
```

---

## 快速使用

### 步骤1：准备期望标签文件

创建 `expected_labels.json`：

```json
{
  "building_video_1": "static",
  "building_video_2": 0.0,
  "dance_video_1": "dynamic",
  "dance_video_2": 1.0,
  "concert_video": "dynamic",
  "walking_video": 0.5
}
```

**标签格式说明**：
- `"static"` 或 `0.0`：期望纯静态
- `"dynamic"` 或 `1.0`：期望高动态
- `0.0-1.0`：期望的具体动态度分数

### 步骤2：运行BadCase检测

```bash
python batch_with_badcase.py \
  --input videos/ \
  --labels expected_labels.json \
  --output badcase_output/ \
  --device cuda
```

### 步骤3：查看结果

```bash
# 查看总结报告
cat badcase_output/badcase_summary.txt

# 查看BadCase视频列表
cat badcase_output/badcase_videos.txt

# 查看JSON详细结果
cat badcase_output/badcase_summary.json
```

---

## 输出结果

### 1. 总结报告 (badcase_summary.txt)

```
======================================================================
劣质视频检测总结 (BadCase Detection Summary)
======================================================================

总视频数: 50
成功处理: 48
处理失败: 2

BadCase数量: 12
正常视频数: 36
BadCase比例: 25.0%

======================================================================
BadCase类型分布:
======================================================================
  期望静态→实际动态（如建筑抖动）: 7
  期望动态→实际静态（如屏幕静止）: 5

======================================================================
严重程度分布:
======================================================================
  severe: 3
  moderate: 5
  mild: 4

======================================================================
BadCase详细列表:
======================================================================

1. building_shaking_001
   类型: static_to_dynamic
   严重程度: severe
   期望: static
   实际动态度: 0.782
   不匹配度: 0.782
   置信度: 89.5%
```

### 2. BadCase视频列表 (badcase_videos.txt)

```
D:\videos\building_shaking_001.mp4
D:\videos\concert_frozen_screen_003.mp4
D:\videos\statue_drifting_005.mp4
...
```

### 3. JSON详细结果 (badcase_summary.json)

```json
{
  "total_videos": 50,
  "badcase_count": 12,
  "badcase_rate": 0.25,
  "type_distribution": {
    "static_to_dynamic": 7,
    "dynamic_to_static": 5
  },
  "badcase_list": [
    {
      "video_name": "building_shaking_001",
      "badcase_type": "static_to_dynamic",
      "severity": "severe",
      "expected_score": 0.0,
      "actual_score": 0.782,
      "mismatch_score": 0.782,
      "confidence": 0.895
    }
  ]
}
```

---

## Python API使用

### 方法1：单个视频检测

```python
from video_processor import VideoProcessor
from badcase_detector import BadCaseAnalyzer

# 创建处理器
processor = VideoProcessor(device='cuda')

# 处理视频
frames = processor.load_video("building_video.mp4")
result = processor.process_video(frames, output_dir="output")

# BadCase检测
analyzer = BadCaseAnalyzer()
badcase_result = analyzer.analyze_with_details(
    result,
    expected_label='static'  # 期望静态
)

# 查看结果
if badcase_result['is_badcase']:
    print(f"?? 检测到BadCase!")
    print(f"类型: {badcase_result['badcase_type']}")
    print(f"不匹配度: {badcase_result['mismatch_score']:.3f}")
    print(badcase_result['description'])
else:
    print("? 质量正常")
```

### 方法2：批量检测

```python
from badcase_detector import BadCaseDetector

# 创建检测器
detector = BadCaseDetector(mismatch_threshold=0.3)

# 批量检测
results = [...]  # 多个视频的处理结果
expected_labels = ['static', 'dynamic', 0.5, ...]  # 对应的期望标签
video_names = ['video1', 'video2', 'video3', ...]

batch_result = detector.batch_detect(results, expected_labels, video_names)

# 查看统计
print(f"BadCase数量: {batch_result['badcase_count']}")
print(f"BadCase比例: {batch_result['badcase_rate']:.1%}")
```

### 方法3：质量过滤

```python
from badcase_detector import QualityFilter

# 创建过滤器
filter = QualityFilter(accept_mismatch=0.3)

# 准备数据
video_results = [
    ('video1.mp4', result1, 'static'),
    ('video2.mp4', result2, 'dynamic'),
    ...
]

# 过滤BadCase
good_videos, bad_videos = filter.filter_videos(video_results, keep_mode='good')

print(f"正常视频: {len(good_videos)}")
print(f"BadCase: {len(bad_videos)}")

# 只保留BadCase（用于人工review）
badcase_videos, _ = filter.filter_videos(video_results, keep_mode='bad')
```

---

## 高级功能

### 1. 筛选特定类型的BadCase

```python
from badcase_detector import BadCaseDetector

detector = BadCaseDetector()
batch_result = detector.batch_detect(results, labels, names)

# 只筛选"期望静态→实际动态"的BadCase
static_badcases = detector.filter_badcases(
    batch_result,
    badcase_types=['static_to_dynamic']
)

print(f"建筑抖动类BadCase: {len(static_badcases)}")

# 只筛选严重的BadCase
severe_badcases = detector.filter_badcases(
    batch_result,
    severity_levels=['severe']
)

print(f"严重BadCase: {len(severe_badcases)}")
```

### 2. 详细诊断

```python
from badcase_detector import BadCaseAnalyzer

analyzer = BadCaseAnalyzer()
badcase_result = analyzer.analyze_with_details(result, 'static')

if badcase_result['is_badcase'] and 'diagnosis' in badcase_result:
    diagnosis = badcase_result['diagnosis']
    print(f"主要问题: {diagnosis['primary_issue']}")
    print(f"贡献因素:")
    for factor in diagnosis['contributing_factors']:
        print(f"  - {factor}")
```

### 3. 导出BadCase列表

```python
from badcase_detector import BadCaseAnalyzer

analyzer = BadCaseAnalyzer()

# 导出为不同格式
analyzer.export_badcase_list(batch_result, 'badcases.json', format='json')
analyzer.export_badcase_list(batch_result, 'badcases.txt', format='txt')
analyzer.export_badcase_list(batch_result, 'badcases.csv', format='csv')
```

---

## 参数配置

### 关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `mismatch_threshold` | 0.3 | 不匹配阈值 | 要求严格→0.2；宽松→0.4 |
| `confidence_threshold` | 0.6 | 最低置信度 | 高质量视频→0.7；低质量→0.5 |

### 调整检测灵敏度

**更严格（减少漏检）**：
```python
detector = BadCaseDetector(
    mismatch_threshold=0.2,  # 降低阈值
    confidence_threshold=0.5  # 降低置信度要求
)
```

**更宽松（减少误检）**：
```python
detector = BadCaseDetector(
    mismatch_threshold=0.4,  # 提高阈值
    confidence_threshold=0.7  # 提高置信度要求
)
```

---

## 应用场景

### 场景1：大规模视频质量筛选

```bash
# 处理1000个视频，筛选出BadCase
python batch_with_badcase.py \
  -i generated_videos/ \
  -l video_labels.json \
  -o quality_check/ \
  --device cuda

# 只保留BadCase结果（节省存储）
python batch_with_badcase.py \
  -i generated_videos/ \
  -l video_labels.json \
  -o quality_check/ \
  --filter-badcase-only
```

### 场景2：人工Review辅助

```python
# 自动筛选需要人工review的视频
filter = QualityFilter(accept_mismatch=0.3)
badcase_videos, _ = filter.filter_videos(results, keep_mode='bad')

# 只review严重的BadCase
severe_cases = detector.filter_badcases(
    batch_result,
    severity_levels=['severe', 'moderate']
)

print(f"需要人工review的视频: {len(severe_cases)}")
```

### 场景3：模型训练数据清洗

```python
# 筛选出质量正常的视频用于训练
good_videos, bad_videos = filter.filter_videos(results, keep_mode='good')

# 保存清洗后的视频列表
with open('clean_dataset.txt', 'w') as f:
    for video in good_videos:
        f.write(f"{video}\n")
```

---

## 常见BadCase示例

### 示例1：建筑抖动

```
视频: ancient_temple.mp4
期望: static (0.0)
实际: 0.78
判定: BadCase - static_to_dynamic (severe)

根因诊断:
- 主要问题: flow_magnitude 异常偏高
- 贡献因素:
  - 光流幅度过大（可能有抖动或飘移）
  - 相机补偿失败率高（特征匹配问题）

建议:
1. 检查视频稳定性，是否存在抖动
2. 验证相机补偿是否正常工作
3. 查看可视化结果，定位异常运动区域
4. 考虑重新生成视频
```

### 示例2：演唱会屏幕静止

```
视频: concert_bigscreen.mp4
期望: dynamic (1.0)
实际: 0.25
判定: BadCase - dynamic_to_static (moderate)

根因诊断:
- 主要问题: 整体运动不足
- 贡献因素:
  - 光流幅度过小（运动幅度不足）
  - 运动区域覆盖小（局部静止）
  - 时序变化小（运动单调或静止）

建议:
1. 检查人物动作是否生成正确
2. 查看关键帧，确认是否存在静止画面
3. 检查大屏幕等应动态区域是否正常
4. 考虑调整生成参数或重新生成
```

---

## 实际使用流程

### 完整工作流程

```bash
# 1. 准备标签文件
vim video_labels.json
# 或从数据库导出标签

# 2. 批量处理与检测
python batch_with_badcase.py \
  -i videos/ \
  -l video_labels.json \
  -o badcase_results/ \
  --device cuda

# 3. 查看BadCase总结
cat badcase_results/badcase_summary.txt

# 4. 人工review严重BadCase
# 根据badcase_videos.txt中的列表逐个检查

# 5. 决策
# - 重新生成BadCase视频
# - 或调整模型参数
# - 或标记为不合格
```

---

## 参数调优指南

### 不匹配阈值调优

**测试不同阈值的效果**：

```python
thresholds = [0.2, 0.3, 0.4, 0.5]
for thresh in thresholds:
    detector = BadCaseDetector(mismatch_threshold=thresh)
    result = detector.batch_detect(results, labels, names)
    print(f"阈值{thresh}: BadCase率={result['badcase_rate']:.1%}")

# 输出示例:
# 阈值0.2: BadCase率=35.2% (太敏感)
# 阈值0.3: BadCase率=25.0% (合适) ?
# 阈值0.4: BadCase率=15.8% (太宽松)
```

**推荐设置**：
- 严格筛选（减少漏检）：0.2
- 平衡（推荐）：0.3
- 宽松筛选（减少误检）：0.4

---

## 常见问题

### Q1: 如何处理没有标签的视频？

**方案1**：跳过BadCase检测
```bash
# 只处理有标签的视频
python batch_with_badcase.py -i videos/ -l labels.json -o output/
# 无标签的视频会被跳过
```

**方案2**：使用默认标签
```python
# 在代码中设置默认标签
expected = labels.get(video_name, 'dynamic')  # 默认期望动态
```

### Q2: BadCase率过高怎么办？

**可能原因**：
1. 阈值设置过严（降低mismatch_threshold）
2. 标签标注不准确（review标签）
3. 视频生成质量确实有问题（改进生成模型）

**诊断方法**：
```python
# 查看不匹配度分布
mismatch_scores = [bc['mismatch_score'] for bc in badcases]
print(f"不匹配度均值: {np.mean(mismatch_scores):.3f}")
print(f"不匹配度中位数: {np.median(mismatch_scores):.3f}")

# 如果中位数接近阈值，说明阈值可能偏严
```

### Q3: 如何验证检测准确性？

**方法**：人工抽样验证

```python
# 随机抽取20个BadCase
import random
sample_badcases = random.sample(badcases, min(20, len(badcases)))

# 人工review，统计准确率
correct = 0
for bc in sample_badcases:
    video_path = bc['video_path']
    # 人工观看视频
    is_actually_bad = input(f"视频 {video_path} 是否确实有问题? (y/n): ")
    if is_actually_bad.lower() == 'y':
        correct += 1

precision = correct / len(sample_badcases)
print(f"BadCase检测精准率: {precision:.1%}")
```

---

## 实战案例

### 案例1：大规模数据集清洗

**场景**：有5000个AIGC生成视频，需要筛选出质量问题视频

```bash
# 1. 批量处理
python batch_with_badcase.py \
  -i aigc_dataset/ \
  -l dataset_labels.json \
  -o quality_check/ \
  --device cuda \
  --mismatch-threshold 0.3

# 2. 导出BadCase列表
# quality_check/badcase_videos.txt

# 3. 统计结果
# 总视频: 5000
# BadCase: 876 (17.5%)
# - 静态→动态: 523
# - 动态→静态: 353
```

**决策**：
- 重新生成876个BadCase视频
- 或标记为低质量样本

### 案例2：模型调优验证

**场景**：调整生成模型参数后，验证改进效果

```python
# 调整前
before_result = batch_detect(before_videos, labels)
print(f"调整前BadCase率: {before_result['badcase_rate']:.1%}")

# 调整后
after_result = batch_detect(after_videos, labels)
print(f"调整后BadCase率: {after_result['badcase_rate']:.1%}")

# 对比
improvement = before_result['badcase_rate'] - after_result['badcase_rate']
print(f"改进: {improvement:.1%}")
```

---

## 与现有系统集成

BadCase检测已完全集成到现有系统中：

```python
# video_processor.py 已包含
self.badcase_detector = BadCaseDetector()
self.badcase_analyzer = BadCaseAnalyzer()

# 可直接使用
processor = VideoProcessor(device='cuda')
# BadCase检测器已自动初始化
```

---

## 总结

### 核心优势

? **自动化检测**：无需人工逐个观看  
? **精准定位**：明确指出问题类型  
? **根因诊断**：分析具体哪个维度异常  
? **批量处理**：支持大规模数据集  
? **灵活配置**：可调整检测灵敏度

### 适用场景

- ? AIGC视频质量控制
- ? 数据集清洗与筛选
- ? 模型效果验证
- ?? 人工review辅助

---

**开始使用**：
```bash
python batch_with_badcase.py -i videos/ -l labels.json -o output/
```

