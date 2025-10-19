# AIGC视频动态度评估系统

## 概述

一个完整的视频动态度评估系统，支持：
- ? **统一动态度评分**：将所有指标整合为0-1标准化分数
- ? **相机运动补偿**：自动去除相机运动影响
- ? **多视角支持**：适用于静态场景和动态场景
- ? **自动场景检测**：智能识别视频类型

### 动态度分数含义

```
0.0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0
 ↑         ↑         ↑         ↑        ↑
纯静态    低动态    中等动态   高动态  极高动态
(建筑)   (旗帜)    (行人)    (跑步)  (跳舞)
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```bash
# 处理单个视频
python video_processor.py -i video.mp4 -o output/

# 批量处理
python video_processor.py -i videos/ -o results/ --batch
```

### Python API

```python
from video_processor import VideoProcessor

# 创建处理器
processor = VideoProcessor(device='cuda')

# 处理视频
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir="output")

# 获取统一动态度分数
score = result['unified_dynamics']['unified_dynamics_score']
category = result['dynamics_classification']['category']

print(f"动态度: {score:.3f} - {category}")
```

---

## 核心功能

### 1. 统一动态度评分

整合5个维度指标，输出0-1标准化分数：

| 维度 | 权重 | 含义 |
|------|------|------|
| 光流幅度 | 35% | 运动强度 |
| 空间覆盖 | 25% | 运动区域占比 |
| 时序变化 | 20% | 时间变化丰富度 |
| 空间一致性 | 10% | 运动均匀性 |
| 相机因子 | 10% | 补偿效果 |

**分类标准**：
- 0.0-0.2: 纯静态（建筑、雕塑）
- 0.2-0.4: 低动态（飘动旗帜）
- 0.4-0.6: 中等动态（行走的人）
- 0.6-0.8: 高动态（跑步、跳舞）
- 0.8-1.0: 极高动态（激烈运动）

### 2. 相机运动补偿

自动检测并去除相机运动影响：
- 使用特征匹配和RANSAC估计相机运动
- 从光流中分离相机运动和物体运动
- 适用于环绕拍摄、手持拍摄等场景

### 3. 自动场景检测

智能识别场景类型并调整评估策略：
- **静态场景**（相机运动）→ 使用残差光流
- **动态场景**（物体运动）→ 使用原始光流

---

## 命令行参数

### 基本参数

```bash
--input, -i          # 输入视频/图像目录
--output, -o         # 输出目录（默认: output）
--device             # 计算设备 cuda/cpu（默认: cuda）
--batch              # 批量处理模式
```

### 相机补偿参数

```bash
--no-camera-compensation      # 禁用相机补偿（默认启用）
--camera-ransac-thresh FLOAT  # RANSAC阈值（默认: 1.0）
--camera-max-features INT     # 最大特征点数（默认: 2000）
```

### 其他参数

```bash
--max_frames INT     # 最大处理帧数
--frame_skip INT     # 帧跳跃间隔（默认: 1）
--fov FLOAT          # 相机视场角度数（默认: 60.0）
--no-visualize       # 禁用可视化生成
```

---

## 输出结果

### JSON结果

```json
{
  "unified_dynamics_score": 0.652,
  "scene_type": "dynamic",
  "dynamics_category": "high_dynamic",
  "confidence": 0.85,
  
  "temporal_stats": {
    "mean_dynamics_score": 0.92,
    "mean_static_ratio": 0.18
  }
}
```

### 文本报告

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
统一动态度评估 (Unified Dynamics Score)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

综合动态度分数: 0.652 / 1.000
场景类型: dynamic
置信度: 85.0%

分类结果: 高动态场景
典型例子: 跑步, 跳舞, 体育运动
```

---

## 应用场景

### 1. 视频分类

```python
if score < 0.2:
    label = "静态建筑视频"
elif score < 0.5:
    label = "低动态视频"
else:
    label = "高动态视频"
```

### 2. 质量筛选

```python
# 筛选纯静态视频
static_videos = [v for v in videos if v['score'] < 0.2]

# 筛选高动态视频
dynamic_videos = [v for v in videos if v['score'] > 0.7]
```

### 3. 数据集标注

```python
# 直接使用0-1分数作为标签
dataset['dynamics_label'] = unified_score
```

---

## 高级配置

### 自定义权重

```python
from unified_dynamics_scorer import UnifiedDynamicsScorer

scorer = UnifiedDynamicsScorer(
    weights={
        'flow_magnitude': 0.5,
        'spatial_coverage': 0.3,
        'temporal_variation': 0.1,
        'spatial_consistency': 0.05,
        'camera_factor': 0.05
    }
)

processor.unified_scorer = scorer
```

### 自定义分类阈值

```python
from unified_dynamics_scorer import DynamicsClassifier

classifier = DynamicsClassifier(
    thresholds={
        'pure_static': 0.10,
        'low_dynamic': 0.30,
        'medium_dynamic': 0.60,
        'high_dynamic': 0.80
    }
)

processor.dynamics_classifier = classifier
```

---

## 项目结构

```
AIGC_detector/
├── video_processor.py              # 主处理器
├── unified_dynamics_scorer.py      # 统一动态度评分
├── static_object_analyzer.py       # 静态物体分析
├── simple_raft.py                  # RAFT光流计算
├── dynamic_motion_compensation/    # 相机补偿模块
│   ├── camera_compensation.py
│   ├── object_motion.py
│   └── se3_utils.py
├── tests/                          # 测试文件
│   ├── test_unified_dynamics.py
│   ├── test_camera_compensation.py
│   └── test_static_dynamics.py
├── requirements.txt                # 依赖
└── README.md                       # 本文档
```

---

## 测试

```bash
# 运行统一动态度测试
python tests/test_unified_dynamics.py

# 运行相机补偿测试
python tests/test_camera_compensation.py

# 运行所有测试
python -m pytest tests/
```

---

## 常见问题

### Q1: 分数与预期不符？

**解决方案**：
```python
# 调整归一化阈值
scorer = UnifiedDynamicsScorer(
    thresholds={'flow_mid': 8.0}
)

# 或指定场景模式
scorer = UnifiedDynamicsScorer(mode='static_scene')
```

### Q2: 相机补偿成功率低？

**解决方案**：
```bash
# 增加特征点
--camera-max-features 3000

# 放宽RANSAC阈值
--camera-ransac-thresh 2.0
```

### Q3: 如何禁用相机补偿？

```bash
python video_processor.py -i video.mp4 -o output/ --no-camera-compensation
```

---

## 技术特性

- **多维度融合**：整合光流、空间、时序等5个维度
- **自适应权重**：根据场景类型智能调整
- **Sigmoid归一化**：平滑映射到0-1范围
- **置信度评估**：评估结果可靠性
- **零额外开销**：基于现有数据，无需额外计算

---

## 依赖要求

- Python >= 3.7
- PyTorch >= 1.6
- OpenCV >= 4.0
- NumPy, SciPy, scikit-learn
- matplotlib, tqdm

详见 `requirements.txt`

---

## 许可证

MIT License

---

## 更新日志

### v1.0 (2025-10-19)
- ? 新增统一动态度评分系统
- ? 集成相机运动补偿
- ? 支持自动场景检测
- ? 完整的测试覆盖

---

## 联系方式

如有问题或建议，欢迎提交 Issue。

---

**快速开始**：
```bash
python video_processor.py -i your_video.mp4 -o output/
```

