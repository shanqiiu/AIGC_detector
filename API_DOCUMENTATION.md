# ? API 使用文档

本文档介绍如何以编程方式使用AIGC视频质量评估系统。

---

## ? 导入模块

```python
from video_processor import VideoProcessor
from badcase_detector import BadCaseDetector, BadCaseAnalyzer
from unified_dynamics_scorer import UnifiedDynamicsScorer
from static_object_analyzer import StaticObjectDynamicsCalculator
from simple_raft import SimpleRAFT
```

---

## ? VideoProcessor (主处理器)

### 初始化

```python
processor = VideoProcessor(
    raft_model_path: str = "pretrained_models/raft-things.pth",
    device: str = 'cuda',
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    enable_visualization: bool = False,
    enable_camera_compensation: bool = True,
    camera_compensation_params: Optional[Dict] = None,
    use_normalized_flow: bool = False,
    flow_threshold_ratio: float = 0.002
)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `raft_model_path` | str | 必需 | RAFT模型文件路径 |
| `device` | str | 'cuda' | 计算设备 ('cuda' 或 'cpu') |
| `max_frames` | int/None | None | 最大处理帧数，None表示处理全部 |
| `frame_skip` | int | 1 | 帧跳跃间隔，1表示不跳帧 |
| `enable_visualization` | bool | False | 是否生成可视化结果 |
| `enable_camera_compensation` | bool | True | 是否启用相机补偿 |
| `camera_compensation_params` | dict/None | None | 相机补偿参数 |
| `use_normalized_flow` | bool | False | 是否启用分辨率归一化 |
| `flow_threshold_ratio` | float | 0.002 | 归一化模式下的静态阈值 |

### 核心方法

#### 1. 加载视频

```python
frames = processor.load_video(
    video_path: str,
    max_frames: Optional[int] = None
) -> List[np.ndarray]
```

**示例**：
```python
frames = processor.load_video("test.mp4")
print(f"加载了 {len(frames)} 帧")
print(f"分辨率: {frames[0].shape}")  # (H, W, C)
```

#### 2. 从图像序列加载

```python
frames = processor.extract_frames_from_images(
    image_dir: str
) -> List[np.ndarray]
```

**示例**：
```python
frames = processor.extract_frames_from_images("frames/")
```

#### 3. 估计相机内参

```python
camera_matrix = processor.estimate_camera_matrix(
    image_shape: tuple,
    fov_degrees: float = 60.0
) -> np.ndarray
```

**示例**：
```python
camera_matrix = processor.estimate_camera_matrix(
    image_shape=frames[0].shape,
    fov_degrees=60.0
)
# 返回 3x3 相机内参矩阵
```

#### 4. 处理视频

```python
result = processor.process_video(
    frames: List[np.ndarray],
    camera_matrix: np.ndarray,
    output_dir: str,
    expected_label: Optional[str] = None
) -> Dict
```

**示例**：
```python
result = processor.process_video(
    frames=frames,
    camera_matrix=camera_matrix,
    output_dir="output/",
    expected_label="high"  # 可选，用于BadCase检测
)
```

**返回结果结构**：
```python
{
    'metadata': {
        'total_frames': int,
        'resolution': [width, height],
        'normalized': bool,
        'camera_compensation': bool,
        'processing_time': float
    },
    'temporal_stats': {
        'mean_dynamics_score': float,
        'std_dynamics_score': float,
        'max_dynamics_score': float,
        'min_dynamics_score': float,
        'mean_static_ratio': float,
        'temporal_stability': float
    },
    'unified_scores': {
        'final_score': float,
        'flow_magnitude_score': float,
        'spatial_coverage_score': float,
        'temporal_variation_score': float,
        'spatial_consistency_score': float,
        'camera_factor_score': float
    },
    # 如果启用BadCase检测
    'badcase_info': {
        'is_badcase': bool,
        'severity': str,  # 'severe', 'moderate', 'minor'
        'mismatch_score': float,
        'expected_label': str,
        'actual_level': str,
        'direction': str  # 'higher', 'lower'
    }
}
```

### 完整示例

```python
from video_processor import VideoProcessor

# 1. 初始化处理器
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device="cuda",
    enable_camera_compensation=True,
    use_normalized_flow=True,  # 推荐开启
    flow_threshold_ratio=0.002
)

# 2. 加载视频
frames = processor.load_video("test.mp4")
print(f"加载了 {len(frames)} 帧，分辨率: {frames[0].shape}")

# 3. 估计相机参数
camera_matrix = processor.estimate_camera_matrix(frames[0].shape, fov=60.0)

# 4. 处理视频
result = processor.process_video(
    frames=frames,
    camera_matrix=camera_matrix,
    output_dir="output/"
)

# 5. 访问结果
print(f"平均动态度: {result['temporal_stats']['mean_dynamics_score']:.3f}")
print(f"统一评分: {result['unified_scores']['final_score']:.3f}")
print(f"时序稳定性: {result['temporal_stats']['temporal_stability']:.3f}")
```

---

## ? BadCaseDetector (质量检测器)

### 初始化

```python
from badcase_detector import BadCaseDetector

detector = BadCaseDetector(
    mismatch_threshold: float = 0.3
)
```

### 核心方法

#### 检测BadCase

```python
result = detector.detect_badcase(
    actual_score: float,
    expected_label: str
) -> Dict
```

**示例**：
```python
detection_result = detector.detect_badcase(
    actual_score=0.15,
    expected_label="high"
)

print(detection_result)
# {
#     'is_badcase': True,
#     'severity': 'severe',
#     'mismatch_score': 0.60,
#     'expected_score': 0.75,
#     'actual_score': 0.15,
#     'expected_label': 'high',
#     'actual_level': 'low',
#     'direction': 'lower'
# }
```

#### 标签到分数映射

```python
score = detector.label_to_score(label: str) -> float
level = detector.score_to_level(score: float) -> str
```

**映射关系**：
```python
# 标签 → 分数
'static' / 'low'    → 0.25
'medium'            → 0.50
'high'              → 0.75

# 分数 → 级别
< 0.35              → 'low'
0.35 - 0.65         → 'medium'
> 0.65              → 'high'
```

---

## ? UnifiedDynamicsScorer (评分系统)

### 初始化

```python
from unified_dynamics_scorer import UnifiedDynamicsScorer

scorer = UnifiedDynamicsScorer(
    mode: str = 'auto',
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    use_normalized_flow: bool = False,
    baseline_diagonal: float = 1469.0
)
```

### 核心方法

#### 计算统一分数

```python
score = scorer.compute_unified_score(
    mean_flow: float,
    dynamic_ratio: float,
    temporal_variation: float,
    spatial_entropy: float,
    camera_motion_magnitude: float
) -> Dict
```

**示例**：
```python
result = scorer.compute_unified_score(
    mean_flow=0.008,
    dynamic_ratio=0.35,
    temporal_variation=0.004,
    spatial_entropy=2.5,
    camera_motion_magnitude=0.012
)

print(result)
# {
#     'final_score': 0.456,
#     'flow_magnitude_score': 0.423,
#     'spatial_coverage_score': 0.512,
#     'temporal_variation_score': 0.389,
#     'spatial_consistency_score': 0.678,
#     'camera_factor_score': 0.234
# }
```

### 自定义权重

```python
custom_weights = {
    'flow_magnitude': 0.40,      # 增加光流权重
    'spatial_coverage': 0.30,
    'temporal_variation': 0.15,
    'spatial_consistency': 0.10,
    'camera_factor': 0.05        # 减少相机权重
}

scorer = UnifiedDynamicsScorer(
    weights=custom_weights,
    use_normalized_flow=True
)
```

### 自定义阈值

```python
custom_thresholds = {
    'flow_low': 0.001,
    'flow_mid': 0.005,
    'flow_high': 0.015,
    'static_ratio': 0.6,
    'temporal_std': 0.001
}

scorer = UnifiedDynamicsScorer(
    use_normalized_flow=True,
    thresholds=custom_thresholds
)
```

---

## ? StaticObjectDynamicsCalculator (静态分析器)

### 初始化

```python
from static_object_analyzer import StaticObjectDynamicsCalculator

calculator = StaticObjectDynamicsCalculator(
    flow_threshold: float = 2.0,
    use_normalized_flow: bool = False,
    flow_threshold_ratio: float = 0.002
)
```

### 核心方法

#### 计算静态物体动态度

```python
result = calculator.calculate_dynamics(
    flows: List[np.ndarray],
    image_shape: tuple
) -> Dict
```

**示例**：
```python
result = calculator.calculate_dynamics(
    flows=flow_list,
    image_shape=(720, 1280, 3)
)

print(result)
# {
#     'per_frame_scores': [0.123, 0.145, ...],
#     'per_frame_static_ratios': [0.65, 0.58, ...],
#     'temporal_stats': {
#         'mean_dynamics_score': 0.234,
#         'std_dynamics_score': 0.082,
#         'max_dynamics_score': 0.512,
#         'min_dynamics_score': 0.089,
#         'cv_dynamics_score': 0.351,
#         'mean_static_ratio': 0.612,
#         'temporal_stability': 0.823
#     }
# }
```

---

## ? SimpleRAFT (光流估计器)

### 初始化

```python
from simple_raft import SimpleRAFT

raft = SimpleRAFT(
    model_path: str,
    device: str = 'cuda'
)
```

### 核心方法

#### 估计光流

```python
flow = raft.estimate_flow(
    frame1: np.ndarray,
    frame2: np.ndarray
) -> np.ndarray
```

**示例**：
```python
raft = SimpleRAFT("pretrained_models/raft-things.pth", device="cuda")

flow = raft.estimate_flow(frame1, frame2)
print(f"光流形状: {flow.shape}")  # (H, W, 2)

# 计算光流幅度
flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
print(f"平均光流: {flow_magnitude.mean():.3f}")
```

---

## ? CameraCompensator (相机补偿)

### 初始化

```python
from dynamic_motion_compensation import CameraCompensator

compensator = CameraCompensator(
    ransac_thresh: float = 1.0,
    max_features: int = 2000
)
```

### 核心方法

#### 补偿相机运动

```python
result = compensator.compensate(
    frame1: np.ndarray,
    frame2: np.ndarray,
    flow: np.ndarray,
    camera_matrix: np.ndarray
) -> Dict
```

**示例**：
```python
result = compensator.compensate(frame1, frame2, flow, camera_matrix)

print(result)
# {
#     'compensated_flow': np.ndarray,  # 补偿后的光流
#     'camera_motion': np.ndarray,      # 估计的相机运动
#     'is_successful': bool,            # 是否成功估计
#     'num_inliers': int                # RANSAC内点数
# }

compensated_flow = result['compensated_flow']
```

---

## ? 批量处理 API

### 批量处理视频

```python
from video_processor import batch_process_videos, load_expected_labels

# 1. 加载标签（可选）
labels = load_expected_labels("labels.json")

# 2. 初始化处理器
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)

# 3. 批量处理
results = batch_process_videos(
    processor=processor,
    video_dir="videos/",
    output_dir="output/",
    fov=60.0,
    badcase_labels=labels  # 可选
)

# 4. 访问结果
if 'badcase_count' in results:  # BadCase模式
    print(f"总视频: {results['total_videos']}")
    print(f"BadCase: {results['badcase_count']}")
    print(f"比例: {results['badcase_rate']:.1%}")
else:  # 普通模式
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"成功: {success}/{len(results)}")
```

---

## ? 完整工作流示例

### 示例1：单视频分析

```python
from video_processor import VideoProcessor
import json

# 初始化
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device="cuda",
    use_normalized_flow=True,
    enable_camera_compensation=True
)

# 加载和处理
frames = processor.load_video("test.mp4")
camera_matrix = processor.estimate_camera_matrix(frames[0].shape)
result = processor.process_video(frames, camera_matrix, "output/")

# 保存自定义结果
custom_result = {
    'video': 'test.mp4',
    'score': result['unified_scores']['final_score'],
    'quality': 'good' if result['unified_scores']['final_score'] > 0.5 else 'bad'
}

with open('custom_result.json', 'w') as f:
    json.dump(custom_result, f, indent=2)
```

### 示例2：批量处理 + 自定义过滤

```python
from video_processor import VideoProcessor, batch_process_videos
import os

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)

# 批量处理
results = batch_process_videos(
    processor, "videos/", "output/", fov=60.0
)

# 自定义过滤：找出低质量视频
low_quality_videos = []
for result in results:
    if result['status'] == 'success':
        score = result['unified_scores']['final_score']
        if score < 0.3:
            low_quality_videos.append({
                'video': result['video_name'],
                'score': score
            })

# 保存低质量视频列表
with open('low_quality_list.txt', 'w') as f:
    for item in low_quality_videos:
        f.write(f"{item['video']}: {item['score']:.3f}\n")
```

### 示例3：自定义评分规则

```python
from video_processor import VideoProcessor
from unified_dynamics_scorer import UnifiedDynamicsScorer

# 自定义评分器
custom_scorer = UnifiedDynamicsScorer(
    mode='auto',
    weights={
        'flow_magnitude': 0.50,       # 更关注光流
        'spatial_coverage': 0.20,
        'temporal_variation': 0.15,
        'spatial_consistency': 0.10,
        'camera_factor': 0.05
    },
    use_normalized_flow=True
)

# 使用自定义评分器
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)
processor.dynamics_scorer = custom_scorer  # 替换默认评分器

# 处理视频
frames = processor.load_video("test.mp4")
camera_matrix = processor.estimate_camera_matrix(frames[0].shape)
result = processor.process_video(frames, camera_matrix, "output/")

print(f"自定义评分: {result['unified_scores']['final_score']:.3f}")
```

---

## ? 错误处理

### 异常类型

```python
try:
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth"
    )
    frames = processor.load_video("test.mp4")
    result = processor.process_video(frames, camera_matrix, "output/")
    
except FileNotFoundError as e:
    print(f"文件不存在: {e}")
    
except RuntimeError as e:
    print(f"运行时错误（可能是CUDA内存不足）: {e}")
    
except Exception as e:
    print(f"未预期的错误: {e}")
```

### 常见错误处理

```python
import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = 'cuda'
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("CUDA不可用，使用CPU")

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device=device
)
```

---

## ? 类型提示

本项目完整支持类型提示，可配合MyPy使用：

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def my_analysis_function(
    video_path: str,
    output_dir: str,
    fov: float = 60.0
) -> Dict[str, float]:
    """
    自定义分析函数
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        fov: 视场角
        
    Returns:
        包含分析结果的字典
    """
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        use_normalized_flow=True
    )
    
    frames: List[np.ndarray] = processor.load_video(video_path)
    camera_matrix: np.ndarray = processor.estimate_camera_matrix(
        frames[0].shape, fov
    )
    result: Dict = processor.process_video(frames, camera_matrix, output_dir)
    
    return {
        'score': result['unified_scores']['final_score'],
        'stability': result['temporal_stats']['temporal_stability']
    }
```

---

## ? 更多资源

- [README.md](README.md) - 完整项目文档
- [QUICK_START.md](QUICK_START.md) - 快速开始
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - 项目概览

---

<div align="center">

**需要帮助？提交 [Issue](https://github.com/your-repo/issues) 或查阅文档**

</div>

