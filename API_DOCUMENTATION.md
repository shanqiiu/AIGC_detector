# ? API ʹ���ĵ�

���ĵ���������Ա�̷�ʽʹ��AIGC��Ƶ��������ϵͳ��

---

## ? ����ģ��

```python
from video_processor import VideoProcessor
from badcase_detector import BadCaseDetector, BadCaseAnalyzer
from unified_dynamics_scorer import UnifiedDynamicsScorer
from static_object_analyzer import StaticObjectDynamicsCalculator
from simple_raft import SimpleRAFT
```

---

## ? VideoProcessor (��������)

### ��ʼ��

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

**����˵��**��

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `raft_model_path` | str | ���� | RAFTģ���ļ�·�� |
| `device` | str | 'cuda' | �����豸 ('cuda' �� 'cpu') |
| `max_frames` | int/None | None | �����֡����None��ʾ����ȫ�� |
| `frame_skip` | int | 1 | ֡��Ծ�����1��ʾ����֡ |
| `enable_visualization` | bool | False | �Ƿ����ɿ��ӻ���� |
| `enable_camera_compensation` | bool | True | �Ƿ������������ |
| `camera_compensation_params` | dict/None | None | ����������� |
| `use_normalized_flow` | bool | False | �Ƿ����÷ֱ��ʹ�һ�� |
| `flow_threshold_ratio` | float | 0.002 | ��һ��ģʽ�µľ�̬��ֵ |

### ���ķ���

#### 1. ������Ƶ

```python
frames = processor.load_video(
    video_path: str,
    max_frames: Optional[int] = None
) -> List[np.ndarray]
```

**ʾ��**��
```python
frames = processor.load_video("test.mp4")
print(f"������ {len(frames)} ֡")
print(f"�ֱ���: {frames[0].shape}")  # (H, W, C)
```

#### 2. ��ͼ�����м���

```python
frames = processor.extract_frames_from_images(
    image_dir: str
) -> List[np.ndarray]
```

**ʾ��**��
```python
frames = processor.extract_frames_from_images("frames/")
```

#### 3. ��������ڲ�

```python
camera_matrix = processor.estimate_camera_matrix(
    image_shape: tuple,
    fov_degrees: float = 60.0
) -> np.ndarray
```

**ʾ��**��
```python
camera_matrix = processor.estimate_camera_matrix(
    image_shape=frames[0].shape,
    fov_degrees=60.0
)
# ���� 3x3 ����ڲξ���
```

#### 4. ������Ƶ

```python
result = processor.process_video(
    frames: List[np.ndarray],
    camera_matrix: np.ndarray,
    output_dir: str,
    expected_label: Optional[str] = None
) -> Dict
```

**ʾ��**��
```python
result = processor.process_video(
    frames=frames,
    camera_matrix=camera_matrix,
    output_dir="output/",
    expected_label="high"  # ��ѡ������BadCase���
)
```

**���ؽ���ṹ**��
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
    # �������BadCase���
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

### ����ʾ��

```python
from video_processor import VideoProcessor

# 1. ��ʼ��������
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device="cuda",
    enable_camera_compensation=True,
    use_normalized_flow=True,  # �Ƽ�����
    flow_threshold_ratio=0.002
)

# 2. ������Ƶ
frames = processor.load_video("test.mp4")
print(f"������ {len(frames)} ֡���ֱ���: {frames[0].shape}")

# 3. �����������
camera_matrix = processor.estimate_camera_matrix(frames[0].shape, fov=60.0)

# 4. ������Ƶ
result = processor.process_video(
    frames=frames,
    camera_matrix=camera_matrix,
    output_dir="output/"
)

# 5. ���ʽ��
print(f"ƽ����̬��: {result['temporal_stats']['mean_dynamics_score']:.3f}")
print(f"ͳһ����: {result['unified_scores']['final_score']:.3f}")
print(f"ʱ���ȶ���: {result['temporal_stats']['temporal_stability']:.3f}")
```

---

## ? BadCaseDetector (���������)

### ��ʼ��

```python
from badcase_detector import BadCaseDetector

detector = BadCaseDetector(
    mismatch_threshold: float = 0.3
)
```

### ���ķ���

#### ���BadCase

```python
result = detector.detect_badcase(
    actual_score: float,
    expected_label: str
) -> Dict
```

**ʾ��**��
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

#### ��ǩ������ӳ��

```python
score = detector.label_to_score(label: str) -> float
level = detector.score_to_level(score: float) -> str
```

**ӳ���ϵ**��
```python
# ��ǩ �� ����
'static' / 'low'    �� 0.25
'medium'            �� 0.50
'high'              �� 0.75

# ���� �� ����
< 0.35              �� 'low'
0.35 - 0.65         �� 'medium'
> 0.65              �� 'high'
```

---

## ? UnifiedDynamicsScorer (����ϵͳ)

### ��ʼ��

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

### ���ķ���

#### ����ͳһ����

```python
score = scorer.compute_unified_score(
    mean_flow: float,
    dynamic_ratio: float,
    temporal_variation: float,
    spatial_entropy: float,
    camera_motion_magnitude: float
) -> Dict
```

**ʾ��**��
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

### �Զ���Ȩ��

```python
custom_weights = {
    'flow_magnitude': 0.40,      # ���ӹ���Ȩ��
    'spatial_coverage': 0.30,
    'temporal_variation': 0.15,
    'spatial_consistency': 0.10,
    'camera_factor': 0.05        # �������Ȩ��
}

scorer = UnifiedDynamicsScorer(
    weights=custom_weights,
    use_normalized_flow=True
)
```

### �Զ�����ֵ

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

## ? StaticObjectDynamicsCalculator (��̬������)

### ��ʼ��

```python
from static_object_analyzer import StaticObjectDynamicsCalculator

calculator = StaticObjectDynamicsCalculator(
    flow_threshold: float = 2.0,
    use_normalized_flow: bool = False,
    flow_threshold_ratio: float = 0.002
)
```

### ���ķ���

#### ���㾲̬���嶯̬��

```python
result = calculator.calculate_dynamics(
    flows: List[np.ndarray],
    image_shape: tuple
) -> Dict
```

**ʾ��**��
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

## ? SimpleRAFT (����������)

### ��ʼ��

```python
from simple_raft import SimpleRAFT

raft = SimpleRAFT(
    model_path: str,
    device: str = 'cuda'
)
```

### ���ķ���

#### ���ƹ���

```python
flow = raft.estimate_flow(
    frame1: np.ndarray,
    frame2: np.ndarray
) -> np.ndarray
```

**ʾ��**��
```python
raft = SimpleRAFT("pretrained_models/raft-things.pth", device="cuda")

flow = raft.estimate_flow(frame1, frame2)
print(f"������״: {flow.shape}")  # (H, W, 2)

# �����������
flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
print(f"ƽ������: {flow_magnitude.mean():.3f}")
```

---

## ? CameraCompensator (�������)

### ��ʼ��

```python
from dynamic_motion_compensation import CameraCompensator

compensator = CameraCompensator(
    ransac_thresh: float = 1.0,
    max_features: int = 2000
)
```

### ���ķ���

#### ��������˶�

```python
result = compensator.compensate(
    frame1: np.ndarray,
    frame2: np.ndarray,
    flow: np.ndarray,
    camera_matrix: np.ndarray
) -> Dict
```

**ʾ��**��
```python
result = compensator.compensate(frame1, frame2, flow, camera_matrix)

print(result)
# {
#     'compensated_flow': np.ndarray,  # ������Ĺ���
#     'camera_motion': np.ndarray,      # ���Ƶ�����˶�
#     'is_successful': bool,            # �Ƿ�ɹ�����
#     'num_inliers': int                # RANSAC�ڵ���
# }

compensated_flow = result['compensated_flow']
```

---

## ? �������� API

### ����������Ƶ

```python
from video_processor import batch_process_videos, load_expected_labels

# 1. ���ر�ǩ����ѡ��
labels = load_expected_labels("labels.json")

# 2. ��ʼ��������
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)

# 3. ��������
results = batch_process_videos(
    processor=processor,
    video_dir="videos/",
    output_dir="output/",
    fov=60.0,
    badcase_labels=labels  # ��ѡ
)

# 4. ���ʽ��
if 'badcase_count' in results:  # BadCaseģʽ
    print(f"����Ƶ: {results['total_videos']}")
    print(f"BadCase: {results['badcase_count']}")
    print(f"����: {results['badcase_rate']:.1%}")
else:  # ��ͨģʽ
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"�ɹ�: {success}/{len(results)}")
```

---

## ? ����������ʾ��

### ʾ��1������Ƶ����

```python
from video_processor import VideoProcessor
import json

# ��ʼ��
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device="cuda",
    use_normalized_flow=True,
    enable_camera_compensation=True
)

# ���غʹ���
frames = processor.load_video("test.mp4")
camera_matrix = processor.estimate_camera_matrix(frames[0].shape)
result = processor.process_video(frames, camera_matrix, "output/")

# �����Զ�����
custom_result = {
    'video': 'test.mp4',
    'score': result['unified_scores']['final_score'],
    'quality': 'good' if result['unified_scores']['final_score'] > 0.5 else 'bad'
}

with open('custom_result.json', 'w') as f:
    json.dump(custom_result, f, indent=2)
```

### ʾ��2���������� + �Զ������

```python
from video_processor import VideoProcessor, batch_process_videos
import os

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)

# ��������
results = batch_process_videos(
    processor, "videos/", "output/", fov=60.0
)

# �Զ�����ˣ��ҳ���������Ƶ
low_quality_videos = []
for result in results:
    if result['status'] == 'success':
        score = result['unified_scores']['final_score']
        if score < 0.3:
            low_quality_videos.append({
                'video': result['video_name'],
                'score': score
            })

# �����������Ƶ�б�
with open('low_quality_list.txt', 'w') as f:
    for item in low_quality_videos:
        f.write(f"{item['video']}: {item['score']:.3f}\n")
```

### ʾ��3���Զ������ֹ���

```python
from video_processor import VideoProcessor
from unified_dynamics_scorer import UnifiedDynamicsScorer

# �Զ���������
custom_scorer = UnifiedDynamicsScorer(
    mode='auto',
    weights={
        'flow_magnitude': 0.50,       # ����ע����
        'spatial_coverage': 0.20,
        'temporal_variation': 0.15,
        'spatial_consistency': 0.10,
        'camera_factor': 0.05
    },
    use_normalized_flow=True
)

# ʹ���Զ���������
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)
processor.dynamics_scorer = custom_scorer  # �滻Ĭ��������

# ������Ƶ
frames = processor.load_video("test.mp4")
camera_matrix = processor.estimate_camera_matrix(frames[0].shape)
result = processor.process_video(frames, camera_matrix, "output/")

print(f"�Զ�������: {result['unified_scores']['final_score']:.3f}")
```

---

## ? ������

### �쳣����

```python
try:
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth"
    )
    frames = processor.load_video("test.mp4")
    result = processor.process_video(frames, camera_matrix, "output/")
    
except FileNotFoundError as e:
    print(f"�ļ�������: {e}")
    
except RuntimeError as e:
    print(f"����ʱ���󣨿�����CUDA�ڴ治�㣩: {e}")
    
except Exception as e:
    print(f"δԤ�ڵĴ���: {e}")
```

### ����������

```python
import torch

# ���CUDA�Ƿ����
if torch.cuda.is_available():
    device = 'cuda'
    print(f"ʹ��GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("CUDA�����ã�ʹ��CPU")

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device=device
)
```

---

## ? ������ʾ

����Ŀ����֧��������ʾ�������MyPyʹ�ã�

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def my_analysis_function(
    video_path: str,
    output_dir: str,
    fov: float = 60.0
) -> Dict[str, float]:
    """
    �Զ����������
    
    Args:
        video_path: ��Ƶ�ļ�·��
        output_dir: ���Ŀ¼
        fov: �ӳ���
        
    Returns:
        ��������������ֵ�
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

## ? ������Դ

- [README.md](README.md) - ������Ŀ�ĵ�
- [QUICK_START.md](QUICK_START.md) - ���ٿ�ʼ
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - ��Ŀ����

---

<div align="center">

**��Ҫ�������ύ [Issue](https://github.com/your-repo/issues) ������ĵ�**

</div>

