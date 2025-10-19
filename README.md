# AIGC��Ƶ��������ϵͳ

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> ���ڹ���������AIGC��Ƶ����������BadCase���ϵͳ

## ? ��Ŀ���

����Ŀ��һ��רΪAIGC��AI Generated Content����Ƶ��Ƶ��Զ�����������ϵͳ��ͨ��������Ƶ�о�̬������쳣�˶�����������������⡣���Ĵ�������**�ֱ��ʹ�ƽ��һ��**��**����˶�����**�������ܹ���ƽ��������ͬ�ֱ�����Ƶ�Ķ�̬������

### ��������

- ? **���ܹ�������** - ����RAFT�ĸ߾��ȹ�������
- ? **�ֱ��ʹ�һ��** - ֧�ֻ�Ϸֱ�����Ƶ�Ĺ�ƽ����
- ? **����˶�����** - ��������˶��������˶�
- ? **BadCase�Զ����** - ���ڶ�̬�Ȳ�ƥ�����������ʶ��
- ? **���ӻ�����** - �ḻ�Ŀ��ӻ���������ڵ��Ժͷ���
- ? **��������** - ��Ч��������Ƶ��������
- ? **�������** - ȫ��Ĳ���������ѡ��

---

## ? ���ٿ�ʼ

### ����Ҫ��

- Python 3.8+
- CUDA 10.2+ (�Ƽ�������GPU����)
- 8GB+ RAM
- 2GB+ GPU�Դ棨ʹ��GPUʱ��

### ��װ����

#### 1. ��¡��Ŀ

```bash
git clone <repository_url>
cd AIGC_detector
```

#### 2. �������⻷�����Ƽ���

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ��
venv\Scripts\activate     # Windows
```

#### 3. ��װ����

```bash
pip install -r requirements.txt
```

#### 4. ����RAFTԤѵ��ģ��

���� [raft-things.pth](https://drive.google.com/file/d/1x1FLCHaGFn_Tr4wMo5f9NLPwKKGDtDa7/view?usp=sharing) �����õ� `pretrained_models/` Ŀ¼��

```bash
mkdir -p pretrained_models
# �����ص� raft-things.pth ���õ���Ŀ¼
```

### �����÷�

#### ����Ƶ����

```bash
python video_processor.py -i video.mp4 -o output/
```

#### ��������

```bash
python video_processor.py -i videos/ -o output/ --batch
```

#### BadCase��⣨�Ƽ���

```bash
python video_processor.py \
    -i videos/ \
    -o output/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

---

## ? ��ϸʹ��˵��

### ʹ�ó���

#### ����1��������Ƶ��������

��������ϸ����������Ƶ�Ķ�̬������

```bash
python video_processor.py \
    -i test_video.mp4 \
    -o results/ \
    --visualize \
    --normalize-by-resolution
```

**���**��
```
results/
������ analysis_report.txt          # �ı���������
������ analysis_results.json        # JSON��ʽ���
������ visualizations/              # ���ӻ����
    ������ frame_0000_analysis.png
    ������ static_ratio_changes.png
    ������ temporal_dynamics.png
    ������ camera_compensation_comparison.png
```

#### ����2��������Ƶ����

�����ڴ�����Ƶ����������

```bash
python video_processor.py \
    -i video_folder/ \
    -o batch_results/ \
    --batch \
    --normalize-by-resolution
```

**���**��
```
batch_results/
������ batch_summary.txt            # ��������
������ batch_summary.json           # JSON��ʽ����
������ video1/                      # ������Ƶ���
��   ������ analysis_report.txt
��   ������ analysis_results.json
������ video2/
    ������ ...
```

#### ����3��BadCase��⣨���Ĺ��ܣ�

���������������Զ����ͷ��ࡣ

```bash
python video_processor.py \
    -i videos/ \
    -o badcase_output/ \
    --batch \
    --badcase-labels labels.json \
    --mismatch-threshold 0.3 \
    --normalize-by-resolution
```

**��ǩ�ļ���ʽ** (`labels.json`):
```json
{
  "video_name1": "high",
  "video_name2": "low",
  "video_name3": "medium"
}
```

**���**��
```
badcase_output/
������ badcase_summary.txt          # BadCase����
������ badcase_summary.json         # JSON��ʽ
������ badcase_videos.txt           # BadCase��Ƶ�б�
������ video_name/
    ������ analysis_report.txt
    ������ analysis_results.json
    ������ badcase_report.txt       # BadCase��ϸ����
```

---

## ?? ��������

### ��������

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `-i, --input` | string | **����** | ������Ƶ�ļ���Ŀ¼ |
| `-o, --output` | string | `output` | ���Ŀ¼ |
| `-m, --raft_model` | string | `pretrained_models/raft-things.pth` | RAFTģ��·�� |
| `--device` | string | `cuda` | �����豸 (cuda/cpu) |
| `--batch` | flag | False | ��������ģʽ |

### BadCase������

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--badcase-labels, -l` | string | None | ������ǩ�ļ���JSON��ʽ�� |
| `--mismatch-threshold` | float | `0.3` | BadCase��ƥ����ֵ |

### �ֱ��ʹ�һ����������Ҫ��?

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--normalize-by-resolution` | flag | False | **���÷ֱ��ʹ�һ����ǿ���Ƽ���** |
| `--flow-threshold-ratio` | float | `0.002` | ��һ����ľ�̬��ֵ���� |

> **Ϊʲô��Ҫ��һ����**  
> ��ͬ�ֱ�����Ƶ����1280x720 vs 750x960���Ĺ���ֵ��Χ����޴�ֱ�ӱȽϻᵼ����������ƽ����һ��ͨ���Խ��߾����׼����ȷ�����ֿɱ��ԡ�

### �����������

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--no-camera-compensation` | flag | False | �������������Ĭ�����ã� |
| `--camera-ransac-thresh` | float | `1.0` | RANSAC��ֵ�����أ� |
| `--camera-max-features` | int | `2000` | ����������� |
| `--fov` | float | `60.0` | ����ӳ��ǣ��ȣ� |

### ������Ʋ���

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--max_frames` | int | None | �����֡�� |
| `--frame_skip` | int | `1` | ֡��Ծ��� |
| `--visualize` | flag | False | ���ɿ��ӻ���� |

---

## ? ������˵��

### �������� (`analysis_report.txt`)

```
====================================================================
��Ƶ��̬�ȷ�������
====================================================================

������Ϣ:
-----------
����֡��: 128
��Ƶ�ֱ���: 1280x720
��һ��ģʽ: ����
����ʱ��: 45.23s

��̬���嶯̬�ȷ���:
-----------
ƽ����̬�ȷ���: 0.245
��׼��: 0.082
���ֵ: 0.512
��Сֵ: 0.089
����ϵ��: 0.335

ʱ���ȶ���: 0.823 (�ȶ�)
```

### JSON��� (`analysis_results.json`)

```json
{
  "metadata": {
    "total_frames": 128,
    "resolution": [1280, 720],
    "normalized": true,
    "processing_time": 45.23
  },
  "temporal_stats": {
    "mean_dynamics_score": 0.245,
    "std_dynamics_score": 0.082,
    "max_dynamics_score": 0.512,
    "min_dynamics_score": 0.089,
    "temporal_stability": 0.823
  },
  "unified_scores": {
    "final_score": 0.627,
    "flow_magnitude_score": 0.234,
    "spatial_coverage_score": 0.456,
    "temporal_variation_score": 0.189
  }
}
```

### BadCase���� (`badcase_report.txt`)

```
====================================================================
BADCASE ��ⱨ��
====================================================================

��Ƶ: example_video.mp4
������̬��: high
ʵ�ʶ�̬��: low
���س̶�: severe
��ƥ���: 0.752

��ϸ����:
-----------
ʵ�ʵ÷�: 0.124 (low)
�����÷�: 0.750 (high)
����: -0.626

�ж�ԭ��:
- �����߶�̬��ʵ�ʵͶ�̬
- ���ܵ�������������
- ����: �������ɻ��������
```

---

## ?? ��Ŀ�ܹ�

### Ŀ¼�ṹ

```
AIGC_detector/
������ video_processor.py              # ����ڣ�ͳһ����
������ badcase_detector.py             # BadCase�����
������ unified_dynamics_scorer.py      # ͳһ����ϵͳ
������ static_object_analyzer.py       # ��̬�������
������ simple_raft.py                  # RAFT������װ
������ dynamic_motion_compensation/    # �������ģ��
��   ������ __init__.py
��   ������ camera_compensation.py
������ third_party/RAFT/               # RAFTԭʼʵ��
������ pretrained_models/              # Ԥѵ��ģ��
��   ������ raft-things.pth
������ requirements.txt                # �����б�
������ docs/                           # �ĵ�
```

### ����ģ��

#### 1. VideoProcessor������������

```python
from video_processor import VideoProcessor

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device="cuda",
    enable_camera_compensation=True,
    use_normalized_flow=True,        # ���ù�һ��
    flow_threshold_ratio=0.002
)
```

#### 2. StaticObjectAnalyzer����̬��������

������Ƶ�о�̬������쳣��̬��

```python
from static_object_analyzer import StaticObjectDynamicsCalculator

calculator = StaticObjectDynamicsCalculator(
    use_normalized_flow=True,
    flow_threshold_ratio=0.002
)
```

#### 3. UnifiedDynamicsScorer��ͳһ��������

�ں϶��ά�ȵĶ�̬��ָ�ꡣ

```python
from unified_dynamics_scorer import UnifiedDynamicsScorer

scorer = UnifiedDynamicsScorer(
    mode='auto',
    use_normalized_flow=True
)
```

#### 4. BadCaseDetector��BadCase�������

�Զ�����������⡣

```python
from badcase_detector import BadCaseDetector

detector = BadCaseDetector(
    mismatch_threshold=0.3
)
```

---

## ? ����ԭ��

### 1. �ֱ��ʹ�һ��

**����**����ͬ�ֱ�����Ƶ�Ĺ���ֵ��Χ��ͬ
- 1280x720��Ƶ������ֵͨ�� 0-30 ����
- 750x960��Ƶ������ֵͨ�� 0-20 ����

**�������**���Խ��߹�һ��
```python
diagonal = sqrt(height? + width?)
normalized_flow = flow_magnitude / diagonal
```

**Ч��**��
- ? ��ͬ�ֱ�����Ƶ���ֿɱ�
- ? ��ֵͳһ��0.002���ֵ��
- ? ������ƽ������

### 2. ����˶�����

**ԭ��**��ʹ��ORB����ƥ�� + RANSAC����ȫ���˶�

```
ԭʼ���� = ����˶� + �����˶�
�����˶� = ԭʼ���� - ����˶�����
```

**Ч��**��
- ? �������ƽ��/��ת�������˶�
- ? �۽������屾����쳣�˶�
- ? ��߾�̬������׼ȷ��

### 3. ͳһ��̬������

�ں�5��ά�ȵ�������

1. **��������** (30%) - �˶�ǿ��
2. **�ռ串��** (25%) - �˶�����ռ��
3. **ʱ��仯** (20%) - �˶�ʱ��ģʽ
4. **�ռ�һ����** (15%) - �˶��ռ�ֲ�
5. **�������** (10%) - ����˶�Ӱ��

���շ�����`[0, 1]` ���䣬Խ�߱�ʾ��̬��Խǿ

---

## ? ����ָ��

### �����ٶ�

| ���� | �ֱ��� | ֡�� | ʱ�� | �ٶ� |
|------|--------|------|------|------|
| GPU (RTX 3090) | 1280x720 | 128 | ~45s | 2.8 FPS |
| GPU (RTX 3090) | 1920x1080 | 128 | ~68s | 1.9 FPS |
| CPU (i9-12900K) | 1280x720 | 128 | ~320s | 0.4 FPS |

### �ڴ�ռ��

- GPU�Դ棺1.5-2.5GB��ȡ���ڷֱ��ʣ�
- ϵͳ�ڴ棺2-4GB

### ׼ȷ��

�����ڲ����Լ���
- BadCase���׼ȷ�ʣ�~87%
- �������ʣ�~8%
- �������ʣ�~5%

---

## ? ���ӻ�ʾ��

### ֡������

![Frame Analysis](docs/images/frame_analysis_example.png)

- ���ϣ�ԭʼ֡
- ���ϣ���̬����mask
- ���£��������ӻ�
- ���£����������Ĺ���

### ʱ�����

![Temporal Analysis](docs/images/temporal_analysis_example.png)

- ��̬����ʱ��仯����
- ��̬��������仯
- �쳣֡���

---

## ? ��������

### Q1: �Ƿ����ʹ��GPU��

**A**: ���Ǳ��룬��ǿ���Ƽ���CPUģʽ�ٶ�ԼΪGPU��1/8��

```bash
# CPUģʽ
python video_processor.py -i video.mp4 --device cpu
```

### Q2: ��δ����Ϸֱ��ʵ���Ƶ����

**A**: �������� `--normalize-by-resolution` ������

```bash
python video_processor.py \
    -i mixed_videos/ \
    --batch \
    --normalize-by-resolution  # �ؼ���
```

### Q3: BadCase������ֵ������ã�

**A**: `--mismatch-threshold` Ĭ��0.3���ɸ������������

- `0.2` - ���ϸ񣨼�����BadCase��
- `0.3` - ƽ�⣨�Ƽ���
- `0.4` - �����ɣ������󱨣�

### Q4: ���ӻ����ռ�ÿռ����ιرգ�

**A**: Ĭ�ϲ����ɿ��ӻ����������ɣ���ʽ��� `--visualize`��

```bash
# �����ɿ��ӻ���Ĭ�ϣ����٣�
python video_processor.py -i video.mp4

# ���ɿ��ӻ�������ռ�ÿռ䣩
python video_processor.py -i video.mp4 --visualize
```

### Q5: ���ֻ������Ƶ��һ����֡��

**A**: ʹ�� `--max_frames` �� `--frame_skip` ������

```bash
# ֻ����ǰ50֡
python video_processor.py -i video.mp4 --max_frames 50

# ÿ��2֡����һ��
python video_processor.py -i video.mp4 --frame_skip 2
```

### Q6: RAFTģ������ʧ����ô�죿

**A**: �ֶ����ز����ã�

1. �� [Google Drive](https://drive.google.com/file/d/1x1FLCHaGFn_Tr4wMo5f9NLPwKKGDtDa7/view?usp=sharing) ����
2. ���õ� `pretrained_models/raft-things.pth`
3. ��֤�ļ���СԼΪ 440MB

---

## ? �߼��÷�

### �Զ�����ֵ

```python
from video_processor import VideoProcessor
from unified_dynamics_scorer import UnifiedDynamicsScorer

# �Զ�����ֵ
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

### �Զ���Ȩ��

```python
# ��������Ȩ��
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

### ��̽ӿ�

```python
from video_processor import VideoProcessor

# ��ʼ��
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device="cuda",
    use_normalized_flow=True
)

# ������Ƶ
frames = processor.load_video("test.mp4")

# �����������
camera_matrix = processor.estimate_camera_matrix(
    frames[0].shape, fov=60.0
)

# ������Ƶ
result = processor.process_video(
    frames, camera_matrix, output_dir="output/"
)

# ���ʽ��
print(f"ƽ����̬��: {result['temporal_stats']['mean_dynamics_score']:.3f}")
print(f"ʱ���ȶ���: {result['temporal_stats']['temporal_stability']:.3f}")
```

---

## ? ʹ��ʾ��

### ʾ��1�������������

```bash
# ���ټ�鵥����Ƶ�������ɿ��ӻ���
python video_processor.py \
    -i suspect_video.mp4 \
    -o quick_check/ \
    --normalize-by-resolution
```

### ʾ��2����ϸ����

```bash
# ��ϸ�������������ӻ�
python video_processor.py \
    -i video_to_analyze.mp4 \
    -o detailed_analysis/ \
    --visualize \
    --normalize-by-resolution
```

### ʾ��3�����������������

```bash
# ����BadCase��⣨�Ƽ����ã�
python video_processor.py \
    -i production_videos/ \
    -o badcase_reports/ \
    --batch \
    --badcase-labels expected_labels.json \
    --mismatch-threshold 0.3 \
    --normalize-by-resolution \
    --device cuda \
    --frame_skip 1
```

### ʾ��4������Դ����

```bash
# CPUģʽ + ������
python video_processor.py \
    -i video.mp4 \
    -o output/ \
    --device cpu \
    --frame_skip 3 \
    --max_frames 60 \
    --normalize-by-resolution
```

---

## ? ����ָ��

��ӭ���ף�����ѭ���²��裺

1. Fork����Ŀ
2. �������Է�֧ (`git checkout -b feature/AmazingFeature`)
3. �ύ���� (`git commit -m 'Add some AmazingFeature'`)
4. ���͵���֧ (`git push origin feature/AmazingFeature`)
5. ����Pull Request

---

## ? ���֤

����Ŀ���� MIT ���֤ - ��� [LICENSE](LICENSE) �ļ�

---

## ? ��ϵ��ʽ

����������飬��ͨ�����·�ʽ��ϵ��

- �ύ [Issue](https://github.com/your-repo/issues)
- �����ʼ�����your-email@example.com

---

## ? ��л

- [RAFT](https://github.com/princeton-vl/RAFT) - ��������ģ��
- PyTorch�Ŷ� - ���ѧϰ���
- OpenCV - ������Ӿ���

---

## ? ������־

### v1.0.0 (2025-10-19)

- ? ��ʼ�汾����
- ? �ֱ��ʹ�һ��֧��
- ? ����˶�����
- ? BadCase�Զ����
- ? ͳһ����ϵͳ
- ? ��������֧��
- ? �������ӻ�����

---

<div align="center">

**? ��������Ŀ�����а������������һ��Star��?**

Made with ?? by AIGC Video Quality Team

</div>

