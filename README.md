# AIGC��Ƶ��̬������ϵͳ

## ����

һ����������Ƶ��̬������ϵͳ��֧�֣�
- ? **ͳһ��̬������**��������ָ������Ϊ0-1��׼������
- ? **����˶�����**���Զ�ȥ������˶�Ӱ��
- ? **���ӽ�֧��**�������ھ�̬�����Ͷ�̬����
- ? **�Զ��������**������ʶ����Ƶ����

### ��̬�ȷ�������

```
0.0 ������������������������������������������������������������������ 1.0
 ��         ��         ��         ��        ��
����̬    �Ͷ�̬    �еȶ�̬   �߶�̬  ���߶�̬
(����)   (����)    (����)    (�ܲ�)  (����)
```

---

## ���ٿ�ʼ

### ��װ����

```bash
pip install -r requirements.txt
```

### ����ʹ��

```bash
# ��������Ƶ
python video_processor.py -i video.mp4 -o output/

# ��������
python video_processor.py -i videos/ -o results/ --batch
```

### Python API

```python
from video_processor import VideoProcessor

# ����������
processor = VideoProcessor(device='cuda')

# ������Ƶ
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir="output")

# ��ȡͳһ��̬�ȷ���
score = result['unified_dynamics']['unified_dynamics_score']
category = result['dynamics_classification']['category']

print(f"��̬��: {score:.3f} - {category}")
```

---

## ���Ĺ���

### 1. ͳһ��̬������

����5��ά��ָ�꣬���0-1��׼��������

| ά�� | Ȩ�� | ���� |
|------|------|------|
| �������� | 35% | �˶�ǿ�� |
| �ռ串�� | 25% | �˶�����ռ�� |
| ʱ��仯 | 20% | ʱ��仯�ḻ�� |
| �ռ�һ���� | 10% | �˶������� |
| ������� | 10% | ����Ч�� |

**�����׼**��
- 0.0-0.2: ����̬�����������ܣ�
- 0.2-0.4: �Ͷ�̬��Ʈ�����ģ�
- 0.4-0.6: �еȶ�̬�����ߵ��ˣ�
- 0.6-0.8: �߶�̬���ܲ������裩
- 0.8-1.0: ���߶�̬�������˶���

### 2. ����˶�����

�Զ���Ⲣȥ������˶�Ӱ�죺
- ʹ������ƥ���RANSAC��������˶�
- �ӹ����з�������˶��������˶�
- �����ڻ������㡢�ֳ�����ȳ���

### 3. �Զ��������

����ʶ�𳡾����Ͳ������������ԣ�
- **��̬����**������˶����� ʹ�òв����
- **��̬����**�������˶����� ʹ��ԭʼ����

---

## �����в���

### ��������

```bash
--input, -i          # ������Ƶ/ͼ��Ŀ¼
--output, -o         # ���Ŀ¼��Ĭ��: output��
--device             # �����豸 cuda/cpu��Ĭ��: cuda��
--batch              # ��������ģʽ
```

### �����������

```bash
--no-camera-compensation      # �������������Ĭ�����ã�
--camera-ransac-thresh FLOAT  # RANSAC��ֵ��Ĭ��: 1.0��
--camera-max-features INT     # �������������Ĭ��: 2000��
```

### ��������

```bash
--max_frames INT     # �����֡��
--frame_skip INT     # ֡��Ծ�����Ĭ��: 1��
--fov FLOAT          # ����ӳ��Ƕ�����Ĭ��: 60.0��
--no-visualize       # ���ÿ��ӻ�����
```

---

## ������

### JSON���

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

### �ı�����

```
��������������������������������������������������������������������������������������������
ͳһ��̬������ (Unified Dynamics Score)
��������������������������������������������������������������������������������������������

�ۺ϶�̬�ȷ���: 0.652 / 1.000
��������: dynamic
���Ŷ�: 85.0%

������: �߶�̬����
��������: �ܲ�, ����, �����˶�
```

---

## Ӧ�ó���

### 1. ��Ƶ����

```python
if score < 0.2:
    label = "��̬������Ƶ"
elif score < 0.5:
    label = "�Ͷ�̬��Ƶ"
else:
    label = "�߶�̬��Ƶ"
```

### 2. ����ɸѡ

```python
# ɸѡ����̬��Ƶ
static_videos = [v for v in videos if v['score'] < 0.2]

# ɸѡ�߶�̬��Ƶ
dynamic_videos = [v for v in videos if v['score'] > 0.7]
```

### 3. ���ݼ���ע

```python
# ֱ��ʹ��0-1������Ϊ��ǩ
dataset['dynamics_label'] = unified_score
```

---

## �߼�����

### �Զ���Ȩ��

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

### �Զ��������ֵ

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

## ��Ŀ�ṹ

```
AIGC_detector/
������ video_processor.py              # ��������
������ unified_dynamics_scorer.py      # ͳһ��̬������
������ static_object_analyzer.py       # ��̬�������
������ simple_raft.py                  # RAFT��������
������ dynamic_motion_compensation/    # �������ģ��
��   ������ camera_compensation.py
��   ������ object_motion.py
��   ������ se3_utils.py
������ tests/                          # �����ļ�
��   ������ test_unified_dynamics.py
��   ������ test_camera_compensation.py
��   ������ test_static_dynamics.py
������ requirements.txt                # ����
������ README.md                       # ���ĵ�
```

---

## ����

```bash
# ����ͳһ��̬�Ȳ���
python tests/test_unified_dynamics.py

# ���������������
python tests/test_camera_compensation.py

# �������в���
python -m pytest tests/
```

---

## ��������

### Q1: ������Ԥ�ڲ�����

**�������**��
```python
# ������һ����ֵ
scorer = UnifiedDynamicsScorer(
    thresholds={'flow_mid': 8.0}
)

# ��ָ������ģʽ
scorer = UnifiedDynamicsScorer(mode='static_scene')
```

### Q2: ��������ɹ��ʵͣ�

**�������**��
```bash
# ����������
--camera-max-features 3000

# �ſ�RANSAC��ֵ
--camera-ransac-thresh 2.0
```

### Q3: ��ν������������

```bash
python video_processor.py -i video.mp4 -o output/ --no-camera-compensation
```

---

## ��������

- **��ά���ں�**�����Ϲ������ռ䡢ʱ���5��ά��
- **����ӦȨ��**�����ݳ����������ܵ���
- **Sigmoid��һ��**��ƽ��ӳ�䵽0-1��Χ
- **���Ŷ�����**����������ɿ���
- **����⿪��**�������������ݣ�����������

---

## ����Ҫ��

- Python >= 3.7
- PyTorch >= 1.6
- OpenCV >= 4.0
- NumPy, SciPy, scikit-learn
- matplotlib, tqdm

��� `requirements.txt`

---

## ���֤

MIT License

---

## ������־

### v1.0 (2025-10-19)
- ? ����ͳһ��̬������ϵͳ
- ? ��������˶�����
- ? ֧���Զ��������
- ? �����Ĳ��Ը���

---

## ��ϵ��ʽ

����������飬��ӭ�ύ Issue��

---

**���ٿ�ʼ**��
```bash
python video_processor.py -i your_video.mp4 -o output/
```

