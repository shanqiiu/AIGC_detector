# ͳһ��̬������ϵͳ - ʹ��ָ��

## ? ����

ͳһ��̬������ϵͳ�����ж�̬�����ָ������Ϊһ��**0-1��׼������**���������������͵���Ƶ��

### ��������

```
0.0 ���������������������������������������������������������������������������������������� 1.0
 ��                    ��                    ��        ��
����̬              �Ͷ�̬              �еȶ�̬    �߶�̬
(����)            (Ʈ������)           (����)    (����)
```

**��׼��ӳ��**��
- **0.0 - 0.2**: ?? ����̬����������ܣ�
- **0.2 - 0.4**: ? �Ͷ�̬����΢�˶�����Ʈ�������ģ�
- **0.4 - 0.6**: ? �еȶ�̬���������ߡ��ճ����
- **0.6 - 0.8**: ? �߶�̬���ܲ������裩
- **0.8 - 1.0**: ? ���߶�̬�������˶��������赸��

---

## ? ����ʹ��

### ����1���Զ����ɣ��Ƽ���

ͳһ��̬���������Զ����ɵ� `video_processor.py` �У�

```bash
# �������м��ɣ�����������
python video_processor.py -i your_video.mp4 -o output/
```

### ����2��Python API

```python
from video_processor import VideoProcessor

# ����������
processor = VideoProcessor(device='cuda')

# ������Ƶ
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir="output")

# ��ȡͳһ��̬�ȷ���
unified_score = result['unified_dynamics']['unified_dynamics_score']
category = result['dynamics_classification']['category']

print(f"��̬�ȷ���: {unified_score:.3f}")
print(f"����: {category}")
```

---

## ? ������

### JSON���

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
    "description": "�߶�̬����",
    "typical_examples": ["�ܲ�", "����", "�����˶�"]
  }
}
```

### �ı�����

```
��������������������������������������������������������������������������������������������
? ͳһ��̬������ (Unified Dynamics Score)
��������������������������������������������������������������������������������������������

�ۺ϶�̬�ȷ���: 0.652 / 1.000
��������: dynamic
���Ŷ�: 85.0%

������: �߶�̬����
��������: �ܲ�, ����, �����˶�

? ��̬��: 0.652 (�߶�̬)
��������: ��̬�����������˶���
��Ҫ����: �˶����� (0.720)

��������:
- 0.0-0.2: ����̬���壨�罨�������ܣ�
- 0.2-0.4: ��΢�˶�����Ʈ�������ģ�
- 0.4-0.6: �е��˶��������ߵ��ˣ�
- 0.6-0.8: ��Ծ�˶������ܲ����赸��
- 0.8-1.0: �����˶���������赸�������˶���
```

---

## ? ����ԭ��

### ��ά��ָ���ں�

ͳһ��̬�ȷ�����5��ά���ں϶��ɣ�

#### 1. �������� (35% Ȩ��)

**����**���˶���ǿ��

```python
# ��̬������ʹ�òв������������
# ��̬������ʹ��ԭʼ����
flow_score = sigmoid(mean_flow_magnitude, threshold=5.0)
```

#### 2. �ռ串�� (25% Ȩ��)

**����**���˶�����ռ��

```python
spatial_score = dynamic_ratio = 1.0 - static_ratio
```

#### 3. ʱ��仯 (20% Ȩ��)

**����**���˶���ʱ��仯�ḻ��

```python
temporal_score = sigmoid(std_dynamics_score, threshold=1.0)
```

#### 4. �ռ�һ���� (10% Ȩ��)

**����**���˶��Ŀռ�����ԣ�����ָ�꣩

```python
consistency_score = 1.0 - mean_consistency_score
```

#### 5. ������� (10% Ȩ��)

**����**���������Ч����������ã�

```python
camera_score = 1.0 - camera_success_rate
```

### ����ӦȨ��

ϵͳ����ݳ��������Զ�����Ȩ�أ�

**��̬����**��������˶�����
```python
weights = {
    'flow_magnitude': 0.40,      # ����ע�в�
    'spatial_coverage': 0.20,
    'temporal_variation': 0.15,
    'spatial_consistency': 0.15,
    'camera_factor': 0.10
}
```

**��̬����**�������˶�����
```python
weights = {
    'flow_magnitude': 0.45,      # ����עԭʼ����
    'spatial_coverage': 0.30,    # ����ע����
    'temporal_variation': 0.15,
    'spatial_consistency': 0.05,
    'camera_factor': 0.05
}
```

### Sigmoid��һ��

ʹ��Sigmoid���������ⷶΧӳ�䵽0-1��

$$
\text{score} = \frac{1}{1 + e^{-k(x - x_0)}}
$$

���У�
- $x$: ԭʼָ��ֵ
- $x_0$: ��ֵ���е㣩
- $k$: ���Ͷ�

---

## ?? �߼�����

### �Զ���Ȩ��

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

### �Զ�����ֵ

```python
custom_thresholds = {
    'flow_low': 0.5,      # �Ͷ�̬��ֵ
    'flow_mid': 3.0,      # �еȶ�̬��ֵ
    'flow_high': 10.0,    # �߶�̬��ֵ
    'static_ratio': 0.6   # ��̬�ж���ֵ
}

scorer = UnifiedDynamicsScorer(thresholds=custom_thresholds)
```

### ָ������ģʽ

```python
# ��̬����ģʽ��ǿ��ʹ�òв������
scorer = UnifiedDynamicsScorer(mode='static_scene')

# ��̬����ģʽ��ǿ��ʹ��ԭʼ������
scorer = UnifiedDynamicsScorer(mode='dynamic_scene')

# �Զ����ģʽ��Ĭ�ϣ�
scorer = UnifiedDynamicsScorer(mode='auto')
```

---

## ? Ӧ�ó���

### ����1����Ƶ����

```python
from unified_dynamics_scorer import DynamicsClassifier

classifier = DynamicsClassifier()

# ������Ƶ
result = processor.process_video(frames, output_dir="output")
unified_score = result['unified_dynamics']['unified_dynamics_score']

# ����
classification = classifier.classify(unified_score)
print(classification['category'])  # 'high_dynamic'
```

### ����2������ɸѡ

```python
# ɸѡ����̬��Ƶ��������
if unified_score < 0.2:
    print("����̬������Ƶ")

# ɸѡ�߶�̬��Ƶ���赸��
if unified_score > 0.7:
    print("�߶�̬������Ƶ")
```

### ����3����������

```python
# ��������
results = []
for video in video_list:
    frames = processor.load_video(video)
    result = processor.process_video(frames)
    results.append(result)

# ����ͳ��
batch_stats = scorer.batch_calculate(
    [r for r in results],
    camera_comp_enabled=True
)

print(f"ƽ����̬��: {batch_stats['mean_score']:.3f}")
print(f"��׼��: {batch_stats['std_score']:.3f}")
```

---

## ? ��������

### �鿴��ά�ȹ���

```python
component_scores = result['unified_dynamics']['component_scores']

for name, score in component_scores.items():
    print(f"{name}: {score:.3f}")

# ���:
# flow_magnitude: 0.680
# spatial_coverage: 0.720
# temporal_variation: 0.450
# spatial_consistency: 0.550
# camera_factor: 0.400
```

### �������Ŷ�

```python
confidence = result['unified_dynamics']['confidence']

if confidence < 0.6:
    print("?? ���ŶȽϵͣ�������ܲ��ȶ�")
    print("�����飺")
    print("- ��Ƶ����")
    print("- ��������׼ȷ��")
    print("- �������Ч��")
```

### �������ͼ��

```python
scene_type = result['unified_dynamics']['scene_type']

if scene_type == 'static':
    print("���Ϊ��̬����������˶���")
    print("ʹ�òв�������㶯̬��")
else:
    print("���Ϊ��̬�����������˶���")
    print("ʹ��ԭʼ�������㶯̬��")
```

---

## ? ���ʵ��

### 1. ��ƵԤ����

? **�Ƽ�**��
- �ȶ���֡��
- �����Ļ���
- �ʵ��ķֱ���

?? **����**��
- �����˶�ģ��
- ���ͷֱ���
- ������

### 2. �������ѡ��

**�����������** (Ĭ��):
- �ʺϣ����ת�����㾲̬����
- �ŵ㣺��׼ȷ�ľ�̬��������

**�����������**:
- �ʺϣ��̶���λ���㶯̬����
- �ŵ㣺ֱ�����������˶�

### 3. �������

**��̬�� < 0.3**:
- �����ǣ��̶���λ���㾲̬����
- ���飺����Ƿ�����Ǿ�̬����

**��̬�� 0.3-0.7**:
- �����ǣ�����������
- �ʺϣ�������ճ���Ƶ

**��̬�� > 0.7**:
- �����ǣ������˶�
- ���飺ȷ���Ƿ����Ԥ��

---

## ? ��������

### Q1: ����ƫ��/ƫ�ͣ�

**ԭ��**��
- ��ֵ���ò���
- �������ͼ�����

**���**��
```python
# ������һ����ֵ
scorer = UnifiedDynamicsScorer(
    thresholds={'flow_mid': 8.0}  # �����ֵ���ͷ���
)

# ��ָ������ģʽ
scorer = UnifiedDynamicsScorer(mode='static_scene')
```

### Q2: ���Ŷȹ��ͣ�

**ԭ��**��
- ʱ���ȶ�
- ��ά�ȷ��������

**���**��
- �����Ƶ����
- ���Ӵ���֡��
- ���������������

### Q3: �����ground truth��ǩ���룿

**����1������ӳ��**
```python
# �����ı�ǩ��0-1
# ͳһ�����Ѿ���0-1����ֱ��ʹ��

# �����Ҫ������Χ
def map_to_label(score):
    # ���磺��0.3-0.7ӳ�䵽0-1
    return (score - 0.3) / 0.4
```

**����2������������ֵ**
```python
from unified_dynamics_scorer import DynamicsClassifier

# �Զ�����ֵ
custom_thresholds = {
    'pure_static': 0.10,
    'low_dynamic': 0.30,
    'medium_dynamic': 0.60,
    'high_dynamic': 0.80
}

classifier = DynamicsClassifier(thresholds=custom_thresholds)
```

---

## ? ����ĵ�

- [��̬�������ԭ��](STATIC_OBJECT_ANALYSIS_EXPLAINED.md)
- [�������ʹ��ָ��](CAMERA_COMPENSATION_GUIDE.md)
- [�����ܽ�](INTEGRATION_SUMMARY.md)

---

## ? �ܽ�

ͳһ��̬������ϵͳ�ṩ�ˣ�

? **ͨ����**������������������Ƶ  
? **��׼��**��0-1ͳһ���������ڱȽ�  
? **�ɽ���**����ά�ȷֽ⣬�������  
? **����Ӧ**���Զ���ⳡ������  
? **������**��������Ȩ�غ���ֵ

**��������**��
```
���ָ�� �� ͳһ�ں� �� ��һ���� �� ������
```

---

**�ĵ��汾**: 1.0  
**������**: 2025-10-19

