# �ع����ͳһ��̬������ϵͳʹ��ָ��

## ? Ŀ¼
- [���ĸĽ�](#���ĸĽ�)
- [���ٿ�ʼ](#���ٿ�ʼ)
- [ϵͳ�ܹ�](#ϵͳ�ܹ�)
- [����˵��](#����˵��)
- [ʹ��ʾ��](#ʹ��ʾ��)
- [API�ο�](#api�ο�)
- [Ǩ��ָ��](#Ǩ��ָ��)

---

## ? ���ĸĽ�

### **1. ͳһ�����ֱ�׼**
- **ͳһ������Χ**: 0-1������ǿ�Ʒֶ�
- **��������Ӧ**: �Զ�ʶ��̬�����Ͷ�̬����
- **��ɸѡ��**: �ܹ�ʶ��"��̬�����ж�����С"����Ƶ

### **2. ˫ģʽ����**
```
��̬����������/���
  ���� ��⾲̬���� �� ����в���� �� �����쳣�˶�

��̬����������/���
  ���� ����������� �� ����������� �� ������������
```

### **3. ��������ɸѡ**
- ɸѡ��̬�����ж�̬�ȹ��͵���Ƶ
- ɸѡ��̬�������쳣�˶����ߵ���Ƶ
- ��������Χ������ȶ��ַ�ʽɸѡ

---

## ? ���ٿ�ʼ

### **��װ����**
```bash
pip install -r requirements.txt
```

### **��򵥵�ʹ��**
```python
from video_processor import VideoProcessor

# ������������ʹ����ϵͳ��
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    use_new_calculator=True  # �����¼�����
)

# ������Ƶ
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir='output/')

# �鿴���
print(f"��̬�ȷ���: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
print(f"��������: {result['dynamics_classification']['scene_type']}")
```

### **���в���**
```bash
# ���ٲ�����ϵͳ
python test_new_system.py

# ��������ʾ��
python example_new_system.py
```

---

## ?? ϵͳ�ܹ�

### **����ģ��**

```
unified_dynamics_calculator.py   # ͳһ��̬�ȼ����������ģ�
������ ͬʱ��⾲̬����Ͷ�̬����
������ �Զ��жϳ�������
������ ���ͳһ��0-1����

video_quality_filter.py         # ��Ƶ����ɸѡ��
������ ɸѡ�Ͷ�̬����Ƶ
������ ɸѡ���쳣��Ƶ
������ ����ɸѡ����

dynamics_config.py              # ���ù���
������ Ԥ�����ã�strict/balanced/lenient��
������ ��ֵ����
������ ����ӳ���

video_processor.py              # �����������Ѹ��£�
������ �����¼�����
������ ������
������ ��������֧��
```

### **������**

```
��Ƶ֡
  ��
�������� (RAFT)
  ��
����˶�����
  ��
�в����
  ��
ͳһ��̬�ȼ�����
  ���� ��⾲̬����
  ���� ��⶯̬����
  ���� �жϳ�������
  ���� ����ͳһ����
  ��
����ɸѡ�� (��ѡ)
  ��
������
```

---

## ?? ����˵��

### **Ԥ������**

ϵͳ�ṩ����Ԥ�����ã�

| ���� | ���ó��� | �ص� |
|------|---------|------|
| `strict` | ����Ҫ��� | ��ֵ�ϸ�ɸѡ���� |
| `balanced` | ͨ�ó�����**Ĭ��**��| ƽ��׼ȷ�ԺͰ����� |
| `lenient` | ���ܸ�����Ƶ | ��ֵ���ɣ�ɸѡ���� |

### **�ؼ���ֵ**

```python
DETECTION_THRESHOLDS = {
    'static_threshold': 0.002,    # ��̬��������ֵ
    'subject_threshold': 0.005,   # ������������ֵ
}

QUALITY_FILTER_THRESHOLDS = {
    'low_dynamic_in_dynamic_scene': 0.3,   # ��̬�����Ͷ�̬��ֵ
    'high_anomaly_in_static_scene': 0.5,   # ��̬�������쳣��ֵ
}
```

### **�Զ�������**

```python
from dynamics_config import get_config

# ���ز��޸�����
config = get_config('balanced')
config['detection']['static_threshold'] = 0.0015  # ���ϸ�
config['quality_filter']['low_dynamic_in_dynamic_scene'] = 0.35

# ʹ���Զ�������
processor = VideoProcessor(
    use_new_calculator=True,
    config_preset='balanced'  # ��������
)
# Ȼ���ֶ��޸�
processor.unified_calculator.static_threshold = 0.0015
```

---

## ? ʹ��ʾ��

### **ʾ��1: ����Ƶ����**

```python
from video_processor import VideoProcessor

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    enable_camera_compensation=True,
    use_normalized_flow=True,
    use_new_calculator=True,
    config_preset='balanced'
)

frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir='output/')

# ������
print(f"����: {result['dynamics_classification']['scene_type']}")
print(f"����: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
print(f"�ȼ�: {result['dynamics_classification']['description']}")
```

### **ʾ��2: ��������ɸѡ**

```python
from video_processor import batch_process_videos
from video_quality_filter import VideoQualityFilter

# ��������
processor = VideoProcessor(use_new_calculator=True)
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

# ɸѡ�Ͷ�̬����Ƶ
quality_filter = VideoQualityFilter()
low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3
)

print(f"�ҵ� {len(low_dynamic_videos)} ����̬�ȹ��͵���Ƶ")
for video in low_dynamic_videos:
    print(f"  {video['video_name']}: {video['score']:.3f}")
    print(f"    {video['reason']}")
```

### **ʾ��3: ����ͳ��**

```python
from video_quality_filter import VideoQualityFilter

# ��ȡͳ����Ϣ
quality_filter = VideoQualityFilter()
stats = quality_filter.get_quality_statistics(results)

print(f"����Ƶ��: {stats['total_videos']}")
print(f"ƽ������: {stats['score_statistics']['mean']:.3f}")
print(f"�������ͷֲ�: {stats['scene_type_distribution']}")
print(f"��̬�ȼ��ֲ�: {stats['category_distribution']}")
```

### **ʾ��4: ������Χɸѡ**

```python
# ɸѡ�еȶ�̬�ȵ���Ƶ��0.35-0.60��
medium_dynamic_videos = quality_filter.filter_by_score_range(
    results,
    min_score=0.35,
    max_score=0.60
)

# ֻɸѡ��̬�������еȶ�̬��Ƶ
medium_dynamic_videos = quality_filter.filter_by_score_range(
    results,
    min_score=0.35,
    max_score=0.60,
    scene_type='dynamic'
)
```

---

## ? API�ο�

### **UnifiedDynamicsCalculator**

ͳһ��̬�ȼ�����

```python
calculator = UnifiedDynamicsCalculator(
    static_threshold=0.002,       # ��̬������ֵ
    subject_threshold=0.005,      # ����������ֵ
    use_normalized_flow=True,     # ʹ�ù�һ��
    scene_auto_detect=True        # �Զ���ⳡ��
)

result = calculator.calculate_unified_dynamics(flows, images)
```

**���ؽ��**:
```python
{
    'unified_dynamics_score': 0.52,  # ͳһ���� (0-1)
    'scene_type': 'dynamic',         # ��������
    'classification': {              # ������Ϣ
        'category': 'medium_dynamic',
        'description': '�еȶ�̬',
        'typical_examples': ['��������', '�ճ��']
    },
    'temporal_stats': {...},         # ʱ��ͳ��
    'interpretation': '...'          # ���ֽ���
}
```

### **VideoQualityFilter**

��Ƶ����ɸѡ��

```python
filter = VideoQualityFilter()

# ɸѡ��̬�����ж�̬�ȹ��͵���Ƶ
low_videos = filter.filter_low_dynamics_in_dynamic_scenes(results, threshold=0.3)

# ɸѡ��̬�������쳣���ߵ���Ƶ
high_videos = filter.filter_high_static_anomaly(results, threshold=0.5)

# ��������Χɸѡ
range_videos = filter.filter_by_score_range(results, min_score=0.2, max_score=0.4)

# ������ɸѡ
category_videos = filter.filter_by_category(results, ['low_dynamic', 'medium_dynamic'])

# ��ȡͳ����Ϣ
stats = filter.get_quality_statistics(results)
```

---

## ? Ǩ��ָ��

### **�Ӿ�ϵͳǨ��**

#### **��ʽ1: ʹ���¼��������Ƽ���**

```python
# �ɴ���
processor = VideoProcessor(
    use_normalized_flow=True
)

# �´��루�����¼�������
processor = VideoProcessor(
    use_normalized_flow=True,
    use_new_calculator=True  # �����һ��
)
```

#### **��ʽ2: ���ּ���**

```python
# ʹ�þɼ������������ݣ�
processor = VideoProcessor(
    use_normalized_flow=True,
    use_new_calculator=False  # ʹ�þ�ϵͳ
)
```

### **�����ʽ�仯**

�¾�ϵͳ�Ľ����ʽ**��ȫ����**���ؼ��ֶα���һ�£�

```python
# ����ϵͳ���е��ֶ�
result['unified_dynamics']['unified_dynamics_score']  # ��̬�ȷ���
result['dynamics_classification']['category']         # ����
result['dynamics_classification']['scene_type']       # ��������
```

---

## ? ���ֱ�׼

### **��̬����������/���**

| ������Χ | �ȼ� | ���� | ����ʾ�� |
|---------|------|------|---------|
| 0.00-0.15 | ����̬ | ��ȫ��ֹ | ���������� |
| 0.15-0.35 | �Ͷ�̬ | ��΢�� | ����Ʈ������Ҷ |
| 0.35-0.60 | �еȶ�̬ | ������ | �ϴ����ҡ�� |
| 0.60-0.85 | �߶�̬ | �쳣�˶� | ǿ�硢�� |
| 0.85-1.00 | ���߶�̬ | �����쳣 | �豸���� |

### **��̬����������/���**

| ������Χ | �ȼ� | ���� | ����ʾ�� |
|---------|------|------|---------|
| 0.00-0.15 | ����̬ | �������� | ��������ֹվ�� |
| 0.15-0.35 | �Ͷ�̬ | ��΢���� | �����ƶ���΢������ |
| 0.35-0.60 | �еȶ�̬ | �������� | ���ߡ��ճ�� |
| 0.60-0.85 | �߶�̬ | ��Ծ���� | �ܲ������� |
| 0.85-1.00 | ���߶�̬ | ���Ҷ��� | �����赸�������˶� |

---

## ? ��������

### **Q1: ���ɸѡ"��̬�����ж�����С"����Ƶ��**

```python
low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3  # С��0.3�Ķ�̬��Ƶ�ᱻɸѡ����
)
```

### **Q2: ��ε���ϵͳ���жȣ�**

ʹ�ò�ͬ��Ԥ�����ã�
- `strict`: �����У�ɸѡ����
- `balanced`: Ĭ��
- `lenient`: �����ɣ�ɸѡ����

### **Q3: �¾�ϵͳ����ͬʱʹ����**

���ԣ����� `use_new_calculator=False` ʹ�þ�ϵͳ��

### **Q4: �����֤��ϵͳ����������**

```bash
python test_new_system.py
```

---

## ? ������־

### v2.0 (�ع���)
- ? ����ͳһ��̬�ȼ�����
- ? ������Ƶ����ɸѡ��
- ? �������ù���ϵͳ
- ? ֧�ֶ�̬�����Ͷ�̬�ȼ��
- ? ͳһ��0-1���ֱ�׼
- ? ������������

---

## ? ֧��

�������⣬��ο���
- ���ٲ���: `python test_new_system.py`
- ����ʾ��: `python example_new_system.py`
- ����ָ��: `python dynamics_config.py`

