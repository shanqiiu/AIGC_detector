# ��һ�������ֵ��������ָ��

## ? �������⣺��һ������ֵ��Ҫ�仯��

**�𰸣��ǵģ�** ���л��ھ�������ֵ����ֵ����Ҫ��Ӧ������

---

## ? ��ֵӰ�����

### ��ֵ��Χ�仯

| ָ�� | ��һ��ǰ�����أ� | ��һ���󣨱����� | �������� |
|------|----------------|----------------|---------|
| mean_dynamics_score | 0.5 ~ 10.0 | 0.0003 ~ 0.007 | �� 1469 |
| std_dynamics_score | 0.1 ~ 3.0 | 0.00007 ~ 0.002 | �� 1469 |
| flow_magnitude | 1.0 ~ 30.0 | 0.0007 ~ 0.02 | �� 1469 |

*���� 1280��720 �ֱ��ʣ�diagonal �� 1469*

---

## ? ���Զ���������ֵ

### 1. StaticObjectDetector ��? ��ʵ�֣�

```python
# ��һ��ǰ
flow_threshold = 2.0  # ����

# ��һ�����Զ���
flow_threshold_ratio = 0.002  # ���ֵ
ʵ����ֵ = 0.002 �� diagonal
  - 1280��720: 0.002 �� 1469 = 2.94 px
  - 640��360:  0.002 �� 734 = 1.47 px  # ����Ӧ��
```

### 2. UnifiedDynamicsScorer ��? ��ʵ�֣�

```python
# ��һ��ǰ
thresholds = {
    'flow_low': 1.0,      # ����
    'flow_mid': 5.0,      # ����
    'flow_high': 15.0,    # ����
    'temporal_std': 1.0   # ����
}

# ��һ�����Զ���baseline=1469��
thresholds = {
    'flow_low': 0.00068,     # 1.0/1469
    'flow_mid': 0.0034,      # 5.0/1469
    'flow_high': 0.0102,     # 15.0/1469
    'temporal_std': 0.00068  # 1.0/1469
}
```

### 3. StaticObjectDynamicsCalculator ��? ��ʵ�֣�

���ж�̬�ȼ��㶼����Թ�һ�����ӡ�

---

## ? ����Ҫ��������ֵ

### 1. DynamicsClassifier

```python
thresholds = {
    'pure_static': 0.15,      # unified_score��ֵ��0-1��Χ��
    'low_dynamic': 0.35,
    'medium_dynamic': 0.60,
    'high_dynamic': 0.85,
}
```

**ԭ��**����Щ�� `unified_dynamics_score` ����ֵ���÷���ʼ���� 0-1 ��Χ�����ܵײ��һ��Ӱ�졣

### 2. BadCaseDetector

```python
mismatch_threshold = 0.3  # unified_score�Ĳ�ֵ��0-1��Χ��
```

**ԭ��**���Ƚϵ��� `unified_score`��ʼ�� 0-1 ��Χ��

**����**������ 0.3 ���䣬�����Ը���ʵ��Ч��΢����0.25 ~ 0.35����

---

## ? �Զ��������

### �����Ĳ���������

```
CLI����: --normalize-by-resolution
    ��
VideoProcessor(use_normalized_flow=True)
    ��
���� StaticObjectDynamicsCalculator(use_normalized_flow=True)
��   ���� StaticObjectDetector(use_normalized_flow=True)
��       ���� ��ֵ: flow_threshold_ratio = 0.002
��
���� UnifiedDynamicsScorer(use_normalized_flow=True)
    ���� ��ֵ: flow_mid = 0.0034 (�Զ�ת��)
```

**�ؼ�**��һ����������������·����ֵ������

---

## ? ��ֵת���ο���

### ������ֵת����baseline: 1280��720, diagonal=1469��

| ��; | ����ֵ��px�� | ��һ��ֵ | ������� |
|------|------------|---------|---------|
| ��̬��� | 2.0 | 0.0014 | flow_threshold_ratio |
| �Ͷ�̬ | 1.0 | 0.00068 | thresholds['flow_low'] |
| �еȶ�̬ | 5.0 | 0.0034 | thresholds['flow_mid'] |
| �߶�̬ | 15.0 | 0.0102 | thresholds['flow_high'] |
| ʱ��仯 | 1.0 | 0.00068 | thresholds['temporal_std'] |

### ��ͬ��׼�ֱ��ʵ�ת��

| ��׼�ֱ��� | �Խ��� | flow_mid | ת���� |
|-----------|-------|----------|--------|
| 1920��1080 | 2203 | 5.0 px | 0.00227 |
| **1280��720** | **1469** | **5.0 px** | **0.0034** ? |
| 960��540 | 1101 | 5.0 px | 0.00454 |
| 640��360 | 734 | 5.0 px | 0.00681 |

**�Ƽ�**��ʹ�� 1280��720 ��Ϊ��׼���еȷֱ��ʣ�������㣩

---

## ? ��֤����

### 1. ��� unified_score �ֲ�

```bash
python batch_with_badcase.py -i videos/ -l labels.json --normalize-by-resolution

# �鿴���
cat output/badcase_summary.json | grep unified_score
```

**�������**��
```json
"unified_score": 0.15,  // ��̬����
"unified_score": 0.42,  // �Ͷ�̬
"unified_score": 0.68,  // �еȶ�̬
```

**�쳣���**����ֵδ��������
```json
"unified_score": 0.0001,  // ? ���ж��ӽ�0
"unified_score": 0.0002,
```

### 2. ��� component_scores

```json
{
  "component_scores": {
    "flow_magnitude": 0.45,      // ? Ӧ���� 0-1 ��Χ
    "spatial_coverage": 0.68,    // ?
    "temporal_variation": 0.32   // ?
  }
}
```

### 3. �ԱȲ���

```bash
# ͬһ��Ƶ������ģʽ
python video_processor.py -i test.mp4 -o output_no_norm/
python video_processor.py -i test.mp4 -o output_norm/ --normalize-by-resolution

# �ȽϽ��
# unified_score Ӧ���������0.1������Ӧ�ò���޴�
```

---

## ?? �ֶ�΢����ֵ���߼���

����Զ���ֵ���ʺ����ĳ����������ֶ�������

### ����1���޸Ĵ���

```python
# ����������ʱָ���Զ�����ֵ
custom_thresholds = {
    'flow_low': 0.0005,    # ���ϸ�ľ�̬�ж�
    'flow_mid': 0.0030,    # ��΢�����еȶ�̬��ֵ
    'flow_high': 0.0120,   # ��΢��߸߶�̬��ֵ
    'static_ratio': 0.5,
    'temporal_std': 0.0008
}

processor = VideoProcessor(
    use_normalized_flow=True,
    ...
)

processor.unified_scorer.thresholds = custom_thresholds
```

### ����2�������ض�����

```python
# ��̬������������- ������
static_scene_thresholds = {
    'flow_mid': 0.0025,   # ������ֵ��������
}

# ��̬���������- ������
dynamic_scene_thresholds = {
    'flow_mid': 0.0045,   # �����ֵ��������
}
```

---

## ? ������ֵ�嵥����һ����

### ģ�飺StaticObjectDetector
| ��ֵ | ԭֵ | ��һ��ֵ | ״̬ |
|------|------|---------|------|
| flow_threshold | 2.0 px | 0.002 (ratio) | ? �ѵ��� |
| min_region_size | 100 px? | 100 px? | ? ���䣨������������ |

### ģ�飺UnifiedDynamicsScorer
| ��ֵ | ԭֵ | ��һ��ֵ | ״̬ |
|------|------|---------|------|
| flow_low | 1.0 px | 0.00068 | ? �Զ����� |
| flow_mid | 5.0 px | 0.0034 | ? �Զ����� |
| flow_high | 15.0 px | 0.0102 | ? �Զ����� |
| temporal_std | 1.0 px | 0.00068 | ? �Զ����� |
| static_ratio | 0.5 | 0.5 | ? ���� |

### ģ�飺DynamicsClassifier
| ��ֵ | ֵ | ״̬ |
|------|---|------|
| pure_static | 0.15 | ? ���� |
| low_dynamic | 0.35 | ? ���� |
| medium_dynamic | 0.60 | ? ���� |
| high_dynamic | 0.85 | ? ���� |

### ģ�飺BadCaseDetector
| ��ֵ | ֵ | ״̬ |
|------|---|------|
| mismatch_threshold | 0.3 | ? ���� |
| confidence_threshold | 0.6 | ? ���� |

---

## ? ����

### ? ����ɵ��Զ�����

1. **StaticObjectDetector**��flow_threshold �� flow_threshold_ratio
2. **UnifiedDynamicsScorer**������ flow_* ��ֵ���ݹ�һ��״̬�Զ�����
3. **��������**��`use_normalized_flow` �Զ����ݵ�������Ҫ��ģ��

### ? �����������ֵ

1. **DynamicsClassifier**������ unified_score��0-1��Χ��
2. **BadCaseDetector**������ unified_score ��ֵ

### ? ʹ�÷���

**ֻ��һ������**��
```bash
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution
```

**ϵͳ���Զ�**��
- ? ���� StaticObjectDetector ��ֵ
- ? ���� UnifiedDynamicsScorer ��ֵ
- ? ���� DynamicsClassifier ��ֵ
- ? ���� BadCaseDetector ��ֵ

**�����ֶ��޸��κ���ֵ��**

---

## ?? ע������

### ������������쳣

֢״��������Ƶ unified_score �� 0 �� �� 1

����ԭ��
1. ��ֵδ�Զ���������������£�
2. ��׼�Խ������ò���
3. ���������쳣

�����
```python
# ��ʱ�ֶ�ָ����ֵ
processor.unified_scorer.thresholds = {
    'flow_mid': 0.0034,  # ����ʵ�ʵ���
    ...
}
```

---

## ? ����ĵ�

- [��ֵ����ָ��](./THRESHOLD_ADJUSTMENT_GUIDE.md) - ��ϸ����
- [��һ��ʵ���ܽ�](./NORMALIZATION_IMPLEMENTATION_SUMMARY.md) - ����ʵ��
- [���ٿ�ʼ](./QUICK_START_NORMALIZATION.md) - ʹ��ʾ��

---

**�ܽ�**��������ֵ��ʵ���Զ���������ֻ����� `--normalize-by-resolution` �������ɣ�?

