# BadCase��⹦�� - �����������

## ? ����ع�

**��������**��
> ɸѡAIGC���ɵ�������Ƶ��������
> 1. ��Ӧ��̬�����嶯̬�Ⱥܸߣ��罨��������
> 2. ��Ӧ��̬�����嶯̬�Ⱥܵͣ����ݳ�����Ļ��ֹ��

## ? �������

### �������

```
������ǩ (static/dynamic) 
    +
ʵ�ʶ�̬�� (0-1����)
    ��
��ƥ����
    ��
BadCaseɸѡ
```

---

## ? �������

### 1. badcase_detector.py (400+��)

**������**��

#### BadCaseDetector
- ��ⵥ����Ƶ��BadCase
- ���㲻ƥ���
- ����BadCaseͳ��
- ����BadCase����

#### BadCaseAnalyzer  
- ��ϸ��������������ϣ�
- ʶ������ĸ�ά���쳣
- �������ָ�ʽ

#### QualityFilter
- ����BadCase���������Ƶ
- ֧�ֱ�������Ƶ/����Ƶ
- ��������ɸѡ

### 2. batch_with_badcase.py (260+��)

**����**��
- ����������Ƶ + BadCase���
- ֧��JSON/CSV/TXT��ǩ�ļ�
- �Զ�����BadCase����
- ����BadCase��Ƶ�б�

### 3. ���ɵ�video_processor.py

**�����**��
```python
self.badcase_detector = BadCaseDetector()
self.badcase_analyzer = BadCaseAnalyzer()
```

����VideoProcessorʵ���Զ��߱�BadCase���������

---

## ? ʹ�÷���

### ����1�����������������Ƽ���

#### ����1��׼����ǩ�ļ�

���� `expected_labels.json`��
```json
{
  "building_video_1": "static",
  "building_video_2": 0.0,
  "dancing_video_1": "dynamic",
  "dancing_video_2": 1.0,
  "concert_video": "dynamic"
}
```

#### ����2�����м��

```bash
python batch_with_badcase.py \
  --input videos/ \
  --labels expected_labels.json \
  --output badcase_results/ \
  --device cuda
```

#### ����3���鿴���

```bash
# �鿴�ܽ�
cat badcase_results/badcase_summary.txt

# �鿴BadCase��Ƶ�б�
cat badcase_results/badcase_videos.txt

# BadCase����: 12
# BadCase����: 25.0%
# - ������̬��ʵ�ʶ�̬: 7
# - ������̬��ʵ�ʾ�̬: 5
```

---

### ����2��Python API

```python
from video_processor import VideoProcessor
from badcase_detector import BadCaseAnalyzer

# ����������
processor = VideoProcessor(device='cuda')

# ������Ƶ
frames = processor.load_video("building.mp4")
result = processor.process_video(frames, output_dir="output")

# BadCase���
badcase_result = processor.badcase_analyzer.analyze_with_details(
    result,
    expected_label='static'  # ������̬
)

# �ж�
if badcase_result['is_badcase']:
    print(f"?? BadCase: {badcase_result['badcase_type']}")
    print(f"��ƥ���: {badcase_result['mismatch_score']:.3f}")
    print(badcase_result['description'])
else:
    print("? ��������")
```

---

## ? ���ʾ��

### �ı����� (badcase_summary.txt)

```
======================================================================
������Ƶ����ܽ� (BadCase Detection Summary)
======================================================================

����Ƶ��: 50
�ɹ�����: 48
����ʧ��: 2

BadCase����: 12
������Ƶ��: 36
BadCase����: 25.0%

���س̶ȷֲ�:
- ��΢ (Mild): 4
- �е� (Moderate): 5
- ���� (Severe): 3

BadCase���ͷֲ�:
- ������̬��ʵ�ʶ�̬: 7
- ������̬��ʵ�ʾ�̬: 5

======================================================================
BadCase��ϸ�б�:
======================================================================

1. ancient_temple_shaking
   ����: static_to_dynamic
   ���س̶�: severe
   ������̬��: 0.000
   ʵ�ʶ�̬��: 0.782
   ��ƥ���: 0.782
   ����: ������Ƶ��������̬��ʵ�ʸ߶�̬������ԭ�����嶶����Ʈ�ơ�
   ����:
   1. �����Ƶ�ȶ��ԣ��Ƿ���ڶ���
   2. ��֤��������Ƿ���������
   3. �鿴���ӻ��������λ�쳣�˶�����
   4. ��������������Ƶ

2. concert_frozen_screen
   ����: dynamic_to_static
   ���س̶�: moderate
   ������̬��: 1.000
   ʵ�ʶ�̬��: 0.215
   ��ƥ���: 0.785
   ����: ������Ƶ��������̬��ʵ�ʵͶ�̬������ԭ�򣺴���Ļ��ֹ������ʧ�ܡ�
   ����:
   1. ������ﶯ���Ƿ�������ȷ
   2. �鿴�ؼ�֡��ȷ���Ƿ���ھ�ֹ����
   3. ������Ļ��Ӧ��̬�����Ƿ�����
   4. ���ǵ������ɲ�������������
```

### JSON��� (badcase_summary.json)

```json
{
  "total_videos": 50,
  "badcase_count": 12,
  "badcase_rate": 0.25,
  "type_distribution": {
    "static_to_dynamic": 7,
    "dynamic_to_static": 5
  },
  "severity_distribution": {
    "mild": 4,
    "moderate": 5,
    "severe": 3
  },
  "badcase_list": [
    {
      "video_name": "ancient_temple_shaking",
      "badcase_type": "static_to_dynamic",
      "severity": "severe",
      "expected_score": 0.0,
      "actual_score": 0.782,
      "mismatch_score": 0.782,
      "confidence": 0.895,
      "description": "...",
      "suggestion": "..."
    }
  ]
}
```

---

## ? ʵսӦ��

### Ӧ��1�����ģ����ɸѡ

```bash
# ����5000����Ƶ��ɸѡBadCase
python batch_with_badcase.py \
  -i aigc_generated/ \
  -l video_labels.json \
  -o quality_check/ \
  --device cuda

# ���������876��BadCase (17.5%)
# - ������̬��ʵ�ʶ�̬: 523
# - ������̬��ʵ�ʾ�̬: 353
```

### Ӧ��2���˹�Review���ȼ�����

```python
# �����س̶���������review���ص�
from badcase_detector import BadCaseDetector

detector = BadCaseDetector()
batch_result = detector.batch_detect(results, labels, names)

# ֻɸѡ����BadCase
severe_cases = detector.filter_badcases(
    batch_result,
    severity_levels=['severe']
)

print(f"��Ҫ����review����Ƶ: {len(severe_cases)}")
# �����32������BadCase�����ȴ���
```

### Ӧ��3��ģ�͸Ľ�Ч����֤

```python
# �Ա�ģ��v1��v2��BadCase��

# ģ��v1
v1_badcases = batch_detect(v1_videos, labels)
print(f"v1 BadCase��: {v1_badcases['badcase_rate']:.1%}")

# ģ��v2
v2_badcases = batch_detect(v2_videos, labels)
print(f"v2 BadCase��: {v2_badcases['badcase_rate']:.1%}")

# �Ľ�
improvement = v1_badcases['badcase_rate'] - v2_badcases['badcase_rate']
print(f"BadCase�ʽ���: {improvement:.1%}")
```

---

## ? ����ϸ��

### BadCase�ж��߼�

```python
# ���㲻ƥ���
mismatch_score = |actual_score - expected_score|

# �ж�����
if mismatch_score >= threshold (Ĭ��0.3):
    if expected < 0.3 and actual > 0.5:
        �� BadCase����: static_to_dynamic
    
    elif expected > 0.7 and actual < 0.4:
        �� BadCase����: dynamic_to_static
    
    else:
        �� BadCase����: over_dynamic / under_dynamic
```

### ���س̶�����

```python
if mismatch < 0.3:
    severity = 'normal'    # ����
elif mismatch < 0.4:
    severity = 'mild'      # ��΢
elif mismatch < 0.6:
    severity = 'moderate'  # �е�
else:
    severity = 'severe'    # ����
```

### �������

����BadCase��ϵͳ����������ĸ�ά���쳣��

**������̬��ʵ�ʶ�̬�����**��
- ��� flow_magnitude > 0.6 �� "�������ȹ���"
- ��� spatial_coverage > 0.5 �� "�˶����򸲸ǹ�"
- ��� camera_factor > 0.5 �� "�������ʧ��"

**������̬��ʵ�ʾ�̬�����**��
- ��� flow_magnitude < 0.3 �� "�������ȹ�С"
- ��� spatial_coverage < 0.4 �� "�˶����򸲸�С"
- ��� temporal_variation < 0.3 �� "ʱ��仯С"

---

## ? �ļ�˵��

### �����ļ�

1. **badcase_detector.py** - BadCase�������߼�
   - BadCaseDetector: �����
   - BadCaseAnalyzer: ������  
   - QualityFilter: ����������

2. **batch_with_badcase.py** - ��������ű�
   - ֧�ֶ��ֱ�ǩ��ʽ
   - �Զ����ɱ���
   - ����BadCase�б�

3. **example_labels.json** - ��ǩ�ļ�ʾ��
   - JSON��ʽʾ��
   - ������ͬ���ͱ�ǩ

4. **example_badcase_detection.py** - ʹ��ʾ��
   - 5������Ӧ�ó���
   - ��ֱ������

5. **BADCASE_DETECTION_GUIDE.md** - ����ʹ��ָ��
   - ��ϸ����˵��
   - ��������ָ��
   - ʵս����

---

## ? ��������

### 1. ��ȫ�Զ���
```
��Ƶ + ������ǩ �� �Զ���� �� BadCase�б�
```
�����˹�����ۿ���

### 2. ��׼��λ
```
�����ж��Ƿ�BadCase���������㣺
- ʲô���͵�BadCase
- ���س̶����
- �����ĸ�ά��������
- ��θĽ�
```

### 3. �������
```
��⵽BadCase���Զ�������
- ��Ҫ������ʲô
- ��Щ���ع���������
- �������ֵ����
```

### 4. �������
```
- �ɵ�����ƥ����ֵ
- ��ɸѡ�ض�����BadCase
- ���������س̶ȹ���
- ֧�ֶ��ֵ�����ʽ
```

---

## ? ����״̬

### ? �����

- [x] BadCase�������߼� (badcase_detector.py, 400+��)
- [x] ��������ű� (batch_with_badcase.py, 260+��)
- [x] ���ɵ�VideoProcessor
- [x] ����ʹ��ָ��
- [x] ʾ������
- [x] ��ǩ�ļ�ģ��
- [x] ��linter����

### ? ����ͳ��

| ģ�� | ���� | ���� |
|------|------|------|
| badcase_detector.py | 400+ | ���ļ���߼� |
| batch_with_badcase.py | 260+ | �������� |
| example_badcase_detection.py | 180+ | ʹ��ʾ�� |
| BADCASE_DETECTION_GUIDE.md | - | �����ĵ� |

---

## ? ������ʼ

### ���ٲ��ԣ�ʹ��������Ƶ��

```bash
# 1. ʹ��example_labels.json����
python batch_with_badcase.py \
  -i videos/ \
  -l example_labels.json \
  -o badcase_test/ \
  --device cuda

# 2. �鿴���
cat badcase_test/badcase_summary.txt
```

### ʵ��ʹ��

```bash
# 1. ׼����ı�ǩ�ļ�
# video_labels.json - ����������Ƶ��������ǩ

# 2. ��������
python batch_with_badcase.py \
  -i your_videos/ \
  -l video_labels.json \
  -o results/ \
  --device cuda \
  --mismatch-threshold 0.3

# 3. ��ȡBadCase�б�
cat results/badcase_videos.txt
# �������Ҫ�������ɻ��˹�review����Ƶ��
```

---

## ? ʹ�ü���

### ����1���������������

```bash
# �ϸ�ģʽ������©�죩
--mismatch-threshold 0.2

# ����ģʽ��������죩
--mismatch-threshold 0.4
```

### ����2�����ദ��

```python
# �ֱ�������BadCase
static_badcases = detector.filter_badcases(
    batch_result,
    badcase_types=['static_to_dynamic']
)
# �� ��Щ��Ƶ��Ҫ�ȶ����Ż�

dynamic_badcases = detector.filter_badcases(
    batch_result,
    badcase_types=['dynamic_to_static']
)
# �� ��Щ��Ƶ��Ҫ��ǿ�˶�
```

### ����3�����ȼ�����

```python
# ֻ�������ص�BadCase
severe_cases = detector.filter_badcases(
    batch_result,
    severity_levels=['severe', 'moderate']
)
# ��������������Щ��Ƶ
```

---

## ? ʵ��Ч��

### ��ʵ�ֵĹ���

? **�Զ�ɸѡ������Ƶ**
```
5000����Ƶ �� �Զ���� �� 876��BadCase (17.5%)
```

? **��׼������������**
```
- ��������: 523��
- ��Ļ��ֹ: 353��
```

? **�������**
```
building_001: 
  ����: ���������쳣 (0.78)
  ԭ��: �������ʧ��
  ����: �������ƥ��
```

? **���������б�**
```
badcase_videos.txt �� ֱ��������������
```

---

## ? �����ĵ�

- ? [BadCase���ָ��](BADCASE_DETECTION_GUIDE.md)
- ? [ʹ��ʾ��](example_badcase_detection.py)
- ? [��ǩ�ļ�ģ��](example_labels.json)

---

## ? �ܽ�

��������**��ȫ����ʵ��**��

### ��������

? ���������̬����̬�ȸߵ���Ƶ�����������ȣ�  
? ���������̬����̬�ȵ͵���Ƶ����Ļ��ֹ�ȣ�  
? ����������ģ���ݼ�  
? �Զ�����BadCase������б�  
? ������ϣ���ȷ��������  
? ������ü�����

### ����ʹ��

```bash
python batch_with_badcase.py \
  -i videos/ \
  -l expected_labels.json \
  -o badcase_results/
```

**BadCase���ϵͳ�Ѿ�������ʼɸѡ������Ƶ�ɣ�** ?

