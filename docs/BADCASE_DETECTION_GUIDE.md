# BadCase���ʹ��ָ��

## ����˵��

BadCase���������ɸѡ����AIGC������Ƶ����������������⣺

### ����A��������̬��ʵ�ʸ߶�̬
**��������**��
- ��������Ƶ���ֶ�����Ʈ��
- �����Ʒ��Ƶ���쳣�˶�
- �羰��Ƶ�о�̬Ԫ���ڻζ�

**����߼�**��
```
������ǩ: static (0.0)
ʵ�ʶ�̬��: > 0.5
�ж�: BadCase - static_to_dynamic
```

### ����B��������̬��ʵ�ʵͶ�̬
**��������**��
- �ݳ������ĻӦ�ö�̬�����־�ֹ
- �����赸��Ƶ�����ｩӲ����
- �����˶���Ƶ֡�ʹ��ͻ򿨶�

**����߼�**��
```
������ǩ: dynamic (1.0)
ʵ�ʶ�̬��: < 0.4
�ж�: BadCase - dynamic_to_static
```

---

## ����ʹ��

### ����1��׼��������ǩ�ļ�

���� `expected_labels.json`��

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

**��ǩ��ʽ˵��**��
- `"static"` �� `0.0`����������̬
- `"dynamic"` �� `1.0`�������߶�̬
- `0.0-1.0`�������ľ��嶯̬�ȷ���

### ����2������BadCase���

```bash
python batch_with_badcase.py \
  --input videos/ \
  --labels expected_labels.json \
  --output badcase_output/ \
  --device cuda
```

### ����3���鿴���

```bash
# �鿴�ܽᱨ��
cat badcase_output/badcase_summary.txt

# �鿴BadCase��Ƶ�б�
cat badcase_output/badcase_videos.txt

# �鿴JSON��ϸ���
cat badcase_output/badcase_summary.json
```

---

## ������

### 1. �ܽᱨ�� (badcase_summary.txt)

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

======================================================================
BadCase���ͷֲ�:
======================================================================
  ������̬��ʵ�ʶ�̬���罨��������: 7
  ������̬��ʵ�ʾ�̬������Ļ��ֹ��: 5

======================================================================
���س̶ȷֲ�:
======================================================================
  severe: 3
  moderate: 5
  mild: 4

======================================================================
BadCase��ϸ�б�:
======================================================================

1. building_shaking_001
   ����: static_to_dynamic
   ���س̶�: severe
   ����: static
   ʵ�ʶ�̬��: 0.782
   ��ƥ���: 0.782
   ���Ŷ�: 89.5%
```

### 2. BadCase��Ƶ�б� (badcase_videos.txt)

```
D:\videos\building_shaking_001.mp4
D:\videos\concert_frozen_screen_003.mp4
D:\videos\statue_drifting_005.mp4
...
```

### 3. JSON��ϸ��� (badcase_summary.json)

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

## Python APIʹ��

### ����1��������Ƶ���

```python
from video_processor import VideoProcessor
from badcase_detector import BadCaseAnalyzer

# ����������
processor = VideoProcessor(device='cuda')

# ������Ƶ
frames = processor.load_video("building_video.mp4")
result = processor.process_video(frames, output_dir="output")

# BadCase���
analyzer = BadCaseAnalyzer()
badcase_result = analyzer.analyze_with_details(
    result,
    expected_label='static'  # ������̬
)

# �鿴���
if badcase_result['is_badcase']:
    print(f"?? ��⵽BadCase!")
    print(f"����: {badcase_result['badcase_type']}")
    print(f"��ƥ���: {badcase_result['mismatch_score']:.3f}")
    print(badcase_result['description'])
else:
    print("? ��������")
```

### ����2���������

```python
from badcase_detector import BadCaseDetector

# ���������
detector = BadCaseDetector(mismatch_threshold=0.3)

# �������
results = [...]  # �����Ƶ�Ĵ�����
expected_labels = ['static', 'dynamic', 0.5, ...]  # ��Ӧ��������ǩ
video_names = ['video1', 'video2', 'video3', ...]

batch_result = detector.batch_detect(results, expected_labels, video_names)

# �鿴ͳ��
print(f"BadCase����: {batch_result['badcase_count']}")
print(f"BadCase����: {batch_result['badcase_rate']:.1%}")
```

### ����3����������

```python
from badcase_detector import QualityFilter

# ����������
filter = QualityFilter(accept_mismatch=0.3)

# ׼������
video_results = [
    ('video1.mp4', result1, 'static'),
    ('video2.mp4', result2, 'dynamic'),
    ...
]

# ����BadCase
good_videos, bad_videos = filter.filter_videos(video_results, keep_mode='good')

print(f"������Ƶ: {len(good_videos)}")
print(f"BadCase: {len(bad_videos)}")

# ֻ����BadCase�������˹�review��
badcase_videos, _ = filter.filter_videos(video_results, keep_mode='bad')
```

---

## �߼�����

### 1. ɸѡ�ض����͵�BadCase

```python
from badcase_detector import BadCaseDetector

detector = BadCaseDetector()
batch_result = detector.batch_detect(results, labels, names)

# ֻɸѡ"������̬��ʵ�ʶ�̬"��BadCase
static_badcases = detector.filter_badcases(
    batch_result,
    badcase_types=['static_to_dynamic']
)

print(f"����������BadCase: {len(static_badcases)}")

# ֻɸѡ���ص�BadCase
severe_badcases = detector.filter_badcases(
    batch_result,
    severity_levels=['severe']
)

print(f"����BadCase: {len(severe_badcases)}")
```

### 2. ��ϸ���

```python
from badcase_detector import BadCaseAnalyzer

analyzer = BadCaseAnalyzer()
badcase_result = analyzer.analyze_with_details(result, 'static')

if badcase_result['is_badcase'] and 'diagnosis' in badcase_result:
    diagnosis = badcase_result['diagnosis']
    print(f"��Ҫ����: {diagnosis['primary_issue']}")
    print(f"��������:")
    for factor in diagnosis['contributing_factors']:
        print(f"  - {factor}")
```

### 3. ����BadCase�б�

```python
from badcase_detector import BadCaseAnalyzer

analyzer = BadCaseAnalyzer()

# ����Ϊ��ͬ��ʽ
analyzer.export_badcase_list(batch_result, 'badcases.json', format='json')
analyzer.export_badcase_list(batch_result, 'badcases.txt', format='txt')
analyzer.export_badcase_list(batch_result, 'badcases.csv', format='csv')
```

---

## ��������

### �ؼ�����

| ���� | Ĭ��ֵ | ˵�� | ���Ž��� |
|------|--------|------|----------|
| `mismatch_threshold` | 0.3 | ��ƥ����ֵ | Ҫ���ϸ��0.2�����ɡ�0.4 |
| `confidence_threshold` | 0.6 | ������Ŷ� | ��������Ƶ��0.7����������0.5 |

### �������������

**���ϸ񣨼���©�죩**��
```python
detector = BadCaseDetector(
    mismatch_threshold=0.2,  # ������ֵ
    confidence_threshold=0.5  # �������Ŷ�Ҫ��
)
```

**�����ɣ�������죩**��
```python
detector = BadCaseDetector(
    mismatch_threshold=0.4,  # �����ֵ
    confidence_threshold=0.7  # ������Ŷ�Ҫ��
)
```

---

## Ӧ�ó���

### ����1�����ģ��Ƶ����ɸѡ

```bash
# ����1000����Ƶ��ɸѡ��BadCase
python batch_with_badcase.py \
  -i generated_videos/ \
  -l video_labels.json \
  -o quality_check/ \
  --device cuda

# ֻ����BadCase�������ʡ�洢��
python batch_with_badcase.py \
  -i generated_videos/ \
  -l video_labels.json \
  -o quality_check/ \
  --filter-badcase-only
```

### ����2���˹�Review����

```python
# �Զ�ɸѡ��Ҫ�˹�review����Ƶ
filter = QualityFilter(accept_mismatch=0.3)
badcase_videos, _ = filter.filter_videos(results, keep_mode='bad')

# ֻreview���ص�BadCase
severe_cases = detector.filter_badcases(
    batch_result,
    severity_levels=['severe', 'moderate']
)

print(f"��Ҫ�˹�review����Ƶ: {len(severe_cases)}")
```

### ����3��ģ��ѵ��������ϴ

```python
# ɸѡ��������������Ƶ����ѵ��
good_videos, bad_videos = filter.filter_videos(results, keep_mode='good')

# ������ϴ�����Ƶ�б�
with open('clean_dataset.txt', 'w') as f:
    for video in good_videos:
        f.write(f"{video}\n")
```

---

## ����BadCaseʾ��

### ʾ��1����������

```
��Ƶ: ancient_temple.mp4
����: static (0.0)
ʵ��: 0.78
�ж�: BadCase - static_to_dynamic (severe)

�������:
- ��Ҫ����: flow_magnitude �쳣ƫ��
- ��������:
  - �������ȹ��󣨿����ж�����Ʈ�ƣ�
  - �������ʧ���ʸߣ�����ƥ�����⣩

����:
1. �����Ƶ�ȶ��ԣ��Ƿ���ڶ���
2. ��֤��������Ƿ���������
3. �鿴���ӻ��������λ�쳣�˶�����
4. ��������������Ƶ
```

### ʾ��2���ݳ�����Ļ��ֹ

```
��Ƶ: concert_bigscreen.mp4
����: dynamic (1.0)
ʵ��: 0.25
�ж�: BadCase - dynamic_to_static (moderate)

�������:
- ��Ҫ����: �����˶�����
- ��������:
  - �������ȹ�С���˶����Ȳ��㣩
  - �˶����򸲸�С���ֲ���ֹ��
  - ʱ��仯С���˶�������ֹ��

����:
1. ������ﶯ���Ƿ�������ȷ
2. �鿴�ؼ�֡��ȷ���Ƿ���ھ�ֹ����
3. ������Ļ��Ӧ��̬�����Ƿ�����
4. ���ǵ������ɲ�������������
```

---

## ʵ��ʹ������

### ������������

```bash
# 1. ׼����ǩ�ļ�
vim video_labels.json
# ������ݿ⵼����ǩ

# 2. ������������
python batch_with_badcase.py \
  -i videos/ \
  -l video_labels.json \
  -o badcase_results/ \
  --device cuda

# 3. �鿴BadCase�ܽ�
cat badcase_results/badcase_summary.txt

# 4. �˹�review����BadCase
# ����badcase_videos.txt�е��б�������

# 5. ����
# - ��������BadCase��Ƶ
# - �����ģ�Ͳ���
# - ����Ϊ���ϸ�
```

---

## ��������ָ��

### ��ƥ����ֵ����

**���Բ�ͬ��ֵ��Ч��**��

```python
thresholds = [0.2, 0.3, 0.4, 0.5]
for thresh in thresholds:
    detector = BadCaseDetector(mismatch_threshold=thresh)
    result = detector.batch_detect(results, labels, names)
    print(f"��ֵ{thresh}: BadCase��={result['badcase_rate']:.1%}")

# ���ʾ��:
# ��ֵ0.2: BadCase��=35.2% (̫����)
# ��ֵ0.3: BadCase��=25.0% (����) ?
# ��ֵ0.4: BadCase��=15.8% (̫����)
```

**�Ƽ�����**��
- �ϸ�ɸѡ������©�죩��0.2
- ƽ�⣨�Ƽ�����0.3
- ����ɸѡ��������죩��0.4

---

## ��������

### Q1: ��δ���û�б�ǩ����Ƶ��

**����1**������BadCase���
```bash
# ֻ�����б�ǩ����Ƶ
python batch_with_badcase.py -i videos/ -l labels.json -o output/
# �ޱ�ǩ����Ƶ�ᱻ����
```

**����2**��ʹ��Ĭ�ϱ�ǩ
```python
# �ڴ���������Ĭ�ϱ�ǩ
expected = labels.get(video_name, 'dynamic')  # Ĭ��������̬
```

### Q2: BadCase�ʹ�����ô�죿

**����ԭ��**��
1. ��ֵ���ù��ϣ�����mismatch_threshold��
2. ��ǩ��ע��׼ȷ��review��ǩ��
3. ��Ƶ��������ȷʵ�����⣨�Ľ�����ģ�ͣ�

**��Ϸ���**��
```python
# �鿴��ƥ��ȷֲ�
mismatch_scores = [bc['mismatch_score'] for bc in badcases]
print(f"��ƥ��Ⱦ�ֵ: {np.mean(mismatch_scores):.3f}")
print(f"��ƥ�����λ��: {np.median(mismatch_scores):.3f}")

# �����λ���ӽ���ֵ��˵����ֵ����ƫ��
```

### Q3: �����֤���׼ȷ�ԣ�

**����**���˹�������֤

```python
# �����ȡ20��BadCase
import random
sample_badcases = random.sample(badcases, min(20, len(badcases)))

# �˹�review��ͳ��׼ȷ��
correct = 0
for bc in sample_badcases:
    video_path = bc['video_path']
    # �˹��ۿ���Ƶ
    is_actually_bad = input(f"��Ƶ {video_path} �Ƿ�ȷʵ������? (y/n): ")
    if is_actually_bad.lower() == 'y':
        correct += 1

precision = correct / len(sample_badcases)
print(f"BadCase��⾫׼��: {precision:.1%}")
```

---

## ʵս����

### ����1�����ģ���ݼ���ϴ

**����**����5000��AIGC������Ƶ����Ҫɸѡ������������Ƶ

```bash
# 1. ��������
python batch_with_badcase.py \
  -i aigc_dataset/ \
  -l dataset_labels.json \
  -o quality_check/ \
  --device cuda \
  --mismatch-threshold 0.3

# 2. ����BadCase�б�
# quality_check/badcase_videos.txt

# 3. ͳ�ƽ��
# ����Ƶ: 5000
# BadCase: 876 (17.5%)
# - ��̬����̬: 523
# - ��̬����̬: 353
```

**����**��
- ��������876��BadCase��Ƶ
- ����Ϊ����������

### ����2��ģ�͵�����֤

**����**����������ģ�Ͳ�������֤�Ľ�Ч��

```python
# ����ǰ
before_result = batch_detect(before_videos, labels)
print(f"����ǰBadCase��: {before_result['badcase_rate']:.1%}")

# ������
after_result = batch_detect(after_videos, labels)
print(f"������BadCase��: {after_result['badcase_rate']:.1%}")

# �Ա�
improvement = before_result['badcase_rate'] - after_result['badcase_rate']
print(f"�Ľ�: {improvement:.1%}")
```

---

## ������ϵͳ����

BadCase�������ȫ���ɵ�����ϵͳ�У�

```python
# video_processor.py �Ѱ���
self.badcase_detector = BadCaseDetector()
self.badcase_analyzer = BadCaseAnalyzer()

# ��ֱ��ʹ��
processor = VideoProcessor(device='cuda')
# BadCase��������Զ���ʼ��
```

---

## �ܽ�

### ��������

? **�Զ������**�������˹�����ۿ�  
? **��׼��λ**����ȷָ����������  
? **�������**�����������ĸ�ά���쳣  
? **��������**��֧�ִ��ģ���ݼ�  
? **�������**���ɵ������������

### ���ó���

- ? AIGC��Ƶ��������
- ? ���ݼ���ϴ��ɸѡ
- ? ģ��Ч����֤
- ?? �˹�review����

---

**��ʼʹ��**��
```bash
python batch_with_badcase.py -i videos/ -l labels.json -o output/
```

