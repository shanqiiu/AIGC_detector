# ��һ�������ֵ����ָ��

## ? ��������

**��һ���ı�����ֵ��Χ**��

| ָ�� | ��һ��ǰ | ��һ���� | �仯 |
|------|---------|---------|------|
| mean_dynamics_score | 0.5 ~ 10.0 ���� | 0.0003 ~ 0.007 | ��С 1000+ �� |
| flow_magnitude | 1 ~ 30 ���� | 0.0007 ~ 0.02 | ��С 1000+ �� |

**Ӱ��**�����л��ھ���ֵ����ֵ��ʧЧ�ˣ�

---

## ? ��Ҫ��������ֵ

### 1. ? StaticObjectDetector - �ѵ���

```python
# ԭ��ֵ������ֵ��
flow_threshold = 2.0  # ����

# ����ֵ�����ֵ��
flow_threshold_ratio = 0.002  # ����ڶԽ���
```

**״̬**��? ��ʵ��

---

### 2. ?? UnifiedDynamicsScorer - ��Ҫ����

#### ��ǰ���루��������ֵ��

```python
self.default_thresholds = {
    'flow_low': 1.0,      # �Ͷ�̬��ֵ������/֡��
    'flow_mid': 5.0,      # �еȶ�̬��ֵ
    'flow_high': 15.0,    # �߶�̬��ֵ
    'static_ratio': 0.5,  # ��̬�����ж���ֵ
}

# ʹ��ʾ��
score = self._sigmoid_normalize(
    raw_value,  # mean_dynamics_score�������һ������ 0.003
    threshold=self.thresholds['flow_mid'],  # 5.0 - ̫��
    steepness=0.5
)
```

**����**��
- ��� `mean_dynamics_score = 0.003`����һ��ֵ��
- ����ֵ���� `5.0`������ֵ��
- sigmoid ��� �� 0.0�����󣡣�

#### �������

```python
class UnifiedDynamicsScorer:
    def __init__(self, 
                 mode: str = 'auto',
                 use_normalized_flow: bool = False,  # ����
                 ...):
        
        if use_normalized_flow:
            # ��һ����ֵ�����ֵ������1280��720 diagonal��1469��
            self.default_thresholds = {
                'flow_low': 0.0007,   # 1.0 / 1469
                'flow_mid': 0.0034,   # 5.0 / 1469
                'flow_high': 0.0102,  # 15.0 / 1469
                'static_ratio': 0.5,  # ����
            }
        else:
            # ������ֵ������ֵ��
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
```

---

### 3. ? DynamicsClassifier - ����Ҫ����

```python
self.default_thresholds = {
    'pure_static': 0.15,    # unified_score����ֵ
    'low_dynamic': 0.35,
    'medium_dynamic': 0.60,
    'high_dynamic': 0.85,
}
```

**ԭ��**����Щ�� `unified_score` ����ֵ��ʼ��0-1��Χ�������ܵײ��һ��Ӱ�졣

**״̬**��? �����޸�

---

### 4. ?? BadCaseDetector - ���۲��䣬ʵ�ʿ�����Ҫ΢��

```python
mismatch_threshold = 0.3  # ��̬�ȷ�������
```

**����**��
- �Ƚϵ��� `unified_score`��0-1��Χ��
- �����ϲ��ܵײ��һ��Ӱ��

**ʵ��**��
- ��� `unified_score` �ļ��㷽ʽ�ı�
- ������Ҫ����ʵ��Ч��΢��

**����**������ 0.3���۲�Ч�����ٵ���

---

## ? ��ֵת����ʽ

### ��׼�ֱ��ʷ����Ƽ���

ѡ��һ����׼�ֱ��ʣ��� 1280��720��������ת�����ӣ�

```python
# ��׼�ֱ���
baseline_w, baseline_h = 1280, 720
baseline_diagonal = sqrt(1280? + 720?) �� 1469

# ת����ʽ
normalized_threshold = absolute_threshold / baseline_diagonal

# ʾ��
flow_low:  1.0 / 1469 �� 0.00068
flow_mid:  5.0 / 1469 �� 0.0034
flow_high: 15.0 / 1469 �� 0.0102
```

### ʵ�ʷֱ��ʷ�Χ��

��������ʵ����Ƶ�ֱ��ʷ�Χ���㣺

```python
# ���ķֱ��ʷ�Χ
resolutions = [(1280, 720), (750, 960), (1080, 1920), ...]

# ����ƽ���Խ���
diagonals = [sqrt(w? + h?) for w, h in resolutions]
avg_diagonal = mean(diagonals)

# ת����ֵ
normalized_threshold = absolute_threshold / avg_diagonal
```

---

## ? �Ƽ���ֵ����

### ����A���Զ����䣨�Ƽ���?

�� `UnifiedDynamicsScorer` ���Զ�����һ��״̬��

```python
class UnifiedDynamicsScorer:
    def __init__(self,
                 mode: str = 'auto',
                 use_normalized_flow: bool = False):
        
        # ���ݹ�һ��״̬ѡ����ֵ
        if use_normalized_flow:
            baseline_diagonal = 1469.0  # 1280��720��׼
            self.default_thresholds = {
                'flow_low': 1.0 / baseline_diagonal,   # �� 0.00068
                'flow_mid': 5.0 / baseline_diagonal,   # �� 0.0034
                'flow_high': 15.0 / baseline_diagonal, # �� 0.0102
                'static_ratio': 0.5,
            }
        else:
            # ԭ����ֵ
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
```

### ����B���û�ָ��

��������в�����

```bash
--unified-flow-mid <float>     # UnifiedScorer���еȶ�̬��ֵ
--unified-flow-high <float>    # UnifiedScorer�ĸ߶�̬��ֵ
```

---

## ?? ʵʩ����

### Step 1: �޸� UnifiedDynamicsScorer

```python
# unified_dynamics_scorer.py

class UnifiedDynamicsScorer:
    def __init__(self,
                 mode: str = 'auto',
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 use_normalized_flow: bool = False):  # ��������
        
        self.mode = mode
        self.use_normalized_flow = use_normalized_flow
        
        # ���ݹ�һ��״̬����Ĭ����ֵ
        if use_normalized_flow:
            # ��һ����ֵ������1280��720��diagonal��1469��
            baseline_diagonal = 1469.0
            self.default_thresholds = {
                'flow_low': 1.0 / baseline_diagonal,    # 0.00068
                'flow_mid': 5.0 / baseline_diagonal,    # 0.0034
                'flow_high': 15.0 / baseline_diagonal,  # 0.0102
                'static_ratio': 0.5,
            }
        else:
            # ������ֵ������ֵ��
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
        
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds
```

### Step 2: �޸� VideoProcessor

```python
# video_processor.py

self.unified_scorer = UnifiedDynamicsScorer(
    mode='static_scene',
    use_normalized_flow=use_normalized_flow  # ���ݹ�һ��״̬
)
```

### Step 3: ��������в�������ѡ��

```python
# �߼��û����ֶ�ָ����ֵ
parser.add_argument('--unified-thresholds', type=str,
                   help='ͳһ��������ֵ��JSON��ʽ��')

# ʹ��
if args.unified_thresholds:
    thresholds = json.loads(args.unified_thresholds)
    processor.unified_scorer.thresholds = thresholds
```

---

## ? ��ֵת����

### ����1280��720��diagonal = 1469��

| ��ֵ���� | ����ֵ�����أ� | ��һ��ֵ�������� | ˵�� |
|---------|--------------|----------------|------|
| `flow_low` | 1.0 | 0.00068 | �Ͷ�̬��ֵ |
| `flow_mid` | 5.0 | 0.0034 | �еȶ�̬��ֵ |
| `flow_high` | 15.0 | 0.0102 | �߶�̬��ֵ |
| `static_ratio` | 0.5 | 0.5 | ���䣨���Ǳ����� |

### ��ͬ��׼�ֱ��ʵ�ת��

| ��׼�ֱ��� | �Խ��� | flow_mid����ֵ | flow_mid��һ��ֵ |
|-----------|-------|--------------|----------------|
| 1920��1080 | 2203 | 5.0 | 0.00227 |
| 1280��720 | 1469 | 5.0 | 0.00340 |
| 640��360 | 734 | 5.0 | 0.00681 |

**����**��ͳһʹ�� 1280��720 ��Ϊ��׼���еȷֱ��ʣ�

---

## ? �����޸�����

### ��С�Ķ��������Ƽ���

ֻ���޸� `video_processor.py` ��ʼ�����֣�

```python
# ��ʼ��ͳһ��̬�������������ݹ�һ��״̬��
self.unified_scorer = UnifiedDynamicsScorer(
    mode='static_scene',
    use_normalized_flow=use_normalized_flow  # ����
)
```

Ȼ���� `UnifiedDynamicsScorer.__init__` �и��� `use_normalized_flow` ������ֵ��

---

## ? ��֤����

### �����ֵ�Ƿ���ȷ

```python
# ����һ����Ƶ
python video_processor.py -i video.mp4 --normalize-by-resolution

# �鿴���
{
  "unified_dynamics": {
    "unified_dynamics_score": 0.58,  # Ӧ���� 0-1 ��Χ��
    "component_scores": {
      "flow_magnitude": 0.45  # Ӧ���� 0-1 ��Χ��
    }
  }
}
```

**�������**��
- ? `unified_dynamics_score` �� 0-1 ��Χ
- ? `component_scores` �� 0-1 ��Χ
- ? ������������̬/��̬��

**�쳣���**����ֵδ��������
- ? ������Ƶ `unified_score` �� 0.0����ֵ̫��
- ? �� `unified_score` �� 1.0����ֵ̫С��

---

## ? ������ֵ�嵥

### �ܹ�һ��Ӱ�죨��Ҫ������??

| ģ�� | ��ֵ | ԭֵ | ��һ��ֵ | ״̬ |
|------|------|------|---------|------|
| StaticObjectDetector | flow_threshold | 2.0 px | 0.0014 | ? �ѵ��� |
| UnifiedDynamicsScorer | flow_low | 1.0 px | 0.00068 | ?? ������ |
| UnifiedDynamicsScorer | flow_mid | 5.0 px | 0.0034 | ?? ������ |
| UnifiedDynamicsScorer | flow_high | 15.0 px | 0.0102 | ?? ������ |
| UnifiedDynamicsScorer | temporal_std | 1.0 px | 0.00068 | ?? ������ |

### ����Ӱ�죨���������?

| ģ�� | ��ֵ | ֵ | ˵�� |
|------|------|----|----|
| UnifiedDynamicsScorer | static_ratio | 0.5 | ����ֵ |
| DynamicsClassifier | pure_static | 0.15 | unified_score��ֵ |
| DynamicsClassifier | low_dynamic | 0.35 | unified_score��ֵ |
| DynamicsClassifier | medium_dynamic | 0.60 | unified_score��ֵ |
| DynamicsClassifier | high_dynamic | 0.85 | unified_score��ֵ |
| BadCaseDetector | mismatch_threshold | 0.3 | unified_score��ֵ |

---

## ? ʵʩ���ȼ�

### P0 - �����޸ģ�������

? **StaticObjectDetector** - �����
- flow_threshold �� flow_threshold_ratio

?? **UnifiedDynamicsScorer** - ��ʵʩ
- ��� `use_normalized_flow` ����
- ���ݹ�һ��״̬ѡ����ֵ

### P1 - �����޸ģ����ڣ�

? **�ĵ�����**
- ˵����һ������ֵ��Ӱ��
- �ṩ��ֵת������/���

? **��֤����**
- ���Թ�һ��ǰ��� unified_score �ֲ�
- ȷ�Ϸ�����������

### P2 - ��ѡ�Ż������ڣ�

? **����Ӧ��ֵ**
- ������Ƶ�ֱ����Զ�����
- �ṩ���ӻ���ֵ���Ź���

---

## ? �Ƽ�����

### ��ʱ������������޸� UnifiedDynamicsScorer ǰ��

```bash
# ����1����ʱ�����ù�һ��������ԭ����ֵ
python batch_with_badcase.py -i videos/ -l labels.json
# ȱ�㣺�ֱ��ʲ���ƽ�����Դ���

# ����2�����ù�һ�� + �ֶ�����ͳһ��������ֵ
# ����Ҫ��ʵʩ������޸ģ�
```

### ��������������޸ĺ�

```bash
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
# ������ֵ�Զ�����
```

---

## ? ����ж���ֵ�Ƿ���Ҫ����

### ��� unified_dynamics_score �ֲ�

```bash
# ����һ����Ƶ
python batch_with_badcase.py -i videos/ -l labels.json --normalize-by-resolution

# �鿴���
cat output/badcase_summary.json
```

**�����ֲ�**����ֵ��ȷ����
```json
[
  {"unified_score": 0.15, ...},  // ��̬
  {"unified_score": 0.42, ...},  // �Ͷ�̬
  {"unified_score": 0.68, ...},  // �еȶ�̬
  {"unified_score": 0.88, ...},  // �߶�̬
]
```

**�쳣�ֲ�**����ֵ���󣩣�
```json
// ������Ƶ���� 0.0 �� 1.0
[
  {"unified_score": 0.0001, ...},  // ? ��ֵ̫��
  {"unified_score": 0.0002, ...},
  {"unified_score": 0.0001, ...},
]
```

---

## ? �����ж�����

### 1. �޸� unified_dynamics_scorer.py

��ӹ�һ����֪��

```python
class UnifiedDynamicsScorer:
    def __init__(self,
                 mode: str = 'auto',
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 use_normalized_flow: bool = False,
                 baseline_diagonal: float = 1469.0):
        
        self.use_normalized_flow = use_normalized_flow
        
        # �Զ�������ֵ
        if use_normalized_flow and thresholds is None:
            self.default_thresholds = {
                'flow_low': 1.0 / baseline_diagonal,
                'flow_mid': 5.0 / baseline_diagonal,
                'flow_high': 15.0 / baseline_diagonal,
                'static_ratio': 0.5,
            }
        else:
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
            }
        
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds
```

### 2. �޸� video_processor.py

���ݹ�һ��״̬��

```python
self.unified_scorer = UnifiedDynamicsScorer(
    mode='static_scene',
    use_normalized_flow=use_normalized_flow  # ����״̬
)
```

### 3. ������֤

```bash
# ������ͬ��Ƶ���ȽϹ�һ��ǰ��
python video_processor.py -i test.mp4 -o output1/
python video_processor.py -i test.mp4 -o output2/ --normalize-by-resolution

# �Ƚ� unified_dynamics_score �Ƿ����
```

---

## ? �ܽ�

| ��ֵ���� | �Ƿ���Ҫ���� | ״̬ | ���ȼ� |
|---------|------------|------|--------|
| StaticObjectDetector.flow_threshold | ? �� | ? ����� | P0 |
| UnifiedDynamicsScorer.flow_* | ? �� | ?? ��ʵʩ | **P0** |
| DynamicsClassifier.* | ? �� | ? �����޸� | - |
| BadCaseDetector.mismatch | ?? ���� | ? ���۲� | P1 |

**�ؼ�**��`UnifiedDynamicsScorer` ����ֵ���������ù�һ����**ǰ������**���������ֻ�ʧЧ��

**����**������������ʵʩ UnifiedDynamicsScorer ����ֵ����Ӧ��

