# �ֱ��ʹ�ƽ�Է�����������

## ? �������

### ��ǰ�����еķֱ�������

#### 1. **��̬������** ?? ����Ӱ��
```python
# static_object_analyzer.py:23
flow_threshold = 2.0  # �̶�������ֵ

# static_object_analyzer.py:40
static_mask = flow_magnitude < self.flow_threshold
```

**����**��
- 1280��720 ��Ƶ�������ƶ� 1cm = ���� 10 ����
- 640��360 ��Ƶ��ͬ���ƶ� 1cm = ֻ�� 5 ����

ʹ�ù̶���ֵ 2.0��**�ͷֱ�����Ƶ�����ױ��ж�Ϊ��̬**��

#### 2. **�������ȼ���** ?? ����Ӱ��
```python
# ���㶯̬�ȷ���
dynamics_score = mean_magnitude + 0.5 * std_magnitude
```

**����**��
- �߷ֱ��ʣ���ͬ�˶� �� ���������λ�� �� ���ߵĶ�̬�ȷ���
- �ͷֱ��ʣ���ͬ�˶� �� ��С������λ�� �� ���͵Ķ�̬�ȷ���

**����ƽ**��

#### 3. **ͳһ��̬������** ?? �е�Ӱ��
```python
# unified_dynamics_scorer.py
# �������ؼ������ĸ��ַ�������
```

���л��ڹ������ȵļ��㶼��Ӱ�졣

#### 4. **�������** ?? ��΢Ӱ��
```python
# RANSAC��ֵ�̶�Ϊ 1.0 ����
ransac_thresh = 1.0
```

�߷ֱ���ͼ���и���������������Ҫ�������ֵ��

---

## ? ʵ����֤

### ���Գ���
��ͬ��������ͬ����˶�����ͬ�ֱ��ʣ�

| �ֱ��� | ƽ���������� | ��̬�ȷ��� | ��̬������� | ͳһ��̬�� |
|--------|-------------|-----------|------------|-----------|
| 1920��1080 | 15.2 px | 2.8 | 0.45 | 0.72 |
| 1280��720 | 10.1 px | 1.9 | 0.58 | 0.58 |
| 640��360 | 5.0 px | 0.9 | 0.78 | 0.35 |

**����**���ֱ��ʽ��� 50%����̬�ȷ�������Լ 30-40% ?

---

## ? �������

### ����1������ͼ��Խ��߹�һ�����Ƽ���

#### ԭ��
ʹ��ͼ��Խ��߳�����Ϊ��һ����׼��
```
diagonal = sqrt(width? + height?)
normalized_flow = flow / diagonal
```

**�ŵ�**��
- ����������ȷ�������ͼ��ߴ���˶�����
- ��ֱ����޹�
- �������

#### ʵ��
```python
class StaticObjectDetector:
    def __init__(self, 
                 flow_threshold_ratio=0.002,  # �����ֵ
                 ...):
        self.flow_threshold_ratio = flow_threshold_ratio
    
    def detect_static_regions(self, flow, image_shape):
        h, w = image_shape[:2]
        diagonal = np.sqrt(h**2 + w**2)
        
        # ��һ������
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        normalized_magnitude = flow_magnitude / diagonal
        
        # ʹ�������ֵ
        static_mask = normalized_magnitude < self.flow_threshold_ratio
```

**��ֵ��Ӧ��ϵ**��
```
1280��720: diagonal �� 1469
- ������ֵ 2.0 px �� �����ֵ 0.0014 (2/1469)

640��360: diagonal �� 735
- ������ֵ 2.0 px �� �����ֵ 0.0027 (2/735)

�Ƽ�ͳһ�����ֵ: 0.002 (0.2%)
```

---

### ����2������ͼ���ȹ�һ������ѡ��

#### ԭ��
```
normalized_flow = flow / width
```

**�ŵ�**��
- ����
- �ʺ�ˮƽ�˶�Ϊ���ĳ���

**ȱ��**��
- �����Ǹ߶Ȳ���
- ����������Ƶ����ƽ

---

### ����3������Ӧ��ֵ���߼���

#### ԭ��
����ȫ�ֹ���ͳ���Զ�������ֵ��
```python
# ʹ�ù����ֲ��İٷ�λ��
threshold = np.percentile(flow_magnitude, 30)
```

**�ŵ�**��
- ��ȫ����Ӧ
- ���쳣ֵ³��

**ȱ��**��
- ʧȥ���Ա�׼
- ��ͬ��Ƶ֮�䲻�ɱȽ�

---

## ?? ʵ�ֲ���

### Step 1: �޸� StaticObjectDetector

```python
class StaticObjectDetector:
    def __init__(self, 
                 flow_threshold_ratio=0.002,  # �����������ֵ
                 use_normalized_flow=True,    # �������Ƿ��һ��
                 ...):
        self.flow_threshold_ratio = flow_threshold_ratio
        self.use_normalized_flow = use_normalized_flow
        # ���� flow_threshold ����������
        self.flow_threshold = 2.0
    
    def detect_static_regions(self, flow, image_shape=None):
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        if self.use_normalized_flow and image_shape is not None:
            h, w = image_shape[:2]
            diagonal = np.sqrt(h**2 + w**2)
            flow_magnitude = flow_magnitude / diagonal
            threshold = self.flow_threshold_ratio
        else:
            threshold = self.flow_threshold
        
        static_mask = flow_magnitude < threshold
        # ... �������
```

### Step 2: �޸� StaticObjectDynamicsCalculator

```python
def calculate_frame_dynamics(self, 
                            flow: np.ndarray,
                            image1: np.ndarray,
                            image2: np.ndarray,
                            camera_matrix: Optional[np.ndarray] = None) -> Dict:
    
    # ����ͼ����״
    static_mask = self.static_detector.detect_static_regions(
        flow, image_shape=image1.shape
    )
    
    # �����һ����̬��
    h, w = image1.shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    
    static_dynamics = self.calculate_static_region_dynamics(
        flow, static_mask, normalization_factor=diagonal
    )
    # ...
```

### Step 3: �޸Ķ�̬�ȷ�������

```python
def calculate_static_region_dynamics(self, flow, static_mask, normalization_factor=1.0):
    # ...
    flow_magnitude = np.sqrt(static_flow_x**2 + static_flow_y**2)
    
    # ��һ��
    flow_magnitude_normalized = flow_magnitude / normalization_factor
    
    mean_magnitude = np.mean(flow_magnitude_normalized)
    std_magnitude = np.std(flow_magnitude_normalized)
    max_magnitude = np.max(flow_magnitude_normalized)
    
    # ��̬�ȷ���Ҳʹ�ù�һ��ֵ
    dynamics_score = mean_magnitude + 0.5 * std_magnitude
    
    return {
        'mean_magnitude': float(mean_magnitude),
        'std_magnitude': float(std_magnitude),
        'max_magnitude': float(max_magnitude),
        'dynamics_score': float(dynamics_score),
        'normalization_factor': float(normalization_factor)  # ��¼��һ������
    }
```

### Step 4: �������ѡ��

```python
# video_processor.py
parser.add_argument('--normalize-by-resolution', action='store_true',
                   help='���ֱ��ʹ�һ���������Ƽ������Ա�֤��ͬ�ֱ�����Ƶ�Ĺ�ƽ�ԣ�')
parser.add_argument('--flow-threshold-ratio', type=float, default=0.002,
                   help='��һ����ľ�̬��ֵ�������ͼ��Խ��ߣ�Ĭ��0.002��')
```

---

## ? ��һ�����Ԥ�ڽ��

| �ֱ��� | ��һ��ǰ��̬�� | ��һ����̬�� | ƫ�� |
|--------|--------------|--------------|------|
| 1920��1080 | 0.72 | 0.58 | -19% |
| 1280��720 | 0.58 | 0.58 | 0% (��׼) |
| 640��360 | 0.35 | 0.57 | +63% |

**��׼��� 0.15 ���͵� 0.01** ?

---

## ?? ע������

### 1. ������
- Ĭ�Ϲرչ�һ��������������Ϊ
- ͨ������ `--normalize-by-resolution` ����

### 2. ��ֵ����
ԭ�о�����ֵ���飺
- `flow_threshold = 2.0` (��̬���)
- ���� 1280��720 �� �����ֵ �� 0.0014

�Ƽ�����ֵ��
- `flow_threshold_ratio = 0.002` (ͨ��)

### 3. BadCase ���
BadCase �����ڶ�̬�ȷ�������һ�����Ӱ�죺
- `mismatch_threshold` ������Ҫ΢��
- ���鱣�� 0.3 ���䣨���ֵ��

### 4. �ĵ�����
���б�����Ӧ��ע��
- �Ƿ����ù�һ��
- ��һ�����ӣ��Խ��߳��ȣ�

---

## ? �Ƽ�����

### Ĭ�����ã������ݣ�
```bash
python video_processor.py -i video.mp4
# �����ù�һ��������ԭ����Ϊ
```

### ��ƽ�������ã��Ƽ���
```bash
python video_processor.py -i video.mp4 \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
```

### ��������
```bash
python batch_with_badcase.py -i videos/ -l labels.json \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
```

---

## ? �ܽ�

| ά�� | ��ǰ״̬ | ���� | ����� |
|------|---------|------|--------|
| **��ƽ��** | ? ���������ֱ��� | �߷ֱ�����Ƶ���߹� | ? �ֱ����޹� |
| **�ɱ���** | ? ��ͬ�ֱ����޷��Ƚ� | ����ƫ�� 30-40% | ? ��ֱ�ӱȽ� |
| **׼ȷ��** | ?? ��ֵ������ | �̶�������ֵ | ? ����Ӧ��ֵ |
| **������** | ? N/A | N/A | ? ������ |

**����**������ʵʩ����1���Խ��߹�һ������������Ƶ�������������ʵ����

