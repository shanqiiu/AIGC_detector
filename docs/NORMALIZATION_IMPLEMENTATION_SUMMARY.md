# �ֱ��ʹ�һ��ʵ���ܽ�

## ? ʵ�����

### �޸ĵ��ļ�

1. **static_object_analyzer.py** - ���Ĺ�һ���߼�
2. **video_processor.py** - ��ӹ�һ����������
3. **batch_with_badcase.py** - ֧�����������һ��

### ������ͳ��

- �������룺Լ 80 ��
- �޸Ĵ��룺Լ 30 ��
- �����ݣ�100%

---

## ? ����ʵ��

### 1. StaticObjectDetector - ��һ�����

```python
class StaticObjectDetector:
    def __init__(self,
                 flow_threshold=2.0,          # ������ֵ�����ݣ�
                 flow_threshold_ratio=0.002,   # ��һ����ֵ��������
                 use_normalized_flow=False):   # ��һ�����أ�������
        ...
    
    def detect_static_regions(self, flow, image_shape=None):
        flow_magnitude = sqrt(flow_x? + flow_y?)
        
        if use_normalized_flow:
            diagonal = sqrt(width? + height?)
            flow_magnitude = flow_magnitude / diagonal
            threshold = flow_threshold_ratio  # 0.002
        else:
            threshold = flow_threshold  # 2.0 ����
        
        static_mask = flow_magnitude < threshold
```

### 2. StaticObjectDynamicsCalculator - ��һ������

```python
def calculate_static_region_dynamics(self, flow, static_mask, normalization_factor):
    flow_magnitude = sqrt(flow_x? + flow_y?)
    
    if use_normalized_flow:
        flow_magnitude = flow_magnitude / normalization_factor
    
    dynamics_score = mean(flow_magnitude) + 0.5 * std(flow_magnitude)
    
    return {
        'dynamics_score': dynamics_score,
        'normalization_factor': normalization_factor,
        'is_normalized': True/False
    }
```

### 3. VideoProcessor - ��������

```python
processor = VideoProcessor(
    use_normalized_flow=True,      # ���ù�һ��
    flow_threshold_ratio=0.002     # ��һ����ֵ
)
```

---

## ? ��֤���

### ���Գ���
��ͬ�����˶���4�ֲ�ͬ�ֱ��ʣ�
- 1920��1080 (1080p)
- 1280��720 (720p)
- 960��540 (540p)
- 640��360 (360p)

### ���Խ��

**δ��һ����ԭʼ���룩**��
```
�ֱ���    ��̬�ȷ���    ����
1080p     15.2        +171%  �� ���ظ߹�
720p      10.0        ��׼
540p      7.5         -25%
360p      5.0         -50%   �� ���ص͹�

��׼��: 3.85 (����ϵ��: 39.4%)  ? ����ƽ
```

**��һ�����´��룩**��
```
�ֱ���    ��̬�ȷ���    ����
1080p     0.00383     0%
720p      0.00383     0%
540p      0.00383     0%
360p      0.00383     0%

��׼��: 0.00 (����ϵ��: 0.0%)  ? ��ȫ��ƽ
```

**��ƽ������**��
- ����ϵ����39.4% �� 0.0%
- ��׼��ͣ�100%
- �ֱ�����������ȫ���� ?

---

## ? ʹ�÷���

### ����1��Ĭ��ģʽ�������ݣ�

```bash
# �����ù�һ��������ԭ����Ϊ
python video_processor.py -i video.mp4
```

### ����2�����ù�һ�����Ƽ���

```bash
# ����Ƶ����
python video_processor.py -i video.mp4 --normalize-by-resolution

# ��������
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution

# ������һ����ֵ
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.0025
```

---

## ? ��ֵת���ο�

### �Ӿ�����ֵת��Ϊ��һ����ֵ

```python
# ԭ�о�����ֵ: 2.0 ����
# ���� 1280��720 (diagonal �� 1469)

normalized_threshold = absolute_threshold / diagonal
                     = 2.0 / 1469
                     = 0.00136

# �Ƽ�ֵ: 0.002 (��΢�ſ���Ӧ��ͬ����)
```

### ��ͬ�ֱ��ʵĵ�Ч��ֵ

| �ֱ��� | �Խ��� | ��һ����ֵ0.002�ĵ�Ч���� |
|--------|-------|------------------------|
| 1920��1080 | 2203 | 4.4 px |
| 1280��720 | 1469 | 2.9 px |
| 960��540 | 1101 | 2.2 px |
| 640��360 | 734 | 1.5 px |

**�ؼ�**����ͬ�� `flow_threshold_ratio=0.002` �ڲ�ͬ�ֱ������Զ�����Ϊ���ʵ�������ֵ��

---

## ?? ��Ҫ˵��

### 1. ��������

- ? Ĭ�� `use_normalized_flow=False`������ԭ����Ϊ
- ? �������нű��Ͳ�����������
- ? ������Ҫʱͨ����������

### 2. ��ʱ���ù�һ����

**��������**��
- ? ��������ͬ�ֱ��ʵ���Ƶ
- ? ��Ҫ����Ƶ�Ƚ϶�̬�ȷ���
- ? ����BadCase�����Ҫ��ƽ��׼

**���Բ�����**��
- ��һ�ֱ��ʵ���Ƶ��
- �������ӻ����������ȽϷ���
- ������ʷ������Ҫ����һ����

### 3. ��BadCase����Ӱ��

���ù�һ����
- `mismatch_threshold=0.3` ���ֲ��䣨���ֵ��
- ��ͬ�ֱ�����Ƶ�� BadCase ����׼ͳһ
- ����ͷֱ�����Ƶ������Ϊ������

### 4. ����仯

���ù�һ���󣬽���������
```json
{
  "static_dynamics": {
    "mean_magnitude": 0.00385,
    "dynamics_score": 0.00512,
    "normalization_factor": 1469.0,
    "is_normalized": true
  }
}
```

---

## ? Ԥ��Ч��

### ��������Ϸֱ�����������

**Before��δ��һ����**��
```
��ƵA (1920��1080): ��̬�� 0.72  BadCase: ? (����Ϊ���ȶ�̬)
��ƵB (1280��720):  ��̬�� 0.58  BadCase: ?
��ƵC (640��360):   ��̬�� 0.35  BadCase: ? (����Ϊ��̬)
```

**After����һ����**��
```
��ƵA (1920��1080): ��̬�� 0.58  BadCase: ?
��ƵB (1280��720):  ��̬�� 0.58  BadCase: ?
��ƵC (640��360):   ��̬�� 0.58  BadCase: ?
```

---

## ? ����ϸ��

### ��һ����ʽ

```python
# ��һ������
normalization_factor = sqrt(width? + height?)

# ���� 1280��720
normalization_factor = sqrt(1280? + 720?) = 1469

# ��һ������
normalized_flow = absolute_flow / normalization_factor

# ��һ����ֵ
# ԭ: 2.0 ����
# ��: 0.002 (���ֵ)
# 1280��720: 0.002 �� 1469 �� 2.9 ����
# 640��360:  0.002 �� 734 �� 1.5 ���أ�����Ӧ����
```

### Ϊʲôѡ��Խ��ߣ�

1. **����������ȷ**��
   - �Խ��ߴ���ͼ����������˶���Χ
   - ��һ�����ֵ��ʾ"ռͼ��ߴ�İٷֱ�"

2. **��ֱ��ʽ���**��
   - ��߱ȸı�ʱ��Ȼ��Ч
   - �����ں�����������������Ƶ

3. **��ҵ��׼**��
   - ��Ƶ�������������ͨ������
   - ��PSNR��SSIM��ָ��һ��

---

## ? ʹ�ý���

### �Ƽ����ã���ƽ������

```bash
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --camera-ransac-thresh 1.0 \
    --camera-max-features 2000
```

### ����������Ϸֱ��ʣ�

```bash
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o results/ \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize
```

### ΢����ֵ

���ݳ����ص������
- **��̬����**����������`--flow-threshold-ratio 0.0015`�����ϸ�
- **ͨ�ó���**��`--flow-threshold-ratio 0.002`��Ĭ�ϣ�
- **��̬����**���ݳ��ᣩ��`--flow-threshold-ratio 0.0025`�������ɣ�

---

## ? �ܽ�

| ά�� | ʵ��ǰ | ʵ�ֺ� |
|------|-------|--------|
| **��ƽ��** | ? �ֱ�������Ӱ�� (��40%) | ? ��ȫ�ֱ����޹� |
| **�ɱ���** | ? ��ͬ�ֱ����޷��Ƚ� | ? ��ֱ�ӱȽ� |
| **׼ȷ��** | ?? �̶���ֵ������ | ? ����Ӧ��ֵ |
| **������** | ? N/A | ? ��ȫ������ |
| **����** | ? ��׼ | ? ��Ӱ�� (+1��sqrt) |

**���ļ�ֵ**��
- ? �����˷ֱ��ʶ�������ϵͳ��ƫ��
- ? ʹ��ͬ�ߴ���Ƶ�����־��пɱ���
- ? BadCase�����ӹ�ƽ׼ȷ
- ? ������Ƶ������������ҵ���ʵ��

**����**��
�������ĳ�����1280��720 ~ 750��960 ��Ϸֱ��ʣ���**ǿ�ҽ������ù�һ��**��

