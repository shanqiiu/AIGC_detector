# �ֱ��ʹ�һ������ʹ��ָ��

## ? ���ⱳ��

������Ƶ�ߴ緶Χ��1280��720 ~ 750��960

**ԭʼ���������**��
- ��ͬ�������˶�����ͬ�ֱ��ʵõ���ͬ�Ķ�̬�ȷ���
- �ͷֱ�����Ƶ��ϵͳ��**�͹�**������Ϊ��̬��
- �߷ֱ�����Ƶ��ϵͳ��**�߹�**������Ϊ��̬��

**�������**�����÷ֱ��ʹ�һ��

---

## ? ���ٿ�ʼ

### ����Ƶ����

```bash
# �����÷�������һ����
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution

# ��������
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize
```

### �������� + BadCase���

```bash
# �Ƽ����ã���ƽ������
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o results/ \
    --normalize-by-resolution

# ��������
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o results/ \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize \
    --camera-ransac-thresh 1.0 \
    --camera-max-features 2000
```

---

## ?? �ؼ�����

| ���� | Ĭ��ֵ | ˵�� | �Ƽ�ֵ |
|------|-------|------|--------|
| `--normalize-by-resolution` | False | ���ù�һ�� | **True�����룩** |
| `--flow-threshold-ratio` | 0.002 | ��һ����ֵ | 0.002��ͨ�ã�|

### ��ֵ����ָ��

���ݳ������͵��� `--flow-threshold-ratio`��

```bash
# ��̬�������������羰��- ���ϸ�
--flow-threshold-ratio 0.0015

# ͨ�ó��� - ƽ��
--flow-threshold-ratio 0.002

# ��̬����������ݳ��ᣩ- ������
--flow-threshold-ratio 0.0025
```

---

## ? Ч���Ա�

### ʾ���������Ϸֱ�����Ƶ

#### Before��δ��һ����?
```
��Ƶ              �ֱ���      ��̬��   �ж�
building_A.mp4   1920��1080   0.72    ��̬�����У�
building_B.mp4   1280��720    0.58    �еȣ���ȷ��
building_C.mp4   640��360     0.35    ��̬�����У�

���⣺��ͬ�Ľ������ͬ�ж������
```

#### After����һ����?
```
��Ƶ              �ֱ���      ��̬��   �ж�
building_A.mp4   1920��1080   0.58    �еȣ���ȷ��
building_B.mp4   1280��720    0.58    �еȣ���ȷ��
building_C.mp4   640��360     0.58    �еȣ���ȷ��

? ��ƽһ�µ��������
```

---

## ? �����֤��һ���Ƿ���Ч

### ������JSON

```json
{
  "static_dynamics": {
    "mean_magnitude": 0.00385,
    "dynamics_score": 0.00512,
    "normalization_factor": 1469.0,    // ��һ�����ӣ��Խ��߳��ȣ�
    "is_normalized": true               // ȷ���ѹ�һ��
  }
}
```

**�ؼ���־**��
- `is_normalized: true` �� ��һ��������
- `normalization_factor` > 1 �� ��ʾ�Խ��߳���
- `mean_magnitude` < 0.01 �� ��һ�����ֵ�����ֵ��

---

## ? ��������

### Q1: �Ƿ���Ҫ���´�����ʷ���ݣ�

**A**: ȡ������������
- �����Ҫ�������ݱȽ� �� �������´���
- ������ڲ��ο� �� ���Ա��ֲ���
- **������ǿ�ҽ������ù�һ��**

### Q2: ��һ����Ӱ��������

**A**: ������Ӱ��
- ������ 1 �� sqrt ���㣨����Խ��ߣ�
- ���ܿ��� < 0.1%

### Q3: �ɵ���ֵ��������

**A**: ��Ҫת��
```python
# ����ֵ�����أ�
old_threshold = 2.0

# �������ĵ��ͷֱ��ʣ���1280��720��ת��
diagonal = np.sqrt(1280**2 + 720**2)  # �� 1469
new_threshold_ratio = old_threshold / diagonal  # �� 0.0014

# �Ƽ�ֵ����΢�ſ�
recommended = 0.002
```

### Q4: �Ƿ�Ӱ��BadCase��⣿

**A**: ����Ӱ��
- ʹ BadCase ������ƽ
- ������ֱ��ʵ��µ�����
- `mismatch_threshold` ���� 0.3 ����

---

## ? Ǩ���嵥

�����Ҫȫ�����ù�һ����

- [ ] ���´���ű������ `--normalize-by-resolution`
- [ ] ���Ե�����Ƶ��ȷ����ֵ���ʣ�Ĭ��0.002��
- [ ] �����ĵ���README��˵����һ������
- [ ] (��ѡ) ���´�����ʷ�����Ա���һ����

---

## ? �ܽ�

**���ĳ�����1280��720 ~ 750��960 ��Ϸֱ��ʣ�**��

? **�������ù�һ��**
- �ֱ��ʷ�Χ��1.7�����죩
- δ��һ���ᵼ�� 30-40% ������ƫ��
- BadCase�����ܵ�����Ӱ��

**�Ƽ�����**��
```bash
python batch_with_badcase.py \
    -i your_videos/ \
    -l labels.json \
    -o results/ \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize
```

**Ԥ������**��
- ? �����ֱ��ʵ��µ�ϵͳ��ƫ��
- ? ������Ƶʹ��ͳһ��׼����
- ? BadCase������׼ȷ�ɿ�
- ? ����������п�ѧ�ԺͿɱ���

---

��ϸ�����ĵ���ο���
- [�ֱ��ʹ�ƽ�Է���](./RESOLUTION_FAIRNESS_ANALYSIS.md)
- [��һ��ʵ���ܽ�](./NORMALIZATION_IMPLEMENTATION_SUMMARY.md)

