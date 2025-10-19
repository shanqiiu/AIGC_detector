# ? �ع����Ż�ʵʩ��ɱ���

## ����
2025-10-19

---

## ? ��ɵ��ع�����

### 1. �����ص�������batch_with_badcase.py + badcase_detector.py��

**����**�������ļ����ڴ����ظ���ͳ�ƺͱ������ɴ���

**���**��
- ͳһ�� `badcase_detector.py::BadCaseAnalyzer`
- ����������`generate_batch_summary()`, `save_batch_report()`
- ���� `batch_with_badcase.py`��267�У�-122�У�-31%��

**����**��
- ? �����ظ�����
- ? ͳһ�����ʽ
- ? ְ���������

---

### 2. �������ģ�龫��

**����**��
- `dynamic_motion_compensation/` ����δʹ�õ�ģ��
- `static_object_analyzer.py` �����ظ��������������

**���**��
- ? ���� `CameraCompensator`�����Ĺ��ܣ�
- ? ɾ�� `ObjectSE3Estimator`��δʹ�ã�
- ? ɾ�� `se3_utils.py`��δʹ�ã�
- ? �Ƴ� `static_object_analyzer.py::CameraMotionEstimator`���ظ���
- ? �ƶ� `cli.py` �� `examples/camera_compensation_demo.py`

**����**��
- ������٣�-420�У�-48%��
- ���������ص�
- ģ��ְ����ȷ

---

### 3. ����ͳһ

**����**��`batch_with_badcase.py` �� `video_processor.py` ������һ��

**���**��
- ��� `--visualize` �� batch_with_badcase.py
- ��� `--camera-ransac-thresh` �� `--camera-max-features`
- ͳһ����������Ĭ��ֵ

**����**��
- ? ����һ����
- ? �û�����ͳһ
- ? �ĵ�ά����

---

### 4. �ֱ��ʹ�һ�������Ĺ��ܣ�?

**����**����ͬ�ֱ�����Ƶ�����������ƽ
- 1280��720 vs 640��360����̬�ȷ���ƫ�� 30-40%
- BadCase����ֱܷ���Ӱ��

**���**��
- ʵ�ֶԽ��߹�һ����`normalized_flow = flow / sqrt(width? + height?)`
- ��Ӳ�����`--normalize-by-resolution`, `--flow-threshold-ratio`
- �����ݣ�Ĭ�Ϲر�

**��֤���**��
```
δ��һ��������ϵ�� 39.4% ? ���ز���ƽ
��һ���󣺱���ϵ�� 0.0%  ? ��ȫ��ƽ
��ƽ��������100%
```

**����**��
- ? �����ֱ���ϵͳ��ƫ��
- ? ��ͬ�ߴ���Ƶ��ֱ�ӱȽ�
- ? BadCase������׼ȷ
- ? ������ҵ���ʵ��

---

## ? �����ع�ͳ��

| ��Ŀ | �ع�ǰ | �ع��� | �仯 |
|------|--------|--------|------|
| `batch_with_badcase.py` | 389�� | 250�� | -139�� (-36%) |
| `badcase_detector.py` | 563�� | 718�� | +155�У��������ܣ�|
| `static_object_analyzer.py` | 528�� | 400�� | -128�� (-24%) |
| `dynamic_motion_compensation/` | 346�� | 86�� | -260�� (-75%) |
| **�ܼ�** | 1826�� | 1454�� | **-372�� (-20%)** |

---

## ? �ؼ��Ľ�

### ��������
- ? ���������ص�
- ? ְ���������
- ? ģ�黯���
- ? ��linter����

### ����������
- ? ����ԭ�й��ܱ���
- ? ���100%����
- ? �����ֱ��ʹ�һ�����ؼ���
- ? ����ͳһ�淶

### ��ƽ����׼ȷ��
- ? ����ֱ��ʲ���ƽ����
- ? BadCase����׼ȷ
- ? ֧�ֻ�Ϸֱ�����������
- ? ������Ƶ����������׼

---

## ? ���������б�ͳһ��

### ��������
```bash
--input, -i              # ����·��
--output, -o            # ���Ŀ¼
--labels, -l            # ��ǩ�ļ���batch_with_badcase.pyר�ã�
--raft_model, -m        # RAFTģ��·��
--device                # cuda/cpu
--fov                   # ����ӳ��ǣ��ȣ�
```

### �����������
```bash
--no-camera-compensation        # �����������
--camera-ransac-thresh <float>  # RANSAC��ֵ�����أ�
--camera-max-features <int>     # �����������
```

### �ֱ��ʹ�һ��������������?
```bash
--normalize-by-resolution       # ���ù�һ�����Ƽ���
--flow-threshold-ratio <float>  # ��һ����ֵ��Ĭ��0.002��
```

### ��������
```bash
--visualize                     # ���ɿ��ӻ�
--mismatch-threshold <float>    # BadCase��ֵ
--filter-badcase-only          # ֻ����BadCase
```

---

## ? �Ƽ�ʹ�÷�ʽ

### ������ĳ�����1280��720 ~ 750��960 ��Ϸֱ��ʣ�

```bash
# �������� + BadCase��⣨�Ƽ����ã�
python batch_with_badcase.py \
    -i D:\my_git_projects\data\Multi-View_Consistency \
    -l labels.json \
    -o output/ \
    --normalize-by-resolution \
    --visualize \
    --device cuda

# ˵����
# --normalize-by-resolution  �� ���룡�����ֱ���Ӱ��
# --visualize               �� ���ɶԱ�ͼ
# ��������ʹ��Ĭ��ֵ����
```

### ����Ƶ����

```bash
python video_processor.py \
    -i video.mp4 \
    -o test_output/ \
    --normalize-by-resolution \
    --visualize
```

---

## ? �ĵ�����

### �û�ָ��
- [���ٿ�ʼ - ��һ��](./docs/QUICK_START_NORMALIZATION.md)
- [����ͳһ����](./docs/PARAMETER_UNIFICATION.md)

### �����ĵ�
- [�ֱ��ʹ�ƽ�Է���](./docs/RESOLUTION_FAIRNESS_ANALYSIS.md)
- [��һ��ʵ���ܽ�](./docs/NORMALIZATION_IMPLEMENTATION_SUMMARY.md)
- [�����������](./docs/CAMERA_COMPENSATION_ANALYSIS.md)
- [�ع��ܽ�](./docs/REFACTORING_SUMMARY.md)

---

## ?? ��Ҫ��ʾ

### ���ڻ�Ϸֱ��ʳ���

**�������ù�һ��** `--normalize-by-resolution`������
- ? �ͷֱ�����Ƶ��ϵͳ�Ե͹�������©��BadCase��
- ? �߷ֱ�����Ƶ��ϵͳ�Ը߹�����������BadCase��
- ? ��������޷�����Ƶ�Ƚ�
- ? �����Ͽ�ѧ������׼

### ������

- ? Ĭ�Ϲرչ�һ��������ԭ����Ϊ
- ? ���нű������޸ļ�������
- ? ������Ҫʱͨ����������

---

## ? ���ļ�ֵ

1. **������������**������ 372 ��������루-20%��
2. **����������**������ͳһ�����ӻ�����
3. **��ƽ�Ա�֤**���ֱ��ʹ�һ��������ϵͳ��ƫ��
4. **��������**����linter������ȫ������

---

## ? ��һ������

1. ʹ�ù�һ��ģʽ���´���������Ƶ��
2. �ȽϹ�һ��ǰ���BadCase�����
3. ����ʵ�����΢�� `flow_threshold_ratio`��0.0015~0.0025��
4. ����ͳһ��������׼����ֵ

**���Ľ���**�������ڿ�ʼ�������µ���������Ӧ���� `--normalize-by-resolution`��

