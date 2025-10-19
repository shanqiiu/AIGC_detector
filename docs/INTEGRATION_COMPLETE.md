# �ļ�������ɱ���

## ? �������

**����**: 2025-10-19

---

## ? ��������

### ���ı��

�� `batch_with_badcase.py` �� BadCase ��⹦�����ϵ� `video_processor.py`

---

## ? ����ǰ��Ա�

### Before��2�������ļ���

```
video_processor.py (803��)
���� ����Ƶ����
���� ��������
���� ���ӻ�
���� CLI (��ͨģʽ)

batch_with_badcase.py (249��)
���� ����Ƶ + BadCase���
���� ���� + BadCase���  �� �ظ�85%
���� ��ǩ����
���� CLI (BadCaseģʽ)    �� �ظ�95%
```

### After��ͳһ��� + wrapper��

```
video_processor.py (930��)
���� ����Ƶ����֧�ֿ�ѡBadCase��
���� ��������֧�ֿ�ѡBadCase��
���� ���ӻ�
���� BadCase���
���� ��ǩ����
���� ͳһCLI

batch_with_badcase.py (77��) - wrapper
���� ����ת�� �� video_processor.py
```

**�������**: (803 + 249) - (930 + 77) = **45��**  
**�ظ�����**: 85% �� 0%

---

## ? �µ�ʹ�÷�ʽ

### ��ʽ1: ͳһ��ڣ��Ƽ���

```bash
# ����Ƶ����
python video_processor.py -i video.mp4 --normalize-by-resolution

# ������������BadCase��
python video_processor.py -i videos/ --batch --normalize-by-resolution

# �������� + BadCase���
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### ��ʽ2: ���ݾ�����

```bash
# ��Ȼ���ã��Զ�ת���� video_processor.py
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution
```

---

## ? ͳһ��Ĳ����б�

### ��������
```bash
--input, -i              # ���루�ļ�/Ŀ¼��
--output, -o            # ���Ŀ¼
--raft_model, -m        # RAFTģ��·��
--device                # cuda/cpu
--fov                   # ����ӳ���
```

### ģʽ����
```bash
--batch                 # ��������ģʽ
```

### BadCase��⣨��ѡ��?
```bash
--badcase-labels, -l    # ��ǩ�ļ�������BadCase��⣩
--mismatch-threshold    # ��ƥ����ֵ��Ĭ��0.3��
```

### �������
```bash
--no-camera-compensation
--camera-ransac-thresh
--camera-max-features
```

### �ֱ��ʹ�һ��
```bash
--normalize-by-resolution  # ���ù�һ�����Ƽ���
--flow-threshold-ratio     # ��һ����ֵ��Ĭ��0.002��
```

### ����
```bash
--visualize             # ���ɿ��ӻ�
--max_frames            # ���֡��
--frame_skip            # ֡��Ծ
```

---

## ? ����ӳ�䣨�ɡ��£�

| �����batch_with_badcase.py�� | �����video_processor.py�� |
|------------------------------|---------------------------|
| `--labels, -l` | `--badcase-labels, -l` |
| `--input, -i` | `--input, -i --batch` |
| �������� | ��ȫ��ͬ |

**�Զ�ת��**: wrapper �Զ��������ӳ��

---

## ? ��������

### 1. �����
- ��һ��ڵ�
- �����ظ�����
- ά���ɱ�����

### 2. �������
```python
# 3��ģʽͳһ����
ģʽ1: ����Ƶ����
ģʽ2: ��������
ģʽ3: ���� + BadCase���

# ͨ���������ʵ��
--batch            �� ����ģʽ
--badcase-labels   �� ����BadCase
```

### 3. ����һ����
- ? �Զ�ͬ���������ֶ�ά��
- ? �������������������÷���Ч
- ? ���������һ�µ�bug

### 4. ������
- ? �ɽű���������
- ? �������Զ�ת��
- ? �����޸����д���

---

## ? �������

### BadCase��Ϊ��ѡģ��

```python
# video_processor.py

def process_single_video(..., expected_label=None):
    # ������Ƶ
    result = processor.process_video(...)
    
    # ��ѡ��BadCase���
    if expected_label is not None:
        badcase_result = processor.badcase_analyzer.analyze_with_details(...)
        # ���BadCase��Ϣ
    
    return result

def batch_process_videos(..., badcase_labels=None):
    for video in videos:
        expected = badcase_labels.get(video_name) if badcase_labels else None
        result = process_single_video(..., expected)
    
    # �����Ƿ��б�ǩѡ�񱨸�����
    if badcase_labels:
        save_badcase_report(...)
    else:
        save_batch_summary(...)
```

**�ؼ�**: һ������������ģʽ��ͨ����ѡ��������

---

## ? ���µ��ĵ�

�Ѵ���/���£�
- `INTEGRATION_COMPLETE.md` (���ĵ�)
- `FINAL_INTEGRATION_ANALYSIS.md` (�����ĵ�)

��Ҫ���£�
- `README.md` - ����ʹ��ʾ��
- `QUICK_START.md` - ��������

---

## ? ��֤����

### ����1: ����Ƶ����
```bash
python video_processor.py -i test.mp4 --normalize-by-resolution
# ? Ӧ����������
```

### ����2: ����������BadCase��
```bash
python video_processor.py -i videos/ --batch --normalize-by-resolution
# ? Ӧ������ batch_summary.txt
```

### ����3: ���� + BadCase
```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
# ? Ӧ������ badcase_summary.txt
```

### ����4: ����wrapper
```bash
python batch_with_badcase.py -i videos/ -l labels.json --normalize-by-resolution
# ? Ӧ��ת���� video_processor.py ����������
```

---

## ? �ܽ�

| ά�� | ����ǰ | ���Ϻ� | �Ľ� |
|------|-------|--------|------|
| �ļ��� | 2��main | 1��main + 1��wrapper | �� |
| �������� | 1052�� | 1007�� | -45�� |
| �ظ����� | 85% | 0% | ? ���� |
| ά����� | 2�� | 1�� | �� |
| ����һ���� | �ֶ�ͬ�� | �Զ�һ�� | ? |
| ������ | N/A | 100% | ? |

**���ļ�ֵ**��
- ? ��һ��ʵ��Դ��Single Source of Truth��
- ? �����Զ�һ��
- ? ����ģ�黯��BadCase��ѡ��
- ? ��ȫ������

---

## ? �Ƽ�ʹ��

**�����ڿ�ʼ��ͳһʹ�� video_processor.py**��

```bash
# ����Ƶ
python video_processor.py -i video.mp4 --normalize-by-resolution

# ��������ͨ��
python video_processor.py -i videos/ --batch --normalize-by-resolution

# ������BadCase��
python video_processor.py \
    -i videos/ \
    --batch \
    -l labels.json \
    --normalize-by-resolution
```

�ɵ� `batch_with_badcase.py` �����Կ��ã�������ʾ����ʹ������ڡ�

