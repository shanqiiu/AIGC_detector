# ? ��Ŀ������� - ͳһ���ʹ��ָ��

## ? �������

**batch_with_badcase.py** �����ϵ� **video_processor.py**������ֻ��һ�����

---

## ? �µ�ʹ�÷�ʽ

### 1. ����Ƶ����

```bash
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution
```

### 2. ������������BadCase��⣩

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --normalize-by-resolution
```

### 3. �������� + BadCase��⣨�Ƽ���?

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### 4. ��������ʾ��

```bash
python video_processor.py \
    -i D:/my_git_projects/data/Multi-View_Consistency \
    --batch \
    --badcase-labels labels.json \
    -o output/ \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize \
    --device cuda \
    --mismatch-threshold 0.3
```

---

## ? ������

**��������Ȼ����**���Զ�ת������

```bash
# �ɷ�ʽ���Կ��ã�
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution

# ����ʾ��ʾ���Զ�ת���� video_processor.py
```

---

## ? ���������б�

### �������
- `--input, -i`: ����·�����ļ�/Ŀ¼��
- `--output, -o`: ���Ŀ¼��Ĭ��: output��

### ģʽ����
- `--batch`: ��������ģʽ

### BadCase��⣨��ѡ��
- `--badcase-labels, -l`: ��ǩ�ļ�������BadCase���
- `--mismatch-threshold`: BadCase��ƥ����ֵ��Ĭ��0.3��

### �ֱ��ʹ�һ����ǿ���Ƽ���?
- `--normalize-by-resolution`: ���ù�һ��
- `--flow-threshold-ratio`: ��һ����ֵ��Ĭ��0.002��

### �������
- `--no-camera-compensation`: �����������
- `--camera-ransac-thresh`: RANSAC��ֵ��Ĭ��1.0��
- `--camera-max-features`: �������������Ĭ��2000��

### ����
- `--visualize`: ���ɿ��ӻ����
- `--device`: cuda/cpu
- `--fov`: ����ӳ���
- `--raft_model, -m`: RAFTģ��·��

---

## ? ����ʹ�ó���

### ����1: ����Ƶ��������

```bash
python video_processor.py \
    -i video.mp4 \
    -o analysis/ \
    --normalize-by-resolution \
    --visualize
```

### ����2: ����BadCaseɸѡ��������Ҫ������

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -l labels.json \
    -o badcase_output/ \
    --normalize-by-resolution
```

**���**��
- `badcase_summary.txt` - BadCaseͳ�Ʊ���
- `badcase_summary.json` - JSON��ʽ���
- `badcase_videos.txt` - BadCase��Ƶ�б�
- `{video_name}/` - ÿ����Ƶ����ϸ����

### ����3: ��Ϸֱ��ʹ�ƽ����

```bash
# ������Ƶ��1280��720 ~ 750��960
python video_processor.py \
    -i mixed_resolution_videos/ \
    --batch \
    -l labels.json \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
```

---

## ? ���ϴ����ĸĽ�

| �Ľ�ά�� | Ч�� |
|---------|------|
| **�����** | ��һ��ڣ�-45���ظ����� |
| **����һ��** | �Զ�ͬ���������ֶ�ά�� |
| **�������** | BadCase��Ϊ��ѡģ�� |
| **������** | ������������� |
| **ά���ɱ�** | 1��main()���2�� |

---

## ? ������֤

### ��֤�����Ƿ�ɹ�

```bash
# ����ͳһ���
python video_processor.py --help | grep badcase-labels
# Ӧ����ʾ: --badcase-labels, -l

# ����wrapper
python batch_with_badcase.py --help
# Ӧ����ʾ��ʾ��Ϣ
```

### ��֤BadCase���

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -l labels.json \
    -o test_output/

# ������
ls test_output/
# Ӧ�ð���: badcase_summary.txt, badcase_summary.json, badcase_videos.txt
```

---

## ?? ��Ҫ��ʾ

### �������Ļ�Ϸֱ��ʳ���

**������ӵĲ���**��
```bash
--normalize-by-resolution  # ȷ���ֱ��ʹ�ƽ��
```

**�����Ƽ�����**��
```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -l labels.json \
    --normalize-by-resolution \
    -o results/
```

---

## ? ����ĵ�

- `INTEGRATION_COMPLETE.md` - ������ϸ˵��
- `FINAL_INTEGRATION_ANALYSIS.md` - ���Ϸ���
- `QUICK_START_NORMALIZATION.md` - ��һ��ʹ��ָ��
- `THRESHOLDS_COMPLETE_GUIDE.md` - ��ֵ����ָ��

---

## ? �ܽ�

**���Ϻ������**��
- ? ��һ�������ģʽ
- ? BadCase����Ϊ��ѡ����
- ? ������ȫͳһ
- ? 100%������
- ? ��������

**�Ƽ�ʹ��**: `video_processor.py` ͳһ��ڣ���������������һ�¡�ά���򵥣�

