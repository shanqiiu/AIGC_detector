# Ǩ��ָ�ϣ��� batch_with_badcase.py �� video_processor.py

## ? ��Ҫ֪ͨ

**`batch_with_badcase.py` �ѱ��Ƴ�**

���й��������ϵ� `video_processor.py`����ʹ���µ�ͳһ��ڡ�

---

## ? ����Ǩ�ƶ���

### ������ �� ������

#### �����÷�
```bash
# ������
python batch_with_badcase.py -i videos/ -l labels.json

# ������
python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

#### ��������
```bash
# ������
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o output/ \
    --visualize \
    --normalize-by-resolution \
    --camera-ransac-thresh 1.0

# �����������ͬ��ֻ�Ǹ��� -l ��������
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    -o output/ \
    --visualize \
    --normalize-by-resolution \
    --camera-ransac-thresh 1.0
```

---

## ? ����ӳ��

| �ɲ��� | �²��� | ˵�� |
|-------|-------|------|
| `-i, --input` | `-i, --input` + `--batch` | ����� --batch |
| `-l, --labels` | `--badcase-labels` �� `-l` | ����������ȷ |
| `-o, --output` | `-o, --output` | ��ͬ |
| `--visualize` | `--visualize` | ��ͬ |
| `--normalize-by-resolution` | `--normalize-by-resolution` | ��ͬ |
| �������в��� | ��ȫ��ͬ | �����޸� |

---

## ? ����Ǩ��

### Step 1: �ҵ�������

```bash
# ������Ŀ�еľ�����
grep -r "batch_with_badcase.py" .
```

### Step 2: �滻����

```bash
# ���Ҳ��滻
��: batch_with_badcase.py
��: video_processor.py --batch

��: --labels
��: --badcase-labels
```

### Step 3: ��֤

```bash
# ����������
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

---

## ? �ع���ɱ�֤

### ���ܶԵ���

| ԭ���� | ��ʵ�� | ״̬ |
|--------|--------|------|
| ���ر�ǩ | `load_expected_labels()` | ? |
| ����Ƶ+BadCase | `process_single_video(..., expected_label)` | ? |
| ����+BadCase | `batch_process_videos(..., badcase_labels)` | ? |
| BadCase���� | `badcase_analyzer.save_batch_report()` | ? |
| ���в��� | ��ȫ���� | ? |

### ��������ͬ

```
���������:
������ badcase_summary.txt
������ badcase_summary.json
������ badcase_videos.txt
������ {video_name}/badcase_report.txt

���������:
������ badcase_summary.txt      ? ��ͬ
������ badcase_summary.json     ? ��ͬ
������ badcase_videos.txt       ? ��ͬ
������ {video_name}/badcase_report.txt  ? ��ͬ
```

---

## ? Ϊʲôɾ����

1. **����100%����** - �޹�����ʧ
2. **�޴�������** - ���ȷ���������ļ�����
3. **ά����** - ��һ��ڣ����������ͬ��
4. **��������** - �����ظ�������DRYԭ��

---

## ? �����ĵ�����

��Ҫ���µ��ĵ���ʾ�������
- `BADCASE_DETECTION_GUIDE.md`
- `BADCASE_FEATURE_SUMMARY.md`
- `QUICK_START_NORMALIZATION.md`
- �����ĵ��е�����ʾ��

**����**: ȫ���滻�ĵ��е�����ʾ��

---

## ? �Ƽ��¹�����

```bash
# ͳһʹ�� video_processor.py

# ����Ƶ
python video_processor.py -i video.mp4 --normalize-by-resolution

# ��������ͨ��
python video_processor.py -i videos/ --batch --normalize-by-resolution

# ������BadCase��
python video_processor.py -i videos/ --batch -l labels.json --normalize-by-resolution
```

**���򵥡���һ�¡���ǿ��**

