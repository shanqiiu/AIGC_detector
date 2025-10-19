# ������Ƶ����ָ��

## ��������

### 1. ��������ģʽ
һ���Դ�������Ŀ¼�µ�������Ƶ�ļ�

### 2. ���ӻ�����
ͨ�����������Ƿ����ɿ��ӻ����ӿ촦���ٶ�

---

## ʹ�÷���

### ������Ƶ����ԭ�й��ܣ�

```bash
# ����ʹ��
python video_processor.py --input video.mp4 --output output/

# ���ÿ��ӻ����ӿ��ٶȣ�
python video_processor.py --input video.mp4 --output output/ --no-visualize

# ��������
python video_processor.py \
    --input video.mp4 \
    --output output/ \
    --max_frames 100 \
    --frame_skip 2 \
    --device cuda \
    --no-visualize
```

---

### ������Ƶ�����¹��ܣ�

```bash
# �������������������ӻ���
python video_processor.py --input videos/ --output batch_output/ --batch

# ���������������ÿ��ӻ����Ƽ���
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize

# �������� + ����֡�������죩
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize \
    --max_frames 100 \
    --frame_skip 2
```

---

## ����˵��

### ��������

| ���� | ˵�� | Ĭ��ֵ |
|------|------|--------|
| `--input` / `-i` | ����·������Ƶ�ļ�/ͼ��Ŀ¼/��ƵĿ¼�� | ���� |
| `--output` / `-o` | ���Ŀ¼ | `output` |
| `--batch` | ������������ģʽ | False |
| `--no-visualize` | ���ÿ��ӻ����� | False��Ĭ�����ɣ� |

### ���ܲ���

| ���� | ˵�� | Ĭ��ֵ | �Ƽ�ֵ�����٣� |
|------|------|--------|--------------|
| `--max_frames` | �����֡�� | None��ȫ���� | 50-100 |
| `--frame_skip` | ֡��Ծ��� | 1 | 2-3 |
| `--device` | �����豸 | cuda | cuda |

### ��������

| ���� | ˵�� | Ĭ��ֵ |
|------|------|--------|
| `--raft_model` / `-m` | RAFTģ��·�� | None |
| `--fov` | ����ӳ��ǣ��ȣ� | 60.0 |

---

## ����ṹ

### ������Ƶ����

```
output/
������ analysis_results.json      # ��ֵ�����JSON��
������ analysis_report.txt         # ���ֱ���
������ visualizations/             # ���ӻ������������ã�
    ������ frame_0000_analysis.png
    ������ temporal_dynamics.png
    ������ static_ratio_changes.png
```

### ��������

```
batch_output/
������ batch_summary.txt           # ���������ܽ�
������ batch_summary.json          # ���������ܽᣨJSON��
������ video1/                     # ��Ƶ1�Ľ��
��   ������ analysis_results.json
��   ������ analysis_report.txt
��   ������ visualizations/         # ��������ã�
������ video2/                     # ��Ƶ2�Ľ��
��   ������ analysis_results.json
��   ������ analysis_report.txt
������ video3/                     # ��Ƶ3�Ľ��
    ������ ...
```

---

## �ٶ��Ż�����

### �������������Ƽ����ã�

```bash
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize \
    --max_frames 50 \
    --frame_skip 2 \
    --device cuda
```

**��������**:
- `--no-visualize`: ��ʡ **30-50%** ʱ��
- `--max_frames 50`: ֻ����ǰ50֡
- `--frame_skip 2`: ÿ��һ֡�����ٶ����� **2��**

### ��������������

```bash
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize \
    --device cuda
```

**�ص�**:
- ��������֡
- ���ÿ��ӻ����ɺ����������ɣ�
- ƽ���ٶȺ�����

### ���������������ӻ���

```bash
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --device cuda
```

**�ص�**:
- �������п��ӻ�
- �ʺ���Ҫ��ϸ�����ĳ���
- �ٶȽ���

---

## ʵ��ʾ��

### ʾ��1: �����������

**����**: ��100����Ƶ��Ҫ����ɸѡ�����������Ƶ

```bash
python video_processor.py \
    --input D:/videos/ \
    --output D:/quick_check/ \
    --batch \
    --no-visualize \
    --max_frames 30 \
    --frame_skip 3
```

**Ԥ��ʱ��**: Լ 1-2 ����/��Ƶ��30֡��GPU��

### ʾ��2: ��ϸ�����ض���Ƶ

**����**: ��ɸѡ����������Ƶ������ϸ����

```bash
python video_processor.py \
    --input D:/videos/problem_video.mp4 \
    --output D:/detailed_analysis/ \
    --device cuda
```

**���**: �����������ӻ��ͱ���

### ʾ��3: ���ģ��������

**����**: ����1000����Ƶ�����ݼ�

```bash
# ��һ�׶Σ����ٴ����ȡ��ֵ
python video_processor.py \
    --input D:/dataset/ \
    --output D:/results/ \
    --batch \
    --no-visualize \
    --max_frames 50 \
    --frame_skip 2

# �ڶ��׶Σ����쳣��Ƶ���´��������ӻ���
python video_processor.py \
    --input D:/dataset/anomaly_video.mp4 \
    --output D:/detailed_results/ \
    --device cuda
```

---

## ���������ܽᱨ��

����������ɺ󣬻����� `batch_summary.txt` �� `batch_summary.json`��

### batch_summary.txt ʾ��

```
======================================================================
������Ƶ�����ܽ�
======================================================================

����Ƶ��: 10
�ɹ�����: 9
����ʧ��: 1

======================================================================
��ϸ���
======================================================================

��Ƶ: video1
  ״̬: ? �ɹ�
  ֡��: 120
  ƽ����̬�ȷ���: 0.856
  ƽ����̬�������: 0.785
  ʱ���ȶ���: 0.912
  ���Ŀ¼: batch_output/video1

��Ƶ: video2
  ״̬: ? �ɹ�
  ֡��: 95
  ƽ����̬�ȷ���: 1.234
  ƽ����̬�������: 0.654
  ʱ���ȶ���: 0.876
  ���Ŀ¼: batch_output/video2

...
```

### batch_summary.json ʾ��

```json
[
  {
    "video_name": "video1",
    "status": "success",
    "frame_count": 120,
    "mean_dynamics_score": 0.856,
    "mean_static_ratio": 0.785,
    "temporal_stability": 0.912,
    "output_dir": "batch_output/video1"
  },
  ...
]
```

---

## ֧�ֵ���Ƶ��ʽ

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
- `.wmv`

---

## ��������

### Q1: �������ٶȴ��������Ƶ��

```bash
python video_processor.py \
    --input videos/ \
    --output output/ \
    --batch \
    --no-visualize \
    --max_frames 30 \
    --frame_skip 3 \
    --device cuda
```

### Q2: ���ֻ������ֵ�������Ҫ���ӻ���

ʹ�� `--no-visualize` ����

### Q3: ����������ĳ����Ƶʧ������ô�죿

���������������Ƶ��ʧ����Ϣ��¼�� `batch_summary.txt` ��

### Q4: ��λָ����ӻ���

���� `--no-visualize` ���ٴ�����������������Ҫ���ӻ�����Ƶ��

```bash
python video_processor.py --input problem_video.mp4 --output detailed/
```

### Q5: ��ε��������㷨��

�޸Ĵ����еĹ����������� `VideoProcessor.__init__` �У���

```python
# ʹ�� TV-L1�����߾��ȣ�
self.raft_predictor = RAFTPredictor(method='tvl1', device=device)

# ʹ�� RAFT�ٷ�����߾��ȣ�
self.raft_predictor = RAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth',
    device=device
)
```

---

## ���ܲο�

���� 1920x1080 ��Ƶ��GPU: RTX 3080

| ���� | �ٶ� | ���� |
|------|------|------|
| ȫ֡ + ���ӻ� | ~3 min/video | ��� |
| ȫ֡ + �޿��ӻ� | ~2 min/video | �� |
| ÿ3֡ + �޿��ӻ� | ~40 sec/video | �� |
| ǰ30֡ + �޿��ӻ� | ~20 sec/video | ���ټ�� |

---

## �ܽ�

- **��������**: `--batch --no-visualize --max_frames 50 --frame_skip 2`
- **����������**: `--batch --no-visualize`
- **��ϸ����**: Ĭ�ϲ����������ӻ���

ѡ����ʵĲ��������ƽ���ٶȺ�������

