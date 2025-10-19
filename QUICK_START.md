# ? ���ٿ�ʼָ��

## 5��������

### 1?? ��װ (1����)

```bash
# ��¡��Ŀ
cd AIGC_detector

# ��װ����
pip install -r requirements.txt

# ׼��RAFTģ��
# ���� raft-things.pth �� pretrained_models/ Ŀ¼
```

### 2?? ׼������ (1����)

```bash
# ������ƵĿ¼
mkdir my_videos

# ���������Ƶ����Ŀ¼
cp /path/to/your/videos/*.mp4 my_videos/

# ����ѡ��׼����ǩ�ļ�
# ���� labels.json����ʽ���£�
```

```json
{
  "video1": "high",
  "video2": "low",
  "video3": "medium"
}
```

### 3?? ���з��� (3����)

#### ����Ƶ����

```bash
python video_processor.py -i my_videos/test.mp4 -o results/
```

#### ��������

```bash
python video_processor.py -i my_videos/ -o results/ --batch
```

#### BadCase��⣨�Ƽ���

```bash
python video_processor.py \
    -i my_videos/ \
    -o results/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

---

## ? �鿴���

### �ı�����

```bash
# �鿴������Ƶ����
cat results/test/analysis_report.txt

# �鿴��������
cat results/badcase_summary.txt
```

### JSON���

```bash
# ʹ��Python��ȡ
import json
with open('results/test/analysis_results.json') as f:
    data = json.load(f)
    print(f"��̬��: {data['temporal_stats']['mean_dynamics_score']}")
```

### ���ӻ����

```bash
# ���ɿ��ӻ�����Ҫ��� --visualize��
python video_processor.py -i test.mp4 -o results/ --visualize

# �鿴���ӻ�
# results/test/visualizations/ Ŀ¼��������ͼ��
```

---

## ? ���������ٲ�

### ���

```bash
python video_processor.py -i video.mp4
```

### ���Ƽ�����Ϸֱ��ʣ�

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --normalize-by-resolution
```

### ��������BadCase��⣩

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### CPUģʽ

```bash
python video_processor.py -i video.mp4 --device cpu
```

### ���ٲ���

```bash
python video_processor.py -i video.mp4 --frame_skip 3 --max_frames 60
```

---

## ?? �ؼ�����

| ���� | ��ʱʹ�� | ʾ��ֵ |
|------|----------|--------|
| `--batch` | �����Ƶ | - |
| `--normalize-by-resolution` | **��Ϸֱ��ʣ����룩** | - |
| `--badcase-labels` | ������� | `labels.json` |
| `--visualize` | ��Ҫͼ�� | - |
| `--device` | CPU/GPUѡ�� | `cpu` �� `cuda` |
| `--mismatch-threshold` | �������ж� | `0.3`��Ĭ�ϣ� |

---

## ? ������

### ��̬�ȷ���

- **0.0 - 0.2**: ���Ͷ�̬����̬����/�������⣩
- **0.2 - 0.4**: �Ͷ�̬����΢�˶���
- **0.4 - 0.6**: �еȶ�̬�������˶���
- **0.6 - 0.8**: �߶�̬�������˶���
- **0.8 - 1.0**: ���߶�̬�������˶���

### BadCase���س̶�

- **severe**: ���ز�ƥ�䣨���������ɣ�
- **moderate**: �жȲ�ƥ�䣨�����飩
- **minor**: ��΢��ƥ�䣨���ܿɽ��ܣ�

### ʱ���ȶ���

- **> 0.8**: �ȶ���������
- **0.6 - 0.8**: һ�㣨�ɽ��ܣ�
- **< 0.6**: ���ȶ������������⣩

---

## ? ��������

### Q: CUDA out of memory

```bash
# ���ٷֱ��ʻ���֡
python video_processor.py -i video.mp4 --frame_skip 2
```

### Q: ����̫��

```bash
# 1. ʹ��GPU
python video_processor.py -i video.mp4 --device cuda

# 2. �����ɿ��ӻ�
python video_processor.py -i video.mp4  # Ĭ�ϲ�����

# 3. ��֡����
python video_processor.py -i video.mp4 --frame_skip 3
```

### Q: ���ֻ���BadCase

```bash
# ʹ��BadCaseģʽ
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json

# �鿴BadCase�б�
cat output/badcase_videos.txt
```

### Q: ��Ϸֱ�����Ƶ��ô��

```bash
# �������ù�һ����
python video_processor.py \
    -i mixed_videos/ \
    --batch \
    --normalize-by-resolution  # �����������Ҫ��
```

---

## ? ������Ϣ

- �����ĵ���[README.md](README.md)
- ����ԭ���鿴 README.md ��"����ԭ��"�½�
- API�ĵ����鿴��ģ���docstring

---

## ? ���͹�����

```bash
# 1. ׼������
mkdir project_videos
cp *.mp4 project_videos/

# 2. ������ǩ����ѡ��
cat > labels.json << EOF
{
  "video1": "high",
  "video2": "medium"
}
EOF

# 3. ���з���
python video_processor.py \
    -i project_videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    -o results/

# 4. �鿴���
cat results/badcase_summary.txt
ls results/badcase_videos.txt

# 5. ����ѡ����ϸ����BadCase
python video_processor.py \
    -i results/badcase_videos.txt�е�ĳ����Ƶ \
    -o detailed_check/ \
    --visualize \
    --normalize-by-resolution
```

---

<div align="center">

**�������Ѿ�׼�����ˣ�?**

�����⣿�鿴 [README.md](README.md) ���ύ Issue

</div>

