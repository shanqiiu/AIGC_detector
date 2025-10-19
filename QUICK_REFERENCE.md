# ���ٲο�

## ? ����������Ƶ����ã�

### �����������Ƽ���
```bash
python video_processor.py \
    --input videos/ \
    --output batch_results/ \
    --batch \
    --no-visualize \
    --max_frames 50 \
    --frame_skip 2
```

### ����������
```bash
python video_processor.py \
    --input videos/ \
    --output batch_results/ \
    --batch \
    --no-visualize
```

---

## ? ������Ƶ����

### ���ټ��
```bash
python video_processor.py --input video.mp4 --output output/ --no-visualize
```

### ���������������ӻ���
```bash
python video_processor.py --input video.mp4 --output output/
```

---

## ? �����ٲ�

| ���� | ���� | �Ƽ�ֵ |
|------|------|--------|
| `--batch` | ��������ģʽ | - |
| `--no-visualize` | ���ÿ��ӻ������٣� | ����ʱʹ�� |
| `--max_frames 50` | ֻ����ǰ50֡ | ���ټ�� |
| `--frame_skip 2` | ÿ��1֡���� | ����2�� |
| `--device cuda` | ʹ��GPU | Ĭ�� |

---

## ? ����ļ�

### ������Ƶ
- `analysis_results.json` - ��ֵ���
- `analysis_report.txt` - ���ֱ���
- `visualizations/` - ���ӻ���������ã�

### ��������
- `batch_summary.txt` - �ܽᱨ��
- `batch_summary.json` - JSON���
- `video1/`, `video2/`... - ����Ƶ���

---

## ? ʹ�ó���

| ���� | ���� |
|------|------|
| 100����Ƶ����ɸѡ | `--batch --no-visualize --max_frames 30 --frame_skip 3` |
| 10����Ƶ��ϸ���� | `--batch --no-visualize` |
| ������Ƶ�������� | ���� `--no-visualize` |

