# ����������� - ���ٲο�

## һ��������

### Ĭ��ʹ�ã��Ƽ���

�������**Ĭ������**��ֱ�����м��ɣ�

```bash
python video_processor.py -i your_video.mp4 -o output/
```

### �����������

����ǹ̶���λ���㣬���Խ���������ٶȣ�

```bash
python video_processor.py -i your_video.mp4 -o output/ --no-camera-compensation
```

## ��������

### ������Ƶ

```bash
# ����ʹ��
python video_processor.py -i video.mp4 -o output/

# ʹ��GPU����
python video_processor.py -i video.mp4 -o output/ --device cuda

# �Զ�������������������ϸ�
python video_processor.py -i video.mp4 -o output/ \
  --camera-ransac-thresh 0.8 \
  --camera-max-features 3000
```

### ͼ������

```bash
python video_processor.py -i image_folder/ -o output/
```

### ��������

```bash
python video_processor.py -i videos/ -o results/ --batch
```

## �����ٲ�

| ���� | ˵�� | Ĭ��ֵ | �Ƽ�ֵ |
|------|------|--------|--------|
| `--no-camera-compensation` | ����������� | ���� | - |
| `--camera-ransac-thresh` | RANSAC��ֵ�����أ� | 1.0 | 0.5-2.0 |
| `--camera-max-features` | ����������� | 2000 | 1000-5000 |

## ����ѡ��

### ? �ʺ������������

- ���ת�����㣨���ơ�ƽ�Ƶȣ�
- ���ӽ���Ƶ
- �ֳֻ����˻�����

### ?? �ɽ����������

- �̶���λ����
- ������˶��ĳ���
- ��Ҫ��촦���ٶ�ʱ

## ������

### �ɹ���ָ��

- **> 80%**: ���㣬�������Ч����
- **60-80%**: ���ã��󲿷�֡�ɹ�����
- **< 60%**: �ϲ������Ҫ�������������

### �в��������

- **< ԭʼ������30%**: ����Ч������
- **30-50%**: ����Ч���е�
- **> 50%**: ����Ч������

## �����ų�

### ���⣺�����ɹ��ʵ�

```bash
# ����������
python video_processor.py -i video.mp4 -o output/ --camera-max-features 3000

# �ſ�RANSAC��ֵ
python video_processor.py -i video.mp4 -o output/ --camera-ransac-thresh 2.0
```

### ���⣺�����ٶ���

```bash
# �����������
python video_processor.py -i video.mp4 -o output/ --no-camera-compensation

# �����������
python video_processor.py -i video.mp4 -o output/ --camera-max-features 1000
```

### ���⣺�в������Ȼ�ܴ�

```bash
# ���ܳ������ʺϵ�Ӧ�Բ��������Խ���
python video_processor.py -i video.mp4 -o output/ --no-camera-compensation
```

## Python API

```python
from video_processor import VideoProcessor

# �������������Ĭ�ϣ�
processor = VideoProcessor(
    device='cuda',
    enable_camera_compensation=True,
    camera_compensation_params={
        'ransac_thresh': 1.0,
        'max_features': 2000
    }
)

# ������Ƶ
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir="output")

# �鿴����ͳ��
if result['camera_compensation_enabled']:
    stats = result['camera_compensation_stats']
    print(f"�ɹ���: {stats['success_rate']:.1%}")
```

## �鿴���

### JSON���

```bash
cat output/analysis_results.json | grep -A 10 "camera_compensation"
```

### �ı�����

```bash
cat output/analysis_report.txt
```

### ���ӻ��Ա�ͼ

```bash
# �鿴�������Ч���Ա�
open output/visualizations/camera_compensation_comparison.png
```

## �����ĵ�

��ϸ��Ϣ��鿴��
- ? [����ʹ��ָ��](CAMERA_COMPENSATION_GUIDE.md)
- ? [����˵��](CAMERA_COMPENSATION_UPDATE.md)
- ? [�����ܽ�](INTEGRATION_SUMMARY.md)

## ����

```bash
# ���в��Խű�
python test_camera_compensation.py

# ʹ��demo���ݲ���
python video_processor.py -i demo_data/ -o demo_output/
```

---

? **��ʾ**: �������Ĭ�����ã����������������ֶ����ã�

