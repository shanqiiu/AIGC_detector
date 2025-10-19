# �����������ʹ��ָ��

## ����

������������Ѽ��ɵ���Ƶ���������У������ڶ��ӽǣ�����˶��������¸�׼ȷ��������Ƶ�������ù���ͨ�����ƺ�ȥ������˶�����Ĺ�����������ʵ�������˶����쳣���Ӷ��ṩ����ȷ�Ķ�̬�ȷ�����

## ����ԭ��

### 1. �����ֽ�
```
ԭʼ���� = ����˶����� + ��ʵ�����˶�����
�в���� = ԭʼ���� - ����˶�����
```

### 2. ����˶�����
- ʹ�� ORB/SIFT ��������ƥ��
- ͨ�� RANSAC ����֡�䵥Ӧ�Ծ���Homography��
- �ӵ�Ӧ�Ծ�������������Ĺ���
- ��ԭʼ�����м�ȥ����������õ��в����

### 3. �в��������
- �в�������ں����ľ�̬���嶯̬�ȷ���
- ��׼ȷ�ط�ӳ�����е���ʵ�˶����쳣

## ʹ�÷���

### ����ʹ�ã��������Ĭ�����ã�

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --device cuda
```

### �����������

�������ʹ��ԭʼ�����������������������

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --no-camera-compensation
```

### �Զ��������������

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --camera-ransac-thresh 0.8 \
  --camera-max-features 3000
```

### ��������ģʽ

```bash
python video_processor.py \
  --input videos/ \
  --output results/ \
  --batch \
  --device cuda
```

## ����˵��

### ���������ز���

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--no-camera-compensation` | flag | False | �������������Ĭ�����ã� |
| `--camera-ransac-thresh` | float | 1.0 | RANSAC��ֵ�����أ���ԽСԽ�ϸ� |
| `--camera-max-features` | int | 2000 | ����������� |

### ��������

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--input/-i` | str | ���� | ������Ƶ��ͼ��Ŀ¼ |
| `--output/-o` | str | 'output' | ���Ŀ¼ |
| `--raft_model/-m` | str | pretrained_models/raft-things.pth | RAFTģ��·�� |
| `--device` | str | 'cuda' | �����豸 (cuda/cpu) |
| `--max_frames` | int | None | �����֡�� |
| `--frame_skip` | int | 1 | ֡��Ծ��� |
| `--fov` | float | 60.0 | ����ӳ��ǣ��ȣ� |
| `--batch` | flag | False | ��������ģʽ |
| `--no-visualize` | flag | False | ���ÿ��ӻ����� |

## ������

### JSON����ļ� (`analysis_results.json`)

�������������JSON���������������Ϣ��

```json
{
  "camera_compensation_enabled": true,
  "camera_compensation_stats": {
    "success_rate": 0.95,
    "successful_frames": 19,
    "total_frames": 20,
    "mean_inliers": 856.3,
    "std_inliers": 123.5,
    "mean_match_ratio": 0.78,
    "std_match_ratio": 0.05
  },
  "frame_results": [
    {
      "frame_index": 0,
      "camera_compensation": {
        "inliers": 850,
        "total_matches": 1100,
        "homography_found": true
      },
      "static_dynamics": {...},
      "global_dynamics": {...}
    }
  ]
}
```

### �ı����� (`analysis_report.txt`)

�����л�����������ͳ����Ϣ��

```
��Ƶ������Ϣ:
- ��֡��: 20
- ����֡��: 19
- �������: ����

����˶�����ͳ��:
- �ɹ���: 95.0% (19/20)
- ƽ���ڵ���: 856.3 �� 123.5
- ƽ��ƥ����: 78.0% �� 5.0%
```

### ���ӻ����

�� `visualizations/` Ŀ¼�»����ɣ�

1. **`camera_compensation_comparison.png`** - �������Ч���Ա�ͼ
   - ��ʾԭʼ��������������Ͳв�����ĶԱ�
   - չʾ����ǰ���Ч������

2. **`temporal_dynamics.png`** - ʱ��̬������
3. **`static_ratio_changes.png`** - ��̬��������仯
4. **`frame_XXXX_analysis.png`** - �ؼ�֡��ϸ����

## Ӧ�ó���

### �ʺ�ʹ����������ĳ���

? **���ת�����㾲̬����**
- ���磺���ƽ�������ܵȾ�̬�������Ƶ
- �����������ȥ������˶���ֻ�����쳣�˶�

? **���ӽ���Ƶ����**
- ��Ҫ��������˶��������˶��ĳ���
- ����AIGC���ɵĶ��ӽ�һ����

? **�˶�����������Ƶ**
- �ֳ���������˻������������˶��ĳ���

### ���ʺ�ʹ����������ĳ���

? **��ȫ��̬����Ƶ**
- �̶���λ���㣬������˶�
- ���Խ��������������ߴ����ٶ�

? **������ʵ�����˶��ĳ���**
- ����������д�����ʵ�˶�����Ӧ�Թ��ƿ��ܲ�׼ȷ
- ����ʹ�ø��߼��������������

## ����ϸ��

### ��Ӧ�Թ���

ʹ�� OpenCV �� `findHomography` ���е�Ӧ�Թ��ƣ�
- ������⣺ORB��Ĭ�ϣ��� SIFT
- ƥ�䷽��������ƥ�䣨BFMatcher��
- RANSAC��ȥ����㣬�����ڵ�

### �������Ž���

**RANSAC��ֵ (`--camera-ransac-thresh`)**
- Ĭ��ֵ��1.0 ����
- �����ȶ������ڵ������Խ��͵� 0.5-0.8
- �������ӡ����ڵ���������ߵ� 1.5-2.0

**����������� (`--camera-max-features`)**
- Ĭ��ֵ��2000
- �߷ֱ�����Ƶ��������ߵ� 3000-5000
- ���ٴ������Խ��͵� 1000-1500

### ����Ӱ��

- �����������������Լ 10-20% �Ĵ���ʱ��
- ��Ҫ��������������ƥ��
- CPUģʽ��Ӱ�������

## ��̽ӿ�

### Python API

```python
from video_processor import VideoProcessor

# ������������Ĭ���������������
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    enable_camera_compensation=True,  # Ĭ��True
    camera_compensation_params={
        'ransac_thresh': 1.0,
        'max_features': 2000
    }
)

# ������Ƶ
frames = processor.load_video("test_video.mp4")

# ������Ƶ
result = processor.process_video(frames, output_dir="output")

# �鿴����������
if result['camera_compensation_enabled']:
    comp_stats = result['camera_compensation_stats']
    print(f"�����ɹ���: {comp_stats['success_rate']:.1%}")
```

### �����������

```python
processor = VideoProcessor(
    enable_camera_compensation=False
)
```

## �����ų�

### ����1����������ɹ��ʵ�

**֢״**��`success_rate` < 0.5

**����ԭ��**��
- �����㲻��
- �����������
- �˶�ģ������

**�������**��
```bash
# ������������
--camera-max-features 3000

# �ſ�RANSAC��ֵ
--camera-ransac-thresh 2.0
```

### ����2���в������Ȼ�ܴ�

**֢״**���в�������Ƚӽ�ԭʼ����

**����ԭ��**��
- �������д�����ʵ�˶�
- ��Ӧ��ģ�Ͳ��ʺϣ����ƽ�泡����
- ����˶����ڸ���

**�������**��
- ���ǽ����������
- ��ʹ�ø��߼��Ĳ�������������ȹ��ƣ�

### ����3�������ٶ���

**֢״**������ʱ�����

**�������**��
```bash
# ������������
--camera-max-features 1000

# ������������
--no-camera-compensation

# ����ÿ��ӻ�
--no-visualize
```

## δ���Ľ�

�ƻ��еĹ��ܣ�
- [ ] ֧����ȹ��Ƶĸ߼�����
- [ ] ֧��SE(3)�����˶�����
- [ ] ֧�ֶ�Ŀ���������
- [ ] ֧��Rolling Shutter����
- [ ] ����Ӧ��������

## �ο�����

- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)
- [Multiple View Geometry in Computer Vision](http://www.robots.ox.ac.uk/~vgg/hzbook/)
- OpenCV Homography Documentation

## ��ϵ��ʽ

����������飬���ύ Issue �� Pull Request��

