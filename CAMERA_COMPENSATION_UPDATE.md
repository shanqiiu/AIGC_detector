# ����������ܼ��ɸ���˵��

## ��������

�ѳɹ��� `dynamic_motion_compensation` ģ�������������ܼ��ɵ� `video_processor.py` ����Ƶ���������С�

## ��Ҫ�Ķ�

### 1. VideoProcessor �����

#### ������ʼ��������
- `enable_camera_compensation` (bool, Ĭ��True): �Ƿ������������
- `camera_compensation_params` (Dict, ��ѡ): ���������������

#### ���������ԣ�
- `self.camera_compensator`: CameraCompensatorʵ����������ã�

### 2. �����������̸���

**ԭ���̣�**
```
RAFT�������� �� ��̬������� �� ������
```

**�����̣����������������**
```
RAFT�������� �� �������(Homography) �� �в���� �� ��̬������� �� ������
                  ��
            ԭʼ�������������
```

### 3. �����в�������

```bash
--no-camera-compensation        # �������������Ĭ�����ã�
--camera-ransac-thresh FLOAT    # RANSAC��ֵ�����أ���Ĭ��1.0
--camera-max-features INT       # �������������Ĭ��2000
```

### 4. ��������ǿ

#### JSON��������ֶΣ�
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
      "camera_compensation": {
        "inliers": 850,
        "total_matches": 1100,
        "homography_found": true
      }
    }
  ]
}
```

#### �ı����������½ڣ�
```
����˶�����ͳ��:
- �ɹ���: 95.0% (19/20)
- ƽ���ڵ���: 856.3 �� 123.5
- ƽ��ƥ����: 78.0% �� 5.0%
```

### 5. ���ӻ�������ǿ

���� `camera_compensation_comparison.png` ���ӻ�ͼ��չʾ��
- ԭʼ֡
- ԭʼ��������
- �����������
- �в��������

ÿ��ͼ��ʾ3���ؼ�֡�ĶԱȡ�

## ʹ��ʾ��

### ����ʹ�ã�Ĭ���������������

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --device cuda
```

### �Զ��������������

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --camera-ransac-thresh 0.8 \
  --camera-max-features 3000
```

### �����������

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --no-camera-compensation
```

### ��������

```bash
python video_processor.py \
  --input videos/ \
  --output results/ \
  --batch
```

## Python API

```python
from video_processor import VideoProcessor

# �������������Ĭ�ϣ�
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    enable_camera_compensation=True,
    camera_compensation_params={
        'ransac_thresh': 1.0,
        'max_features': 2000
    }
)

# ������Ƶ
frames = processor.load_video("test.mp4")
result = processor.process_video(frames, output_dir="output")

# �鿴�������ͳ��
if result['camera_compensation_enabled']:
    stats = result['camera_compensation_stats']
    print(f"��������ɹ���: {stats['success_rate']:.1%}")
```

## ����ϸ��

### �������ԭ��

1. **���������ƥ��**
   - ʹ��ORB�����������Ĭ�ϣ�
   - BFMatcher��������ƥ��

2. **��Ӧ�Թ���**
   - RANSAC�㷨����֡�䵥Ӧ�Ծ���
   - ȥ����㣬�����ڵ�

3. **�����ֽ�**
   ```
   ԭʼ���� = ����˶����� + ��ʵ�����˶�����
   �в���� = ԭʼ���� - ����˶�����
   ```

4. **��������**
   - ʹ�òв�������о�̬���嶯̬�ȷ���
   - ��׼ȷ�ط�ӳ��ʵ�������˶����쳣

### ����Ӱ��

- ����Լ10-20%�Ĵ���ʱ��
- ��Ҫ��������������ƥ��
- ��ͨ�����������Ż�����

## ������

- ? �����ݣ�Ĭ�������������������ͨ����������
- ? ���й��ܲ���Ӱ�죺����ԭ�й�����������
- ? �����ʽ���ݣ������ֶβ�Ӱ�����н����߼�

## ���Խ���

### ���Գ���1�����ת�����㾲̬����
```bash
python video_processor.py \
  --input test_data/ \
  --output test_output/ \
  --device cuda
```

**Ԥ�ڽ����**
- ��������ɹ��� > 80%
- �в������������С��ԭʼ����
- ��̬���嶯̬�ȷ��� < 1.0

### ���Գ���2���̶���λ����
```bash
python video_processor.py \
  --input fixed_camera_video.mp4 \
  --output test_output/ \
  --no-camera-compensation
```

**Ԥ�ڽ����**
- �����ٶȸ���
- ��̬�ȷ��������ò���ʱ�������Ϊ������˶���

### ���Գ���3����������
```bash
python video_processor.py \
  --input videos/ \
  --output batch_results/ \
  --batch \
  --device cuda
```

**Ԥ�ڽ����**
- ������Ƶ�ɹ�����
- �������������ܽ��ļ�
- ÿ����Ƶ�����������ͳ����Ϣ

## ����ĵ�

- [CAMERA_COMPENSATION_GUIDE.md](CAMERA_COMPENSATION_GUIDE.md) - ��ϸʹ��ָ��
- [dynamic_motion_compensation/README.md](dynamic_motion_compensation/README.md) - �ײ�ģ��˵��

## ��һ���ƻ�

- [ ] ��Ӹ�������������ӻ�ѡ��
- [ ] ֧���Զ������������ѡ��SIFT/SURF/ORB��
- [ ] ������������������ָ��
- [ ] �Ż�������������
- [ ] ֧����ȹ��Ƶĸ߼�����

## ��������

2025-10-19

