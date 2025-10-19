# batch_with_badcase.py �� video_processor.py ���Ϸ���

## ? �����ص�����

### ��ͬ����

| ���� | video_processor.py | batch_with_badcase.py | �ص��� |
|------|-------------------|---------------------|--------|
| ����Ƶ���� | ? `process_single_video()` | ? `process_with_badcase_detection()` | 90% |
| �������� | ? `batch_process_videos()` | ? `batch_process_with_badcase()` | 85% |
| �������� | ? `main()` + argparse | ? `main()` + argparse | 95% |
| VideoProcessor��ʼ�� | ? | ? | 100% |

### ���й���

| ���� | video_processor.py | batch_with_badcase.py |
|------|-------------------|---------------------|
| ͼ�����д��� | ? | ? |
| BadCase��� | ? | ? |
| ��ǩ���� | ? | ? |
| BadCase���� | ? | ? |

**����**��Լ **85%** �����ص����������ϣ�

---

## ? ���Ϸ���

### �������� BadCase �����Ϊ��ѡģ��

```
ͳһ���: video_processor.py

������ ģʽ1: ����Ƶ����
��   python video_processor.py -i video.mp4
��
������ ģʽ2: ������������BadCase��
��   python video_processor.py -i videos/ --batch
��
������ ģʽ3: �������� + BadCase���
    python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

---

## ?? ʵʩ����

### Step 1: �ϲ����������߼�

```python
# video_processor.py (���Ϻ�)

def batch_process_videos(processor, input_dir, output_dir, camera_fov, 
                        badcase_labels=None):  # �� ��������
    """
    ����������Ƶ
    
    Args:
        badcase_labels: ��ѡ��������ǩ�ֵ䣬����BadCase���
    """
    
    # ... ������Ƶ�ļ� ...
    
    results = []
    for video_path in video_files:
        if badcase_labels is not None:
            # BadCaseģʽ
            expected = badcase_labels.get(video_name, 'dynamic')
            result = process_with_badcase_detection(
                processor, video_path, expected, output_dir, camera_fov
            )
        else:
            # ��ͨģʽ
            result = process_single_video(
                processor, video_path, output_dir, camera_fov
            )
        results.append(result)
    
    # ������
    if badcase_labels is not None:
        # BadCase����
        summary = processor.badcase_analyzer.generate_batch_summary(results)
        processor.badcase_analyzer.save_batch_report(summary, results, output_dir)
    else:
        # ��ͨ����
        save_batch_summary(results, output_dir)
    
    return results
```

### Step 2: ͳһ��������

```python
# video_processor.py main()

parser.add_argument('--batch', action='store_true',
                   help='��������ģʽ')

# BadCase��ز�������ѡ��
parser.add_argument('--badcase-labels', '-l', default=None,
                   help='������ǩ�ļ���JSON��������BadCase���')
parser.add_argument('--mismatch-threshold', type=float, default=0.3,
                   help='BadCase��ƥ����ֵ')

# ʹ��
if args.batch:
    if args.badcase_labels:
        # BadCaseģʽ
        labels = load_labels(args.badcase_labels)
        processor.setup_badcase_detector(args.mismatch_threshold)
    batch_process_videos(..., badcase_labels=labels if args.badcase_labels else None)
```

### Step 3: ���� batch_with_badcase.py

```python
# batch_with_badcase.py (��Ϊwrapper)

"""
������wrapper���ض��� video_processor.py
����ֱ��ʹ��: python video_processor.py --batch --badcase-labels labels.json
"""

import sys
import subprocess

# ת������
args = sys.argv[1:]
# --labels �� --badcase-labels
args = [arg.replace('--labels', '--badcase-labels') for arg in args]
# ��� --batch
if '--batch' not in args:
    args.insert(0, '--batch')

# ���� video_processor.py
subprocess.run(['python', 'video_processor.py'] + args)
```

---

## ? ��������

| ά�� | ����ǰ | ���Ϻ� | �Ľ� |
|------|-------|--------|------|
| �������� | 249 + 803 = 1052 | ~850 | -200�� (-19%) |
| ά����� | 2��main() | 1��main() | �� |
| ����һ���� | ���ֶ�ͬ�� | �Զ�һ�� | ? |
| �û����� | 2������ | 1������ | �� |

---

## ? ���Ϻ��ʹ�÷�ʽ

### ����Ƶ����

```bash
python video_processor.py -i video.mp4 --normalize-by-resolution
```

### ������������BadCase��

```bash
python video_processor.py -i videos/ --batch --normalize-by-resolution
```

### �������� + BadCase���

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### ���ݾ����ͨ��wrapper��

```bash
# ��Ȼ���ã��Զ�ת��
python batch_with_badcase.py -i videos/ -l labels.json
```

---

## ?? ע������

### 1. ������

- ? ���� `batch_with_badcase.py` ��Ϊwrapper
- ? �������Զ�ת���������
- ? �������нű������޸�

### 2. ����������

- ? ���й��ܱ���
- ? BadCase����Ϊ��ѡģ��
- ? ��ͨ�������������

### 3. ������֯

```
video_processor.py (ͳһ���)
������ ��: VideoProcessor (���Ĵ���)
������ ����: process_single_video (����Ƶ)
������ ����: batch_process_videos (������֧��BadCase)
������ ����: load_labels (��ǩ����)
������ ����: main (ͳһCLI)

batch_with_badcase.py (����wrapper)
������ �ض��� video_processor.py
```

---

## ? ���Ϻ�Ĳ����б�

```bash
# ��������
--input, -i              # ����
--output, -o            # ���
--device                # �豸
--raft_model, -m        # ģ��

# ģʽ����
--batch                 # ����ģʽ

# BadCase��⣨��ѡ��
--badcase-labels, -l    # ��ǩ�ļ�������BadCase��
--mismatch-threshold    # ��ƥ����ֵ

# �������
--no-camera-compensation
--camera-ransac-thresh
--camera-max-features

# �ֱ��ʹ�һ��
--normalize-by-resolution
--flow-threshold-ratio

# ����
--visualize
--fov
```

---

## ? �Ƽ�ʵʩ����

1. ? �� video_processor.py ����� BadCase ���֧��
2. ? �������������߼�
3. ? ͳһ��������
4. ? �� batch_with_badcase.py ��Ϊ����wrapper
5. ? �����ĵ�

**�Ƿ���Ҫ������ʵʩ������Ϸ�����**

