# �����㷨ʹ��ָ��

## ���ٿ�ʼ

���� `simple_raft.py` ֧�����ֹ����㷨��ͨ��ͳһ�ӿ�ʹ�ã�

### ����1: Farneback���Ƽ���ʼʹ�ã�

```python
from simple_raft import SimpleRAFTPredictor

# ��� - �����κζ�������
predictor = SimpleRAFTPredictor(method='farneback')
flow = predictor.predict_flow(image1, image2)
```

**�ص�**:
- ? �ٶȿ� (~50ms/frame)
- ? OpenCV���ã�������ⰲװ
- ? �ܼ�����Ե���������
- ?? �����е�

---

### ����2: TV-L1�����������Ƽ���

```python
from simple_raft import SimpleRAFTPredictor

# ��Ҫ��װ: pip install opencv-contrib-python
predictor = SimpleRAFTPredictor(method='tvl1')
flow = predictor.predict_flow(image1, image2)
```

**�ص�**:
- ? ���ȸߣ��߽�����
- ? CPU�Ѻ�
- ? �ܼ��΢С����������
- ?? �ٶȽ��� (~200ms/frame)
- ?? ��Ҫ��װ opencv-contrib-python

---

### ����3: RAFT�ٷ�����߾��ȣ�

```python
from simple_raft import SimpleRAFTPredictor

# ��Ҫ: 
# 1. third_party/RAFT Ŀ¼���ٷ����룩
# 2. pretrained_models/raft-things.pth��Ԥѵ��ģ�ͣ�
predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth',
    device='cuda'  # �� 'cpu'
)
flow = predictor.predict_flow(image1, image2)
```

**�ص�**:
- ? �������
- ? С�˶��������ǿ
- ? �߽�������
- ?? ��ҪGPU��CPUҲ���õ�������
- ?? ��Ҫ����ģ���ļ� (~150MB)
- ?? �ٶ��е� (~100ms/frame on GPU)

---

## ����ʾ��

```python
from simple_raft import SimpleRAFTPredictor
import cv2
import numpy as np

# ����ͼ��
image1 = cv2.imread('frame1.png')
image2 = cv2.imread('frame2.png')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# ѡ��һ�ַ���
predictor = SimpleRAFTPredictor(method='farneback')  # �� 'tvl1' �� 'raft'

# Ԥ�����
flow = predictor.predict_flow(image1, image2)
print(f"������״: {flow.shape}")  # (2, H, W)

# Ԥ������
images = [image1, image2, image3, ...]
flows = predictor.predict_flow_sequence(images)
```

---

## �����д�����ʹ��

### video_processor.py ���л�

```python
# ��ǰĬ�ϣ�Farneback��
self.raft_predictor = SimpleRAFTPredictor()

# �л���TV-L1
self.raft_predictor = SimpleRAFTPredictor(method='tvl1')

# �л���RAFT�ٷ�
self.raft_predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth'
)
```

### demo.py ���л�

ֻ���޸�һ�У�

```python
# ԭ��
flow_predictor = SimpleRAFTPredictor()

# ��Ϊ
flow_predictor = SimpleRAFTPredictor(method='tvl1')  # �� 'raft'
```

---

## ���ܶԱ�

| ���� | �ٶ� | ���� | GPU���� | �������� |
|------|------|------|---------|---------|
| **Farneback** | ???? | ??? | �� | �� |
| **TV-L1** | ?? | ???? | �� | opencv-contrib-python |
| **RAFT** | ??? (GPU) | ????? | �Ƽ� | ģ���ļ� + �ٷ����� |

---

## ѡ����

### ����1: ���ٿ���/��ʾ
```python
predictor = SimpleRAFTPredictor(method='farneback')
```
- �������ã����伴��
- �ܼ�����Ե���������

### ����2: ��������
```python
# ��װ: pip install opencv-contrib-python
predictor = SimpleRAFTPredictor(method='tvl1')
```
- ������������
- CPU�Ѻã������

### ����3: �о�/���¾���
```python
predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth',
    device='cuda'
)
```
- ѧ����׼����
- �ܼ����΢С������

---

## RAFT ����ָ��

���ѡ��ʹ�� RAFT �ٷ�ģ�ͣ���Ҫ��

### 1. ���عٷ�����
```bash
# �Ѿ������� AIGC_detector/third_party/RAFT/
```

### 2. ����Ԥѵ��ģ��
���� `raft-things.pth` �����õ� `pretrained_models/` Ŀ¼

- �ٷ���ַ: https://github.com/princeton-vl/RAFT
- ģ�ʹ�С: ~150MB

### 3. ʹ��
```python
predictor = SimpleRAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth'
)
```

---

## �����ų�

### TV-L1 �޷�ʹ��
```bash
# ����: opencv-contrib-pythonδ��װ
pip install opencv-contrib-python
```

### RAFT ����ʧ��
��飺
1. `third_party/RAFT/core/` Ŀ¼����
2. `pretrained_models/raft-things.pth` �ļ�����
3. ���Զ����˵� Farneback

### NumPy �汾����
```bash
# ������� NumPy 2.x ����������
pip install "numpy<2.0"
```

---

## �ܽ�

- **��ʼ**: ʹ�� `method='farneback'`���������ã�
- **����**: �л��� `method='tvl1'`�����߾��ȣ�
- **����**: ʹ�� `method='raft'`����߾��ȣ���Ҫ���ã�

���з���ʹ����ͬ�Ľӿڣ����������л���

