# RAFT����ģ��ʹ��ָ��

## ����

����Ŀ֧��**���ַ�ʽ**ʹ��RAFT�������ƣ�

1. **OpenCV����** (�Ƽ����ڿ��ٲ���) - �������ش�ģ��
2. **RAFT�ٷ�Ԥѵ��ģ��** (�Ƽ���������) - ���ȸ���

---

## ��ʽ1: ʹ��OpenCV���� (�Ƽ�)

### �ŵ�
- ? �������ش�ģ���ļ�
- ? ��װ�򵥣�������
- ? �ٶȿ�
- ? �ʺϿ���ԭ�ͺͲ���

### ʹ�÷���

```python
from raft_model_simple import RAFTPredictor

# ����Ԥ���� (�Զ�ʹ��OpenCV)
predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)

# Ԥ�����
flow = predictor.predict_flow(image1, image2)
```

### ��ǰ��Ŀʹ�����

��ǰ��Ŀ������ģ����ʹ���˹�����
- `static_object_analyzer.py` - ��̬���嶯̬�ȷ���
- `dynamic_motion_compensation/` - ��̬�˶�����

��Щģ��**�Ѿ�ʹ����OpenCV����**��ͨ�� `simple_raft.py`�����������ã�

---

## ��ʽ2: ʹ��RAFT�ٷ�Ԥѵ��ģ��

### �ŵ�
- ? �������ȸ���
- ? �Ը��ӳ���³���Ը���
- ? ѧ��/����������

### ȱ��
- ?? ��Ҫ���ش�ģ���ļ� (~150MB)
- ?? ��Ҫ����GPU�ڴ�
- ?? �����ٶȽ���

---

## ����RAFT�ٷ�Ԥѵ��Ȩ��

### ����1: �ӹٷ��ֿ�����

���� RAFT �ٷ� GitHub:
```
https://github.com/princeton-vl/RAFT
```

����������һģ��:

| ģ������ | ѵ�����ݼ� | ���ó��� | ��С |
|---------|----------|---------|------|
| **raft-things.pth** | Things3D | ͨ�ó��� | ~150MB |
| **raft-sintel.pth** | Sintel | ��Ӱ������ | ~150MB |
| **raft-chairs.pth** | FlyingChairs | �򵥳��� | ~150MB |
| **raft-kitti.pth** | KITTI | �Զ���ʻ | ~150MB |

**�Ƽ�**: `raft-things.pth` (�����������)

### ����2: ֱ����������

```bash
# raft-things.pth
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth

# raft-sintel.pth  
wget https://dl.dropboxusercontent.com/s/kqdjpd17kkb8syk/raft-sintel.pth
```

### ����3: ʹ�� gdown (Google Drive)

```bash
pip install gdown

# raft-things
gdown 1M5QHhdMI6oWF3Bv8Y1oW8vvVGMxM8Gru -O raft-things.pth

# raft-sintel
gdown 1Sxb0RDsJ7JBz9NJ6wj4QzHXlZ7PdYJKd -O raft-sintel.pth
```

---

## ʹ��RAFT�ٷ�ģ��

### ����1: ����ģ���ļ�

�����ص� `.pth` �ļ��ŵ�������һλ��:
- `AIGC_detector/raft-things.pth`
- `AIGC_detector/pretrained_models/raft-things.pth`

### ����2: ʹ��ģ��

```python
from raft_model_simple import RAFTPredictor

# ָ��ģ��·��
predictor = RAFTPredictor(
    model_path='raft-things.pth',
    device='cuda'  # �� 'cpu'
)

# Ԥ�����
flow = predictor.predict_flow(image1, image2)
```

### ����3: (��ѡ) �������д���

�����������ģ����ʹ��RAFT�ٷ�ģ��:

```python
# �� static_object_analyzer.py ��
from raft_model_simple import RAFTPredictor

# �滻ԭ���� SimpleRAFTPredictor
self.flow_predictor = RAFTPredictor(
    model_path='raft-things.pth',
    device='cuda'
)
```

---

## ����Ա�

### ��ǰʵ�� (raft_model.py)
- **390�д���** - ����ʵ����RAFT�ܹ�
- ����: ResidualBlock, FeatureEncoder, CorrBlock, UpdateBlock��
- ? ���Ӷȸߣ���ά��
- ? ��Ҫ�Լ�ѵ����ת��Ȩ�ظ�ʽ

### �򻯰� (raft_model_simple.py)
- **~200�д���** - ֻ�Ǽ�������Ԥ����
- ֱ��ʹ�ùٷ�Ԥѵ��Ȩ��
- ? ���׶�
- ? �ٷ�Ȩ�ؿ��伴��
- ? ֧��OpenCV�󱸷���

---

## ���ܶԱ�

| ���� | ���� | �ٶ� | GPU�ڴ� | ģ�ʹ�С |
|-----|------|------|---------|---------|
| OpenCV Farneback | �е� | �� (~50ms) | ����GPU | 0 MB |
| RAFT�ٷ�ģ�� | �� | �е� (~100ms) | ~2GB | ~150 MB |
| �Լ�ѵ��RAFT | �� | �е� | ~2GB | ~150 MB |

---

## ����

### ���ڿ���/����
```python
# ʹ�� OpenCV - ���١���
predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)
```

### ��������/����
```python
# ʹ�� RAFT�ٷ�ģ�� - �߾���
predictor = RAFTPredictor(model_path='raft-things.pth')
```

---

## ��������

### Q1: raft_model.py ��ɾ��

**? �Ѽ򻯣�** ��Ŀ��ɾ��390�е�����RAFTʵ�֣�
- ~~`raft_model.py`~~ - ��ɾ�������ڸ��ӣ�
- `raft_model_simple.py` - �򻯰棨�Ƽ�ʹ�ã�
- `simple_raft.py` - OpenCVʵ�֣���ǰʹ���У�

### Q2: �ٷ�Ȩ���ļ���ʽ?

RAFT�ٷ�Ȩ���Ǳ�׼�� PyTorch `.pth` �ļ�:
```python
torch.load('raft-things.pth')
```

### Q3: ģ�ͼ���ʧ��?

�������ʧ�ܣ���**�Զ����˵�OpenCV����**:
```python
predictor = RAFTPredictor(
    model_path='raft-things.pth',
    use_opencv_fallback=True  # ����ʧ���Զ�ʹ��OpenCV
)
```

### Q4: GPU�ڴ治��?

ʹ��CPUģʽ:
```python
predictor = RAFTPredictor(model_path='raft-things.pth', device='cpu')
```

��ʹ��OpenCV:
```python
predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)
```

---

## �ܽ�

? **�Ƽ�����**: 
- �������д���ʹ�� OpenCV ����
- ������߾��ȣ����� `raft-things.pth` ��ʹ�� `raft_model_simple.py`
- ����Ҫ�Լ�ʵ�� RAFT �ܹ�����

? **����**: 
- ����ʵ�� RAFT (390�д���)
- �Լ�ѵ�� RAFT ģ��
- ���ӵ�Ȩ�ظ�ʽת��

