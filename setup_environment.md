# ��������ָ��

## ��ǰ����

�������� **NumPy 2.x �� OpenCV ������**�����⣺
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2
```

## �������

### ����1: ���� NumPy���Ƽ���

```bash
pip install "numpy<2.0"
```

### ����2: ��װ��������

```bash
# ж�س�ͻ�İ�
pip uninstall numpy opencv-python opencv-contrib-python -y

# ����ȷ˳����װ
pip install "numpy<2.0"
pip install opencv-python opencv-contrib-python

# ��ֱ��ʹ��requirements.txt
pip install -r requirements.txt
```

### ����3: �����µ����⻷��

```bash
# �����»���
conda create -n aigc_detector python=3.9 -y
conda activate aigc_detector

# ��װ����
pip install -r requirements.txt
```

---

## ���޸��Ĵ�������

���� NumPy �汾���⣬�һ��޸������´������⣺

### 1. ? ����������������
```python
# ֮ǰ������
static_flow = flow[static_mask]
magnitude = np.sqrt(static_flow[:, 0]**2 + static_flow[:, 1]**2)

# ���ڣ���ȷ��
static_flow_x = flow[:, :, 0][static_mask]
static_flow_y = flow[:, :, 1][static_mask]
magnitude = np.sqrt(static_flow_x**2 + static_flow_y**2)
```

### 2. ? ���������ļ�ȥ��
Windows���ļ�����Сд�����е����ظ�

### 3. ? �����ļ����UTF-8��������

---

## ��װ����

1. **���� NumPy**
```bash
pip install "numpy<2.0"
```

2. **��װȱʧ������**
```bash
pip install opencv-contrib-python scikit-learn
```

3. **��֤��װ**
```bash
python -c "import cv2; import numpy as np; print('OpenCV:', cv2.__version__); print('NumPy:', np.__version__)"
```

���������
```
OpenCV: 4.x.x
NumPy: 1.x.x (< 2.0)
```

---

## ���в���

��װ��ɺ����У�

```bash
# ������Ƶ����
python video_processor.py -i videos/test.mp4 -o output_single/ --no-visualize

# ��������
python video_processor.py -i videos/ -o results/ --batch --no-visualize --max_frames 50 --frame_skip 3
```

---

## �����嵥

�Ѹ��µ� `requirements.txt`:
- `numpy>=1.21.0,<2.0.0` �� **���ư汾�������������**
- `opencv-python>=4.5.0`
- `opencv-contrib-python>=4.5.0` �� **������֧��TV-L1**
- `scikit-learn>=0.24.0` �� **������RANSAC��Ҫ**
- torch, scipy, matplotlib ��

---

## ���ٿ�ʼ

```bash
# 1. �޸�����
pip install "numpy<2.0" opencv-contrib-python scikit-learn

# 2. ���Ե�����Ƶ
python video_processor.py -i videos/test.mp4 -o test_output/ --no-visualize

# 3. ������������ģʽ��
python video_processor.py -i videos/ -o batch_results/ --batch --no-visualize --max_frames 30 --frame_skip 3
```

