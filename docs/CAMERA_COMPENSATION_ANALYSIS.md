# �������ģ���������

## ? `dynamic_motion_compensation` �ļ��з���

### ? �ļ��ṹ

```
dynamic_motion_compensation/
������ __init__.py                 # ����ʼ��
������ camera_compensation.py       # CameraCompensator ��
������ object_motion.py            # ObjectSE3Estimator ��
������ se3_utils.py                # SE(3) ��ѧ����
������ cli.py                      # ����CLI����
������ requirements.txt            # ����
```

### ? ���Ĺ���

#### 1. **CameraCompensator** (camera_compensation.py)
**����**: �ӹ����з�������˶���������ʵ�˶�

**��������**:
```python
ԭʼ���� (RAFT) = ����˶����� + ������ʵ�˶�����
�в���� = ԭʼ���� - ����˶�����  # �õ��������ʵ�˶�
```

**ʵ�ַ���**:
1. ʹ�� ORB/SIFT ����ƥ����Ƶ�Ӧ�Ծ��� H
2. ���� H �����������Ĺ��� (camera_flow)
3. ��ԭʼ�����м�ȥ camera_flow���õ��в���� (residual_flow)

**���**:
```python
{
    'homography': H,              # ��Ӧ�Ծ���
    'camera_flow': cam_flow,      # ����˶�����
    'residual_flow': residual,    # �в������������ʵ�˶���
    'inliers': inliers,           # �ڵ�����
    'total_matches': total        # ��ƥ�����
}
```

#### 2. **ObjectSE3Estimator** (object_motion.py)
**����**: �������ͼ������������Ƹ����˶���SE(3)�任��

**ʹ�ó���**: 
- ��Ҫ���ͼ
- ��Ҫ����ָ�����
- ���ڸ���ȷ��ÿ�����˶�����

**״̬**: ?? **δ��������ʹ��**

#### 3. **se3_utils.py**
SE(3)������任����ѧ���߿⣺
- `skew()`: ���Գƾ���
- `se3_exp()`: SE(3)ָ��ӳ��
- `project_points()`: 3D��ͶӰ
- `homography_from_RTn()`: ��R,T,n���㵥Ӧ��

**״̬**: ���� ObjectSE3Estimator ʹ�ã�������**δʹ��**

#### 4. **cli.py**
�����������й��ߣ��������ߴ�����Ƶ�����油�������

**״̬**: �������ߣ���������**���е�������**

---

## ? �����д���Ĺ�ϵ

### ��ǰʹ�����

�� `video_processor.py` �У�
```python
# ����
from dynamic_motion_compensation.camera_compensation import CameraCompensator

# ��ʼ��
self.camera_compensator = CameraCompensator(**params)

# ʹ��
comp_result = self.camera_compensator.compensate(flow, frame1, frame2)
flows.append(comp_result['residual_flow'])  # ʹ�òв����
```

**����**: ? `CameraCompensator` ��������ʹ��

---

### ?? �����ص�����

�� `static_object_analyzer.py` �У�������һ������˶���������

```python
class CameraMotionEstimator:
    """����˶�������"""
    # Ҳ�ǻ�������ƥ�� + ��Ӧ�Ծ���
    # ������ CameraCompensator ������ͬ
    
class StaticObjectDetector:
    def compensate_camera_motion(self, flow, homography):
        """��������˶�"""
        # Ҳ�Ǽ���в����
```

**����**: 
1. ����ģ��ʵ������ͬ�Ĺ���
2. `static_object_analyzer.py` �����������**δ��ʵ��ʹ��**
3. ��Ϊ����� `video_processor.py` �Ѿ����˲���

---

## ? �Ƿ��Ҫ��

### ? ��Ҫ�Ĳ���

**CameraCompensator (camera_compensation.py)**
- **��Ҫ��**: ????? (5/5)
- **ԭ��**: 
  - ��������Ǻ��Ĺ��ܣ����ڷ�������˶��������˶�
  - ����������̬�ĳ������罨�������������˶�������ж��Ƿ����쳣����
  - ������߶�̬��������׼ȷ��
- **����**: **����������ʹ��**

### ? �Ǳ�Ҫ�Ĳ���

**ObjectSE3Estimator (object_motion.py)**
- **��Ҫ��**: ?����� (1/5)
- **ԭ��**:
  - ��Ҫ��������ͼ����������
  - ��ǰ����δʹ��
  - ���Ӹ��Ӷȵ���ʵ������
- **����**: **�����Ƴ�����Ϊʵ���Թ���**

**se3_utils.py**
- **��Ҫ��**: ?����� (1/5)
- **ԭ��**: ����δʹ�õ� ObjectSE3Estimator ����
- **����**: �� ObjectSE3Estimator һ���Ƴ�

**cli.py**
- **��Ҫ��**: ??���� (2/5)
- **ԭ��**: 
  - �������ߣ������ڵ��Ժ���֤
  - �����������ѽ�
  - �����ѱ� video_processor.py ����
- **����**: ��Ϊ�������߱����������� examples/

---

## ? �� static_object_analyzer.py ������

| ���� | dynamic_motion_compensation | static_object_analyzer | ʵ��ʹ�� |
|------|----------------------------|------------------------|---------|
| ������� | ? CameraCompensator | ? CameraMotionEstimator | ǰ�� |
| ��Ӧ�Թ��� | ? | ? | ǰ�� |
| ������� | ? | ? | ǰ�� |
| �в�������� | ? | ? (StaticObjectDetector) | ǰ�� |

**����**: `static_object_analyzer.py` �е�����˶����ƴ�����**�����**�������Ƴ���

---

## ? �ع�����

### ����1: ����ܹ����Ƽ���

```
����:
������ dynamic_motion_compensation/
��   ������ __init__.py
��   ������ camera_compensation.py   # ���ģ�����
��   ������ requirements.txt

�Ƴ�:
������ object_motion.py              # δʹ��
������ se3_utils.py                  # δʹ��
������ cli.py                        # ���� examples/camera_compensation_demo.py

�ع�:
������ static_object_analyzer.py
��   ������ �Ƴ� CameraMotionEstimator �ࣨ�ظ����ܣ�
```

### ����2: ��ȫ����

�� `CameraCompensator` ֱ�����ϵ� `static_object_analyzer.py`:
- �Ƴ� `dynamic_motion_compensation` �ļ���
- �� `static_object_analyzer.py` �б���һ��ͳһ�����������
- �򻯵����ϵ

### ����3: ������״��������

```python
# dynamic_motion_compensation/__init__.py
"""
����˶�����ģ��

���Ĺ���: CameraCompensator (ʹ����)
ʵ�鹦��: ObjectSE3Estimator, se3_utils (δʹ��)
"""
```

---

## ? �ܽ�

| ���� | �� |
|------|------|
| **�Ƿ��Ҫ��** | **���ֱ�Ҫ**: CameraCompensator �Ǻ��Ĺ��� |
| **�Ƿ�ʹ�ã�** | CameraCompensator ? ʹ����<br>����ģ�� ? δʹ�� |
| **�Ƿ����ࣿ** | ? �� static_object_analyzer.py ���ڹ����ص� |
| **����** | 1. ���� CameraCompensator<br>2. �Ƴ�δʹ�õ� ObjectSE3Estimator ��<br>3. ���� static_object_analyzer.py �е��ظ����� |

---

## ? �Ƽ��ж�

### ����ִ�У������ȼ���
1. ? ���� `CameraCompensator` - **���Ĺ��ܣ���Ҫ**
2. ? �Ƴ� `static_object_analyzer.py::CameraMotionEstimator` - �ظ�
3. ? �Ƴ� `static_object_analyzer.py::StaticObjectDetector.compensate_camera_motion()` - �ظ�

### ��ѡ�Ż��������ȼ���
4. ? �Ƴ� `ObjectSE3Estimator` + `se3_utils.py` - δʹ��
5. ? �ƶ� `cli.py` �� `examples/` - ��������
6. ? ����ĵ�˵�������������Ҫ��


