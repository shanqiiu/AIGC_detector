# ? �ĵ�����

���ٵ����������ĵ�

---

## ? ��������

### 1. [README.md](../README.md) - **�����￪ʼ**
- ��Ŀ����
- ��װ����
- ����ʹ��
- ����˵��
- FAQ

**�ʺ�**�������û�

---

### 2. [QUICK_START.md](../QUICK_START.md) - **5��������**
- ���ٰ�װ
- ��������
- ������
- ��������

**�ʺ�**����������õ��û�

---

## ? ��ϸ�ĵ�

### 3. [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - **��Ŀ����**
- �����ܹ�
- ���ļ���ԭ��
- ����ָ��
- δ���滮

**�ʺ�**���������˽���Ŀ���û�

---

### 4. [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - **API�ĵ�**
- ��̽ӿ����
- �������뷽��
- ����ʾ��
- ������

**�ʺ�**�������ߣ���Ҫ��̼��ɵ��û�

---

## ? ��ʹ�ó���

### ����...

#### ? ��������һ����Ƶ
�� [QUICK_START.md](../QUICK_START.md) - "����Ƶ����"

```bash
python video_processor.py -i video.mp4
```

---

#### ? ������������Ƶ
�� [README.md](../README.md) - "����2��������Ƶ����"

```bash
python video_processor.py -i videos/ --batch --normalize-by-resolution
```

---

#### ? ����������⣨BadCase��
�� [README.md](../README.md) - "����3��BadCase���"

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

---

#### ? �����Ϸֱ�����Ƶ
�� [README.md](../README.md) - "�ֱ��ʹ�һ������"  
�� [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - "�ֱ��ʹ�һ��"

**�ؼ�����**��`--normalize-by-resolution`

---

#### ? �ô��뼯�ɵ��ҵ���Ŀ
�� [API_DOCUMENTATION.md](../API_DOCUMENTATION.md)

```python
from video_processor import VideoProcessor

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)
result = processor.process_video(frames, camera_matrix, "output/")
```

---

#### ? �Զ������ֹ���
�� [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - "UnifiedDynamicsScorer"

```python
custom_weights = {
    'flow_magnitude': 0.40,
    'spatial_coverage': 0.30,
    # ...
}
scorer = UnifiedDynamicsScorer(weights=custom_weights)
```

---

#### ? ��⼼��ԭ��
�� [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - "���ļ���"  
�� [README.md](../README.md) - "����ԭ��"

- �������ƣ�RAFT��
- �ֱ��ʹ�һ��
- ����˶�����
- ͳһ����ϵͳ

---

#### ? �Ż������ٶ�
�� [README.md](../README.md) - "����ָ��"  
�� [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - "�����Ż�����"

```bash
# GPU����
python video_processor.py -i video.mp4 --device cuda

# ��֡����
python video_processor.py -i video.mp4 --frame_skip 2
```

---

#### ? ���CUDA�ڴ治��
�� [QUICK_START.md](../QUICK_START.md) - "Q: CUDA out of memory"

```bash
# ��֡��ʹ��CPU
python video_processor.py -i video.mp4 --frame_skip 2
python video_processor.py -i video.mp4 --device cpu
```

---

#### ? ���������
�� [README.md](../README.md) - "������˵��"  
�� [QUICK_START.md](../QUICK_START.md) - "������"

**�ļ�˵��**��
- `analysis_report.txt` - �ı�����
- `analysis_results.json` - �ṹ������
- `badcase_report.txt` - BadCase����

---

## ? ����������

### �̳���

| �ĵ� | ���� | �Ѷ� |
|------|------|------|
| [QUICK_START.md](../QUICK_START.md) | �������� | ? ���� |
| [README.md](../README.md) | ȫ��ָ�� | ?? ���� |
| [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) | ��̽ӿ� | ??? �߼� |

### �ο���

| �ĵ� | ���� | ��; |
|------|------|------|
| [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) | ��Ŀ���� | ���ܹ� |
| [README.md](../README.md) | �����ĵ� | ���Ĳ��� |
| [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) | API�ĵ� | ���Ľӿ� |

---

## ? �ĵ���ϵͼ

```
                    ������������������������������
                    �� DOCS_INDEX  �� �� ��������
                    ��  (����ҳ)   ��
                    ���������������Щ�������������
                           ��
           ���������������������������������੤������������������������������
           ��               ��               ��
      ����������������������    ��������������������������   ����������������������
      �� QUICK   ��    ��  README   ��   ��PROJECT  ��
      �� START   ��    �� (�����ĵ�) ��   ��OVERVIEW ��
      ����������������������    ���������������Щ���������   ����������������������
                            ��
                      ��������������������������
                      ��    API    ��
                      ��DOCUMENTATION��
                      ��������������������������
```

### �Ƽ��Ķ�˳��

1. **��һ��ʹ��**��QUICK_START.md �� README.md
2. **�������**��PROJECT_OVERVIEW.md
3. **��������**��API_DOCUMENTATION.md

---

## ? ��������

### ���Ĺ���

- [��װָ��](../README.md#��װ����)
- [�����÷�](../README.md#�����÷�)
- [��������](../README.md#����2������Ƶ����)
- [BadCase���](../README.md#����3badcase���)
- [�ֱ��ʹ�һ��](../README.md#�ֱ��ʹ�һ��������Ҫ)

### ��������

- [�����ܹ�](../PROJECT_OVERVIEW.md#�����ܹ�)
- [��������](../PROJECT_OVERVIEW.md#1-��������raft)
- [��һ��ԭ��](../PROJECT_OVERVIEW.md#2-�ֱ��ʹ�һ��)
- [�������](../PROJECT_OVERVIEW.md#3-����˶�����)
- [����ϵͳ](../PROJECT_OVERVIEW.md#4-ͳһ����ϵͳ)

### API�ӿ�

- [VideoProcessor](../API_DOCUMENTATION.md#videoprocessor-��������)
- [BadCaseDetector](../API_DOCUMENTATION.md#badcasedetector-���������)
- [UnifiedDynamicsScorer](../API_DOCUMENTATION.md#unifieddynamicsscorer-����ϵͳ)
- [����ʾ��](../API_DOCUMENTATION.md#����������ʾ��)

### ������

- [��������FAQ](../README.md#��������)
- [����������](../QUICK_START.md#��������)
- [������](../API_DOCUMENTATION.md#������)

---

## ? ��ȡ����

### �ĵ�û�н��������⣿

1. **��������Issue**
   - �鿴 [GitHub Issues](https://github.com/your-repo/issues)

2. **����ǰ׼��**
   - ��Ļ�����Python�汾��CUDA�汾�ȣ�
   - �����Ĵ�����Ϣ
   - ���ֲ���

3. **�ύ��Issue**
   - [����Bug](https://github.com/your-repo/issues/new?labels=bug)
   - [���ܽ���](https://github.com/your-repo/issues/new?labels=enhancement)

4. **ֱ����ϵ**
   - Email: your-email@example.com

---

## ? �ĵ����¼�¼

### ���°汾 (2025-10-19)

- ? ������README�ĵ�
- ? ���ٿ�ʼָ��
- ? ��Ŀ�����ĵ�
- ? APIʹ���ĵ�
- ? �ĵ�����ҳ

### δ���ƻ�

- [ ] ��Ƶ�̳�
- [ ] ��Ӣ��˫���ĵ�
- [ ] �������ʾ��
- [ ] ����������չ

---

## ? �ĵ�����

�ĵ��в�����ĵط���

- �ύ [Documentation Issue](https://github.com/your-repo/issues/new?labels=documentation)
- ֱ����PR�Ľ��ĵ�

���ǳ����Ľ��ĵ�������

---

<div align="center">

**�ҵ�����Ҫ������**

����������⣬��ӭ [��Issue](https://github.com/your-repo/issues) ����������ĵ�

Made with ?? by AIGC Video Quality Team

</div>

