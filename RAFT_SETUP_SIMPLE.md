# RAFTģ�Ϳ�������

## һ����װ

### 1. ����Ԥѵ��ģ��

```bash
# ����ģ��Ŀ¼
mkdir -p pretrained_models

# ����RAFT-Thingsģ�ͣ��Ƽ���
wget https://drive.google.com/file/d/1YWQtFl9ewNKGFAx3x4WRJCkI_zMYNfUy/view \
  -O pretrained_models/raft-things.pth

# ��ʹ��gdown�������㣩
pip install gdown
gdown --id 1YWQtFl9ewNKGFAx3x4WRJCkI_zMYNfUy -O pretrained_models/raft-things.pth
```

### 2. ��֤��װ

```python
from simple_raft import SimpleRAFTPredictor

# ���Լ���
predictor = SimpleRAFTPredictor(
    model_path="pretrained_models/raft-things.pth",
    device='cuda'
)
print("RAFTģ�ͼ��سɹ���")
```

---

## ��ѡģ��

| ģ�� | ���ó��� | �������� |
|------|----------|----------|
| raft-things.pth | ͨ�ó������Ƽ��� | [����](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) |
| raft-sintel.pth | �����˶� | [����](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) |
| raft-kitti.pth | �Զ���ʻ | [����](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) |

---

## ��������

### Q: ģ������̫����

**����1**��ʹ�þ���վ��
```bash
# ʹ�ù��ھ�������У�
```

**����2**���ֶ�����
1. ���� Google Drive ����
2. �ֶ����ص� `pretrained_models/` Ŀ¼

### Q: CUDA�ڴ治�㣿

```bash
# ʹ��CPUģʽ
python video_processor.py -i video.mp4 -o output/ --device cpu
```

### Q: �Ҳ���ģ���ļ���

���·����
```bash
ls pretrained_models/
# Ӧ�ÿ���: raft-things.pth
```

---

## �������ã��״�ʹ�ã�

```bash
# 1. ��װ����
pip install -r requirements.txt

# 2. ����ģ��
pip install gdown
gdown --id 1YWQtFl9ewNKGFAx3x4WRJCkI_zMYNfUy \
  -O pretrained_models/raft-things.pth

# 3. ��������
python video_processor.py -i demo_data/ -o test_output/

# �ɹ���
```

---

**��Ҫ������** �鿴 [README.md](README.md) ���ύ Issue

