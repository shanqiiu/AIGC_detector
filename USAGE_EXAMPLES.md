# ʹ��ʾ���볣������

## ? ��ȷ�÷�

### ����1������������ƵĿ¼���Ƽ���

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

**˵��**��
- `-i videos/` - ָ��������Ƶ�ļ���Ŀ¼
- `--batch` - **������Ӵ˲���**����ʾ��������ģʽ
- ϵͳ���Զ��ҵ�Ŀ¼�е����� `.mp4` ��Ƶ�ļ�

---

### ����2����������Ƶ�ļ�

```bash
python video_processor.py \
    -i videos/test.mp4 \
    --normalize-by-resolution \
    --visualize
```

**˵��**��
- ֱ��ָ����Ƶ�ļ�·��
- ����Ҫ `--batch` ����

---

### ����3������ͼ������

```bash
python video_processor.py \
    -i image_frames/ \
    --normalize-by-resolution
```

**˵��**��
- `image_frames/` Ŀ¼Ӧ���� `.jpg` �� `.png` ͼ���ļ�
- ����Ҫ `--batch` ����
- ϵͳ�ᰴ�ļ����������

---

## ? ��������

### ����1��������� --batch ����

**��������**��
```bash
python video_processor.py -i videos/ --badcase-labels labels.json
```

**������Ϣ**��
```
���ڴ�Ŀ¼����ͼ��: videos/
������ɣ��� 0 ֡
IndexError: list index out of range
```

**ԭ��**��
- û�� `--batch` ����ʱ��ϵͳ��Ϊ `videos/` ��ͼ������Ŀ¼
- �� `videos/` ���� `.mp4` �ļ�������ͼ�����Լ����� 0 ֡

**�޸�**��
```bash
# ��� --batch ����
python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

---

### ����2��Ŀ¼·������

**��������**��
```bash
python video_processor.py -i video/ --batch
```

**������Ϣ**��
```
FileNotFoundError: [Errno 2] No such file or directory: 'video/'
```

**�޸�**��
```bash
# ȷ��Ŀ¼���ƣ�Ӧ���� videos/ ������ video/
python video_processor.py -i videos/ --batch
```

---

### ����3����ǩ�ļ�·������

**��������**��
```bash
python video_processor.py -i videos/ --batch --badcase-labels label.json
```

**������Ϣ**��
```
FileNotFoundError: ��ǩ�ļ�������: label.json
```

**�޸�**��
```bash
# ȷ���ļ�����Ӧ���� labels.json
python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

---

## ? ��������ο�

### ��򵥣�����Ƶ��

```bash
python video_processor.py -i test.mp4
```

### ��ã����� + ��һ����

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --normalize-by-resolution
```

### ��������BadCase + ���ӻ� + ��һ����

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### CPUģʽ

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --device cpu \
    --normalize-by-resolution
```

### ���ٲ��ԣ���֡��

```bash
python video_processor.py \
    -i test.mp4 \
    --frame_skip 3 \
    --max_frames 60
```

---

## ? �������˵��

| �������� | �Ƿ���Ҫ --batch | ʾ�� |
|---------|-----------------|------|
| ������Ƶ�ļ� | ? ����Ҫ | `-i test.mp4` |
| ��ƵĿ¼�������� | ? **����** | `-i videos/ --batch` |
| ͼ������Ŀ¼ | ? ����Ҫ | `-i frames/` |

---

## ? ������������

### ��������������Ƶ

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

**�ؼ���**��
- ? ������ `--batch`
- ? Ŀ¼���� `.mp4` �ļ�
- ? `labels.json` ����

### ������Ե�����Ƶ

```bash
python video_processor.py -i videos/test.mp4 --normalize-by-resolution
```

**�ؼ���**��
- ? ֱ��ָ���ļ�·��
- ? ��Ҫ `--batch`

### ���봦��ͼ������

```bash
python video_processor.py -i my_frames/
```

**�ؼ���**��
- ? Ŀ¼���� `.jpg` �� `.png`
- ? ��Ҫ `--batch`

---

## ? ��֤�����Ƿ���ȷ

### ����嵥

1. **����ģʽ**
   - [ ] ������Ŀ¼��
   - [ ] ���� `.mp4` �ļ���
   - [ ] ����� `--batch` ������

2. **����Ƶģʽ**
   - [ ] �����ǵ����ļ���
   - [ ] �ļ����ڣ�
   - [ ] û�� `--batch` ������

3. **BadCase���**
   - [ ] `labels.json` ���ڣ�
   - [ ] ����� `--badcase-labels labels.json`��

4. **��һ�����Ƽ���**
   - [ ] ����� `--normalize-by-resolution`��

---

## ? ��������

### ����1����֤����

```bash
python video_processor.py --help
```

### ����2����������Ƶ

```bash
python video_processor.py \
    -i videos/test.mp4 \
    -o test_output/
```

### ����3����������

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -o batch_output/
```

---

**��ס**������������ƵĿ¼��**������� `--batch` ����**��


