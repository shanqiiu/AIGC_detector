# ����ͳһ����

## �������

### ��ǰ�����Ա�

| ���� | video_processor.py | batch_with_badcase.py | �Ƿ�ͳһ |
|------|-------------------|---------------------|---------|
| `--visualize` | ? �����ã�Ĭ��False�� | ? Ӳ����False | ? |
| `--no-camera-compensation` | ? | ? | ? |
| `--camera-ransac-thresh` | ? | ? ȱʧ | ? |
| `--camera-max-features` | ? | ? ȱʧ | ? |
| `--mismatch-threshold` | ? | ? (BadCaseר��) | N/A |

### ���ڵ�����

1. **���ӻ�����ȱʧ**��`batch_with_badcase.py` Ӳ���� `enable_visualization=False`���û��޷�ѡ�����ɿ��ӻ�
2. **�����������������**��ȱ�� RANSAC ��ֵ��������������
3. **�������ݲ�һ��**��һ��ֱ�Ӵ�������һ�����ֵ�

## ͳһ����

### �޸� batch_with_badcase.py

```python
# 1. ��Ӳ�������215�к�
parser.add_argument('--no-camera-compensation', dest='camera_compensation',
                   action='store_false',
                   help='�����������')
parser.add_argument('--camera-ransac-thresh', type=float, default=1.0,
                   help='�������RANSAC��ֵ�����أ�')
parser.add_argument('--camera-max-features', type=int, default=2000,
                   help='������������������')
parser.add_argument('--visualize', action='store_true',
                   help='���ɿ��ӻ�����������Ӵ���ʱ�䣩')
parser.add_argument('--filter-badcase-only', action='store_true',
                   help='ֻ����BadCase��Ƶ���')
parser.set_defaults(camera_compensation=True, visualize=False)

# 2. ׼�����������������229��ǰ��
camera_compensation_params = {
    'ransac_thresh': args.camera_ransac_thresh,
    'max_features': args.camera_max_features
}

# 3. ����������ʱ����������������230�У�
processor = VideoProcessor(
    raft_model_path=args.raft_model,
    device=args.device,
    enable_visualization=args.visualize,  # ��Ϊ������
    enable_camera_compensation=args.camera_compensation,
    camera_compensation_params=camera_compensation_params  # ����
)
```

### ͳһ��Ĳ����б�

| ���� | ���� | Ĭ��ֵ | ˵�� |
|------|------|--------|------|
| `--input, -i` | str | ���� | ������ƵĿ¼ |
| `--output, -o` | str | output | ���Ŀ¼ |
| `--labels, -l` | str | ���� | ������ǩ�ļ� |
| `--raft_model, -m` | str | pretrained_models/raft-things.pth | RAFTģ��·�� |
| `--device` | str | cuda | �����豸 |
| `--fov` | float | 60.0 | ����ӳ��� |
| `--mismatch-threshold` | float | 0.3 | BadCase��ƥ����ֵ |
| `--no-camera-compensation` | flag | False | ����������� |
| `--camera-ransac-thresh` | float | 1.0 | RANSAC��ֵ |
| `--camera-max-features` | int | 2000 | ����������� |
| `--visualize` | flag | False | ���ɿ��ӻ� |
| `--filter-badcase-only` | flag | False | ֻ����BadCase |

## ʹ��ʾ��

### ����ʹ�ã��޿��ӻ���
```bash
python batch_with_badcase.py -i videos/ -l labels.json -o results/
```

### ���ÿ��ӻ�
```bash
python batch_with_badcase.py -i videos/ -l labels.json -o results/ --visualize
```

### ���������������
```bash
python batch_with_badcase.py -i videos/ -l labels.json \
    --camera-ransac-thresh 0.5 \
    --camera-max-features 3000
```

### ��������
```bash
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o results/ \
    --visualize \
    --camera-ransac-thresh 0.8 \
    --camera-max-features 2500 \
    --mismatch-threshold 0.25 \
    --device cuda
```

## ����

1. ? **����һ����**�������ű�ʹ����ͬ�Ĳ���������Ĭ��ֵ
2. ? **�����**���û��ɸ�������ѡ���Ƿ����ɿ��ӻ�
3. ? **�ɿ���**����ϸ���������������
4. ? **��ά����**��ͳһ�Ĳ������ݷ�ʽ

## ������

- Ĭ����Ϊ���䣺`visualize=False`����Ӱ�����нű�
- ������������Ĭ��ֵ���������Կ���������

