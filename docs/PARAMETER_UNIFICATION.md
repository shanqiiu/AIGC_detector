# 参数统一方案

## 问题分析

### 当前参数对比

| 参数 | video_processor.py | batch_with_badcase.py | 是否统一 |
|------|-------------------|---------------------|---------|
| `--visualize` | ? 可配置（默认False） | ? 硬编码False | ? |
| `--no-camera-compensation` | ? | ? | ? |
| `--camera-ransac-thresh` | ? | ? 缺失 | ? |
| `--camera-max-features` | ? | ? 缺失 | ? |
| `--mismatch-threshold` | ? | ? (BadCase专用) | N/A |

### 存在的问题

1. **可视化参数缺失**：`batch_with_badcase.py` 硬编码 `enable_visualization=False`，用户无法选择生成可视化
2. **相机补偿参数不完整**：缺少 RANSAC 阈值和特征点数配置
3. **参数传递不一致**：一个直接传参数，一个传字典

## 统一方案

### 修改 batch_with_badcase.py

```python
# 1. 添加参数（第215行后）
parser.add_argument('--no-camera-compensation', dest='camera_compensation',
                   action='store_false',
                   help='禁用相机补偿')
parser.add_argument('--camera-ransac-thresh', type=float, default=1.0,
                   help='相机补偿RANSAC阈值（像素）')
parser.add_argument('--camera-max-features', type=int, default=2000,
                   help='相机补偿最大特征点数')
parser.add_argument('--visualize', action='store_true',
                   help='生成可视化结果（会增加处理时间）')
parser.add_argument('--filter-badcase-only', action='store_true',
                   help='只保留BadCase视频结果')
parser.set_defaults(camera_compensation=True, visualize=False)

# 2. 准备相机补偿参数（第229行前）
camera_compensation_params = {
    'ransac_thresh': args.camera_ransac_thresh,
    'max_features': args.camera_max_features
}

# 3. 创建处理器时传入完整参数（第230行）
processor = VideoProcessor(
    raft_model_path=args.raft_model,
    device=args.device,
    enable_visualization=args.visualize,  # 改为可配置
    enable_camera_compensation=args.camera_compensation,
    camera_compensation_params=camera_compensation_params  # 新增
)
```

### 统一后的参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input, -i` | str | 必需 | 输入视频目录 |
| `--output, -o` | str | output | 输出目录 |
| `--labels, -l` | str | 必需 | 期望标签文件 |
| `--raft_model, -m` | str | pretrained_models/raft-things.pth | RAFT模型路径 |
| `--device` | str | cuda | 计算设备 |
| `--fov` | float | 60.0 | 相机视场角 |
| `--mismatch-threshold` | float | 0.3 | BadCase不匹配阈值 |
| `--no-camera-compensation` | flag | False | 禁用相机补偿 |
| `--camera-ransac-thresh` | float | 1.0 | RANSAC阈值 |
| `--camera-max-features` | int | 2000 | 最大特征点数 |
| `--visualize` | flag | False | 生成可视化 |
| `--filter-badcase-only` | flag | False | 只保留BadCase |

## 使用示例

### 基础使用（无可视化）
```bash
python batch_with_badcase.py -i videos/ -l labels.json -o results/
```

### 启用可视化
```bash
python batch_with_badcase.py -i videos/ -l labels.json -o results/ --visualize
```

### 调整相机补偿参数
```bash
python batch_with_badcase.py -i videos/ -l labels.json \
    --camera-ransac-thresh 0.5 \
    --camera-max-features 3000
```

### 完整配置
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

## 收益

1. ? **参数一致性**：两个脚本使用相同的参数命名和默认值
2. ? **灵活性**：用户可根据需求选择是否生成可视化
3. ? **可控性**：精细控制相机补偿参数
4. ? **可维护性**：统一的参数传递方式

## 向后兼容

- 默认行为不变：`visualize=False`，不影响现有脚本
- 新增参数都有默认值，旧命令仍可正常工作

