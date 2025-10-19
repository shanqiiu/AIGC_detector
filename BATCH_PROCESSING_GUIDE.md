# 批量视频处理指南

## 新增功能

### 1. 批量处理模式
一次性处理整个目录下的所有视频文件

### 2. 可视化控制
通过参数控制是否生成可视化，加快处理速度

---

## 使用方法

### 单个视频处理（原有功能）

```bash
# 基本使用
python video_processor.py --input video.mp4 --output output/

# 禁用可视化（加快速度）
python video_processor.py --input video.mp4 --output output/ --no-visualize

# 完整参数
python video_processor.py \
    --input video.mp4 \
    --output output/ \
    --max_frames 100 \
    --frame_skip 2 \
    --device cuda \
    --no-visualize
```

---

### 批量视频处理（新功能）

```bash
# 基本批量处理（包含可视化）
python video_processor.py --input videos/ --output batch_output/ --batch

# 快速批量处理（禁用可视化，推荐）
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize

# 批量处理 + 限制帧数（更快）
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize \
    --max_frames 100 \
    --frame_skip 2
```

---

## 参数说明

### 基本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` / `-i` | 输入路径（视频文件/图像目录/视频目录） | 必需 |
| `--output` / `-o` | 输出目录 | `output` |
| `--batch` | 启用批量处理模式 | False |
| `--no-visualize` | 禁用可视化生成 | False（默认生成） |

### 性能参数

| 参数 | 说明 | 默认值 | 推荐值（快速） |
|------|------|--------|--------------|
| `--max_frames` | 最大处理帧数 | None（全部） | 50-100 |
| `--frame_skip` | 帧跳跃间隔 | 1 | 2-3 |
| `--device` | 计算设备 | cuda | cuda |

### 其他参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--raft_model` / `-m` | RAFT模型路径 | None |
| `--fov` | 相机视场角（度） | 60.0 |

---

## 输出结构

### 单个视频处理

```
output/
├── analysis_results.json      # 数值结果（JSON）
├── analysis_report.txt         # 文字报告
└── visualizations/             # 可视化结果（如果启用）
    ├── frame_0000_analysis.png
    ├── temporal_dynamics.png
    └── static_ratio_changes.png
```

### 批量处理

```
batch_output/
├── batch_summary.txt           # 批量处理总结
├── batch_summary.json          # 批量处理总结（JSON）
├── video1/                     # 视频1的结果
│   ├── analysis_results.json
│   ├── analysis_report.txt
│   └── visualizations/         # （如果启用）
├── video2/                     # 视频2的结果
│   ├── analysis_results.json
│   └── analysis_report.txt
└── video3/                     # 视频3的结果
    └── ...
```

---

## 速度优化建议

### 快速批量处理（推荐配置）

```bash
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize \
    --max_frames 50 \
    --frame_skip 2 \
    --device cuda
```

**性能提升**:
- `--no-visualize`: 节省 **30-50%** 时间
- `--max_frames 50`: 只处理前50帧
- `--frame_skip 2`: 每隔一帧处理，速度提升 **2倍**

### 高质量批量处理

```bash
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --no-visualize \
    --device cuda
```

**特点**:
- 处理所有帧
- 禁用可视化（可后续单独生成）
- 平衡速度和质量

### 完整分析（带可视化）

```bash
python video_processor.py \
    --input videos/ \
    --output batch_output/ \
    --batch \
    --device cuda
```

**特点**:
- 生成所有可视化
- 适合需要详细分析的场景
- 速度较慢

---

## 实际示例

### 示例1: 快速批量检测

**场景**: 有100个视频需要快速筛选出有问题的视频

```bash
python video_processor.py \
    --input D:/videos/ \
    --output D:/quick_check/ \
    --batch \
    --no-visualize \
    --max_frames 30 \
    --frame_skip 3
```

**预计时间**: 约 1-2 分钟/视频（30帧，GPU）

### 示例2: 详细分析特定视频

**场景**: 对筛选出的问题视频进行详细分析

```bash
python video_processor.py \
    --input D:/videos/problem_video.mp4 \
    --output D:/detailed_analysis/ \
    --device cuda
```

**输出**: 包含完整可视化和报告

### 示例3: 大规模批量处理

**场景**: 处理1000个视频的数据集

```bash
# 第一阶段：快速处理获取数值
python video_processor.py \
    --input D:/dataset/ \
    --output D:/results/ \
    --batch \
    --no-visualize \
    --max_frames 50 \
    --frame_skip 2

# 第二阶段：对异常视频重新处理（带可视化）
python video_processor.py \
    --input D:/dataset/anomaly_video.mp4 \
    --output D:/detailed_results/ \
    --device cuda
```

---

## 批量处理总结报告

批量处理完成后，会生成 `batch_summary.txt` 和 `batch_summary.json`：

### batch_summary.txt 示例

```
======================================================================
批量视频处理总结
======================================================================

总视频数: 10
成功处理: 9
处理失败: 1

======================================================================
详细结果
======================================================================

视频: video1
  状态: ? 成功
  帧数: 120
  平均动态度分数: 0.856
  平均静态区域比例: 0.785
  时序稳定性: 0.912
  输出目录: batch_output/video1

视频: video2
  状态: ? 成功
  帧数: 95
  平均动态度分数: 1.234
  平均静态区域比例: 0.654
  时序稳定性: 0.876
  输出目录: batch_output/video2

...
```

### batch_summary.json 示例

```json
[
  {
    "video_name": "video1",
    "status": "success",
    "frame_count": 120,
    "mean_dynamics_score": 0.856,
    "mean_static_ratio": 0.785,
    "temporal_stability": 0.912,
    "output_dir": "batch_output/video1"
  },
  ...
]
```

---

## 支持的视频格式

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
- `.wmv`

---

## 常见问题

### Q1: 如何最快速度处理大量视频？

```bash
python video_processor.py \
    --input videos/ \
    --output output/ \
    --batch \
    --no-visualize \
    --max_frames 30 \
    --frame_skip 3 \
    --device cuda
```

### Q2: 如何只生成数值结果，不要可视化？

使用 `--no-visualize` 参数

### Q3: 批量处理中某个视频失败了怎么办？

会继续处理其他视频，失败信息记录在 `batch_summary.txt` 中

### Q4: 如何恢复可视化？

先用 `--no-visualize` 快速处理，后续单独处理需要可视化的视频：

```bash
python video_processor.py --input problem_video.mp4 --output detailed/
```

### Q5: 如何调整光流算法？

修改代码中的光流方法（在 `VideoProcessor.__init__` 中）：

```python
# 使用 TV-L1（更高精度）
self.raft_predictor = RAFTPredictor(method='tvl1', device=device)

# 使用 RAFT官方（最高精度）
self.raft_predictor = RAFTPredictor(
    method='raft',
    model_path='pretrained_models/raft-things.pth',
    device=device
)
```

---

## 性能参考

基于 1920x1080 视频，GPU: RTX 3080

| 配置 | 速度 | 精度 |
|------|------|------|
| 全帧 + 可视化 | ~3 min/video | 最高 |
| 全帧 + 无可视化 | ~2 min/video | 高 |
| 每3帧 + 无可视化 | ~40 sec/video | 中 |
| 前30帧 + 无可视化 | ~20 sec/video | 快速检测 |

---

## 总结

- **快速批量**: `--batch --no-visualize --max_frames 50 --frame_skip 2`
- **高质量批量**: `--batch --no-visualize`
- **详细分析**: 默认参数（带可视化）

选择合适的参数组合以平衡速度和质量！

