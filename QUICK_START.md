# ? 快速开始指南

## 5分钟上手

### 1?? 安装 (1分钟)

```bash
# 克隆项目
cd AIGC_detector

# 安装依赖
pip install -r requirements.txt

# 准备RAFT模型
# 下载 raft-things.pth 到 pretrained_models/ 目录
```

### 2?? 准备数据 (1分钟)

```bash
# 创建视频目录
mkdir my_videos

# 复制你的视频到此目录
cp /path/to/your/videos/*.mp4 my_videos/

# （可选）准备标签文件
# 创建 labels.json，格式如下：
```

```json
{
  "video1": "high",
  "video2": "low",
  "video3": "medium"
}
```

### 3?? 运行分析 (3分钟)

#### 单视频分析

```bash
python video_processor.py -i my_videos/test.mp4 -o results/
```

#### 批量分析

```bash
python video_processor.py -i my_videos/ -o results/ --batch
```

#### BadCase检测（推荐）

```bash
python video_processor.py \
    -i my_videos/ \
    -o results/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

---

## ? 查看结果

### 文本报告

```bash
# 查看单个视频报告
cat results/test/analysis_report.txt

# 查看批量汇总
cat results/badcase_summary.txt
```

### JSON结果

```bash
# 使用Python读取
import json
with open('results/test/analysis_results.json') as f:
    data = json.load(f)
    print(f"动态度: {data['temporal_stats']['mean_dynamics_score']}")
```

### 可视化结果

```bash
# 生成可视化（需要添加 --visualize）
python video_processor.py -i test.mp4 -o results/ --visualize

# 查看可视化
# results/test/visualizations/ 目录下有所有图表
```

---

## ? 常用命令速查

### 最简单

```bash
python video_processor.py -i video.mp4
```

### 最推荐（混合分辨率）

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --normalize-by-resolution
```

### 最完整（BadCase检测）

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### CPU模式

```bash
python video_processor.py -i video.mp4 --device cpu
```

### 快速采样

```bash
python video_processor.py -i video.mp4 --frame_skip 3 --max_frames 60
```

---

## ?? 关键参数

| 参数 | 何时使用 | 示例值 |
|------|----------|--------|
| `--batch` | 多个视频 | - |
| `--normalize-by-resolution` | **混合分辨率（必须）** | - |
| `--badcase-labels` | 质量检测 | `labels.json` |
| `--visualize` | 需要图表 | - |
| `--device` | CPU/GPU选择 | `cpu` 或 `cuda` |
| `--mismatch-threshold` | 调整敏感度 | `0.3`（默认） |

---

## ? 结果解读

### 动态度分数

- **0.0 - 0.2**: 极低动态（静态场景/严重问题）
- **0.2 - 0.4**: 低动态（轻微运动）
- **0.4 - 0.6**: 中等动态（正常运动）
- **0.6 - 0.8**: 高动态（明显运动）
- **0.8 - 1.0**: 极高动态（剧烈运动）

### BadCase严重程度

- **severe**: 严重不匹配（需重新生成）
- **moderate**: 中度不匹配（建议检查）
- **minor**: 轻微不匹配（可能可接受）

### 时序稳定性

- **> 0.8**: 稳定（正常）
- **0.6 - 0.8**: 一般（可接受）
- **< 0.6**: 不稳定（可能有问题）

---

## ? 常见问题

### Q: CUDA out of memory

```bash
# 减少分辨率或跳帧
python video_processor.py -i video.mp4 --frame_skip 2
```

### Q: 处理太慢

```bash
# 1. 使用GPU
python video_processor.py -i video.mp4 --device cuda

# 2. 不生成可视化
python video_processor.py -i video.mp4  # 默认不生成

# 3. 跳帧处理
python video_processor.py -i video.mp4 --frame_skip 3
```

### Q: 如何只检测BadCase

```bash
# 使用BadCase模式
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json

# 查看BadCase列表
cat output/badcase_videos.txt
```

### Q: 混合分辨率视频怎么办

```bash
# 必须启用归一化！
python video_processor.py \
    -i mixed_videos/ \
    --batch \
    --normalize-by-resolution  # 这个参数很重要！
```

---

## ? 更多信息

- 完整文档：[README.md](README.md)
- 技术原理：查看 README.md 的"技术原理"章节
- API文档：查看各模块的docstring

---

## ? 典型工作流

```bash
# 1. 准备数据
mkdir project_videos
cp *.mp4 project_videos/

# 2. 创建标签（可选）
cat > labels.json << EOF
{
  "video1": "high",
  "video2": "medium"
}
EOF

# 3. 运行分析
python video_processor.py \
    -i project_videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    -o results/

# 4. 查看结果
cat results/badcase_summary.txt
ls results/badcase_videos.txt

# 5. （可选）详细分析BadCase
python video_processor.py \
    -i results/badcase_videos.txt中的某个视频 \
    -o detailed_check/ \
    --visualize \
    --normalize-by-resolution
```

---

<div align="center">

**现在你已经准备好了！?**

有问题？查看 [README.md](README.md) 或提交 Issue

</div>

