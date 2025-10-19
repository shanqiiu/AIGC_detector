# 快速参考

## ? 批量处理视频（最常用）

### 快速批量（推荐）
```bash
python video_processor.py \
    --input videos/ \
    --output batch_results/ \
    --batch \
    --no-visualize \
    --max_frames 50 \
    --frame_skip 2
```

### 高质量批量
```bash
python video_processor.py \
    --input videos/ \
    --output batch_results/ \
    --batch \
    --no-visualize
```

---

## ? 单个视频处理

### 快速检测
```bash
python video_processor.py --input video.mp4 --output output/ --no-visualize
```

### 完整分析（带可视化）
```bash
python video_processor.py --input video.mp4 --output output/
```

---

## ? 参数速查

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `--batch` | 批量处理模式 | - |
| `--no-visualize` | 禁用可视化（加速） | 批量时使用 |
| `--max_frames 50` | 只处理前50帧 | 快速检测 |
| `--frame_skip 2` | 每隔1帧处理 | 加速2倍 |
| `--device cuda` | 使用GPU | 默认 |

---

## ? 输出文件

### 单个视频
- `analysis_results.json` - 数值结果
- `analysis_report.txt` - 文字报告
- `visualizations/` - 可视化（如果启用）

### 批量处理
- `batch_summary.txt` - 总结报告
- `batch_summary.json` - JSON结果
- `video1/`, `video2/`... - 各视频结果

---

## ? 使用场景

| 场景 | 命令 |
|------|------|
| 100个视频快速筛选 | `--batch --no-visualize --max_frames 30 --frame_skip 3` |
| 10个视频详细分析 | `--batch --no-visualize` |
| 单个视频完整报告 | 不加 `--no-visualize` |

