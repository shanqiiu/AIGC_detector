# 相机补偿功能 - 快速参考

## 一分钟上手

### 默认使用（推荐）

相机补偿**默认启用**，直接运行即可：

```bash
python video_processor.py -i your_video.mp4 -o output/
```

### 禁用相机补偿

如果是固定机位拍摄，可以禁用以提高速度：

```bash
python video_processor.py -i your_video.mp4 -o output/ --no-camera-compensation
```

## 常用命令

### 单个视频

```bash
# 基本使用
python video_processor.py -i video.mp4 -o output/

# 使用GPU加速
python video_processor.py -i video.mp4 -o output/ --device cuda

# 自定义相机补偿参数（更严格）
python video_processor.py -i video.mp4 -o output/ \
  --camera-ransac-thresh 0.8 \
  --camera-max-features 3000
```

### 图像序列

```bash
python video_processor.py -i image_folder/ -o output/
```

### 批量处理

```bash
python video_processor.py -i videos/ -o results/ --batch
```

## 参数速查

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--no-camera-compensation` | 禁用相机补偿 | 启用 | - |
| `--camera-ransac-thresh` | RANSAC阈值（像素） | 1.0 | 0.5-2.0 |
| `--camera-max-features` | 最大特征点数 | 2000 | 1000-5000 |

## 场景选择

### ? 适合启用相机补偿

- 相机转动拍摄（环绕、平移等）
- 多视角视频
- 手持或无人机拍摄

### ?? 可禁用相机补偿

- 固定机位拍摄
- 无相机运动的场景
- 需要最快处理速度时

## 输出解读

### 成功率指标

- **> 80%**: 优秀，相机补偿效果好
- **60-80%**: 良好，大部分帧成功补偿
- **< 60%**: 较差，可能需要调整参数或禁用

### 残差光流幅度

- **< 原始光流的30%**: 补偿效果显著
- **30-50%**: 补偿效果中等
- **> 50%**: 补偿效果有限

## 故障排除

### 问题：补偿成功率低

```bash
# 增加特征点
python video_processor.py -i video.mp4 -o output/ --camera-max-features 3000

# 放宽RANSAC阈值
python video_processor.py -i video.mp4 -o output/ --camera-ransac-thresh 2.0
```

### 问题：处理速度慢

```bash
# 禁用相机补偿
python video_processor.py -i video.mp4 -o output/ --no-camera-compensation

# 或减少特征点
python video_processor.py -i video.mp4 -o output/ --camera-max-features 1000
```

### 问题：残差光流仍然很大

```bash
# 可能场景不适合单应性补偿，尝试禁用
python video_processor.py -i video.mp4 -o output/ --no-camera-compensation
```

## Python API

```python
from video_processor import VideoProcessor

# 启用相机补偿（默认）
processor = VideoProcessor(
    device='cuda',
    enable_camera_compensation=True,
    camera_compensation_params={
        'ransac_thresh': 1.0,
        'max_features': 2000
    }
)

# 处理视频
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir="output")

# 查看补偿统计
if result['camera_compensation_enabled']:
    stats = result['camera_compensation_stats']
    print(f"成功率: {stats['success_rate']:.1%}")
```

## 查看结果

### JSON结果

```bash
cat output/analysis_results.json | grep -A 10 "camera_compensation"
```

### 文本报告

```bash
cat output/analysis_report.txt
```

### 可视化对比图

```bash
# 查看相机补偿效果对比
open output/visualizations/camera_compensation_comparison.png
```

## 完整文档

详细信息请查看：
- ? [完整使用指南](CAMERA_COMPENSATION_GUIDE.md)
- ? [更新说明](CAMERA_COMPENSATION_UPDATE.md)
- ? [集成总结](INTEGRATION_SUMMARY.md)

## 测试

```bash
# 运行测试脚本
python test_camera_compensation.py

# 使用demo数据测试
python video_processor.py -i demo_data/ -o demo_output/
```

---

? **提示**: 相机补偿默认启用，大多数情况下无需手动配置！

