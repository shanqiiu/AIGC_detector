# 相机补偿功能使用指南

## 概述

相机补偿功能已集成到视频处理流程中，用于在多视角（相机运动）场景下更准确地评估视频质量。该功能通过估计和去除相机运动引起的光流，保留真实的物体运动或异常，从而提供更精确的动态度分析。

## 工作原理

### 1. 光流分解
```
原始光流 = 相机运动光流 + 真实物体运动光流
残差光流 = 原始光流 - 相机运动光流
```

### 2. 相机运动估计
- 使用 ORB/SIFT 特征检测和匹配
- 通过 RANSAC 估计帧间单应性矩阵（Homography）
- 从单应性矩阵计算相机引起的光流
- 从原始光流中减去相机光流，得到残差光流

### 3. 残差光流分析
- 残差光流用于后续的静态物体动态度分析
- 更准确地反映场景中的真实运动或异常

## 使用方法

### 基本使用（相机补偿默认启用）

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --device cuda
```

### 禁用相机补偿

如果您想使用原始光流（不进行相机补偿）：

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --no-camera-compensation
```

### 自定义相机补偿参数

```bash
python video_processor.py \
  --input videos/test_video.mp4 \
  --output results/test_output \
  --camera-ransac-thresh 0.8 \
  --camera-max-features 3000
```

### 批量处理模式

```bash
python video_processor.py \
  --input videos/ \
  --output results/ \
  --batch \
  --device cuda
```

## 参数说明

### 相机补偿相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--no-camera-compensation` | flag | False | 禁用相机补偿（默认启用） |
| `--camera-ransac-thresh` | float | 1.0 | RANSAC阈值（像素），越小越严格 |
| `--camera-max-features` | int | 2000 | 最大特征点数 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input/-i` | str | 必需 | 输入视频或图像目录 |
| `--output/-o` | str | 'output' | 输出目录 |
| `--raft_model/-m` | str | pretrained_models/raft-things.pth | RAFT模型路径 |
| `--device` | str | 'cuda' | 计算设备 (cuda/cpu) |
| `--max_frames` | int | None | 最大处理帧数 |
| `--frame_skip` | int | 1 | 帧跳跃间隔 |
| `--fov` | float | 60.0 | 相机视场角（度） |
| `--batch` | flag | False | 批量处理模式 |
| `--no-visualize` | flag | False | 禁用可视化生成 |

## 输出结果

### JSON结果文件 (`analysis_results.json`)

启用相机补偿后，JSON结果会包含额外的信息：

```json
{
  "camera_compensation_enabled": true,
  "camera_compensation_stats": {
    "success_rate": 0.95,
    "successful_frames": 19,
    "total_frames": 20,
    "mean_inliers": 856.3,
    "std_inliers": 123.5,
    "mean_match_ratio": 0.78,
    "std_match_ratio": 0.05
  },
  "frame_results": [
    {
      "frame_index": 0,
      "camera_compensation": {
        "inliers": 850,
        "total_matches": 1100,
        "homography_found": true
      },
      "static_dynamics": {...},
      "global_dynamics": {...}
    }
  ]
}
```

### 文本报告 (`analysis_report.txt`)

报告中会包含相机补偿统计信息：

```
视频基本信息:
- 总帧数: 20
- 分析帧数: 19
- 相机补偿: 启用

相机运动补偿统计:
- 成功率: 95.0% (19/20)
- 平均内点数: 856.3 ± 123.5
- 平均匹配率: 78.0% ± 5.0%
```

### 可视化结果

在 `visualizations/` 目录下会生成：

1. **`camera_compensation_comparison.png`** - 相机补偿效果对比图
   - 显示原始光流、相机光流和残差光流的对比
   - 展示补偿前后的效果差异

2. **`temporal_dynamics.png`** - 时序动态度曲线
3. **`static_ratio_changes.png`** - 静态区域比例变化
4. **`frame_XXXX_analysis.png`** - 关键帧详细分析

## 应用场景

### 适合使用相机补偿的场景

? **相机转动拍摄静态场景**
- 例如：环绕建筑物、雕塑等静态物体的视频
- 相机补偿可以去除相机运动，只保留异常运动

? **多视角视频评估**
- 需要区分相机运动和物体运动的场景
- 评估AIGC生成的多视角一致性

? **运动相机拍摄的视频**
- 手持相机、无人机拍摄等有相机运动的场景

### 不适合使用相机补偿的场景

? **完全静态的视频**
- 固定机位拍摄，无相机运动
- 可以禁用相机补偿以提高处理速度

? **大量真实物体运动的场景**
- 如果场景中有大量真实运动，单应性估计可能不准确
- 建议使用更高级的相机补偿方法

## 技术细节

### 单应性估计

使用 OpenCV 的 `findHomography` 进行单应性估计：
- 特征检测：ORB（默认）或 SIFT
- 匹配方法：暴力匹配（BFMatcher）
- RANSAC：去除外点，保留内点

### 参数调优建议

**RANSAC阈值 (`--camera-ransac-thresh`)**
- 默认值：1.0 像素
- 场景稳定、无遮挡：可以降低到 0.5-0.8
- 场景复杂、有遮挡：可以提高到 1.5-2.0

**最大特征点数 (`--camera-max-features`)**
- 默认值：2000
- 高分辨率视频：可以提高到 3000-5000
- 快速处理：可以降低到 1000-1500

### 性能影响

- 启用相机补偿会增加约 10-20% 的处理时间
- 主要开销在特征检测和匹配
- CPU模式下影响更明显

## 编程接口

### Python API

```python
from video_processor import VideoProcessor

# 创建处理器（默认启用相机补偿）
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    enable_camera_compensation=True,  # 默认True
    camera_compensation_params={
        'ransac_thresh': 1.0,
        'max_features': 2000
    }
)

# 加载视频
frames = processor.load_video("test_video.mp4")

# 处理视频
result = processor.process_video(frames, output_dir="output")

# 查看相机补偿结果
if result['camera_compensation_enabled']:
    comp_stats = result['camera_compensation_stats']
    print(f"补偿成功率: {comp_stats['success_rate']:.1%}")
```

### 禁用相机补偿

```python
processor = VideoProcessor(
    enable_camera_compensation=False
)
```

## 故障排除

### 问题1：相机补偿成功率低

**症状**：`success_rate` < 0.5

**可能原因**：
- 特征点不足
- 场景纹理过少
- 运动模糊严重

**解决方案**：
```bash
# 增加特征点数
--camera-max-features 3000

# 放宽RANSAC阈值
--camera-ransac-thresh 2.0
```

### 问题2：残差光流仍然很大

**症状**：残差光流幅度接近原始光流

**可能原因**：
- 场景中有大量真实运动
- 单应性模型不适合（如非平面场景）
- 相机运动过于复杂

**解决方案**：
- 考虑禁用相机补偿
- 或使用更高级的补偿方法（如深度估计）

### 问题3：处理速度慢

**症状**：处理时间过长

**解决方案**：
```bash
# 减少特征点数
--camera-max-features 1000

# 或禁用相机补偿
--no-camera-compensation

# 或禁用可视化
--no-visualize
```

## 未来改进

计划中的功能：
- [ ] 支持深度估计的高级补偿
- [ ] 支持SE(3)刚体运动估计
- [ ] 支持多目标独立补偿
- [ ] 支持Rolling Shutter补偿
- [ ] 自适应参数调整

## 参考资料

- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)
- [Multiple View Geometry in Computer Vision](http://www.robots.ox.ac.uk/~vgg/hzbook/)
- OpenCV Homography Documentation

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

