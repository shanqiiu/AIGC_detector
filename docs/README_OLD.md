# 静态物体动态度分析系统

本系统专门用于解决相机转动拍摄静态建筑视频中RAFT光流计算偏高的问题，通过区分相机运动和真实物体运动，仅计算静态物体的动态度。

## 问题背景

在使用RAFT进行相机转动拍摄静态建筑的视频测试时，由于相机运动会导致整个场景产生光流，使得RAFT计算出的动态度偏高。本系统通过以下技术解决这个问题：

1. **相机运动估计**: 使用特征匹配和单应性矩阵估计相机运动
2. **运动补偿**: 从原始光流中减去相机运动分量
3. **静态区域检测**: 识别场景中的静态物体区域
4. **动态度计算**: 仅针对静态物体计算真实的动态度

## 系统架构

```
├── raft_model.py              # RAFT光流估计模型
├── static_object_analyzer.py  # 静态物体分析核心算法
├── video_processor.py         # 视频处理主程序
├── test_static_dynamics.py    # 测试脚本
├── requirements.txt           # 依赖包列表
└── README.md                 # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 核心功能

### 1. 相机运动估计 (`CameraMotionEstimator`)

- 使用ORB/SIFT特征检测器提取特征点
- 通过特征匹配和RANSAC算法估计单应性矩阵
- 分解单应性矩阵得到相机运动参数

### 2. 静态物体检测 (`StaticObjectDetector`)

- 基于光流幅度阈值检测静态区域
- 使用相机运动补偿消除相机运动影响
- 结合图像梯度信息细化静态区域边界

### 3. 动态度计算 (`StaticObjectDynamicsCalculator`)

- 计算静态区域的光流统计量
- 提供多种动态度指标：平均幅度、标准差、最大值等
- 支持时序分析和稳定性评估

## 使用方法

### 命令行使用

```bash
# 处理视频文件
python video_processor.py -i video.mp4 -o output_dir

# 处理图像序列
python video_processor.py -i image_directory/ -o output_dir

# 指定更多参数
python video_processor.py \
    -i video.mp4 \
    -o output_dir \
    --max_frames 100 \
    --frame_skip 2 \
    --fov 60 \
    --device cuda
```

### 参数说明

- `--input, -i`: 输入视频文件或图像目录路径
- `--output, -o`: 输出目录路径 (默认: output)
- `--raft_model, -m`: RAFT预训练模型路径 (可选)
- `--max_frames`: 最大处理帧数 (可选)
- `--frame_skip`: 帧跳跃间隔 (默认: 1)
- `--device`: 计算设备 cuda/cpu (默认: cuda)
- `--fov`: 相机视场角度数 (默认: 60.0)

### 编程接口使用

```python
from video_processor import VideoProcessor
from static_object_analyzer import StaticObjectDynamicsCalculator

# 创建处理器
processor = VideoProcessor(device='cuda')

# 加载视频
frames = processor.load_video('video.mp4')

# 处理并分析
result = processor.process_video(frames, output_dir='output')

# 获取结果
temporal_stats = result['temporal_stats']
print(f"平均动态度分数: {temporal_stats['mean_dynamics_score']:.3f}")
```

## 输出结果

系统会在输出目录中生成以下文件：

```
output/
├── analysis_results.json      # 数值分析结果
├── analysis_report.txt        # 文字分析报告
└── visualizations/           # 可视化结果
    ├── frame_0000_analysis.png
    ├── frame_0025_analysis.png
    ├── temporal_dynamics.png
    └── static_ratio_changes.png
```

### 结果解读

#### 动态度分数 (Dynamics Score)
- **< 1.0**: 静态物体动态度低，相机运动补偿效果良好
- **1.0-2.0**: 动态度中等，存在轻微残余运动
- **> 2.0**: 动态度高，可能存在补偿误差或真实物体运动

#### 静态区域比例 (Static Ratio)
- **> 0.7**: 场景主要由静态物体组成，适合分析
- **0.5-0.7**: 静态和动态区域比例适中
- **< 0.5**: 动态区域较多，分析结果可能不够准确

#### 时序稳定性 (Temporal Stability)
- **> 0.8**: 稳定性高，结果可靠
- **0.6-0.8**: 稳定性中等
- **< 0.6**: 稳定性低，结果波动较大

## 算法原理

### 1. RAFT光流估计

使用改进的RAFT网络计算帧间光流：
- 特征提取网络提取多尺度特征
- 相关性计算构建相关性金字塔
- 迭代更新网络细化光流预测

### 2. 相机运动补偿

```python
# 单应性变换补偿相机运动
transformed_coords = homography @ coords
camera_flow = transformed_coords - original_coords
compensated_flow = original_flow - camera_flow
```

### 3. 静态区域检测

```python
# 基于光流幅度检测静态区域
flow_magnitude = sqrt(flow_x² + flow_y²)
static_mask = flow_magnitude < threshold

# 形态学操作去除噪声
static_mask = morphology_close(morphology_open(static_mask))
```

### 4. 动态度计算

```python
# 计算静态区域动态度
static_flow = compensated_flow[static_mask]
dynamics_score = mean(magnitude) + 0.5 * std(magnitude)
```

## 测试

运行测试脚本验证系统功能：

```bash
python test_static_dynamics.py
```

测试包括：
- 相机运动补偿测试
- 静态区域检测测试  
- 动态度计算测试
- 报告生成测试
- 端到端功能测试

## 应用场景

本系统特别适用于：

1. **建筑物检测**: 相机绕建筑物转动拍摄的视频分析
2. **静态场景监控**: 需要检测静态场景中微小变化的应用
3. **视频质量评估**: 评估视频中静态内容的稳定性
4. **相机运动分析**: 分析和补偿相机运动对光流计算的影响

## 技术特点

- **高精度**: 通过相机运动补偿显著提高静态物体动态度计算精度
- **鲁棒性**: 支持多种特征检测器和匹配算法
- **可视化**: 提供丰富的可视化结果和分析报告
- **灵活性**: 支持视频文件和图像序列输入
- **可扩展**: 模块化设计，易于扩展和定制

## 注意事项

1. **相机标定**: 如果有精确的相机内参，可以提供更准确的结果
2. **场景选择**: 静态物体占比高的场景分析效果更好
3. **计算资源**: RAFT模型需要一定的GPU计算资源
4. **参数调优**: 可根据具体场景调整阈值参数

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 使用CPU模式
   python video_processor.py -i video.mp4 --device cpu
   ```

2. **特征匹配失败**
   - 检查输入图像质量
   - 尝试不同的特征检测器
   - 调整特征检测参数

3. **动态度异常高**
   - 检查相机运动估计是否准确
   - 调整静态区域检测阈值
   - 验证输入视频质量

## 贡献

欢迎提交Issue和Pull Request来改进系统功能。

## 许可证

本项目采用MIT许可证。