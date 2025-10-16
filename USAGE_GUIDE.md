# 静态物体动态度分析系统 - 使用指南

## 系统概述

本系统专门解决**相机转动拍摄静态建筑视频中RAFT光流计算偏高**的问题。通过区分相机运动和真实物体运动，系统能够仅计算静态物体的动态度，为视频质量评估和AIGC检测提供准确的指标。

## 核心技术

### 1. 相机运动估计与补偿
- 使用特征匹配算法（ORB/SIFT）检测关键点
- 通过RANSAC算法估计单应性矩阵
- 从原始光流中减去相机运动分量

### 2. 静态区域检测
- 基于补偿后光流幅度的阈值检测
- 结合图像梯度信息细化边界
- 形态学操作去除噪声

### 3. 动态度量化
- 计算静态区域的光流统计量
- 提供多维度动态度指标
- 支持时序稳定性分析

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision opencv-python matplotlib scipy scikit-image scikit-learn tqdm numpy

# 验证安装
python3 test_static_dynamics.py
```

### 2. 运行演示

```bash
# 运行内置演示
python3 demo.py
```

演示将创建一个模拟相机转动的建筑场景，展示系统如何：
- 检测相机运动
- 补偿运动影响  
- 识别静态区域
- 计算真实动态度

### 3. 处理真实视频

```bash
# 处理视频文件
python3 video_processor.py -i your_video.mp4 -o output_dir

# 处理图像序列
python3 video_processor.py -i image_directory/ -o output_dir

# 自定义参数
python3 video_processor.py \
    -i video.mp4 \
    -o results \
    --max_frames 100 \
    --frame_skip 2 \
    --fov 60 \
    --device cpu
```

## 输出结果解读

### 1. 数值指标

#### 动态度分数 (Dynamics Score)
- **< 1.0**: 优秀 - 静态物体动态度低，相机运动补偿效果良好
- **1.0-2.0**: 良好 - 存在轻微残余运动，可接受
- **> 2.0**: 需要关注 - 可能存在补偿误差或真实物体运动

#### 静态区域比例 (Static Ratio)  
- **> 0.7**: 理想 - 场景主要由静态物体组成
- **0.5-0.7**: 适中 - 静态和动态区域比例平衡
- **< 0.5**: 不理想 - 动态内容过多，分析可能不准确

#### 时序稳定性 (Temporal Stability)
- **> 0.8**: 高稳定性 - 结果可靠一致
- **0.6-0.8**: 中等稳定性 - 结果基本可信
- **< 0.6**: 低稳定性 - 结果波动较大

### 2. 可视化结果

系统生成多种可视化图表：

- **关键帧分析**: 显示原始光流、补偿后光流、静态区域检测
- **时序曲线**: 展示动态度和静态比例随时间变化
- **统计分布**: 分析光流幅度分布特征

### 3. 输出文件结构

```
output_directory/
├── analysis_results.json      # 完整数值结果
├── analysis_report.txt        # 文字分析报告  
└── visualizations/           # 可视化图表
    ├── frame_xxxx_analysis.png
    ├── temporal_dynamics.png
    └── static_ratio_changes.png
```

## 应用场景

### 1. 建筑物视频分析
适用于：
- 房地产展示视频
- 建筑监控录像
- 无人机航拍建筑

### 2. AIGC视频检测
- 检测AI生成视频中的异常动态
- 评估视频时序一致性
- 识别不自然的物体运动

### 3. 视频质量评估
- 相机抖动检测
- 运动补偿效果评估
- 视频稳定性分析

## 参数调优指南

### 1. 相机运动估计参数

```python
# 在static_object_analyzer.py中调整
estimator = CameraMotionEstimator(
    feature_detector='ORB',     # 或 'SIFT'
    max_features=1000,          # 增加以提高精度
    ransac_threshold=1.0,       # 降低以提高严格性
    ransac_max_trials=1000      # 增加以提高鲁棒性
)
```

### 2. 静态区域检测参数

```python
# 调整检测阈值
detector = StaticObjectDetector(
    flow_threshold=2.0,         # 降低以更严格检测
    consistency_threshold=0.8,   # 提高以要求更高一致性
    min_region_size=100         # 调整最小区域大小
)
```

### 3. 动态度计算参数

```python
calculator = StaticObjectDynamicsCalculator(
    temporal_window=5,          # 时序窗口大小
    spatial_kernel_size=5,      # 空间核大小
    dynamics_threshold=1.0      # 动态度阈值
)
```

## 常见问题解决

### 1. 相机运动估计失败

**症状**: 报告显示"相机运动估计返回空结果"

**解决方案**:
- 检查输入图像质量和对比度
- 尝试不同的特征检测器（SIFT vs ORB）
- 调整特征检测参数
- 确保相邻帧间有足够的重叠区域

### 2. 静态区域检测不准确

**症状**: 静态区域比例异常低或高

**解决方案**:
- 调整`flow_threshold`参数
- 检查相机运动补偿是否有效
- 验证输入视频的相机运动类型
- 考虑场景特点调整参数

### 3. 动态度分数异常高

**症状**: 明显静态的场景显示高动态度

**解决方案**:
- 检查相机内参估计是否准确
- 验证相机运动模型是否适合（平移vs旋转）
- 调整光流计算参数
- 检查输入视频是否有真实物体运动

### 4. 性能优化

**内存不足**:
```bash
# 使用CPU模式
python3 video_processor.py -i video.mp4 --device cpu

# 限制处理帧数
python3 video_processor.py -i video.mp4 --max_frames 50

# 跳帧处理
python3 video_processor.py -i video.mp4 --frame_skip 3
```

**处理速度慢**:
- 使用GPU加速（如果可用）
- 减少处理帧数
- 降低输入分辨率
- 调整光流计算精度

## 技术限制

### 1. 相机运动类型
- 主要支持平移和轻微旋转
- 复杂的3D运动可能导致补偿不准确
- 快速运动可能导致特征匹配失败

### 2. 场景要求
- 需要足够的纹理特征进行匹配
- 过于均匀的场景可能影响效果
- 强烈光照变化可能影响检测

### 3. 计算资源
- 完整RAFT模型需要GPU支持
- 大分辨率视频需要较多内存
- 长视频处理时间较长

## 扩展开发

### 1. 自定义光流算法

```python
# 在simple_raft.py中实现自定义算法
class CustomFlowEstimator:
    def estimate_flow(self, img1, img2):
        # 实现自定义光流算法
        pass
```

### 2. 添加新的动态度指标

```python
# 在static_object_analyzer.py中扩展
def calculate_custom_dynamics(self, flow, mask):
    # 实现自定义动态度计算
    pass
```

### 3. 集成到现有系统

```python
from static_object_analyzer import StaticObjectDynamicsCalculator

# 在您的项目中使用
calculator = StaticObjectDynamicsCalculator()
result = calculator.calculate_frame_dynamics(flow, img1, img2)
dynamics_score = result['static_dynamics']['dynamics_score']
```

## 联系支持

如果遇到技术问题或需要定制开发，请：

1. 查看测试输出和错误信息
2. 检查输入数据格式和质量
3. 参考本指南的故障排除部分
4. 提供详细的错误描述和环境信息

---

**版本**: 1.0  
**更新日期**: 2025-10-16  
**兼容性**: Python 3.7+, PyTorch 1.9+