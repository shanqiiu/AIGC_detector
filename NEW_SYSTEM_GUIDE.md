# 重构后的统一动态度评估系统使用指南

## ? 目录
- [核心改进](#核心改进)
- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [API参考](#api参考)
- [迁移指南](#迁移指南)

---

## ? 核心改进

### **1. 统一的评分标准**
- **统一分数范围**: 0-1，不再强制分段
- **场景自适应**: 自动识别静态场景和动态场景
- **可筛选性**: 能够识别"动态场景中动作很小"的视频

### **2. 双模式分析**
```
静态场景（建筑/静物）
  └→ 检测静态区域 → 计算残差光流 → 评估异常运动

动态场景（人物/动物）
  └→ 检测主体区域 → 计算主体光流 → 评估动作幅度
```

### **3. 灵活的质量筛选**
- 筛选动态场景中动态度过低的视频
- 筛选静态场景中异常运动过高的视频
- 按分数范围、分类等多种方式筛选

---

## ? 快速开始

### **安装依赖**
```bash
pip install -r requirements.txt
```

### **最简单的使用**
```python
from video_processor import VideoProcessor

# 创建处理器（使用新系统）
processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    use_new_calculator=True  # 启用新计算器
)

# 处理视频
frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir='output/')

# 查看结果
print(f"动态度分数: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
print(f"场景类型: {result['dynamics_classification']['scene_type']}")
```

### **运行测试**
```bash
# 快速测试新系统
python test_new_system.py

# 运行完整示例
python example_new_system.py
```

---

## ?? 系统架构

### **核心模块**

```
unified_dynamics_calculator.py   # 统一动态度计算器（核心）
├── 同时检测静态区域和动态区域
├── 自动判断场景类型
└── 输出统一的0-1分数

video_quality_filter.py         # 视频质量筛选器
├── 筛选低动态度视频
├── 筛选高异常视频
└── 生成筛选报告

dynamics_config.py              # 配置管理
├── 预设配置（strict/balanced/lenient）
├── 阈值参数
└── 评分映射表

video_processor.py              # 主处理器（已更新）
├── 集成新计算器
├── 向后兼容
└── 批量处理支持
```

### **数据流**

```
视频帧
  ↓
光流计算 (RAFT)
  ↓
相机运动补偿
  ↓
残差光流
  ↓
统一动态度计算器
  ├→ 检测静态区域
  ├→ 检测动态区域
  ├→ 判断场景类型
  └→ 计算统一分数
  ↓
质量筛选器 (可选)
  ↓
输出结果
```

---

## ?? 配置说明

### **预设配置**

系统提供三种预设配置：

| 配置 | 适用场景 | 特点 |
|------|---------|------|
| `strict` | 质量要求高 | 阈值严格，筛选更多 |
| `balanced` | 通用场景（**默认**）| 平衡准确性和包容性 |
| `lenient` | 接受更多视频 | 阈值宽松，筛选较少 |

### **关键阈值**

```python
DETECTION_THRESHOLDS = {
    'static_threshold': 0.002,    # 静态区域检测阈值
    'subject_threshold': 0.005,   # 主体区域检测阈值
}

QUALITY_FILTER_THRESHOLDS = {
    'low_dynamic_in_dynamic_scene': 0.3,   # 动态场景低动态阈值
    'high_anomaly_in_static_scene': 0.5,   # 静态场景高异常阈值
}
```

### **自定义配置**

```python
from dynamics_config import get_config

# 加载并修改配置
config = get_config('balanced')
config['detection']['static_threshold'] = 0.0015  # 更严格
config['quality_filter']['low_dynamic_in_dynamic_scene'] = 0.35

# 使用自定义配置
processor = VideoProcessor(
    use_new_calculator=True,
    config_preset='balanced'  # 基础配置
)
# 然后手动修改
processor.unified_calculator.static_threshold = 0.0015
```

---

## ? 使用示例

### **示例1: 单视频处理**

```python
from video_processor import VideoProcessor

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    device='cuda',
    enable_camera_compensation=True,
    use_normalized_flow=True,
    use_new_calculator=True,
    config_preset='balanced'
)

frames = processor.load_video("video.mp4")
result = processor.process_video(frames, output_dir='output/')

# 输出结果
print(f"场景: {result['dynamics_classification']['scene_type']}")
print(f"分数: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
print(f"等级: {result['dynamics_classification']['description']}")
```

### **示例2: 批量处理并筛选**

```python
from video_processor import batch_process_videos
from video_quality_filter import VideoQualityFilter

# 批量处理
processor = VideoProcessor(use_new_calculator=True)
results = batch_process_videos(processor, 'videos/', 'output/', 60.0)

# 筛选低动态度视频
quality_filter = VideoQualityFilter()
low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3
)

print(f"找到 {len(low_dynamic_videos)} 个动态度过低的视频")
for video in low_dynamic_videos:
    print(f"  {video['video_name']}: {video['score']:.3f}")
    print(f"    {video['reason']}")
```

### **示例3: 质量统计**

```python
from video_quality_filter import VideoQualityFilter

# 获取统计信息
quality_filter = VideoQualityFilter()
stats = quality_filter.get_quality_statistics(results)

print(f"总视频数: {stats['total_videos']}")
print(f"平均分数: {stats['score_statistics']['mean']:.3f}")
print(f"场景类型分布: {stats['scene_type_distribution']}")
print(f"动态等级分布: {stats['category_distribution']}")
```

### **示例4: 分数范围筛选**

```python
# 筛选中等动态度的视频（0.35-0.60）
medium_dynamic_videos = quality_filter.filter_by_score_range(
    results,
    min_score=0.35,
    max_score=0.60
)

# 只筛选动态场景的中等动态视频
medium_dynamic_videos = quality_filter.filter_by_score_range(
    results,
    min_score=0.35,
    max_score=0.60,
    scene_type='dynamic'
)
```

---

## ? API参考

### **UnifiedDynamicsCalculator**

统一动态度计算器

```python
calculator = UnifiedDynamicsCalculator(
    static_threshold=0.002,       # 静态区域阈值
    subject_threshold=0.005,      # 主体区域阈值
    use_normalized_flow=True,     # 使用归一化
    scene_auto_detect=True        # 自动检测场景
)

result = calculator.calculate_unified_dynamics(flows, images)
```

**返回结果**:
```python
{
    'unified_dynamics_score': 0.52,  # 统一分数 (0-1)
    'scene_type': 'dynamic',         # 场景类型
    'classification': {              # 分类信息
        'category': 'medium_dynamic',
        'description': '中等动态',
        'typical_examples': ['正常行走', '日常活动']
    },
    'temporal_stats': {...},         # 时序统计
    'interpretation': '...'          # 文字解释
}
```

### **VideoQualityFilter**

视频质量筛选器

```python
filter = VideoQualityFilter()

# 筛选动态场景中动态度过低的视频
low_videos = filter.filter_low_dynamics_in_dynamic_scenes(results, threshold=0.3)

# 筛选静态场景中异常过高的视频
high_videos = filter.filter_high_static_anomaly(results, threshold=0.5)

# 按分数范围筛选
range_videos = filter.filter_by_score_range(results, min_score=0.2, max_score=0.4)

# 按分类筛选
category_videos = filter.filter_by_category(results, ['low_dynamic', 'medium_dynamic'])

# 获取统计信息
stats = filter.get_quality_statistics(results)
```

---

## ? 迁移指南

### **从旧系统迁移**

#### **方式1: 使用新计算器（推荐）**

```python
# 旧代码
processor = VideoProcessor(
    use_normalized_flow=True
)

# 新代码（启用新计算器）
processor = VideoProcessor(
    use_normalized_flow=True,
    use_new_calculator=True  # 添加这一行
)
```

#### **方式2: 保持兼容**

```python
# 使用旧计算器（向后兼容）
processor = VideoProcessor(
    use_normalized_flow=True,
    use_new_calculator=False  # 使用旧系统
)
```

### **结果格式变化**

新旧系统的结果格式**完全兼容**，关键字段保持一致：

```python
# 两种系统都有的字段
result['unified_dynamics']['unified_dynamics_score']  # 动态度分数
result['dynamics_classification']['category']         # 分类
result['dynamics_classification']['scene_type']       # 场景类型
```

---

## ? 评分标准

### **静态场景（建筑/静物）**

| 分数范围 | 等级 | 含义 | 典型示例 |
|---------|------|------|---------|
| 0.00-0.15 | 纯静态 | 完全静止 | 建筑、雕塑 |
| 0.15-0.35 | 低动态 | 轻微振动 | 旗帜飘动、树叶 |
| 0.35-0.60 | 中等动态 | 明显振动 | 较大幅度摇动 |
| 0.60-0.85 | 高动态 | 异常运动 | 强风、震动 |
| 0.85-1.00 | 极高动态 | 严重异常 | 设备故障 |

### **动态场景（人物/动物）**

| 分数范围 | 等级 | 含义 | 典型示例 |
|---------|------|------|---------|
| 0.00-0.15 | 纯静态 | 几乎不动 | 静坐、静止站立 |
| 0.15-0.35 | 低动态 | 轻微动作 | 缓慢移动、微调姿势 |
| 0.35-0.60 | 中等动态 | 正常动作 | 行走、日常活动 |
| 0.60-0.85 | 高动态 | 活跃动作 | 跑步、跳舞 |
| 0.85-1.00 | 极高动态 | 剧烈动作 | 快速舞蹈、体育运动 |

---

## ? 常见问题

### **Q1: 如何筛选"动态场景中动作很小"的视频？**

```python
low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
    results,
    threshold=0.3  # 小于0.3的动态视频会被筛选出来
)
```

### **Q2: 如何调整系统敏感度？**

使用不同的预设配置：
- `strict`: 更敏感，筛选更多
- `balanced`: 默认
- `lenient`: 更宽松，筛选更少

### **Q3: 新旧系统可以同时使用吗？**

可以！设置 `use_new_calculator=False` 使用旧系统。

### **Q4: 如何验证新系统工作正常？**

```bash
python test_new_system.py
```

---

## ? 更新日志

### v2.0 (重构版)
- ? 新增统一动态度计算器
- ? 新增视频质量筛选器
- ? 新增配置管理系统
- ? 支持动态场景低动态度检测
- ? 统一的0-1评分标准
- ? 保持向后兼容性

---

## ? 支持

如有问题，请参考：
- 快速测试: `python test_new_system.py`
- 完整示例: `python example_new_system.py`
- 配置指南: `python dynamics_config.py`

