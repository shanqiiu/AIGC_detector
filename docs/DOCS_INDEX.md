# ? 文档索引

快速导航到所需文档

---

## ? 新手入门

### 1. [README.md](../README.md) - **从这里开始**
- 项目介绍
- 安装步骤
- 基础使用
- 参数说明
- FAQ

**适合**：所有用户

---

### 2. [QUICK_START.md](../QUICK_START.md) - **5分钟上手**
- 快速安装
- 常用命令
- 结果解读
- 常见问题

**适合**：想快速试用的用户

---

## ? 详细文档

### 3. [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - **项目总览**
- 技术架构
- 核心技术原理
- 性能指标
- 未来规划

**适合**：想深入了解项目的用户

---

### 4. [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - **API文档**
- 编程接口详解
- 核心类与方法
- 代码示例
- 错误处理

**适合**：开发者，需要编程集成的用户

---

## ? 按使用场景

### 我想...

#### ? 快速评估一个视频
→ [QUICK_START.md](../QUICK_START.md) - "单视频分析"

```bash
python video_processor.py -i video.mp4
```

---

#### ? 批量处理多个视频
→ [README.md](../README.md) - "场景2：批量视频处理"

```bash
python video_processor.py -i videos/ --batch --normalize-by-resolution
```

---

#### ? 检测质量问题（BadCase）
→ [README.md](../README.md) - "场景3：BadCase检测"

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

---

#### ? 处理混合分辨率视频
→ [README.md](../README.md) - "分辨率归一化参数"  
→ [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - "分辨率归一化"

**关键参数**：`--normalize-by-resolution`

---

#### ? 用代码集成到我的项目
→ [API_DOCUMENTATION.md](../API_DOCUMENTATION.md)

```python
from video_processor import VideoProcessor

processor = VideoProcessor(
    raft_model_path="pretrained_models/raft-things.pth",
    use_normalized_flow=True
)
result = processor.process_video(frames, camera_matrix, "output/")
```

---

#### ? 自定义评分规则
→ [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - "UnifiedDynamicsScorer"

```python
custom_weights = {
    'flow_magnitude': 0.40,
    'spatial_coverage': 0.30,
    # ...
}
scorer = UnifiedDynamicsScorer(weights=custom_weights)
```

---

#### ? 理解技术原理
→ [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - "核心技术"  
→ [README.md](../README.md) - "技术原理"

- 光流估计（RAFT）
- 分辨率归一化
- 相机运动补偿
- 统一评分系统

---

#### ? 优化处理速度
→ [README.md](../README.md) - "性能指标"  
→ [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - "性能优化建议"

```bash
# GPU加速
python video_processor.py -i video.mp4 --device cuda

# 跳帧处理
python video_processor.py -i video.mp4 --frame_skip 2
```

---

#### ? 解决CUDA内存不足
→ [QUICK_START.md](../QUICK_START.md) - "Q: CUDA out of memory"

```bash
# 跳帧或使用CPU
python video_processor.py -i video.mp4 --frame_skip 2
python video_processor.py -i video.mp4 --device cpu
```

---

#### ? 理解输出结果
→ [README.md](../README.md) - "输出结果说明"  
→ [QUICK_START.md](../QUICK_START.md) - "结果解读"

**文件说明**：
- `analysis_report.txt` - 文本报告
- `analysis_results.json` - 结构化数据
- `badcase_report.txt` - BadCase详情

---

## ? 按内容类型

### 教程类

| 文档 | 内容 | 难度 |
|------|------|------|
| [QUICK_START.md](../QUICK_START.md) | 快速上手 | ? 入门 |
| [README.md](../README.md) | 全面指南 | ?? 进阶 |
| [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) | 编程接口 | ??? 高级 |

### 参考类

| 文档 | 内容 | 用途 |
|------|------|------|
| [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) | 项目总览 | 理解架构 |
| [README.md](../README.md) | 完整文档 | 查阅参数 |
| [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) | API文档 | 查阅接口 |

---

## ? 文档关系图

```
                    ┌─────────────┐
                    │ DOCS_INDEX  │ ← 你在这里
                    │  (导航页)   │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      ┌────▼────┐    ┌─────▼─────┐   ┌────▼────┐
      │ QUICK   │    │  README   │   │PROJECT  │
      │ START   │    │ (核心文档) │   │OVERVIEW │
      └─────────┘    └──────┬────┘   └─────────┘
                            │
                      ┌─────▼─────┐
                      │    API    │
                      │DOCUMENTATION│
                      └───────────┘
```

### 推荐阅读顺序

1. **第一次使用**：QUICK_START.md → README.md
2. **深入理解**：PROJECT_OVERVIEW.md
3. **开发集成**：API_DOCUMENTATION.md

---

## ? 快速链接

### 核心功能

- [安装指南](../README.md#安装步骤)
- [基础用法](../README.md#基础用法)
- [批量处理](../README.md#场景2批量视频处理)
- [BadCase检测](../README.md#场景3badcase检测)
- [分辨率归一化](../README.md#分辨率归一化参数重要)

### 技术深入

- [技术架构](../PROJECT_OVERVIEW.md#技术架构)
- [光流估计](../PROJECT_OVERVIEW.md#1-光流估计raft)
- [归一化原理](../PROJECT_OVERVIEW.md#2-分辨率归一化)
- [相机补偿](../PROJECT_OVERVIEW.md#3-相机运动补偿)
- [评分系统](../PROJECT_OVERVIEW.md#4-统一评分系统)

### API接口

- [VideoProcessor](../API_DOCUMENTATION.md#videoprocessor-主处理器)
- [BadCaseDetector](../API_DOCUMENTATION.md#badcasedetector-质量检测器)
- [UnifiedDynamicsScorer](../API_DOCUMENTATION.md#unifieddynamicsscorer-评分系统)
- [完整示例](../API_DOCUMENTATION.md#完整工作流示例)

### 问题解决

- [常见问题FAQ](../README.md#常见问题)
- [快速问题解决](../QUICK_START.md#常见问题)
- [错误处理](../API_DOCUMENTATION.md#错误处理)

---

## ? 获取帮助

### 文档没有解决你的问题？

1. **搜索现有Issue**
   - 查看 [GitHub Issues](https://github.com/your-repo/issues)

2. **提问前准备**
   - 你的环境（Python版本、CUDA版本等）
   - 完整的错误信息
   - 复现步骤

3. **提交新Issue**
   - [报告Bug](https://github.com/your-repo/issues/new?labels=bug)
   - [功能建议](https://github.com/your-repo/issues/new?labels=enhancement)

4. **直接联系**
   - Email: your-email@example.com

---

## ? 文档更新记录

### 最新版本 (2025-10-19)

- ? 完整的README文档
- ? 快速开始指南
- ? 项目总览文档
- ? API使用文档
- ? 文档索引页

### 未来计划

- [ ] 视频教程
- [ ] 中英文双语文档
- [ ] 更多代码示例
- [ ] 常见问题扩展

---

## ? 文档反馈

文档有不清楚的地方？

- 提交 [Documentation Issue](https://github.com/your-repo/issues/new?labels=documentation)
- 直接提PR改进文档

我们持续改进文档质量！

---

<div align="center">

**找到你需要的了吗？**

如果还有问题，欢迎 [提Issue](https://github.com/your-repo/issues) 或查阅其他文档

Made with ?? by AIGC Video Quality Team

</div>

