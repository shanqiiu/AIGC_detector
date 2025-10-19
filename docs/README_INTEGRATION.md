# ? 项目整合完成 - 统一入口使用指南

## ? 整合完成

**batch_with_badcase.py** 已整合到 **video_processor.py**，现在只需一个命令！

---

## ? 新的使用方式

### 1. 单视频分析

```bash
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution
```

### 2. 批量分析（无BadCase检测）

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --normalize-by-resolution
```

### 3. 批量分析 + BadCase检测（推荐）?

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### 4. 完整配置示例

```bash
python video_processor.py \
    -i D:/my_git_projects/data/Multi-View_Consistency \
    --batch \
    --badcase-labels labels.json \
    -o output/ \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize \
    --device cuda \
    --mismatch-threshold 0.3
```

---

## ? 向后兼容

**旧命令仍然可用**（自动转发）：

```bash
# 旧方式（仍可用）
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    --normalize-by-resolution

# 会显示提示并自动转发到 video_processor.py
```

---

## ? 完整参数列表

### 必需参数
- `--input, -i`: 输入路径（文件/目录）
- `--output, -o`: 输出目录（默认: output）

### 模式控制
- `--batch`: 批量处理模式

### BadCase检测（可选）
- `--badcase-labels, -l`: 标签文件，启用BadCase检测
- `--mismatch-threshold`: BadCase不匹配阈值（默认0.3）

### 分辨率归一化（强烈推荐）?
- `--normalize-by-resolution`: 启用归一化
- `--flow-threshold-ratio`: 归一化阈值（默认0.002）

### 相机补偿
- `--no-camera-compensation`: 禁用相机补偿
- `--camera-ransac-thresh`: RANSAC阈值（默认1.0）
- `--camera-max-features`: 最大特征点数（默认2000）

### 其他
- `--visualize`: 生成可视化结果
- `--device`: cuda/cpu
- `--fov`: 相机视场角
- `--raft_model, -m`: RAFT模型路径

---

## ? 典型使用场景

### 场景1: 单视频质量分析

```bash
python video_processor.py \
    -i video.mp4 \
    -o analysis/ \
    --normalize-by-resolution \
    --visualize
```

### 场景2: 批量BadCase筛选（您的主要场景）

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -l labels.json \
    -o badcase_output/ \
    --normalize-by-resolution
```

**输出**：
- `badcase_summary.txt` - BadCase统计报告
- `badcase_summary.json` - JSON格式结果
- `badcase_videos.txt` - BadCase视频列表
- `{video_name}/` - 每个视频的详细分析

### 场景3: 混合分辨率公平评估

```bash
# 您的视频：1280×720 ~ 750×960
python video_processor.py \
    -i mixed_resolution_videos/ \
    --batch \
    -l labels.json \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002
```

---

## ? 整合带来的改进

| 改进维度 | 效果 |
|---------|------|
| **代码简化** | 单一入口，-45行重复代码 |
| **参数一致** | 自动同步，无需手动维护 |
| **功能灵活** | BadCase作为可选模块 |
| **向后兼容** | 旧命令继续工作 |
| **维护成本** | 1个main()替代2个 |

---

## ? 功能验证

### 验证整合是否成功

```bash
# 测试统一入口
python video_processor.py --help | grep badcase-labels
# 应该显示: --badcase-labels, -l

# 测试wrapper
python batch_with_badcase.py --help
# 应该显示提示信息
```

### 验证BadCase检测

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -l labels.json \
    -o test_output/

# 检查输出
ls test_output/
# 应该包含: badcase_summary.txt, badcase_summary.json, badcase_videos.txt
```

---

## ?? 重要提示

### 对于您的混合分辨率场景

**必须添加的参数**：
```bash
--normalize-by-resolution  # 确保分辨率公平性
```

**完整推荐命令**：
```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -l labels.json \
    --normalize-by-resolution \
    -o results/
```

---

## ? 相关文档

- `INTEGRATION_COMPLETE.md` - 整合详细说明
- `FINAL_INTEGRATION_ANALYSIS.md` - 整合分析
- `QUICK_START_NORMALIZATION.md` - 归一化使用指南
- `THRESHOLDS_COMPLETE_GUIDE.md` - 阈值完整指南

---

## ? 总结

**整合后的优势**：
- ? 单一命令，多种模式
- ? BadCase检测变为可选功能
- ? 参数完全统一
- ? 100%向后兼容
- ? 代码更简洁

**推荐使用**: `video_processor.py` 统一入口，功能完整、参数一致、维护简单！

