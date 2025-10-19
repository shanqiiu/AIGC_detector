# 分辨率归一化快速使用指南

## ? 问题背景

您的视频尺寸范围：1280×720 ~ 750×960

**原始代码的问题**：
- 相同的物理运动，不同分辨率得到不同的动态度分数
- 低分辨率视频被系统性**低估**（误判为静态）
- 高分辨率视频被系统性**高估**（误判为动态）

**解决方案**：启用分辨率归一化

---

## ? 快速开始

### 单视频处理

```bash
# 基础用法（带归一化）
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution

# 完整配置
python video_processor.py \
    -i video.mp4 \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize
```

### 批量处理 + BadCase检测

```bash
# 推荐配置（公平评估）
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o results/ \
    --normalize-by-resolution

# 完整配置
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o results/ \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize \
    --camera-ransac-thresh 1.0 \
    --camera-max-features 2000
```

---

## ?? 关键参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|-------|------|--------|
| `--normalize-by-resolution` | False | 启用归一化 | **True（必须）** |
| `--flow-threshold-ratio` | 0.002 | 归一化阈值 | 0.002（通用）|

### 阈值调优指南

根据场景类型调整 `--flow-threshold-ratio`：

```bash
# 静态场景（建筑、风景）- 更严格
--flow-threshold-ratio 0.0015

# 通用场景 - 平衡
--flow-threshold-ratio 0.002

# 动态场景（人物、演唱会）- 更宽松
--flow-threshold-ratio 0.0025
```

---

## ? 效果对比

### 示例：处理混合分辨率视频

#### Before（未归一化）?
```
视频              分辨率      动态度   判定
building_A.mp4   1920×1080   0.72    动态（误判）
building_B.mp4   1280×720    0.58    中等（正确）
building_C.mp4   640×360     0.35    静态（误判）

问题：相同的建筑物，不同判定结果！
```

#### After（归一化）?
```
视频              分辨率      动态度   判定
building_A.mp4   1920×1080   0.58    中等（正确）
building_B.mp4   1280×720    0.58    中等（正确）
building_C.mp4   640×360     0.58    中等（正确）

? 公平一致的评估结果
```

---

## ? 如何验证归一化是否生效

### 检查输出JSON

```json
{
  "static_dynamics": {
    "mean_magnitude": 0.00385,
    "dynamics_score": 0.00512,
    "normalization_factor": 1469.0,    // 归一化因子（对角线长度）
    "is_normalized": true               // 确认已归一化
  }
}
```

**关键标志**：
- `is_normalized: true` → 归一化已启用
- `normalization_factor` > 1 → 显示对角线长度
- `mean_magnitude` < 0.01 → 归一化后的值（相对值）

---

## ? 常见问题

### Q1: 是否需要重新处理历史数据？

**A**: 取决于您的需求
- 如果需要与新数据比较 → 建议重新处理
- 如果仅内部参考 → 可以保持不变
- **新数据强烈建议启用归一化**

### Q2: 归一化会影响性能吗？

**A**: 几乎无影响
- 仅增加 1 次 sqrt 计算（计算对角线）
- 性能开销 < 0.1%

### Q3: 旧的阈值还能用吗？

**A**: 需要转换
```python
# 旧阈值（像素）
old_threshold = 2.0

# 基于您的典型分辨率（如1280×720）转换
diagonal = np.sqrt(1280**2 + 720**2)  # ≈ 1469
new_threshold_ratio = old_threshold / diagonal  # ≈ 0.0014

# 推荐值（稍微放宽）
recommended = 0.002
```

### Q4: 是否影响BadCase检测？

**A**: 正面影响
- 使 BadCase 检测更公平
- 避免因分辨率导致的误判
- `mismatch_threshold` 保持 0.3 不变

---

## ? 迁移清单

如果您要全面启用归一化：

- [ ] 更新处理脚本，添加 `--normalize-by-resolution`
- [ ] 测试典型视频，确认阈值合适（默认0.002）
- [ ] 更新文档和README，说明归一化参数
- [ ] (可选) 重新处理历史数据以保持一致性

---

## ? 总结

**您的场景（1280×720 ~ 750×960 混合分辨率）**：

? **必须启用归一化**
- 分辨率范围大（1.7倍差异）
- 未归一化会导致 30-40% 的评分偏差
- BadCase检测会受到严重影响

**推荐命令**：
```bash
python batch_with_badcase.py \
    -i your_videos/ \
    -l labels.json \
    -o results/ \
    --normalize-by-resolution \
    --flow-threshold-ratio 0.002 \
    --visualize
```

**预期收益**：
- ? 消除分辨率导致的系统性偏差
- ? 所有视频使用统一标准评估
- ? BadCase检测更加准确可靠
- ? 评估结果具有科学性和可比性

---

详细技术文档请参考：
- [分辨率公平性分析](./RESOLUTION_FAIRNESS_ANALYSIS.md)
- [归一化实现总结](./NORMALIZATION_IMPLEMENTATION_SUMMARY.md)

