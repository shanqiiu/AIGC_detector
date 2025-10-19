# 迁移指南：从 batch_with_badcase.py 到 video_processor.py

## ? 重要通知

**`batch_with_badcase.py` 已被移除**

所有功能已整合到 `video_processor.py`，请使用新的统一入口。

---

## ? 命令迁移对照

### 旧命令 → 新命令

#### 基础用法
```bash
# 旧命令
python batch_with_badcase.py -i videos/ -l labels.json

# 新命令
python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

#### 完整参数
```bash
# 旧命令
python batch_with_badcase.py \
    -i videos/ \
    -l labels.json \
    -o output/ \
    --visualize \
    --normalize-by-resolution \
    --camera-ransac-thresh 1.0

# 新命令（几乎相同，只是改了 -l 参数名）
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    -o output/ \
    --visualize \
    --normalize-by-resolution \
    --camera-ransac-thresh 1.0
```

---

## ? 参数映射

| 旧参数 | 新参数 | 说明 |
|-------|-------|------|
| `-i, --input` | `-i, --input` + `--batch` | 需添加 --batch |
| `-l, --labels` | `--badcase-labels` 或 `-l` | 参数名更明确 |
| `-o, --output` | `-o, --output` | 相同 |
| `--visualize` | `--visualize` | 相同 |
| `--normalize-by-resolution` | `--normalize-by-resolution` | 相同 |
| 其他所有参数 | 完全相同 | 无需修改 |

---

## ? 快速迁移

### Step 1: 找到旧命令

```bash
# 搜索项目中的旧命令
grep -r "batch_with_badcase.py" .
```

### Step 2: 替换命令

```bash
# 查找并替换
旧: batch_with_badcase.py
新: video_processor.py --batch

旧: --labels
新: --badcase-labels
```

### Step 3: 验证

```bash
# 测试新命令
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

---

## ? 重构完成保证

### 功能对等性

| 原功能 | 新实现 | 状态 |
|--------|--------|------|
| 加载标签 | `load_expected_labels()` | ? |
| 单视频+BadCase | `process_single_video(..., expected_label)` | ? |
| 批量+BadCase | `batch_process_videos(..., badcase_labels)` | ? |
| BadCase报告 | `badcase_analyzer.save_batch_report()` | ? |
| 所有参数 | 完全集成 | ? |

### 输出结果相同

```
旧命令输出:
├── badcase_summary.txt
├── badcase_summary.json
├── badcase_videos.txt
└── {video_name}/badcase_report.txt

新命令输出:
├── badcase_summary.txt      ? 相同
├── badcase_summary.json     ? 相同
├── badcase_videos.txt       ? 相同
└── {video_name}/badcase_report.txt  ? 相同
```

---

## ? 为什么删除？

1. **功能100%整合** - 无功能损失
2. **无代码依赖** - 检查确认无其他文件导入
3. **维护简化** - 单一入口，避免参数不同步
4. **代码质量** - 消除重复，符合DRY原则

---

## ? 更新文档引用

需要更新的文档（示例命令）：
- `BADCASE_DETECTION_GUIDE.md`
- `BADCASE_FEATURE_SUMMARY.md`
- `QUICK_START_NORMALIZATION.md`
- 其他文档中的命令示例

**建议**: 全局替换文档中的命令示例

---

## ? 推荐新工作流

```bash
# 统一使用 video_processor.py

# 单视频
python video_processor.py -i video.mp4 --normalize-by-resolution

# 批量（普通）
python video_processor.py -i videos/ --batch --normalize-by-resolution

# 批量（BadCase）
python video_processor.py -i videos/ --batch -l labels.json --normalize-by-resolution
```

**更简单、更一致、更强大！**

