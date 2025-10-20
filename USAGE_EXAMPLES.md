# 使用示例与常见错误

## ? 正确用法

### 场景1：批量处理视频目录（推荐）

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

**说明**：
- `-i videos/` - 指定包含视频文件的目录
- `--batch` - **必须添加此参数**，表示批量处理模式
- 系统会自动找到目录中的所有 `.mp4` 视频文件

---

### 场景2：处理单个视频文件

```bash
python video_processor.py \
    -i videos/test.mp4 \
    --normalize-by-resolution \
    --visualize
```

**说明**：
- 直接指定视频文件路径
- 不需要 `--batch` 参数

---

### 场景3：处理图像序列

```bash
python video_processor.py \
    -i image_frames/ \
    --normalize-by-resolution
```

**说明**：
- `image_frames/` 目录应包含 `.jpg` 或 `.png` 图像文件
- 不需要 `--batch` 参数
- 系统会按文件名排序加载

---

## ? 常见错误

### 错误1：忘记添加 --batch 参数

**错误命令**：
```bash
python video_processor.py -i videos/ --badcase-labels labels.json
```

**错误信息**：
```
正在从目录加载图像: videos/
加载完成，共 0 帧
IndexError: list index out of range
```

**原因**：
- 没有 `--batch` 参数时，系统认为 `videos/` 是图像序列目录
- 但 `videos/` 里是 `.mp4` 文件，不是图像，所以加载了 0 帧

**修复**：
```bash
# 添加 --batch 参数
python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

---

### 错误2：目录路径错误

**错误命令**：
```bash
python video_processor.py -i video/ --batch
```

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'video/'
```

**修复**：
```bash
# 确认目录名称，应该是 videos/ 而不是 video/
python video_processor.py -i videos/ --batch
```

---

### 错误3：标签文件路径错误

**错误命令**：
```bash
python video_processor.py -i videos/ --batch --badcase-labels label.json
```

**错误信息**：
```
FileNotFoundError: 标签文件不存在: label.json
```

**修复**：
```bash
# 确认文件名，应该是 labels.json
python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

---

## ? 快速命令参考

### 最简单（单视频）

```bash
python video_processor.py -i test.mp4
```

### 最常用（批量 + 归一化）

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --normalize-by-resolution
```

### 最完整（BadCase + 可视化 + 归一化）

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### CPU模式

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --device cpu \
    --normalize-by-resolution
```

### 快速测试（跳帧）

```bash
python video_processor.py \
    -i test.mp4 \
    --frame_skip 3 \
    --max_frames 60
```

---

## ? 参数组合说明

| 输入类型 | 是否需要 --batch | 示例 |
|---------|-----------------|------|
| 单个视频文件 | ? 不需要 | `-i test.mp4` |
| 视频目录（批量） | ? **必需** | `-i videos/ --batch` |
| 图像序列目录 | ? 不需要 | `-i frames/` |

---

## ? 根据您的需求

### 我想批量处理视频

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution
```

**关键点**：
- ? 必须有 `--batch`
- ? 目录包含 `.mp4` 文件
- ? `labels.json` 存在

### 我想测试单个视频

```bash
python video_processor.py -i videos/test.mp4 --normalize-by-resolution
```

**关键点**：
- ? 直接指定文件路径
- ? 不要 `--batch`

### 我想处理图像序列

```bash
python video_processor.py -i my_frames/
```

**关键点**：
- ? 目录包含 `.jpg` 或 `.png`
- ? 不要 `--batch`

---

## ? 验证命令是否正确

### 检查清单

1. **批量模式**
   - [ ] 输入是目录？
   - [ ] 包含 `.mp4` 文件？
   - [ ] 添加了 `--batch` 参数？

2. **单视频模式**
   - [ ] 输入是单个文件？
   - [ ] 文件存在？
   - [ ] 没有 `--batch` 参数？

3. **BadCase检测**
   - [ ] `labels.json` 存在？
   - [ ] 添加了 `--badcase-labels labels.json`？

4. **归一化（推荐）**
   - [ ] 添加了 `--normalize-by-resolution`？

---

## ? 立即测试

### 测试1：验证环境

```bash
python video_processor.py --help
```

### 测试2：处理单个视频

```bash
python video_processor.py \
    -i videos/test.mp4 \
    -o test_output/
```

### 测试3：批量处理

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    -o batch_output/
```

---

**记住**：批量处理视频目录，**必须添加 `--batch` 参数**！


