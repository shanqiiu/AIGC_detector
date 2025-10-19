# batch_with_badcase.py 与 video_processor.py 整合分析

## ? 功能重叠分析

### 相同功能

| 功能 | video_processor.py | batch_with_badcase.py | 重叠度 |
|------|-------------------|---------------------|--------|
| 单视频处理 | ? `process_single_video()` | ? `process_with_badcase_detection()` | 90% |
| 批量处理 | ? `batch_process_videos()` | ? `batch_process_with_badcase()` | 85% |
| 参数解析 | ? `main()` + argparse | ? `main()` + argparse | 95% |
| VideoProcessor初始化 | ? | ? | 100% |

### 独有功能

| 功能 | video_processor.py | batch_with_badcase.py |
|------|-------------------|---------------------|
| 图像序列处理 | ? | ? |
| BadCase检测 | ? | ? |
| 标签加载 | ? | ? |
| BadCase报告 | ? | ? |

**结论**：约 **85%** 功能重叠，可以整合！

---

## ? 整合方案

### 方案：将 BadCase 检测作为可选模块

```
统一入口: video_processor.py

├── 模式1: 单视频分析
│   python video_processor.py -i video.mp4
│
├── 模式2: 批量分析（无BadCase）
│   python video_processor.py -i videos/ --batch
│
└── 模式3: 批量分析 + BadCase检测
    python video_processor.py -i videos/ --batch --badcase-labels labels.json
```

---

## ?? 实施方案

### Step 1: 合并批量处理逻辑

```python
# video_processor.py (整合后)

def batch_process_videos(processor, input_dir, output_dir, camera_fov, 
                        badcase_labels=None):  # ← 新增参数
    """
    批量处理视频
    
    Args:
        badcase_labels: 可选，期望标签字典，启用BadCase检测
    """
    
    # ... 查找视频文件 ...
    
    results = []
    for video_path in video_files:
        if badcase_labels is not None:
            # BadCase模式
            expected = badcase_labels.get(video_name, 'dynamic')
            result = process_with_badcase_detection(
                processor, video_path, expected, output_dir, camera_fov
            )
        else:
            # 普通模式
            result = process_single_video(
                processor, video_path, output_dir, camera_fov
            )
        results.append(result)
    
    # 保存结果
    if badcase_labels is not None:
        # BadCase报告
        summary = processor.badcase_analyzer.generate_batch_summary(results)
        processor.badcase_analyzer.save_batch_report(summary, results, output_dir)
    else:
        # 普通报告
        save_batch_summary(results, output_dir)
    
    return results
```

### Step 2: 统一参数解析

```python
# video_processor.py main()

parser.add_argument('--batch', action='store_true',
                   help='批量处理模式')

# BadCase相关参数（可选）
parser.add_argument('--badcase-labels', '-l', default=None,
                   help='期望标签文件（JSON），启用BadCase检测')
parser.add_argument('--mismatch-threshold', type=float, default=0.3,
                   help='BadCase不匹配阈值')

# 使用
if args.batch:
    if args.badcase_labels:
        # BadCase模式
        labels = load_labels(args.badcase_labels)
        processor.setup_badcase_detector(args.mismatch_threshold)
    batch_process_videos(..., badcase_labels=labels if args.badcase_labels else None)
```

### Step 3: 废弃 batch_with_badcase.py

```python
# batch_with_badcase.py (简化为wrapper)

"""
兼容性wrapper，重定向到 video_processor.py
建议直接使用: python video_processor.py --batch --badcase-labels labels.json
"""

import sys
import subprocess

# 转换参数
args = sys.argv[1:]
# --labels → --badcase-labels
args = [arg.replace('--labels', '--badcase-labels') for arg in args]
# 添加 --batch
if '--batch' not in args:
    args.insert(0, '--batch')

# 调用 video_processor.py
subprocess.run(['python', 'video_processor.py'] + args)
```

---

## ? 整合收益

| 维度 | 整合前 | 整合后 | 改进 |
|------|-------|--------|------|
| 代码行数 | 249 + 803 = 1052 | ~850 | -200行 (-19%) |
| 维护入口 | 2个main() | 1个main() | 简化 |
| 参数一致性 | 需手动同步 | 自动一致 | ? |
| 用户体验 | 2个命令 | 1个命令 | 简化 |

---

## ? 整合后的使用方式

### 单视频分析

```bash
python video_processor.py -i video.mp4 --normalize-by-resolution
```

### 批量分析（无BadCase）

```bash
python video_processor.py -i videos/ --batch --normalize-by-resolution
```

### 批量分析 + BadCase检测

```bash
python video_processor.py \
    -i videos/ \
    --batch \
    --badcase-labels labels.json \
    --normalize-by-resolution \
    --visualize
```

### 兼容旧命令（通过wrapper）

```bash
# 仍然可用，自动转发
python batch_with_badcase.py -i videos/ -l labels.json
```

---

## ?? 注意事项

### 1. 向后兼容

- ? 保留 `batch_with_badcase.py` 作为wrapper
- ? 旧命令自动转换到新入口
- ? 所有现有脚本无需修改

### 2. 功能完整性

- ? 所有功能保留
- ? BadCase检测变为可选模块
- ? 普通批量处理更轻量

### 3. 代码组织

```
video_processor.py (统一入口)
├── 类: VideoProcessor (核心处理)
├── 函数: process_single_video (单视频)
├── 函数: batch_process_videos (批量，支持BadCase)
├── 函数: load_labels (标签加载)
└── 函数: main (统一CLI)

batch_with_badcase.py (兼容wrapper)
└── 重定向到 video_processor.py
```

---

## ? 整合后的参数列表

```bash
# 基础参数
--input, -i              # 输入
--output, -o            # 输出
--device                # 设备
--raft_model, -m        # 模型

# 模式控制
--batch                 # 批量模式

# BadCase检测（可选）
--badcase-labels, -l    # 标签文件（启用BadCase）
--mismatch-threshold    # 不匹配阈值

# 相机补偿
--no-camera-compensation
--camera-ransac-thresh
--camera-max-features

# 分辨率归一化
--normalize-by-resolution
--flow-threshold-ratio

# 其他
--visualize
--fov
```

---

## ? 推荐实施步骤

1. ? 在 video_processor.py 中添加 BadCase 检测支持
2. ? 整合批量处理逻辑
3. ? 统一参数解析
4. ? 将 batch_with_badcase.py 改为兼容wrapper
5. ? 更新文档

**是否需要我立即实施这个整合方案？**

