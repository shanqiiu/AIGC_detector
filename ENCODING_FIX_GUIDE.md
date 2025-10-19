# Encoding Fix Guide - 编码修复指南

## Problem Description - 问题描述

Generated text files (`.txt`, `.md`) show garbled characters in some editors.

生成的文本文件在某些编辑器中显示为乱码。

---

## Root Cause - 原因分析

1. **Python Default Encoding** - Windows Python may use GBK by default
2. **Editor Settings** - Some editors open files with non-UTF-8 encoding by default

Windows系统下Python可能默认使用GBK编码，某些编辑器也可能不使用UTF-8打开文件。

---

## Solutions - 解决方案

### Solution 1: Fix Existing Files (Recommended) - 修复已有文件（推荐）

Use the encoding fix tool:

```bash
# Fix all .txt files in results directory
# 修复results目录下所有txt文件
python fix_encoding.py --dir results --pattern *.txt

# Fix specific directory
# 修复特定目录
python fix_encoding.py --dir results/video1 --pattern *.txt

# Fix Markdown files
# 修复Markdown文件
python fix_encoding.py --dir . --pattern *.md
```

### Solution 2: One-Click Fix - 一键修复

Double-click to run:

```
fix_all_encoding.bat
```

This will automatically fix all .txt and .md files.

这会自动修复所有txt和md文件的编码。

### Solution 3: Regenerate Results - 重新生成结果

```bash
# Delete old results
# 删除旧结果
Remove-Item -Recurse -Force results

# Run again
# 重新运行
python video_processor.py -i videos/ -o results/ --batch --no-visualize
```

### Solution 4: Editor Settings - 编辑器设置

#### VS Code
1. Open file / 打开文件
2. Click encoding in bottom-right / 点击右下角编码
3. "Reopen with Encoding" → "GBK" or "Chinese Simplified GB18030"
4. "Save with Encoding" → "UTF-8"

#### Notepad++
1. Encoding menu / 编码菜单
2. Convert to UTF-8 (without BOM) / 转为UTF-8编码（无BOM）
3. Save / 保存

---

## Code Fixes (Completed) - 代码修复（已完成）

All Python files now have UTF-8 declaration:

所有Python文件已添加UTF-8声明：

```python
# -*- coding: utf-8 -*-
```

All file operations use UTF-8:

所有文件操作使用UTF-8：

```python
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
```

---

## Quick Start - 快速开始

### Step 1: Fix old files - 修复旧文件

```bash
fix_all_encoding.bat
```

### Step 2: Test - 测试

Open any .txt file and verify Chinese characters display correctly.

打开任意txt文件，验证中文显示正常。

---

## Verification - 验证

Check file encoding:

检查文件编码：

```python
import chardet

with open('file.txt', 'rb') as f:
    result = chardet.detect(f.read())
    print(f"Encoding: {result['encoding']}")
```

Expected output: `Encoding: utf-8`

预期输出：utf-8

---

## Summary - 总结

- **All new files** will be in UTF-8 (no garbled text)
- **Old files** can be fixed with `fix_all_encoding.bat`
- **Code** has been updated to always use UTF-8

- **所有新文件**都会使用UTF-8编码（无乱码）
- **旧文件**可用fix_all_encoding.bat修复
- **代码**已更新为始终使用UTF-8
