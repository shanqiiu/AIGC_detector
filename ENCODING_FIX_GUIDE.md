# Encoding Fix Guide - �����޸�ָ��

## Problem Description - ��������

Generated text files (`.txt`, `.md`) show garbled characters in some editors.

���ɵ��ı��ļ���ĳЩ�༭������ʾΪ���롣

---

## Root Cause - ԭ�����

1. **Python Default Encoding** - Windows Python may use GBK by default
2. **Editor Settings** - Some editors open files with non-UTF-8 encoding by default

Windowsϵͳ��Python����Ĭ��ʹ��GBK���룬ĳЩ�༭��Ҳ���ܲ�ʹ��UTF-8���ļ���

---

## Solutions - �������

### Solution 1: Fix Existing Files (Recommended) - �޸������ļ����Ƽ���

Use the encoding fix tool:

```bash
# Fix all .txt files in results directory
# �޸�resultsĿ¼������txt�ļ�
python fix_encoding.py --dir results --pattern *.txt

# Fix specific directory
# �޸��ض�Ŀ¼
python fix_encoding.py --dir results/video1 --pattern *.txt

# Fix Markdown files
# �޸�Markdown�ļ�
python fix_encoding.py --dir . --pattern *.md
```

### Solution 2: One-Click Fix - һ���޸�

Double-click to run:

```
fix_all_encoding.bat
```

This will automatically fix all .txt and .md files.

����Զ��޸�����txt��md�ļ��ı��롣

### Solution 3: Regenerate Results - �������ɽ��

```bash
# Delete old results
# ɾ���ɽ��
Remove-Item -Recurse -Force results

# Run again
# ��������
python video_processor.py -i videos/ -o results/ --batch --no-visualize
```

### Solution 4: Editor Settings - �༭������

#### VS Code
1. Open file / ���ļ�
2. Click encoding in bottom-right / ������½Ǳ���
3. "Reopen with Encoding" �� "GBK" or "Chinese Simplified GB18030"
4. "Save with Encoding" �� "UTF-8"

#### Notepad++
1. Encoding menu / ����˵�
2. Convert to UTF-8 (without BOM) / תΪUTF-8���루��BOM��
3. Save / ����

---

## Code Fixes (Completed) - �����޸�������ɣ�

All Python files now have UTF-8 declaration:

����Python�ļ������UTF-8������

```python
# -*- coding: utf-8 -*-
```

All file operations use UTF-8:

�����ļ�����ʹ��UTF-8��

```python
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
```

---

## Quick Start - ���ٿ�ʼ

### Step 1: Fix old files - �޸����ļ�

```bash
fix_all_encoding.bat
```

### Step 2: Test - ����

Open any .txt file and verify Chinese characters display correctly.

������txt�ļ�����֤������ʾ������

---

## Verification - ��֤

Check file encoding:

����ļ����룺

```python
import chardet

with open('file.txt', 'rb') as f:
    result = chardet.detect(f.read())
    print(f"Encoding: {result['encoding']}")
```

Expected output: `Encoding: utf-8`

Ԥ�������utf-8

---

## Summary - �ܽ�

- **All new files** will be in UTF-8 (no garbled text)
- **Old files** can be fixed with `fix_all_encoding.bat`
- **Code** has been updated to always use UTF-8

- **�������ļ�**����ʹ��UTF-8���루�����룩
- **���ļ�**����fix_all_encoding.bat�޸�
- **����**�Ѹ���Ϊʼ��ʹ��UTF-8
