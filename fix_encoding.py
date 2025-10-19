# -*- coding: utf-8 -*-
"""
修复文本文件编码问题
将GBK编码的文件转换为UTF-8编码
"""

import os
import glob
from pathlib import Path


def fix_file_encoding(file_path, source_encoding='gbk', target_encoding='utf-8'):
    """修复单个文件的编码"""
    try:
        # 尝试用GBK读取
        with open(file_path, 'r', encoding=source_encoding) as f:
            content = f.read()
        
        # 用UTF-8写回
        with open(file_path, 'w', encoding=target_encoding) as f:
            f.write(content)
        
        print(f"? 修复: {file_path}")
        return True
    except Exception as e:
        print(f"? 失败: {file_path} - {e}")
        return False


def fix_directory_encoding(directory, file_pattern='*.txt'):
    """修复目录下所有匹配文件的编码"""
    print(f"扫描目录: {directory}")
    print(f"文件模式: {file_pattern}")
    print("=" * 60)
    
    # 查找所有匹配的文件
    files = []
    for root, dirs, filenames in os.walk(directory):
        pattern_path = os.path.join(root, file_pattern)
        files.extend(glob.glob(pattern_path))
    
    if not files:
        print(f"未找到匹配的文件")
        return
    
    print(f"找到 {len(files)} 个文件\n")
    
    # 修复每个文件
    success_count = 0
    for file_path in files:
        if fix_file_encoding(file_path):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"完成! 成功修复 {success_count}/{len(files)} 个文件")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='修复文本文件编码')
    parser.add_argument('--dir', default='results', help='目标目录')
    parser.add_argument('--pattern', default='*.txt', help='文件模式')
    parser.add_argument('--source', default='gbk', help='源编码')
    parser.add_argument('--target', default='utf-8', help='目标编码')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"错误: 目录不存在 - {args.dir}")
    else:
        fix_directory_encoding(args.dir, args.pattern)

