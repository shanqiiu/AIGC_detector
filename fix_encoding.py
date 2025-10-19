# -*- coding: utf-8 -*-
"""
�޸��ı��ļ���������
��GBK������ļ�ת��ΪUTF-8����
"""

import os
import glob
from pathlib import Path


def fix_file_encoding(file_path, source_encoding='gbk', target_encoding='utf-8'):
    """�޸������ļ��ı���"""
    try:
        # ������GBK��ȡ
        with open(file_path, 'r', encoding=source_encoding) as f:
            content = f.read()
        
        # ��UTF-8д��
        with open(file_path, 'w', encoding=target_encoding) as f:
            f.write(content)
        
        print(f"? �޸�: {file_path}")
        return True
    except Exception as e:
        print(f"? ʧ��: {file_path} - {e}")
        return False


def fix_directory_encoding(directory, file_pattern='*.txt'):
    """�޸�Ŀ¼������ƥ���ļ��ı���"""
    print(f"ɨ��Ŀ¼: {directory}")
    print(f"�ļ�ģʽ: {file_pattern}")
    print("=" * 60)
    
    # ��������ƥ����ļ�
    files = []
    for root, dirs, filenames in os.walk(directory):
        pattern_path = os.path.join(root, file_pattern)
        files.extend(glob.glob(pattern_path))
    
    if not files:
        print(f"δ�ҵ�ƥ����ļ�")
        return
    
    print(f"�ҵ� {len(files)} ���ļ�\n")
    
    # �޸�ÿ���ļ�
    success_count = 0
    for file_path in files:
        if fix_file_encoding(file_path):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"���! �ɹ��޸� {success_count}/{len(files)} ���ļ�")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='�޸��ı��ļ�����')
    parser.add_argument('--dir', default='results', help='Ŀ��Ŀ¼')
    parser.add_argument('--pattern', default='*.txt', help='�ļ�ģʽ')
    parser.add_argument('--source', default='gbk', help='Դ����')
    parser.add_argument('--target', default='utf-8', help='Ŀ�����')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"����: Ŀ¼������ - {args.dir}")
    else:
        fix_directory_encoding(args.dir, args.pattern)

