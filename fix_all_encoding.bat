@echo off
chcp 65001 >nul
echo ========================================
echo ��������һ���޸�����
echo ========================================
echo.

echo �����޸�����ļ�����...
python fix_encoding.py --dir results --pattern *.txt

echo.
echo �����޸�Markdown�ĵ�����...
python fix_encoding.py --dir . --pattern *.md

echo.
echo ========================================
echo �޸���ɣ�
echo ========================================
pause

