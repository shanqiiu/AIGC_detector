@echo off
chcp 65001 >nul
echo ========================================
echo 编码问题一键修复工具
echo ========================================
echo.

echo 正在修复结果文件编码...
python fix_encoding.py --dir results --pattern *.txt

echo.
echo 正在修复Markdown文档编码...
python fix_encoding.py --dir . --pattern *.md

echo.
echo ========================================
echo 修复完成！
echo ========================================
pause

