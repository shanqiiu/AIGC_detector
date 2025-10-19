# ����������

## ����� ?

1. ? ɾ��18�������ĵ�
2. ? ���������README_NEW.md
3. ? �����ĵ�����DOCS_INDEX.md
4. ? �����򻯰�RAFT_SETUP_SIMPLE.md
5. ? ���������ܽ�CLEANUP_SUMMARY.md

---

## ���ֶ���ɵĲ���

### ����1: �滻��README

```bash
# ���ݾ�README
mv README.md README_OLD.md

# ʹ����README
mv README_NEW.md README.md
```

### ����2: ��RAFT�����ĵ�����ѡ��

```bash
# ���RAFT_SETUP_SIMPLE.md���㹻
rm RAFT_SETUP_GUIDE.md  # ɾ���ɵ���ϸ�汾
```

### ����3: ɾ����Ŀ���棨��ѡ��

```bash
# ������Ŀ�����Ƶ�����λ��
mv �¶Ƚ�չ�㱨_AIGC��Ƶ����������Ŀ.md ../docs/
```

### ����4: ��������ļ����Ƽ���

```bash
# ����testsĿ¼
mkdir -p tests

# �ƶ������ļ�
mv test_unified_dynamics.py tests/
mv test_camera_compensation.py tests/
mv test_static_dynamics.py tests/

# ����ʾ���ļ�λ��
mv example_unified_dynamics.py examples/
```

### ����5: ������ʱ�ļ�

```bash
# ɾ��ʾ�����
rm -rf demo_output/
rm -rf test_output*/

# ����Python����
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## �����ļ��ṹ

```
AIGC_detector/
��
������ ? �����ĵ�
��   ������ README.md                      # ���ĵ����£�
��   ������ DOCS_INDEX.md                  # �ĵ�����
��   ������ UNIFIED_DYNAMICS_GUIDE.md      # ͳһ������ϸָ��
��   ������ CAMERA_COMPENSATION_GUIDE.md   # �������ָ��
��   ������ STATIC_OBJECT_ANALYSIS_EXPLAINED.md  # ����ԭ��
��   ������ RAFT_SETUP_SIMPLE.md           # RAFT��������
��
������ ? ���Ĵ���
��   ������ video_processor.py
��   ������ unified_dynamics_scorer.py
��   ������ static_object_analyzer.py
��   ������ simple_raft.py
��   ������ dynamic_motion_compensation/
��       ������ camera_compensation.py
��       ������ object_motion.py
��       ������ se3_utils.py
��       ������ cli.py
��
������ ? ������ʾ��
��   ������ tests/
��   ��   ������ test_unified_dynamics.py
��   ��   ������ test_camera_compensation.py
��   ��   ������ test_static_dynamics.py
��   ������ examples/
��       ������ example_unified_dynamics.py
��
������ ? ������ģ��
��   ������ pretrained_models/
��   ��   ������ raft-things.pth
��   ������ demo_data/
��   ������ test_data/
��   ������ third_party/RAFT/
��
������ ? ����
��   ������ requirements.txt
��
������ ? ����
    ������ CLEANUP_SUMMARY.md      # �����ܽ�
    ������ FINAL_CLEANUP_STEPS.md  # ���ĵ�
    ������ README_OLD.md           # ��README����
```

---

## ��֤������

```bash
# ͳ���ĵ�����
ls *.md | wc -l
# Ԥ��: 8�����������ݺ���ʱ�ĵ���

# ͳ�ƺ��Ĵ���
ls *.py | wc -l
# Ԥ��: 4�����ļ�

# ������
ls tests/*.py | wc -l
# Ԥ��: 3�������ļ�
```

---

## ��ɺ�ĺô�

? **�ĵ�����**: ��23�����ٵ�5�������ĵ�
? **�ṹ����**: �����ĵ������롢���Է���
? **����ά��**: �������࣬����ά���ɱ�
? **��������**: README�ṩ���б�Ҫ��Ϣ
? **����ѧϰ**: ��ϸ�ĵ��������

---

## ��ѡ������.gitignore

```bash
# ����.gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# ��ʱ���
*_output/
results/
*.log

# �����ļ�
*_OLD.*
*_backup.*

# IDE
.vscode/
.idea/
*.swp

# ���ݣ����ļ���
videos/*.mp4
results/*.mp4
EOF
```

---

## ִ��һ������ű�

�����Զ�������ű���

```bash
#!/bin/bash
# cleanup.sh

echo "��ʼ����..."

# 1. �滻README
mv README.md README_OLD.md
mv README_NEW.md README.md
echo "? README�Ѹ���"

# 2. ��������ļ�
mkdir -p tests examples
mv test_*.py tests/ 2>/dev/null
mv example_*.py examples/ 2>/dev/null
echo "? �����ļ�������"

# 3. ������ʱ�ļ�
rm -rf demo_output test_output*
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "? ��ʱ�ļ�������"

# 4. ɾ���򻯰�RAFT�ĵ�����ѡ��
# rm RAFT_SETUP_GUIDE.md

echo "������ɣ�"
echo ""
echo "�ĵ��ṹ:"
tree -L 2 -I '__pycache__|*.pyc'
```

ʹ�÷���:
```bash
chmod +x cleanup.sh
./cleanup.sh
```

---

## ��ɼ���嵥

- [ ] README.md���滻
- [ ] �����ļ����ƶ���tests/
- [ ] ʾ���ļ����ƶ���examples/
- [ ] ��ʱ���������
- [ ] Python������ɾ��
- [ ] .gitignore�Ѵ�������ѡ��
- [ ] ��֤���й�������

---

**׼�����������ܼ�����Ŀ�ṹ�ɣ�** ?

