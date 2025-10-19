# 文档索引

## 核心文档

### 1. README.md
**主要文档** - 快速开始、API使用、命令行参数

### 2. UNIFIED_DYNAMICS_GUIDE.md
**统一动态度评分详细指南** - 技术原理、高级配置、应用场景

### 3. CAMERA_COMPENSATION_GUIDE.md
**相机补偿使用指南** - 工作原理、参数调优、故障排除

### 4. STATIC_OBJECT_ANALYSIS_EXPLAINED.md
**静态物体分析技术原理** - 算法细节、数学推导、性能优化

### 5. RAFT_SETUP_GUIDE.md
**RAFT模型设置指南** - 模型下载、环境配置

---

## 文档使用建议

### 快速上手
1. 阅读 **README.md** 快速开始部分
2. 运行示例命令
3. 查看输出结果

### 深入理解
1. **UNIFIED_DYNAMICS_GUIDE.md** - 理解评分系统
2. **STATIC_OBJECT_ANALYSIS_EXPLAINED.md** - 理解技术原理
3. **CAMERA_COMPENSATION_GUIDE.md** - 优化相机补偿

### 问题排查
1. README常见问题部分
2. CAMERA_COMPENSATION_GUIDE故障排除
3. 查看测试文件示例

---

## 测试文件

- `tests/test_unified_dynamics.py` - 统一动态度测试
- `tests/test_camera_compensation.py` - 相机补偿测试
- `tests/test_static_dynamics.py` - 静态分析测试
- `example_unified_dynamics.py` - 使用示例

---

## 文档结构

```
AIGC_detector/
├── README.md                              # 主文档
├── UNIFIED_DYNAMICS_GUIDE.md              # 统一评分指南
├── CAMERA_COMPENSATION_GUIDE.md           # 相机补偿指南
├── STATIC_OBJECT_ANALYSIS_EXPLAINED.md    # 技术原理
├── RAFT_SETUP_GUIDE.md                    # RAFT设置
└── DOCS_INDEX.md                          # 本文档
```

---

**建议阅读顺序**：
README.md → UNIFIED_DYNAMICS_GUIDE.md → 其他文档（按需）

