# RAFT模型快速设置

## 一键安装

### 1. 下载预训练模型

```bash
# 创建模型目录
mkdir -p pretrained_models

# 下载RAFT-Things模型（推荐）
wget https://drive.google.com/file/d/1YWQtFl9ewNKGFAx3x4WRJCkI_zMYNfUy/view \
  -O pretrained_models/raft-things.pth

# 或使用gdown（更方便）
pip install gdown
gdown --id 1YWQtFl9ewNKGFAx3x4WRJCkI_zMYNfUy -O pretrained_models/raft-things.pth
```

### 2. 验证安装

```python
from simple_raft import SimpleRAFTPredictor

# 测试加载
predictor = SimpleRAFTPredictor(
    model_path="pretrained_models/raft-things.pth",
    device='cuda'
)
print("RAFT模型加载成功！")
```

---

## 可选模型

| 模型 | 适用场景 | 下载链接 |
|------|----------|----------|
| raft-things.pth | 通用场景（推荐） | [下载](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) |
| raft-sintel.pth | 复杂运动 | [下载](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) |
| raft-kitti.pth | 自动驾驶 | [下载](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) |

---

## 常见问题

### Q: 模型下载太慢？

**方案1**：使用镜像站点
```bash
# 使用国内镜像（如果有）
```

**方案2**：手动下载
1. 访问 Google Drive 链接
2. 手动下载到 `pretrained_models/` 目录

### Q: CUDA内存不足？

```bash
# 使用CPU模式
python video_processor.py -i video.mp4 -o output/ --device cpu
```

### Q: 找不到模型文件？

检查路径：
```bash
ls pretrained_models/
# 应该看到: raft-things.pth
```

---

## 完整设置（首次使用）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型
pip install gdown
gdown --id 1YWQtFl9ewNKGFAx3x4WRJCkI_zMYNfUy \
  -O pretrained_models/raft-things.pth

# 3. 测试运行
python video_processor.py -i demo_data/ -o test_output/

# 成功！
```

---

**需要帮助？** 查看 [README.md](README.md) 或提交 Issue

