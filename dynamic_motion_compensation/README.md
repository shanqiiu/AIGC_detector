# 动态物体多视角光流抵消（Global Shutter 版）

本模块提供在无遮挡假设下，针对动态物体的多视角（相机运动）光流抵消能力：
- 先用特征匹配与RANSAC估计帧间单应性，计算并减去相机引起的光流。
- 可选：若有深度与对象掩码，使用PnP估计对象的SE(3)刚体运动，进一步抵消其视差。
- 输出残差光流与统计指标，作为“动态度”更稳健的衡量。

> 注意：本模块不考虑滚动快门，仅适用于Global Shutter或近似场景。

## 文件结构

```
dynamic_motion_compensation/
├── se3_utils.py                # SE(3)与投影工具（备选）
├── camera_compensation.py      # 相机侧单应性补偿与残差光流
├── object_motion.py            # （可选）对象SE(3)估计（PnP+深度）
└── cli.py                      # 命令行入口，端到端运行
```

## 安装依赖

```bash
pip install -r requirements.txt
```

如仓库根目录已有`requirements.txt`且已安装，可跳过。

## 快速开始

- 输入可为视频文件或图像序列目录；默认用仓库已有RAFT/简化RAFT估计光流。
- 输出包括原始光流、相机光流与残差光流的`npy`文件，以及若干可视化。

```bash
python -m dynamic_motion_compensation.cli \
  --input demo_data/ \
  --output demo_global_mc \
  --device cpu \
  --fov 60
```

参数说明：
- `--input/-i`：视频文件或图像目录
- `--output/-o`：输出目录
- `--device`：`cuda`或`cpu`
- `--raft_model`：可选的RAFT权重文件
- `--max_frames`/`--frame_skip`：控制帧数与采样率
- `--fov`：估算相机内参（无标定时）
- `--depth`/`--masks`：可选；若提供，则对每帧执行对象PnP以估计SE(3)

## 输出内容

- `flows.npy`：原始RAFT光流（列表式对象数组）
- `camera_flows.npy`：由单应性估计的相机光流
- `residual_flows.npy`：残差光流（多视角视差已抵消）
- `stats.json`：每帧残差统计，含均值与90分位数
- `vis/flow_mag_xxxx.png` 与 `vis/residual_mag_xxxx.png`：幅度可视化

## 与主项目集成

- 可直接调用`dynamic_motion_compensation.camera_compensation.CameraCompensator`来获得`residual_flow`；再将其作为你的“动态度”输入。
- 如果已有深度与目标掩码，可调用`ObjectSE3Estimator.estimate_with_depth`进一步在对象坐标系抵消刚体运动，残差更接近非刚性/高频运动。

## 实施建议

- 无遮挡假设下，可收紧RANSAC阈值（如0.8px）提升稳健性。
- 大位移场景建议保留光流的金字塔或特征引导；本模块对光流源不做限制。
- 深度质量决定对象PnP效果；若无深度，可先仅用相机补偿得到残差动态度。

## 许可

MIT
