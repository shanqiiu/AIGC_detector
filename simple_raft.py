"""
简化版RAFT模型，用于演示静态物体动态度计算
支持多种光流算法：Farneback, TV-L1等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple


class SimpleRAFT:
    """简化版RAFT光流估计 - 支持多种光流算法"""
    
    def __init__(self, device='cpu', method='farneback'):
        """
        初始化光流估计器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
            method: 光流方法
                - 'farneback': 快速，精度中等 (默认)
                - 'tvl1': 较慢，精度高，边界清晰
        """
        self.device = device
        self.method = method
        
        # 如果使用TV-L1，创建光流对象
        if method == 'tvl1':
            try:
                self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
                    tau=0.25,          # 时间步长
                    lambda_=0.15,      # 数据项权重
                    theta=0.3,         # 平滑项权重  
                    nscales=5,         # 金字塔层数
                    warps=5,           # Warp次数
                    epsilon=0.01,      # 停止阈值
                    innerIterations=30,
                    outerIterations=10,
                    scaleStep=0.8,
                    gamma=0.0,
                    useInitialFlow=False
                )
                print(f"? 使用TV-L1光流算法（高精度）")
            except AttributeError:
                print("? opencv-contrib-python未安装，TV-L1不可用")
                print("  自动回退到Farneback算法")
                print("  安装命令：pip install opencv-contrib-python")
                self.method = 'farneback'
        
        if method == 'farneback':
            print(f"? 使用Farneback光流算法（快速）")
    
    def estimate_flow_opencv(self, image1, image2):
        """使用OpenCV的光流估计作为RAFT的替代"""
        # 转换为灰度图
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = image2
        
        # 根据选择的方法计算光流
        if self.method == 'tvl1':
            # TV-L1光流（变分方法，边界保持好，精度高）
            flow = self.tvl1.calc(gray1, gray2, None)
            
        else:  # 'farneback' 或默认
            # Farneback光流算法（速度快，精度中等）
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 
                pyr_scale=0.5,     # 金字塔缩放
                levels=5,          # 金字塔层数（增加以处理大位移）
                winsize=15,        # 窗口大小
                iterations=3,      # 每层迭代次数
                poly_n=7,          # 多项式扩展邻域（增加平滑性）
                poly_sigma=1.5,    # 高斯标准差
                flags=0
            )
        
        return flow
    
    def predict_flow(self, image1, image2):
        """预测光流"""
        # 使用OpenCV估计光流
        flow = self.estimate_flow_opencv(image1, image2)
        
        # 转换为PyTorch张量格式 (2, H, W)
        if isinstance(flow, np.ndarray):
            flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        else:
            # 如果OpenCV返回None，创建零光流
            h, w = image1.shape[:2]
            flow_tensor = torch.zeros(2, h, w, dtype=torch.float32)
        
        return flow_tensor.numpy()


class SimpleRAFTPredictor:
    """简化版RAFT预测器"""
    
    def __init__(self, model_path=None, device='cpu', method='farneback'):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径（兼容参数，此实现中不使用）
            device: 计算设备
            method: 光流方法
                - 'farneback': 快速，精度中等 (默认)
                - 'tvl1': 较慢，精度高，边界清晰
        
        示例:
            # 使用Farneback（快速）
            predictor = SimpleRAFTPredictor(method='farneback')
            
            # 使用TV-L1（高精度）
            predictor = SimpleRAFTPredictor(method='tvl1')
        """
        self.device = device
        self.method = method
        self.model = SimpleRAFT(device, method)
    
    def predict_flow(self, image1, image2):
        """预测光流"""
        return self.model.predict_flow(image1, image2)
    
    def predict_flow_sequence(self, images):
        """预测图像序列的光流"""
        flows = []
        for i in range(len(images) - 1):
            flow = self.predict_flow(images[i], images[i + 1])
            flows.append(flow)
        return flows


if __name__ == '__main__':
    print("=" * 60)
    print("简化版RAFT光流预测器 - 使用示例")
    print("=" * 60)
    
    # 方法1: 使用Farneback（默认）
    print("\n方法1: Farneback光流（快速）")
    predictor_fast = SimpleRAFTPredictor(method='farneback')
    
    # 方法2: 使用TV-L1（高精度）
    print("\n方法2: TV-L1光流（高精度）")
    predictor_accurate = SimpleRAFTPredictor(method='tvl1')
    
    print("\n" + "=" * 60)
    print("选择建议：")
    print("- 快速原型/演示 → method='farneback'")
    print("- 生产环境/高精度 → method='tvl1'")
    print("=" * 60)
