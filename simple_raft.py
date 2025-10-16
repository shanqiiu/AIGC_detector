"""
简化版RAFT模型，用于演示静态物体动态度计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple


class SimpleRAFT:
    """简化版RAFT光流估计"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
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
        
        # 使用Farneback光流算法
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
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
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = SimpleRAFT(device)
    
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