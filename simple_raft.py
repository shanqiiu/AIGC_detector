"""
统一光流估计模块
支持多种光流算法：Farneback, TV-L1, RAFT官方模型

使用示例:
    # 快速 - Farneback
    predictor = SimpleRAFTPredictor(method='farneback')
    
    # 高精度 - TV-L1
    predictor = SimpleRAFTPredictor(method='tvl1')
    
    # 最高精度 - RAFT官方
    predictor = SimpleRAFTPredictor(
        method='raft',
        model_path='pretrained_models/raft-things.pth'
    )
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional


class SimpleRAFT:
    """统一光流估计器 - 支持 Farneback, TV-L1, RAFT"""
    
    def __init__(self, device='cpu', method='farneback', model_path=None):
        """
        初始化光流估计器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
            method: 光流方法
                - 'farneback': 快速，精度中等 (默认，OpenCV内置)
                - 'tvl1': 较慢，精度高，边界清晰 (需要 opencv-contrib-python)
                - 'raft': 最高精度 (需要模型文件和 third_party/RAFT)
            model_path: RAFT模型路径 (仅当 method='raft' 时使用)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.method = method
        self.model_path = model_path
        self.raft_model = None
        
        # 初始化不同的光流方法
        if method == 'tvl1':
            self._init_tvl1()
        elif method == 'raft':
            self._init_raft()
        elif method == 'farneback':
            print(f"使用Farneback光流算法（快速）")
        else:
            print(f"警告: 未知方法 '{method}'，使用默认Farneback")
            self.method = 'farneback'
    
    def _init_tvl1(self):
        """初始化TV-L1光流"""
        try:
            # 使用默认参数创建，避免不同OpenCV版本的参数名称差异
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            # 可以设置的通用参数
            self.tvl1.setTau(0.25)
            self.tvl1.setLambda(0.15)
            self.tvl1.setTheta(0.3)
            self.tvl1.setScalesNumber(5)
            self.tvl1.setWarpingsNumber(5)
            self.tvl1.setEpsilon(0.01)
            print(f"使用TV-L1光流算法（高精度）")
        except AttributeError:
            print("错误: opencv-contrib-python未安装，TV-L1不可用")
            print("  安装命令: pip install opencv-contrib-python")
            print("  自动回退到Farneback算法")
            self.method = 'farneback'
    
    def _init_raft(self):
        """初始化RAFT官方模型"""
        if not self.model_path or not Path(self.model_path).exists():
            print(f"错误: RAFT模型文件不存在: {self.model_path}")
            print("  自动回退到Farneback算法")
            self.method = 'farneback'
            return
        
        try:
            # 添加RAFT官方代码路径
            raft_core_path = Path(__file__).parent / 'third_party' / 'RAFT' / 'core'
            if not raft_core_path.exists():
                raft_core_path = Path(__file__).parent.parent / 'third_party' / 'RAFT' / 'core'
            
            if not raft_core_path.exists():
                raise FileNotFoundError("third_party/RAFT/core 目录不存在")
            
            sys.path.insert(0, str(raft_core_path))
            
            # 导入RAFT
            from raft import RAFT  # type: ignore
            import argparse
            
            # 创建args对象
            args = argparse.Namespace()
            args.small = False
            args.mixed_precision = False
            args.alternate_corr = False
            args.dropout = 0
            args.corr_levels = 4
            args.corr_radius = 4
            
            self.raft_model = RAFT(args)
            
            # 加载预训练权重
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # 处理可能的state_dict包装
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            # 移除可能的module.前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            
            self.raft_model.load_state_dict(new_state_dict, strict=False)
            self.raft_model.to(self.device)
            self.raft_model.eval()
            
            print(f"成功加载RAFT官方模型: {self.model_path}")
            
        except Exception as e:
            print(f"错误: 加载RAFT模型失败: {e}")
            print("  自动回退到Farneback算法")
            self.method = 'farneback'
    
    def predict_flow(self, image1, image2):
        """
        预测光流
        
        Args:
            image1: 第一帧图像 (H, W, 3) numpy数组，RGB格式，0-255
            image2: 第二帧图像 (H, W, 3) numpy数组，RGB格式，0-255
            
        Returns:
            flow: 光流 (2, H, W) numpy数组
        """
        if self.method == 'raft' and self.raft_model is not None:
            return self._predict_flow_raft(image1, image2)
        else:
            return self._predict_flow_opencv(image1, image2)
    
    def _predict_flow_raft(self, image1, image2):
        """使用RAFT模型预测光流"""
        with torch.no_grad():
            # 预处理
            img1 = self._preprocess_image_raft(image1)
            img2 = self._preprocess_image_raft(image2)
            
            # 预测
            _, flow_up = self.raft_model(img1, img2, iters=20, test_mode=True)
            
            # 转换为numpy (2, H, W)
            flow = flow_up[0].cpu().numpy()
            return flow
    
    def _preprocess_image_raft(self, img):
        """预处理图像为RAFT输入格式"""
        # 转换为tensor: (H, W, 3) -> (1, 3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0)
        
        # Padding到8的倍数
        _, _, h, w = img_tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        
        return img_tensor.to(self.device)
    
    def _predict_flow_opencv(self, image1, image2):
        """使用OpenCV预测光流"""
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
        
        # 转换为PyTorch张量格式 (2, H, W)
        if isinstance(flow, np.ndarray):
            flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        else:
            # 如果OpenCV返回None，创建零光流
            h, w = image1.shape[:2]
            flow_tensor = torch.zeros(2, h, w, dtype=torch.float32)
        
        return flow_tensor.numpy()


class SimpleRAFTPredictor:
    """统一光流预测器接口"""
    
    def __init__(self, model_path=None, device='cpu', method='farneback'):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径（仅当 method='raft' 时需要）
            device: 计算设备 ('cpu' 或 'cuda')
            method: 光流方法
                - 'farneback': 快速，精度中等 (默认)
                - 'tvl1': 较慢，精度高，边界清晰
                - 'raft': 最高精度（需要模型文件）
        
        示例:
            # 使用Farneback（快速，默认）
            predictor = SimpleRAFTPredictor(method='farneback')
            
            # 使用TV-L1（高精度）
            predictor = SimpleRAFTPredictor(method='tvl1')
            
            # 使用RAFT官方模型（最高精度）
            predictor = SimpleRAFTPredictor(
                method='raft',
                model_path='pretrained_models/raft-things.pth',
                device='cuda'
            )
        """
        self.device = device
        self.method = method
        self.model = SimpleRAFT(device, method, model_path)
    
    def predict_flow(self, image1, image2):
        """
        预测光流
        
        Args:
            image1: 第一帧图像 (H, W, 3) numpy数组
            image2: 第二帧图像 (H, W, 3) numpy数组
            
        Returns:
            flow: 光流 (2, H, W) numpy数组
        """
        return self.model.predict_flow(image1, image2)
    
    def predict_flow_sequence(self, images):
        """
        预测图像序列的光流
        
        Args:
            images: 图像列表，每个图像为 (H, W, 3) numpy数组
            
        Returns:
            flows: 光流列表，每个光流为 (2, H, W) numpy数组
        """
        flows = []
        for i in range(len(images) - 1):
            flow = self.predict_flow(images[i], images[i + 1])
            flows.append(flow)
        return flows


if __name__ == '__main__':
    print("=" * 70)
    print("统一光流预测器 - 使用示例")
    print("=" * 70)
    
    # 方法1: 使用Farneback（默认，最快）
    print("\n方法1: Farneback光流（快速，OpenCV内置）")
    predictor_fast = SimpleRAFTPredictor(method='farneback')
    
    # 方法2: 使用TV-L1（高精度）
    print("\n方法2: TV-L1光流（高精度，需要opencv-contrib-python）")
    predictor_accurate = SimpleRAFTPredictor(method='tvl1')
    
    # 方法3: 使用RAFT官方模型（最高精度）
    print("\n方法3: RAFT官方模型（最高精度，需要模型文件）")
    model_path = 'pretrained_models/raft-things.pth'
    if Path(model_path).exists():
        predictor_best = SimpleRAFTPredictor(
            method='raft',
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        print(f"  模型文件不存在: {model_path}")
        print("  请下载RAFT预训练权重:")
        print("  - 访问: https://github.com/princeton-vl/RAFT")
        print("  - 下载: raft-things.pth")
        print("  - 放置到: pretrained_models/")
    
    print("\n" + "=" * 70)
    print("选择建议：")
    print("  快速原型/演示        → method='farneback'")
    print("  生产环境/高精度      → method='tvl1'")
    print("  研究/极致精度        → method='raft'")
    print("=" * 70)
