"""
使用RAFT官方预训练权重的光流估计模块

支持直接加载官方预训练模型：
- raft-things.pth (在Things3D数据集上训练)
- raft-sintel.pth (在Sintel数据集上训练)

官方模型下载：
https://github.com/princeton-vl/RAFT
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path


class RAFTPredictor:
    """
    RAFT光流预测器 - 使用官方预训练权重
    
    使用方法:
        # 方法1: 使用官方预训练权重
        predictor = RAFTPredictor(model_path='raft-things.pth')
        
        # 方法2: 使用OpenCV作为后备方案（无需下载模型）
        predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)
    """
    
    def __init__(self, model_path=None, device='cuda', use_opencv_fallback=True):
        """
        初始化RAFT预测器
        
        Args:
            model_path: RAFT预训练权重路径 (如 'raft-things.pth')
            device: 运行设备 ('cuda' 或 'cpu')
            use_opencv_fallback: 如果加载失败，是否使用OpenCV作为后备方案
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.use_opencv = False
        
        # 尝试加载RAFT官方模型
        if model_path and Path(model_path).exists():
            try:
                self._load_official_raft(model_path)
                print(f"? 成功加载RAFT官方模型: {model_path}")
            except Exception as e:
                print(f"? 加载RAFT模型失败: {e}")
                if use_opencv_fallback:
                    print("  使用OpenCV光流作为后备方案")
                    self.use_opencv = True
        else:            
            if use_opencv_fallback:
                print("  使用OpenCV光流作为后备方案")
                self.use_opencv = True
            else:
                raise ValueError("需要提供有效的模型路径或启用OpenCV后备方案")
    
    def _load_official_raft(self, model_path):
        """加载RAFT官方预训练模型"""
        try:
            # 添加RAFT官方代码路径（需要添加core目录，因为RAFT使用相对导入）
            raft_core_path = Path(__file__).parent / 'third_party' / 'RAFT' / 'core'
            if not raft_core_path.exists():
                # 如果在当前目录找不到，尝试上级目录
                raft_core_path = Path(__file__).parent.parent / 'third_party' / 'RAFT' / 'core'
            
            if raft_core_path.exists():
                sys.path.insert(0, str(raft_core_path))
            
            try:
                from raft import RAFT  # type: ignore
                import argparse
                
                # 创建args对象，包含RAFT需要的所有参数
                args = argparse.Namespace()
                args.small = False
                args.mixed_precision = False
                args.alternate_corr = False
                args.dropout = 0
                args.corr_levels = 4
                args.corr_radius = 4
                
                self.model = RAFT(args)
            except ImportError as e:
                raise ImportError(
                    f"无法导入RAFT: {e}\n"
                    "请确保 third_party/RAFT 目录存在且包含官方代码\n"
                    "或使用OpenCV: RAFTPredictor(model_path=None, use_opencv_fallback=True)"
                )
            
            # 加载预训练权重
            state_dict = torch.load(model_path, map_location=self.device)
            
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
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"无法加载RAFT模型: {e}")
    
    def predict_flow(self, image1, image2):
        """
        预测两帧图像之间的光流
        
        Args:
            image1: 第一帧图像 (H, W, 3) numpy数组，RGB格式，0-255
            image2: 第二帧图像 (H, W, 3) numpy数组，RGB格式，0-255
            
        Returns:
            flow: 光流 (2, H, W) numpy数组
        """
        if self.use_opencv:
            return self._predict_flow_opencv(image1, image2)
        else:
            return self._predict_flow_raft(image1, image2)
    
    def _predict_flow_raft(self, image1, image2):
        """使用RAFT模型预测光流"""
        with torch.no_grad():
            # 预处理
            img1 = self._preprocess_image(image1)
            img2 = self._preprocess_image(image2)
            
            # 预测
            _, flow_up = self.model(img1, img2, iters=20, test_mode=True)
            
            # 转换为numpy
            flow = flow_up[0].cpu().numpy()
            return flow
    
    def _predict_flow_opencv(self, image1, image2):
        """使用OpenCV预测光流（后备方案）"""
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
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )
        
        # 转换为 (2, H, W) 格式
        return flow.transpose(2, 0, 1)
    
    def _preprocess_image(self, img):
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


def download_raft_weights(model_name='raft-things'):
    """
    下载RAFT官方预训练权重
    
    Args:
        model_name: 'raft-things' 或 'raft-sintel'
    
    Returns:
        model_path: 下载的模型路径
    """
    import urllib.request
    from pathlib import Path
    
    # 官方模型URL
    model_urls = {
        'raft-things': 'https://drive.google.com/uc?export=download&id=1M5QHhdMI6oWF3Bv8Y1oW8vvVGMxM8Gru',
        'raft-sintel': 'https://drive.google.com/uc?export=download&id=1Sxb0RDsJ7JBz9NJ6wj4QzHXlZ7PdYJKd',
    }
    
    if model_name not in model_urls:
        raise ValueError(f"未知模型: {model_name}，可选: {list(model_urls.keys())}")
    
    # 下载路径
    save_dir = Path(__file__).parent / 'pretrained_models'
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / f"{model_name}.pth"
    
    if model_path.exists():
        print(f"? 模型已存在: {model_path}")
        return str(model_path)
    
    print(f"正在下载 {model_name}...")
    print(f"URL: {model_urls[model_name]}")
    print(f"保存到: {model_path}")
    
    try:
        urllib.request.urlretrieve(model_urls[model_name], model_path)
        print(f"? 下载完成!")
        return str(model_path)
    except Exception as e:
        print(f"? 下载失败: {e}")
        print("\n请手动下载:")
        print(f"1. 访问 https://github.com/princeton-vl/RAFT")
        print(f"2. 下载 {model_name}.pth")
        print(f"3. 放置到: {save_dir}")
        return None


if __name__ == '__main__':
    # 使用示例
    print("=" * 60)
    print("RAFT光流预测器 - 使用示例")
    print("=" * 60)
    
    # 方式1: 使用OpenCV (无需下载模型)
    # print("\n方式1: 使用OpenCV光流 (推荐用于快速测试)")
    # predictor = RAFTPredictor(model_path=None, use_opencv_fallback=True)
    
    # 方式2: 使用RAFT官方模型
    print("\n方式2: 使用RAFT官方预训练模型")
    model_path = 'pretrained_models/raft-things.pth' 
    if Path(model_path).exists():
        predictor = RAFTPredictor(model_path=model_path)
    else:
        print(f"模型文件不存在: {model_path}")
        print("请下载RAFT预训练权重:")
        print("- 访问: https://github.com/princeton-vl/RAFT")
        print("- 下载: raft-things.pth 或 raft-sintel.pth")
        print("- 放置到当前目录")
    
    print("\n" + "=" * 60)

