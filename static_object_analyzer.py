# -*- coding: utf-8 -*-
"""
静态物体动态度分析器（重构版）
用于分析静态区域的运动特征，计算动态度分数

重要：相机运动补偿已在 video_processor.py 中使用 CameraCompensator 完成
      本模块接收的是已补偿后的残差光流，无需再次补偿
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from scipy import ndimage
import matplotlib.pyplot as plt


class StaticObjectDetector:
    """静态物体检测器（支持分辨率归一化）"""
    
    def __init__(self, 
                 flow_threshold=2.0,
                 flow_threshold_ratio=0.002,
                 use_normalized_flow=False,
                 consistency_threshold=0.8,
                 min_region_size=100):
        """
        Args:
            flow_threshold: 绝对像素阈值（向后兼容，默认2.0）
            flow_threshold_ratio: 归一化阈值，相对于图像对角线（默认0.002）
            use_normalized_flow: 是否使用归一化（默认False保持兼容）
            consistency_threshold: 一致性阈值
            min_region_size: 最小区域大小
        """
        self.flow_threshold = flow_threshold
        self.flow_threshold_ratio = flow_threshold_ratio
        self.use_normalized_flow = use_normalized_flow
        self.consistency_threshold = consistency_threshold
        self.min_region_size = min_region_size
    
    def detect_static_regions(self, flow, image_shape=None):
        """
        检测静态区域
        
        Args:
            flow: 光流数组 (H, W, 2)
            image_shape: 图像形状 (H, W, C)，用于归一化
        
        注意：输入的 flow 应该已经是经过相机补偿的残差光流
        """
        # 计算光流幅度
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # 确定阈值
        if self.use_normalized_flow and image_shape is not None:
            h, w = image_shape[:2]
            diagonal = np.sqrt(h**2 + w**2)
            # 归一化光流幅度
            flow_magnitude_normalized = flow_magnitude / diagonal
            threshold = self.flow_threshold_ratio
            static_mask = flow_magnitude_normalized < threshold
        else:
            # 使用绝对阈值（向后兼容）
            threshold = self.flow_threshold
            static_mask = flow_magnitude < threshold
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        static_mask = cv2.morphologyEx(static_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel)
        
        # 移除小区域
        static_mask = self.remove_small_regions(static_mask, self.min_region_size)
        
        return static_mask.astype(bool)
    
    def remove_small_regions(self, mask, min_size):
        """移除小区域"""
        labeled, num_labels = ndimage.label(mask)
        
        for i in range(1, num_labels + 1):
            region_size = np.sum(labeled == i)
            if region_size < min_size:
                mask[labeled == i] = 0
                
        return mask
    
    def refine_static_regions(self, static_mask, image, flow):
        """基于图像特征细化静态区域"""
        # 计算图像梯度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 在高梯度区域（边缘）更严格地判断静态区域
        edge_mask = gradient_magnitude > np.percentile(gradient_magnitude, 75)
        
        # 在边缘区域使用更严格的阈值
        flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        
        # 根据是否归一化使用不同的严格阈值
        if self.use_normalized_flow:
            h, w = image.shape[:2]
            diagonal = np.sqrt(h**2 + w**2)
            flow_magnitude_normalized = flow_magnitude / diagonal
            strict_static_mask = flow_magnitude_normalized < (self.flow_threshold_ratio * 0.5)
        else:
            strict_static_mask = flow_magnitude < (self.flow_threshold * 0.5)
        
        # 组合结果 - 确保维度匹配
        refined_mask = static_mask.copy()
        
        # 检查维度是否匹配，如果不匹配则调整
        if edge_mask.shape != refined_mask.shape:
            if edge_mask.shape[0] != refined_mask.shape[0] or edge_mask.shape[1] != refined_mask.shape[1]:
                target_height, target_width = flow.shape[:2]
                
                if edge_mask.shape != (target_height, target_width):
                    edge_mask = cv2.resize(edge_mask.astype(np.uint8), 
                                         (target_width, target_height), 
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                
                if refined_mask.shape != (target_height, target_width):
                    refined_mask = cv2.resize(refined_mask.astype(np.uint8), 
                                            (target_width, target_height), 
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        refined_mask[edge_mask] = strict_static_mask[edge_mask]

        return refined_mask


class StaticObjectDynamicsCalculator:
    """静态物体动态度计算器（支持分辨率归一化）"""
    
    def __init__(self, 
                 temporal_window=5,
                 spatial_kernel_size=5,
                 dynamics_threshold=1.0,
                 use_normalized_flow=False,
                 flow_threshold_ratio=0.002):
        """
        Args:
            temporal_window: 时序窗口
            spatial_kernel_size: 空间核大小
            dynamics_threshold: 动态度阈值
            use_normalized_flow: 是否使用分辨率归一化
            flow_threshold_ratio: 归一化阈值比例
        """
        self.temporal_window = temporal_window
        self.spatial_kernel_size = spatial_kernel_size
        self.dynamics_threshold = dynamics_threshold
        self.use_normalized_flow = use_normalized_flow
        self.static_detector = StaticObjectDetector(
            use_normalized_flow=use_normalized_flow,
            flow_threshold_ratio=flow_threshold_ratio
        )
        
    def calculate_frame_dynamics(self, 
                                flow: np.ndarray,
                                image1: np.ndarray,
                                image2: np.ndarray,
                                camera_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        计算单帧的静态物体动态度
        
        注意：输入的 flow 应该已经是经过相机补偿的残差光流
        camera_matrix 参数保留用于兼容性，但不再使用
        """
        
        # 检测静态区域（传递图像形状用于归一化）
        static_mask = self.static_detector.detect_static_regions(flow, image1.shape)
        
        # 细化静态区域
        refined_static_mask = self.static_detector.refine_static_regions(
            static_mask, image1, flow
        )
        
        # 计算归一化因子
        normalization_factor = 1.0
        if self.use_normalized_flow:
            h, w = image1.shape[:2]
            normalization_factor = np.sqrt(h**2 + w**2)
        
        # 计算静态区域的动态度（传递归一化因子）
        static_dynamics = self.calculate_static_region_dynamics(
            flow, refined_static_mask, normalization_factor
        )
        
        # 计算全局动态度指标（传递归一化因子）
        global_dynamics = self.calculate_global_dynamics(
            flow, refined_static_mask, normalization_factor
        )
        
        return {
            'static_mask': refined_static_mask,
            'compensated_flow': flow,  # 保留字段名以兼容，实际已是残差流
            'static_dynamics': static_dynamics,
            'global_dynamics': global_dynamics,
            'camera_motion': None,  # 相机补偿在外层完成，此处不再估计
            'original_flow': flow  # 传入的就是已补偿的流
        }
    
    def calculate_static_region_dynamics(self, flow, static_mask, normalization_factor=1.0):
        """
        计算静态区域的动态度
        
        Args:
            flow: 光流数组
            static_mask: 静态区域掩码
            normalization_factor: 归一化因子（图像对角线长度）
        """
        if not np.any(static_mask):
            return {
                'mean_magnitude': 0.0,
                'std_magnitude': 0.0,
                'max_magnitude': 0.0,
                'dynamics_score': 0.0,
                'normalization_factor': float(normalization_factor),
                'is_normalized': self.use_normalized_flow
            }
        
        # 提取静态区域的光流
        static_flow_x = flow[:, :, 0][static_mask]
        static_flow_y = flow[:, :, 1][static_mask]
        
        # 计算光流幅度
        flow_magnitude = np.sqrt(static_flow_x**2 + static_flow_y**2)
        
        # 应用归一化
        if self.use_normalized_flow and normalization_factor > 0:
            flow_magnitude = flow_magnitude / normalization_factor
        
        # 计算统计量
        mean_magnitude = np.mean(flow_magnitude)
        std_magnitude = np.std(flow_magnitude)
        max_magnitude = np.max(flow_magnitude)
        
        # 计算动态度分数（考虑幅度和一致性）
        dynamics_score = mean_magnitude + 0.5 * std_magnitude
        
        return {
            'mean_magnitude': float(mean_magnitude),
            'std_magnitude': float(std_magnitude),
            'max_magnitude': float(max_magnitude),
            'dynamics_score': float(dynamics_score),
            'normalization_factor': float(normalization_factor),
            'is_normalized': self.use_normalized_flow
        }
    
    def calculate_global_dynamics(self, flow, static_mask, normalization_factor=1.0):
        """
        计算全局动态度指标
        
        Args:
            flow: 光流数组
            static_mask: 静态区域掩码
            normalization_factor: 归一化因子
        """
        
        h, w = flow.shape[:2]
        total_pixels = h * w
        static_pixels = np.sum(static_mask)
        dynamic_pixels = total_pixels - static_pixels
        
        # 静态区域比例
        static_ratio = static_pixels / total_pixels
        
        # 动态区域的平均光流幅度
        if dynamic_pixels > 0:
            dynamic_flow_x = flow[:, :, 0][~static_mask]
            dynamic_flow_y = flow[:, :, 1][~static_mask]
            dynamic_magnitude = np.sqrt(dynamic_flow_x**2 + dynamic_flow_y**2)
            
            # 应用归一化
            if self.use_normalized_flow and normalization_factor > 0:
                dynamic_magnitude = dynamic_magnitude / normalization_factor
            
            mean_dynamic_magnitude = np.mean(dynamic_magnitude)
        else:
            mean_dynamic_magnitude = 0.0
        
        # 全局一致性分数
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # 应用归一化
        if self.use_normalized_flow and normalization_factor > 0:
            flow_magnitude = flow_magnitude / normalization_factor
        
        consistency_score = 1.0 - (np.std(flow_magnitude) / (np.mean(flow_magnitude) + 1e-6))
        
        return {
            'static_ratio': float(static_ratio),
            'dynamic_ratio': float(1.0 - static_ratio),
            'mean_dynamic_magnitude': float(mean_dynamic_magnitude),
            'consistency_score': float(max(0.0, consistency_score))
        }
    
    def calculate_temporal_dynamics(self, 
                                  flows: List[np.ndarray],
                                  images: List[np.ndarray],
                                  camera_matrix: Optional[np.ndarray] = None) -> Dict:
        """计算时序动态度"""
        
        if len(flows) != len(images) - 1:
            raise ValueError("光流数量应该比图像数量少1")
        
        frame_results = []
        
        # 计算每帧的动态度
        for i, flow in enumerate(flows):
            result = self.calculate_frame_dynamics(
                flow, images[i], images[i+1], camera_matrix
            )
            frame_results.append(result)
        
        # 计算时序统计量
        temporal_stats = self.calculate_temporal_statistics(frame_results)
        
        return {
            'frame_results': frame_results,
            'temporal_stats': temporal_stats
        }
    
    def calculate_temporal_statistics(self, frame_results):
        """计算时序统计量"""
        dynamics_scores = [r['static_dynamics']['dynamics_score'] for r in frame_results]
        static_ratios = [r['global_dynamics']['static_ratio'] for r in frame_results]
        consistency_scores = [r['global_dynamics']['consistency_score'] for r in frame_results]
        
        return {
            'mean_dynamics_score': float(np.mean(dynamics_scores)),
            'std_dynamics_score': float(np.std(dynamics_scores)),
            'max_dynamics_score': float(np.max(dynamics_scores)),
            'min_dynamics_score': float(np.min(dynamics_scores)),
            'mean_static_ratio': float(np.mean(static_ratios)),
            'std_static_ratio': float(np.std(static_ratios)),
            'mean_consistency_score': float(np.mean(consistency_scores)),
            'temporal_stability': float(1.0 / (1.0 + np.std(dynamics_scores)))
        }
    
    def visualize_results(self, 
                         image: np.ndarray,
                         flow: np.ndarray,
                         result: Dict,
                         save_path: Optional[str] = None):
        """可视化分析结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 残差光流
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        im1 = axes[0, 1].imshow(flow_magnitude, cmap='jet')
        axes[0, 1].set_title('Residual Flow Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 光流向量
        step = 16
        y, x = np.mgrid[step//2:flow.shape[0]:step, step//2:flow.shape[1]:step]
        fx = flow[::step, ::step, 0]
        fy = flow[::step, ::step, 1]
        axes[0, 2].imshow(image)
        axes[0, 2].quiver(x, y, fx, fy, color='red', alpha=0.7)
        axes[0, 2].set_title('Flow Vectors')
        axes[0, 2].axis('off')
        
        # 静态区域掩码
        axes[1, 0].imshow(result['static_mask'], cmap='gray')
        axes[1, 0].set_title('Static Regions Mask')
        axes[1, 0].axis('off')
        
        # 静态区域叠加
        overlay = image.copy()
        if len(overlay.shape) == 3:
            overlay[result['static_mask']] = [0, 255, 0]  # 绿色标记静态区域
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Static Regions Overlay')
        axes[1, 1].axis('off')
        
        # 动态度分布
        static_mask = result['static_mask']
        compensated_flow = result['compensated_flow']
        if np.any(static_mask):
            static_flow_x = compensated_flow[:, :, 0][static_mask]
            static_flow_y = compensated_flow[:, :, 1][static_mask]
            static_magnitude = np.sqrt(static_flow_x**2 + static_flow_y**2)
            axes[1, 2].hist(static_magnitude, bins=50, alpha=0.7, label='Static Regions')
        
        if np.any(~static_mask):
            dynamic_flow_x = compensated_flow[:, :, 0][~static_mask]
            dynamic_flow_y = compensated_flow[:, :, 1][~static_mask]
            dynamic_magnitude = np.sqrt(dynamic_flow_x**2 + dynamic_flow_y**2)
            axes[1, 2].hist(dynamic_magnitude, bins=50, alpha=0.7, label='Dynamic Regions')
        
        axes[1, 2].set_title('Flow Magnitude Distribution')
        axes[1, 2].set_xlabel('Flow Magnitude')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, result: Dict) -> str:
        """生成分析报告"""
        static_dynamics = result['static_dynamics']
        global_dynamics = result['global_dynamics']
        
        report = f"""
静态物体动态度分析报告
====================

静态区域动态度:
- 平均光流幅度: {static_dynamics['mean_magnitude']:.3f}
- 光流幅度标准差: {static_dynamics['std_magnitude']:.3f}
- 最大光流幅度: {static_dynamics['max_magnitude']:.3f}
- 动态度分数: {static_dynamics['dynamics_score']:.3f}

全局动态度指标:
- 静态区域比例: {global_dynamics['static_ratio']:.3f}
- 动态区域比例: {global_dynamics['dynamic_ratio']:.3f}
- 动态区域平均光流幅度: {global_dynamics['mean_dynamic_magnitude']:.3f}
- 一致性分数: {global_dynamics['consistency_score']:.3f}

分析结论:
"""
        
        # 添加分析结论
        if static_dynamics['dynamics_score'] < 1.0:
            report += "- 静态物体动态度较低，质量良好\n"
        elif static_dynamics['dynamics_score'] < 2.0:
            report += "- 静态物体动态度中等，可能存在轻微的运动\n"
        else:
            report += "- 静态物体动态度较高，可能存在真实物体运动或抖动\n"
        
        if global_dynamics['static_ratio'] > 0.7:
            report += "- 场景中大部分区域为静态，适合进行静态物体动态度分析\n"
        else:
            report += "- 场景中动态区域较多，需要谨慎解释静态物体动态度结果\n"
        
        return report

