"""
静态物体动态度分析器
用于区分相机运动和真实物体运动，计算静态物体的动态度
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from scipy import ndimage
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt


class CameraMotionEstimator:
    """相机运动估计器"""
    
    def __init__(self, 
                 feature_detector='ORB',
                 max_features=1000,
                 ransac_threshold=1.0,
                 ransac_max_trials=1000):
        self.max_features = max_features
        self.ransac_threshold = ransac_threshold
        self.ransac_max_trials = ransac_max_trials
        
        # 初始化特征检测器
        if feature_detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif feature_detector == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif feature_detector == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create()
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
            
        # 特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def detect_features(self, image):
        """检测图像特征点"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """匹配特征点"""
        if desc1 is None or desc2 is None:
            return []
            
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def estimate_homography(self, kp1, kp2, matches, min_matches=10):
        """估计单应性矩阵"""
        if len(matches) < min_matches:
            return None, None
            
        # 提取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC估计单应性矩阵
        homography, mask = cv2.findHomography(
            pts1, pts2, 
            cv2.RANSAC, 
            self.ransac_threshold,
            maxIters=self.ransac_max_trials
        )
        
        return homography, mask
    
    def decompose_homography(self, homography, camera_matrix):
        """分解单应性矩阵得到相机运动参数"""
        if homography is None:
            return None
            
        # 分解单应性矩阵
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homography, camera_matrix)
        
        # 选择最合理的解
        best_solution = None
        max_inliers = 0
        
        for i in range(num):
            # 检查解的合理性
            R = Rs[i]
            T = Ts[i]
            N = Ns[i]
            
            # 检查旋转矩阵的有效性
            if np.abs(np.linalg.det(R) - 1.0) < 0.1:
                inliers = np.sum(N[:, 2] > 0)  # 法向量z分量应为正
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_solution = {'R': R, 'T': T, 'N': N}
                    
        return best_solution
    
    def estimate_camera_motion(self, image1, image2, camera_matrix=None):
        """估计相机运动"""
        # 检测特征点
        kp1, desc1 = self.detect_features(image1)
        kp2, desc2 = self.detect_features(image2)
        
        # 匹配特征点
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 10:
            return None
            
        # 估计单应性矩阵
        homography, mask = self.estimate_homography(kp1, kp2, matches)
        
        if homography is None:
            return None
            
        # 如果有相机内参，分解单应性矩阵
        if camera_matrix is not None:
            motion = self.decompose_homography(homography, camera_matrix)
            if motion is not None:
                motion['homography'] = homography
                motion['matches'] = matches
                motion['mask'] = mask
                return motion
        
        # 否则只返回单应性矩阵
        return {
            'homography': homography,
            'matches': matches,
            'mask': mask
        }


class StaticObjectDetector:
    """静态物体检测器"""
    
    def __init__(self, 
                 flow_threshold=2.0,
                 consistency_threshold=0.8,
                 min_region_size=100):
        self.flow_threshold = flow_threshold
        self.consistency_threshold = consistency_threshold
        self.min_region_size = min_region_size
        
    def compensate_camera_motion(self, flow, homography):
        """补偿相机运动"""
        if homography is None:
            return flow
            
        h, w = flow.shape[:2]
        
        # 创建坐标网格
        y, x = np.mgrid[0:h, 0:w]
        coords = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
        
        # 应用单应性变换
        transformed_coords = (homography @ coords.T).T
        transformed_coords = transformed_coords[:, :2] / transformed_coords[:, 2:3]
        transformed_coords = transformed_coords.reshape(h, w, 2)
        
        # 计算相机运动引起的光流
        camera_flow = transformed_coords - np.stack([x, y], axis=-1)
        
        # 补偿相机运动
        compensated_flow = flow - camera_flow
        
        return compensated_flow
    
    def detect_static_regions(self, flow, homography=None):
        """检测静态区域"""
        # 补偿相机运动
        if homography is not None:
            compensated_flow = self.compensate_camera_motion(flow, homography)
        else:
            compensated_flow = flow.copy()
            
        # 计算光流幅度
        flow_magnitude = np.sqrt(compensated_flow[:, :, 0]**2 + compensated_flow[:, :, 1]**2)
        
        # 基于阈值检测静态区域
        static_mask = flow_magnitude < self.flow_threshold
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        static_mask = cv2.morphologyEx(static_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel)
        
        # 移除小区域
        static_mask = self.remove_small_regions(static_mask, self.min_region_size)
        
        return static_mask.astype(bool), compensated_flow
    
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
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 在高梯度区域（边缘）更严格地判断静态区域
        edge_mask = gradient_magnitude > np.percentile(gradient_magnitude, 75)
        
        # 在边缘区域使用更严格的阈值
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        strict_static_mask = flow_magnitude < (self.flow_threshold * 0.5)
        
        # 组合结果
        refined_mask = static_mask.copy()
        refined_mask[edge_mask] = strict_static_mask[edge_mask]
        
        return refined_mask


class StaticObjectDynamicsCalculator:
    """静态物体动态度计算器"""
    
    def __init__(self, 
                 temporal_window=5,
                 spatial_kernel_size=5,
                 dynamics_threshold=1.0):
        self.temporal_window = temporal_window
        self.spatial_kernel_size = spatial_kernel_size
        self.dynamics_threshold = dynamics_threshold
        self.camera_estimator = CameraMotionEstimator()
        self.static_detector = StaticObjectDetector()
        
    def calculate_frame_dynamics(self, 
                                flow: np.ndarray,
                                image1: np.ndarray,
                                image2: np.ndarray,
                                camera_matrix: Optional[np.ndarray] = None) -> Dict:
        """计算单帧的静态物体动态度"""
        
        # 估计相机运动
        camera_motion = self.camera_estimator.estimate_camera_motion(
            image1, image2, camera_matrix
        )
        
        homography = camera_motion['homography'] if camera_motion else None
        
        # 检测静态区域
        static_mask, compensated_flow = self.static_detector.detect_static_regions(
            flow, homography
        )
        
        # 细化静态区域
        refined_static_mask = self.static_detector.refine_static_regions(
            static_mask, image1, compensated_flow
        )
        
        # 计算静态区域的动态度
        static_dynamics = self.calculate_static_region_dynamics(
            compensated_flow, refined_static_mask
        )
        
        # 计算全局动态度指标
        global_dynamics = self.calculate_global_dynamics(
            compensated_flow, refined_static_mask
        )
        
        return {
            'static_mask': refined_static_mask,
            'compensated_flow': compensated_flow,
            'static_dynamics': static_dynamics,
            'global_dynamics': global_dynamics,
            'camera_motion': camera_motion,
            'original_flow': flow
        }
    
    def calculate_static_region_dynamics(self, flow, static_mask):
        """计算静态区域的动态度"""
        if not np.any(static_mask):
            return {
                'mean_magnitude': 0.0,
                'std_magnitude': 0.0,
                'max_magnitude': 0.0,
                'dynamics_score': 0.0
            }
        
        # 提取静态区域的光流
        static_flow = flow[static_mask]
        
        # 计算光流幅度
        flow_magnitude = np.sqrt(static_flow[:, 0]**2 + static_flow[:, 1]**2)
        
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
            'dynamics_score': float(dynamics_score)
        }
    
    def calculate_global_dynamics(self, flow, static_mask):
        """计算全局动态度指标"""
        h, w = flow.shape[:2]
        total_pixels = h * w
        static_pixels = np.sum(static_mask)
        dynamic_pixels = total_pixels - static_pixels
        
        # 静态区域比例
        static_ratio = static_pixels / total_pixels
        
        # 动态区域的平均光流幅度
        if dynamic_pixels > 0:
            dynamic_flow = flow[~static_mask]
            dynamic_magnitude = np.sqrt(dynamic_flow[:, 0]**2 + dynamic_flow[:, 1]**2)
            mean_dynamic_magnitude = np.mean(dynamic_magnitude)
        else:
            mean_dynamic_magnitude = 0.0
        
        # 全局一致性分数
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
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
        
        # 原始光流
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        im1 = axes[0, 1].imshow(flow_magnitude, cmap='jet')
        axes[0, 1].set_title('Original Flow Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 补偿后的光流
        compensated_flow = result['compensated_flow']
        compensated_magnitude = np.sqrt(compensated_flow[:, :, 0]**2 + compensated_flow[:, :, 1]**2)
        im2 = axes[0, 2].imshow(compensated_magnitude, cmap='jet')
        axes[0, 2].set_title('Compensated Flow Magnitude')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])
        
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
        static_flow = compensated_flow[result['static_mask']]
        if len(static_flow) > 0:
            static_magnitude = np.sqrt(static_flow[:, 0]**2 + static_flow[:, 1]**2)
            axes[1, 2].hist(static_magnitude, bins=50, alpha=0.7, label='Static Regions')
        
        dynamic_flow = compensated_flow[~result['static_mask']]
        if len(dynamic_flow) > 0:
            dynamic_magnitude = np.sqrt(dynamic_flow[:, 0]**2 + dynamic_flow[:, 1]**2)
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
            report += "- 静态物体动态度较低，相机运动补偿效果良好\n"
        elif static_dynamics['dynamics_score'] < 2.0:
            report += "- 静态物体动态度中等，可能存在轻微的运动或补偿误差\n"
        else:
            report += "- 静态物体动态度较高，可能存在真实物体运动或相机运动补偿不准确\n"
        
        if global_dynamics['static_ratio'] > 0.7:
            report += "- 场景中大部分区域为静态，适合进行静态物体动态度分析\n"
        else:
            report += "- 场景中动态区域较多，需要谨慎解释静态物体动态度结果\n"
        
        return report