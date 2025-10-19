# -*- coding: utf-8 -*-
"""
视频处理器 - 处理相机转动拍摄静态建筑的视频
计算静态物体的动态度
"""

import cv2
import numpy as np
import os
import json
import glob
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from simple_raft import SimpleRAFTPredictor as RAFTPredictor
from static_object_analyzer import StaticObjectDynamicsCalculator
from dynamic_motion_compensation.camera_compensation import CameraCompensator


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, 
                 raft_model_path: Optional[str] = None,
                 device: str = 'cuda',
                 max_frames: Optional[int] = None,
                 frame_skip: int = 1,
                 enable_visualization: bool = True,
                 enable_camera_compensation: bool = True,
                 camera_compensation_params: Optional[Dict] = None):
        """
        初始化视频处理器
        
        Args:
            raft_model_path: RAFT模型路径
            device: 计算设备
            max_frames: 最大处理帧数
            frame_skip: 帧跳跃间隔
            enable_visualization: 是否生成可视化结果
            enable_camera_compensation: 是否启用相机补偿（默认True）
            camera_compensation_params: 相机补偿参数字典
        """
        self.raft_predictor = RAFTPredictor(raft_model_path, device, method='raft')
        self.dynamics_calculator = StaticObjectDynamicsCalculator()
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        self.enable_visualization = enable_visualization
        self.enable_camera_compensation = enable_camera_compensation
        
        # 初始化相机补偿器
        if camera_compensation_params is None:
            camera_compensation_params = {}
        self.camera_compensator = CameraCompensator(**camera_compensation_params) if enable_camera_compensation else None
        
    def load_video(self, video_path: str) -> List[np.ndarray]:
        """加载视频帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        print(f"正在加载视频: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_skip == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if self.max_frames and len(frames) >= self.max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"加载完成，共 {len(frames)} 帧")
        return frames
    
    def extract_frames_from_images(self, image_dir: str) -> List[np.ndarray]:
        """从图像目录加载帧"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        image_files.sort()
        frames = []
        
        print(f"正在从目录加载图像: {image_dir}")
        
        for i, img_path in enumerate(image_files):
            if self.max_frames and i >= self.max_frames:
                break
                
            if i % self.frame_skip == 0:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img_rgb)
        
        print(f"加载完成，共 {len(frames)} 帧")
        return frames
    
    def estimate_camera_matrix(self, frame_shape: Tuple[int, int], fov: float = 60.0) -> np.ndarray:
        """估计相机内参矩阵"""
        h, w = frame_shape[:2]
        
        # 基于视场角估计焦距
        focal_length = w / (2 * np.tan(np.radians(fov / 2)))
        
        # 构建相机内参矩阵
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return camera_matrix
    
    def process_video(self, 
                     frames: List[np.ndarray],
                     camera_matrix: Optional[np.ndarray] = None,
                     output_dir: str = 'output') -> Dict:
        """处理视频序列"""
        
        if len(frames) < 2:
            raise ValueError("至少需要2帧图像")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果没有提供相机矩阵，估计一个
        if camera_matrix is None:
            camera_matrix = self.estimate_camera_matrix(frames[0].shape)
            print("使用估计的相机内参矩阵")
        
        print("开始计算光流...")
        
        # 计算所有帧间的光流
        flows = []  # 用于分析的光流（可能是补偿后的）
        original_flows = []  # 原始光流
        camera_compensation_results = []  # 存储相机补偿结果
        
        for i in tqdm(range(len(frames) - 1), desc="计算光流"):
            flow = self.raft_predictor.predict_flow(frames[i], frames[i + 1])
            # 确保flow格式正确 (2, H, W) -> (H, W, 2)
            if flow.shape[0] == 2:
                flow = flow.transpose(1, 2, 0)  # (2, H, W) -> (H, W, 2)
            elif len(flow.shape) == 2:
                # 如果是2D，可能需要添加通道维度
                print(f"警告: 帧{i}的光流形状异常: {flow.shape}")
                continue
            
            original_flows.append(flow.copy())  # 保存原始光流
            
            # 应用相机补偿
            if self.enable_camera_compensation and self.camera_compensator is not None:
                comp_result = self.camera_compensator.compensate(flow, frames[i], frames[i + 1])
                camera_compensation_results.append(comp_result)
                # 使用残差光流（补偿后的光流）用于分析
                flows.append(comp_result['residual_flow'])
            else:
                flows.append(flow)
                camera_compensation_results.append(None)
        
        print("开始分析静态物体动态度...")
        
        # 计算时序动态度
        temporal_result = self.dynamics_calculator.calculate_temporal_dynamics(
            flows, frames, camera_matrix
        )
        
        # 添加相机补偿信息到结果中
        temporal_result['camera_compensation_enabled'] = self.enable_camera_compensation
        temporal_result['camera_compensation_results'] = camera_compensation_results
        temporal_result['original_flows'] = original_flows  # 保存原始光流供可视化使用
        
        # 保存结果
        self.save_results(temporal_result, frames, flows, output_dir)
        
        return temporal_result
    
    def save_results(self, 
                    result: Dict,
                    frames: List[np.ndarray],
                    flows: List[np.ndarray],
                    output_dir: str):
        """保存分析结果"""
        
        # 保存数值结果
        numeric_result = {
            'temporal_stats': result['temporal_stats'],
            'frame_count': len(frames),
            'flow_count': len(flows),
            'camera_compensation_enabled': result.get('camera_compensation_enabled', False)
        }
        
        # 添加相机补偿统计信息
        if result.get('camera_compensation_enabled', False):
            camera_comp_results = result.get('camera_compensation_results', [])
            comp_stats = self._calculate_camera_compensation_stats(camera_comp_results)
            numeric_result['camera_compensation_stats'] = comp_stats
        
        # 添加每帧的数值结果
        numeric_result['frame_results'] = []
        for i, frame_result in enumerate(result['frame_results']):
            numeric_frame_result = {
                'frame_index': i,
                'static_dynamics': frame_result['static_dynamics'],
                'global_dynamics': frame_result['global_dynamics']
            }
            
            # 添加相机补偿信息
            if result.get('camera_compensation_enabled', False) and i < len(result.get('camera_compensation_results', [])):
                comp_result = result['camera_compensation_results'][i]
                if comp_result is not None:
                    numeric_frame_result['camera_compensation'] = {
                        'inliers': comp_result['inliers'],
                        'total_matches': comp_result['total_matches'],
                        'homography_found': comp_result['homography'] is not None
                    }
            
            numeric_result['frame_results'].append(numeric_frame_result)
        
        # 保存JSON结果
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
            json.dump(numeric_result, f, indent=2, ensure_ascii=False)
        
        # 生成并保存报告
        report = self.generate_video_report(result)
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存可视化结果（如果启用）
        if self.enable_visualization:
            self.save_visualizations(result, frames, flows, output_dir)
        
        print(f"结果已保存到: {output_dir}")
    
    def _calculate_camera_compensation_stats(self, camera_comp_results: List[Optional[Dict]]) -> Dict:
        """计算相机补偿统计信息"""
        if not camera_comp_results:
            return {}
        
        valid_results = [r for r in camera_comp_results if r is not None and r['homography'] is not None]
        
        if not valid_results:
            return {
                'success_rate': 0.0,
                'mean_inliers': 0.0,
                'mean_match_ratio': 0.0
            }
        
        total_frames = len(camera_comp_results)
        successful_frames = len(valid_results)
        
        inliers = [r['inliers'] for r in valid_results]
        match_ratios = [r['inliers'] / max(r['total_matches'], 1) for r in valid_results]
        
        return {
            'success_rate': successful_frames / total_frames,
            'successful_frames': successful_frames,
            'total_frames': total_frames,
            'mean_inliers': float(np.mean(inliers)) if inliers else 0.0,
            'std_inliers': float(np.std(inliers)) if inliers else 0.0,
            'mean_match_ratio': float(np.mean(match_ratios)) if match_ratios else 0.0,
            'std_match_ratio': float(np.std(match_ratios)) if match_ratios else 0.0
        }
    
    def save_visualizations(self, 
                           result: Dict,
                           frames: List[np.ndarray],
                           flows: List[np.ndarray],
                           output_dir: str):
        """保存可视化结果"""
        
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 保存关键帧的详细分析
        key_frames = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-2]
        key_frames = [i for i in key_frames if i < len(result['frame_results'])]
        
        for i in key_frames:
            frame_result = result['frame_results'][i]
            fig = self.dynamics_calculator.visualize_results(
                frames[i], flows[i], frame_result
            )
            fig.savefig(os.path.join(vis_dir, f'frame_{i:04d}_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # 绘制时序动态度曲线
        self.plot_temporal_dynamics(result, os.path.join(vis_dir, 'temporal_dynamics.png'))
        
        # 绘制静态区域比例变化
        self.plot_static_ratio_changes(result, os.path.join(vis_dir, 'static_ratio_changes.png'))
        
        # 如果启用了相机补偿，绘制补偿效果对比
        if result.get('camera_compensation_enabled', False):
            self.plot_camera_compensation_comparison(result, frames, os.path.join(vis_dir, 'camera_compensation_comparison.png'))
    
    def plot_temporal_dynamics(self, result: Dict, save_path: str):
        """绘制时序动态度曲线"""
        frame_results = result['frame_results']
        
        dynamics_scores = [r['static_dynamics']['dynamics_score'] for r in frame_results]
        static_ratios = [r['global_dynamics']['static_ratio'] for r in frame_results]
        consistency_scores = [r['global_dynamics']['consistency_score'] for r in frame_results]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 动态度分数
        axes[0].plot(dynamics_scores, 'b-', linewidth=2, label='Dynamics Score')
        axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Low Threshold')
        axes[0].axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='High Threshold')
        axes[0].set_ylabel('Dynamics Score')
        axes[0].set_title('Static Object Dynamics Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 静态区域比例
        axes[1].plot(static_ratios, 'g-', linewidth=2, label='Static Ratio')
        axes[1].axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Good Threshold')
        axes[1].set_ylabel('Static Ratio')
        axes[1].set_title('Static Region Ratio Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 一致性分数
        axes[2].plot(consistency_scores, 'm-', linewidth=2, label='Consistency Score')
        axes[2].set_ylabel('Consistency Score')
        axes[2].set_xlabel('Frame Index')
        axes[2].set_title('Flow Consistency Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_camera_compensation_comparison(self, result: Dict, frames: List[np.ndarray], save_path: str):
        """绘制相机补偿效果对比"""
        original_flows = result.get('original_flows', [])
        camera_comp_results = result.get('camera_compensation_results', [])
        
        if not original_flows or not camera_comp_results:
            return
        
        # 选择几个关键帧进行对比
        key_frames = [0, len(original_flows)//4, len(original_flows)//2, 3*len(original_flows)//4, len(original_flows)-1]
        key_frames = [i for i in key_frames if i < len(original_flows) and camera_comp_results[i] is not None]
        
        if not key_frames:
            return
        
        # 创建对比图
        n_frames = min(3, len(key_frames))  # 最多显示3帧
        fig, axes = plt.subplots(n_frames, 4, figsize=(20, 5*n_frames))
        
        if n_frames == 1:
            axes = axes.reshape(1, -1)
        
        for idx, frame_idx in enumerate(key_frames[:n_frames]):
            original_flow = original_flows[frame_idx]
            comp_result = camera_comp_results[frame_idx]
            
            # 计算幅度
            original_mag = np.sqrt(original_flow[:, :, 0]**2 + original_flow[:, :, 1]**2)
            camera_mag = np.sqrt(comp_result['camera_flow'][:, :, 0]**2 + comp_result['camera_flow'][:, :, 1]**2)
            residual_mag = np.sqrt(comp_result['residual_flow'][:, :, 0]**2 + comp_result['residual_flow'][:, :, 1]**2)
            
            # 原始帧
            axes[idx, 0].imshow(frames[frame_idx])
            axes[idx, 0].set_title(f'Frame {frame_idx}')
            axes[idx, 0].axis('off')
            
            # 原始光流
            im1 = axes[idx, 1].imshow(original_mag, cmap='jet', vmin=0, vmax=np.percentile(original_mag, 95))
            axes[idx, 1].set_title(f'Original Flow\n(mean={original_mag.mean():.2f})')
            axes[idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)
            
            # 相机光流
            im2 = axes[idx, 2].imshow(camera_mag, cmap='jet', vmin=0, vmax=np.percentile(original_mag, 95))
            axes[idx, 2].set_title(f'Camera Flow\n(inliers={comp_result["inliers"]})')
            axes[idx, 2].axis('off')
            plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)
            
            # 残差光流
            im3 = axes[idx, 3].imshow(residual_mag, cmap='jet', vmin=0, vmax=np.percentile(residual_mag, 95))
            axes[idx, 3].set_title(f'Residual Flow\n(mean={residual_mag.mean():.2f})')
            axes[idx, 3].axis('off')
            plt.colorbar(im3, ax=axes[idx, 3], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_static_ratio_changes(self, result: Dict, save_path: str):
        """绘制静态区域比例变化"""
        frame_results = result['frame_results']
        
        static_ratios = [r['global_dynamics']['static_ratio'] for r in frame_results]
        dynamic_ratios = [r['global_dynamics']['dynamic_ratio'] for r in frame_results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(static_ratios))
        ax.fill_between(x, 0, static_ratios, alpha=0.7, color='green', label='Static Regions')
        ax.fill_between(x, static_ratios, 1.0, alpha=0.7, color='red', label='Dynamic Regions')
        
        ax.set_ylabel('Region Ratio')
        ax.set_xlabel('Frame Index')
        ax.set_title('Static vs Dynamic Region Ratio Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_video_report(self, result: Dict) -> str:
        """生成视频分析报告"""
        temporal_stats = result['temporal_stats']
        frame_count = len(result['frame_results'])
        camera_comp_enabled = result.get('camera_compensation_enabled', False)
        
        report = f"""
相机转动拍摄静态建筑视频 - 静态物体动态度分析报告
================================================

视频基本信息:
- 总帧数: {frame_count}
- 分析帧数: {len(result['frame_results'])}
- 相机补偿: {'启用' if camera_comp_enabled else '禁用'}

"""
        
        # 添加相机补偿统计
        if camera_comp_enabled and 'camera_compensation_results' in result:
            comp_stats = self._calculate_camera_compensation_stats(result['camera_compensation_results'])
            if comp_stats:
                report += f"""相机运动补偿统计:
- 成功率: {comp_stats['success_rate']:.1%} ({comp_stats['successful_frames']}/{comp_stats['total_frames']})
- 平均内点数: {comp_stats['mean_inliers']:.1f} ± {comp_stats['std_inliers']:.1f}
- 平均匹配率: {comp_stats['mean_match_ratio']:.1%} ± {comp_stats['std_match_ratio']:.1%}

"""
        
        report += f"""时序动态度统计:
- 平均动态度分数: {temporal_stats['mean_dynamics_score']:.3f}
- 动态度分数标准差: {temporal_stats['std_dynamics_score']:.3f}
- 最大动态度分数: {temporal_stats['max_dynamics_score']:.3f}
- 最小动态度分数: {temporal_stats['min_dynamics_score']:.3f}

静态区域分析:
- 平均静态区域比例: {temporal_stats['mean_static_ratio']:.3f}
- 静态区域比例标准差: {temporal_stats['std_static_ratio']:.3f}

一致性分析:
- 平均一致性分数: {temporal_stats['mean_consistency_score']:.3f}
- 时序稳定性: {temporal_stats['temporal_stability']:.3f}

综合评估:
"""
        
        # 添加综合评估
        mean_dynamics = temporal_stats['mean_dynamics_score']
        mean_static_ratio = temporal_stats['mean_static_ratio']
        temporal_stability = temporal_stats['temporal_stability']
        
        if mean_dynamics < 1.0:
            report += "✓ 静态物体动态度低，相机运动补偿效果良好\n"
        elif mean_dynamics < 2.0:
            report += "⚠ 静态物体动态度中等，存在轻微残余运动\n"
        else:
            report += "✗ 静态物体动态度高，可能存在补偿误差或真实物体运动\n"
        
        if mean_static_ratio > 0.7:
            report += "✓ 场景主要由静态物体组成，适合进行静态物体动态度分析\n"
        elif mean_static_ratio > 0.5:
            report += "⚠ 场景中静态和动态区域比例适中\n"
        else:
            report += "✗ 场景中动态区域较多，静态物体动态度分析可能不够准确\n"
        
        if temporal_stability > 0.8:
            report += "✓ 时序稳定性高，动态度计算结果可靠\n"
        elif temporal_stability > 0.6:
            report += "⚠ 时序稳定性中等\n"
        else:
            report += "✗ 时序稳定性低，结果可能存在较大波动\n"
        
        # 添加建议
        report += "\n建议:\n"
        
        if mean_dynamics > 1.5:
            report += "- 考虑调整相机运动估计参数或使用更精确的相机标定\n"
            report += "- 检查是否存在真实的物体运动（如风吹、振动等）\n"
        
        if mean_static_ratio < 0.6:
            report += "- 当前场景动态内容较多，建议选择更静态的场景进行测试\n"
        
        if temporal_stability < 0.7:
            report += "- 动态度计算结果波动较大，建议增加时序平滑或调整参数\n"
        
        return report


def process_single_video(processor, video_path, output_dir, camera_fov):
    """处理单个视频"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    
    print(f"\n{'='*70}")
    print(f"处理视频: {video_name}")
    print(f"{'='*70}")
    
    try:
        # 加载视频
        frames = processor.load_video(video_path)
        
        # 估计相机内参
        camera_matrix = processor.estimate_camera_matrix(frames[0].shape, camera_fov)
        
        # 处理视频
        result = processor.process_video(frames, camera_matrix, video_output_dir)
        
        # 返回简要结果
        temporal_stats = result['temporal_stats']
        return {
            'video_name': video_name,
            'status': 'success',
            'frame_count': len(frames),
            'mean_dynamics_score': temporal_stats['mean_dynamics_score'],
            'mean_static_ratio': temporal_stats['mean_static_ratio'],
            'temporal_stability': temporal_stats['temporal_stability'],
            'output_dir': video_output_dir
        }
    except Exception as e:
        print(f"错误: 处理视频失败 - {e}")
        return {
            'video_name': video_name,
            'status': 'failed',
            'error': str(e)
        }


def batch_process_videos(processor, input_dir, output_dir, camera_fov):
    """批量处理目录下的所有视频"""
    # 查找所有视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    # 去重（Windows系统下大小写不敏感会导致重复）
    video_files = list(set(video_files))
    
    if not video_files:
        print(f"错误: 在 {input_dir} 中未找到视频文件")
        return []
    
    video_files.sort()
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"\n进度: {i}/{len(video_files)}")
        result = process_single_video(processor, video_path, output_dir, camera_fov)
        results.append(result)
    
    # 保存批量处理总结
    save_batch_summary(results, output_dir)
    
    return results


def save_batch_summary(results, output_dir):
    """保存批量处理总结"""
    summary_path = os.path.join(output_dir, 'batch_summary.txt')
    
    # 统计成功和失败数量
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("批量视频处理总结\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"总视频数: {len(results)}\n")
        f.write(f"成功处理: {success_count}\n")
        f.write(f"处理失败: {failed_count}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("详细结果\n")
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            f.write(f"视频: {result['video_name']}\n")
            if result['status'] == 'success':
                f.write(f"  状态: ✓ 成功\n")
                f.write(f"  帧数: {result['frame_count']}\n")
                f.write(f"  平均动态度分数: {result['mean_dynamics_score']:.3f}\n")
                f.write(f"  平均静态区域比例: {result['mean_static_ratio']:.3f}\n")
                f.write(f"  时序稳定性: {result['temporal_stability']:.3f}\n")
                f.write(f"  输出目录: {result['output_dir']}\n")
            else:
                f.write(f"  状态: ✗ 失败\n")
                f.write(f"  错误: {result['error']}\n")
            f.write("\n")
    
    # 同时保存JSON格式
    summary_json_path = os.path.join(output_dir, 'batch_summary.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n批量处理总结已保存到: {summary_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='静态物体动态度分析')
    parser.add_argument('--input', '-i', required=True, 
                       help='输入视频文件、图像目录或视频目录路径（批量模式）')
    parser.add_argument('--output', '-o', default='output',
                       help='输出目录路径')
    parser.add_argument('--raft_model', '-m', default="pretrained_models/raft-things.pth",
                       help='RAFT模型路径')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='最大处理帧数')
    parser.add_argument('--frame_skip', type=int, default=1,
                       help='帧跳跃间隔')
    parser.add_argument('--device', default='cuda',
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='相机视场角 (度)')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式：处理输入目录下的所有视频')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                       help='禁用可视化生成（加快处理速度）')
    parser.add_argument('--no-camera-compensation', dest='camera_compensation', action='store_false',
                       help='禁用相机补偿（默认启用）')
    parser.add_argument('--camera-ransac-thresh', type=float, default=1.0,
                       help='相机补偿RANSAC阈值（像素）')
    parser.add_argument('--camera-max-features', type=int, default=2000,
                       help='相机补偿最大特征点数')
    parser.set_defaults(visualize=True, camera_compensation=True)
    
    args = parser.parse_args()
    
    # 准备相机补偿参数
    camera_compensation_params = {
        'ransac_thresh': args.camera_ransac_thresh,
        'max_features': args.camera_max_features
    }
    
    # 创建视频处理器
    processor = VideoProcessor(
        raft_model_path=args.raft_model,
        device=args.device,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        enable_visualization=args.visualize,
        enable_camera_compensation=args.camera_compensation,
        camera_compensation_params=camera_compensation_params
    )
    
    # 批量处理模式
    if args.batch:
        if not os.path.isdir(args.input):
            raise ValueError(f"批量模式需要提供目录路径: {args.input}")
        
        results = batch_process_videos(processor, args.input, args.output, args.fov)
        
        # 打印总结
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"\n{'='*70}")
        print(f"批量处理完成!")
        print(f"总计: {len(results)} 个视频")
        print(f"成功: {success_count} 个")
        print(f"失败: {len(results) - success_count} 个")
        print(f"结果保存到: {args.output}")
        print(f"{'='*70}")
        
    # 单个视频/图像目录处理模式
    else:
        # 加载视频或图像
        if os.path.isfile(args.input):
            # 单个视频文件
            frames = processor.load_video(args.input)
        elif os.path.isdir(args.input):
            # 图像序列目录
            frames = processor.extract_frames_from_images(args.input)
        else:
            raise ValueError(f"输入路径无效: {args.input}")
        
        # 估计相机内参
        camera_matrix = processor.estimate_camera_matrix(frames[0].shape, args.fov)
        
        # 处理视频
        result = processor.process_video(frames, camera_matrix, args.output)
        
        # 打印简要结果
        temporal_stats = result['temporal_stats']
        print(f"\n分析完成!")
        print(f"平均静态物体动态度分数: {temporal_stats['mean_dynamics_score']:.3f}")
        print(f"平均静态区域比例: {temporal_stats['mean_static_ratio']:.3f}")
        print(f"时序稳定性: {temporal_stats['temporal_stability']:.3f}")
        print(f"详细结果已保存到: {args.output}")


if __name__ == '__main__':
    main()