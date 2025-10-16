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

try:
    from raft_model import RAFTPredictor
except ImportError:
    from simple_raft import SimpleRAFTPredictor as RAFTPredictor
from static_object_analyzer import StaticObjectDynamicsCalculator


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, 
                 raft_model_path: Optional[str] = None,
                 device: str = 'cuda',
                 max_frames: Optional[int] = None,
                 frame_skip: int = 1):
        """
        初始化视频处理器
        
        Args:
            raft_model_path: RAFT模型路径
            device: 计算设备
            max_frames: 最大处理帧数
            frame_skip: 帧跳跃间隔
        """
        self.raft_predictor = RAFTPredictor(raft_model_path, device)
        self.dynamics_calculator = StaticObjectDynamicsCalculator()
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        
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
        flows = []
        for i in tqdm(range(len(frames) - 1), desc="计算光流"):
            flow = self.raft_predictor.predict_flow(frames[i], frames[i + 1])
            flows.append(flow.transpose(1, 2, 0))  # 转换为 (H, W, 2)
        
        print("开始分析静态物体动态度...")
        
        # 计算时序动态度
        temporal_result = self.dynamics_calculator.calculate_temporal_dynamics(
            flows, frames, camera_matrix
        )
        
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
            'flow_count': len(flows)
        }
        
        # 添加每帧的数值结果
        numeric_result['frame_results'] = []
        for i, frame_result in enumerate(result['frame_results']):
            numeric_frame_result = {
                'frame_index': i,
                'static_dynamics': frame_result['static_dynamics'],
                'global_dynamics': frame_result['global_dynamics']
            }
            numeric_result['frame_results'].append(numeric_frame_result)
        
        # 保存JSON结果
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
            json.dump(numeric_result, f, indent=2, ensure_ascii=False)
        
        # 生成并保存报告
        report = self.generate_video_report(result)
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存可视化结果
        self.save_visualizations(result, frames, flows, output_dir)
        
        print(f"结果已保存到: {output_dir}")
    
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
        
        report = f"""
相机转动拍摄静态建筑视频 - 静态物体动态度分析报告
================================================

视频基本信息:
- 总帧数: {frame_count}
- 分析帧数: {len(result['frame_results'])}

时序动态度统计:
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='静态物体动态度分析')
    parser.add_argument('--input', '-i', required=True, 
                       help='输入视频文件或图像目录路径')
    parser.add_argument('--output', '-o', default='output',
                       help='输出目录路径')
    parser.add_argument('--raft_model', '-m', default=None,
                       help='RAFT模型路径')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='最大处理帧数')
    parser.add_argument('--frame_skip', type=int, default=1,
                       help='帧跳跃间隔')
    parser.add_argument('--device', default='cuda',
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='相机视场角 (度)')
    
    args = parser.parse_args()
    
    # 创建视频处理器
    processor = VideoProcessor(
        raft_model_path=args.raft_model,
        device=args.device,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip
    )
    
    # 加载视频或图像
    if os.path.isfile(args.input):
        # 视频文件
        frames = processor.load_video(args.input)
    elif os.path.isdir(args.input):
        # 图像目录
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