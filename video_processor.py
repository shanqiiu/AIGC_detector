# -*- coding: utf-8 -*-
"""
Video Processor - Unified Dynamics Assessment System
Processes videos to calculate unified dynamics scores (0-1 range)
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
from dynamic_motion_compensation.camera_compensation import CameraCompensator
from unified_dynamics_calculator import UnifiedDynamicsCalculator
from video_quality_filter import VideoQualityFilter
from dynamics_config import get_config
from badcase_detector import BadCaseDetector, BadCaseAnalyzer


class VideoProcessor:
    """Video Processor with Unified Dynamics Assessment"""
    
    def __init__(self, 
                 raft_model_path: Optional[str] = None,
                 device: str = 'cuda',
                 max_frames: Optional[int] = None,
                 frame_skip: int = 1,
                 enable_visualization: bool = True,
                 enable_camera_compensation: bool = True,
                 camera_compensation_params: Optional[Dict] = None,
                 config_preset: str = 'balanced'):
        """
        Initialize Video Processor
        
        Args:
            raft_model_path: Path to RAFT model
            device: Computing device ('cuda' or 'cpu')
            max_frames: Maximum number of frames to process
            frame_skip: Frame skip interval
            enable_visualization: Whether to generate visualizations
            enable_camera_compensation: Whether to enable camera compensation
            camera_compensation_params: Camera compensation parameters
            config_preset: Configuration preset ('strict', 'balanced', 'lenient')
        """
        self.raft_predictor = RAFTPredictor(raft_model_path, device, method='raft')
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        self.enable_visualization = enable_visualization
        self.enable_camera_compensation = enable_camera_compensation
        
        # Load configuration
        self.config = get_config(config_preset)
        
        # Initialize camera compensator
        if camera_compensation_params is None:
            camera_compensation_params = {}
        self.camera_compensator = CameraCompensator(**camera_compensation_params) if enable_camera_compensation else None
        
        # Initialize unified dynamics calculator
        self.dynamics_calculator = UnifiedDynamicsCalculator(
            static_threshold=self.config['detection']['static_threshold'],
            subject_threshold=self.config['detection']['subject_threshold'],
            use_normalized_flow=self.config['flow']['use_normalized_flow'],
            scene_auto_detect=self.config['scene_classification']['scene_auto_detect']
        )
        
        # Initialize quality filter
        self.quality_filter = VideoQualityFilter()
        
        # Initialize BadCase detector
        self.badcase_detector = BadCaseDetector()
        self.badcase_analyzer = BadCaseAnalyzer()
        
    def load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        print(f"Loading video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if self.max_frames and len(frames) >= self.max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"Loaded {len(frames)} frames")
        return frames
    
    def extract_frames_from_images(self, image_dir: str) -> List[np.ndarray]:
        """Load frames from image directory"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        image_files.sort()
        frames = []
        
        print(f"Loading images from: {image_dir}")
        
        for i, img_path in enumerate(image_files):
            if self.max_frames and i >= self.max_frames:
                break
                
            if i % self.frame_skip == 0:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img_rgb)
        
        print(f"Loaded {len(frames)} frames")
        return frames
    
    def estimate_camera_matrix(self, frame_shape: Tuple[int, int], fov: float = 60.0) -> np.ndarray:
        """Estimate camera intrinsic matrix"""
        h, w = frame_shape[:2]
        focal_length = w / (2 * np.tan(np.radians(fov / 2)))
        
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
        """Process video sequence"""
        
        if len(frames) < 2:
            raise ValueError("At least 2 frames required")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if camera_matrix is None:
            camera_matrix = self.estimate_camera_matrix(frames[0].shape)
            print("Using estimated camera matrix")
        
        print("Computing optical flow...")
        
        # Calculate optical flow
        flows = []
        original_flows = []
        camera_compensation_results = []
        
        for i in tqdm(range(len(frames) - 1), desc="Computing flow"):
            flow = self.raft_predictor.predict_flow(frames[i], frames[i + 1])
            
            # Ensure flow format (2, H, W) -> (H, W, 2)
            if flow.shape[0] == 2:
                flow = flow.transpose(1, 2, 0)
            elif len(flow.shape) == 2:
                print(f"Warning: abnormal flow shape at frame {i}: {flow.shape}")
                continue
            
            original_flows.append(flow.copy())
            
            # Apply camera compensation
            if self.enable_camera_compensation and self.camera_compensator is not None:
                comp_result = self.camera_compensator.compensate(flow, frames[i], frames[i + 1])
                camera_compensation_results.append(comp_result)
                flows.append(comp_result['residual_flow'])
            else:
                flows.append(flow)
                camera_compensation_results.append(None)
        
        print("Analyzing unified dynamics...")
        
        # Calculate unified dynamics
        unified_result = self.dynamics_calculator.calculate_unified_dynamics(
            flows, frames, camera_matrix
        )
        
        # Add camera compensation info
        unified_result['camera_compensation_enabled'] = self.enable_camera_compensation
        unified_result['camera_compensation_results'] = camera_compensation_results
        unified_result['original_flows'] = original_flows
        
        # Build result structure
        result = {
            'frame_results': unified_result['frame_results'],
            'temporal_stats': unified_result['temporal_stats'],
            'unified_dynamics': {
                'unified_dynamics_score': unified_result['unified_dynamics_score'],
                'scene_type': unified_result['scene_type'],
                'confidence': unified_result['temporal_stats'].get('temporal_stability', 0.0),
                'interpretation': unified_result['interpretation']
            },
            'dynamics_classification': unified_result['classification'],
            'camera_compensation_enabled': self.enable_camera_compensation,
            'camera_compensation_results': camera_compensation_results,
            'original_flows': original_flows
        }
        
        # Save results
        self.save_results(result, frames, flows, output_dir)
        
        return result
    
    def save_results(self, 
                    result: Dict,
                    frames: List[np.ndarray],
                    flows: List[np.ndarray],
                    output_dir: str):
        """Save analysis results"""
        
        # Prepare numeric results
        numeric_result = {
            'temporal_stats': result['temporal_stats'],
            'frame_count': len(frames),
            'flow_count': len(flows),
            'camera_compensation_enabled': result.get('camera_compensation_enabled', False),
            'unified_dynamics_score': result['unified_dynamics']['unified_dynamics_score'],
            'scene_type': result['unified_dynamics']['scene_type'],
            'dynamics_category': result['dynamics_classification']['category'],
            'dynamics_category_id': result['dynamics_classification']['category_id'],
            'confidence': result['unified_dynamics']['confidence']
        }
        
        # Add camera compensation stats
        if result.get('camera_compensation_enabled', False):
            camera_comp_results = result.get('camera_compensation_results', [])
            comp_stats = self._calculate_camera_compensation_stats(camera_comp_results)
            numeric_result['camera_compensation_stats'] = comp_stats
        
        # Add frame results
        numeric_result['frame_results'] = []
        for i, frame_result in enumerate(result['frame_results']):
            numeric_frame_result = {
                'frame_index': i,
                'static_metrics': frame_result['static_metrics'],
                'dynamic_metrics': frame_result['dynamic_metrics'],
                'global_metrics': frame_result['global_metrics'],
                'static_ratio': frame_result['static_ratio'],
                'dynamic_ratio': frame_result['dynamic_ratio']
            }
            
            if result.get('camera_compensation_enabled', False) and i < len(result.get('camera_compensation_results', [])):
                comp_result = result['camera_compensation_results'][i]
                if comp_result is not None:
                    numeric_frame_result['camera_compensation'] = {
                        'inliers': comp_result['inliers'],
                        'total_matches': comp_result['total_matches'],
                        'homography_found': comp_result['homography'] is not None
                    }
            
            numeric_result['frame_results'].append(numeric_frame_result)
        
        # Save JSON
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
            json.dump(numeric_result, f, indent=2, ensure_ascii=False)
        
        # Generate and save report
        report = self.generate_video_report(result)
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save visualizations
        if self.enable_visualization:
            self.save_visualizations(result, frames, flows, output_dir)
        
        print(f"Results saved to: {output_dir}")
    
    def _calculate_camera_compensation_stats(self, camera_comp_results: List[Optional[Dict]]) -> Dict:
        """Calculate camera compensation statistics"""
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
        """Save visualizations"""
        
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save key frame analyses
        key_frames = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-2]
        key_frames = [i for i in key_frames if i < len(result['frame_results'])]
        
        for i in key_frames:
            self._visualize_frame(i, frames[i], flows[i], result['frame_results'][i], vis_dir)
        
        # Plot temporal dynamics
        self.plot_temporal_dynamics(result, os.path.join(vis_dir, 'temporal_dynamics.png'))
        
        # Plot camera compensation comparison if enabled
        if result.get('camera_compensation_enabled', False):
            self.plot_camera_compensation_comparison(result, frames, os.path.join(vis_dir, 'camera_compensation.png'))
    
    def _visualize_frame(self, frame_idx: int, image: np.ndarray, flow: np.ndarray, frame_result: Dict, vis_dir: str):
        """Visualize single frame analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f'Frame {frame_idx}')
        axes[0, 0].axis('off')
        
        # Flow magnitude
        flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        im1 = axes[0, 1].imshow(flow_mag, cmap='jet')
        axes[0, 1].set_title('Flow Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Flow vectors
        step = 16
        y, x = np.mgrid[step//2:flow.shape[0]:step, step//2:flow.shape[1]:step]
        fx = flow[::step, ::step, 0]
        fy = flow[::step, ::step, 1]
        axes[0, 2].imshow(image)
        axes[0, 2].quiver(x, y, fx, fy, color='red', alpha=0.7)
        axes[0, 2].set_title('Flow Vectors')
        axes[0, 2].axis('off')
        
        # Metrics text
        static_score = frame_result['static_metrics']['score']
        dynamic_score = frame_result['dynamic_metrics']['score']
        static_ratio = frame_result['static_ratio']
        dynamic_ratio = frame_result['dynamic_ratio']
        
        metrics_text = f"""
Frame Metrics:
- Static score: {static_score:.4f}
- Dynamic score: {dynamic_score:.4f}
- Static ratio: {static_ratio:.2%}
- Dynamic ratio: {dynamic_ratio:.2%}
        """
        
        axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
        axes[1, 0].axis('off')
        
        # Static region visualization
        # (Simplified - can be enhanced)
        axes[1, 1].imshow(flow_mag < 0.002, cmap='gray')
        axes[1, 1].set_title('Static Regions')
        axes[1, 1].axis('off')
        
        # Dynamic region visualization
        axes[1, 2].imshow(flow_mag > 0.005, cmap='hot')
        axes[1, 2].set_title('Dynamic Regions')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'frame_{frame_idx:04d}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_dynamics(self, result: Dict, save_path: str):
        """Plot temporal dynamics curves"""
        frame_results = result['frame_results']
        
        static_scores = [r['static_metrics']['score'] for r in frame_results]
        dynamic_scores = [r['dynamic_metrics']['score'] for r in frame_results]
        static_ratios = [r['static_ratio'] for r in frame_results]
        dynamic_ratios = [r['dynamic_ratio'] for r in frame_results]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Dynamics scores
        axes[0].plot(static_scores, 'b-', linewidth=2, label='Static Score')
        axes[0].plot(dynamic_scores, 'r-', linewidth=2, label='Dynamic Score')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Dynamics Scores Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Region ratios
        axes[1].plot(static_ratios, 'g-', linewidth=2, label='Static Ratio')
        axes[1].plot(dynamic_ratios, 'm-', linewidth=2, label='Dynamic Ratio')
        axes[1].set_ylabel('Ratio')
        axes[1].set_title('Region Ratios Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Stacked area
        axes[2].fill_between(range(len(static_ratios)), 0, static_ratios, alpha=0.7, color='green', label='Static')
        axes[2].fill_between(range(len(static_ratios)), static_ratios, 1.0, alpha=0.7, color='red', label='Dynamic')
        axes[2].set_ylabel('Proportion')
        axes[2].set_xlabel('Frame Index')
        axes[2].set_title('Static vs Dynamic Regions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_camera_compensation_comparison(self, result: Dict, frames: List[np.ndarray], save_path: str):
        """Plot camera compensation comparison"""
        original_flows = result.get('original_flows', [])
        camera_comp_results = result.get('camera_compensation_results', [])
        
        if not original_flows or not camera_comp_results:
            return
        
        # Select key frames
        key_frames = [0, len(original_flows)//2, len(original_flows)-1]
        key_frames = [i for i in key_frames if i < len(original_flows) and camera_comp_results[i] is not None]
        
        if not key_frames:
            return
        
        n_frames = min(3, len(key_frames))
        fig, axes = plt.subplots(n_frames, 3, figsize=(15, 5*n_frames))
        
        if n_frames == 1:
            axes = axes.reshape(1, -1)
        
        for idx, frame_idx in enumerate(key_frames[:n_frames]):
            original_flow = original_flows[frame_idx]
            comp_result = camera_comp_results[frame_idx]
            
            original_mag = np.sqrt(original_flow[:, :, 0]**2 + original_flow[:, :, 1]**2)
            camera_mag = np.sqrt(comp_result['camera_flow'][:, :, 0]**2 + comp_result['camera_flow'][:, :, 1]**2)
            residual_mag = np.sqrt(comp_result['residual_flow'][:, :, 0]**2 + comp_result['residual_flow'][:, :, 1]**2)
            
            vmax = np.percentile(original_mag, 95)
            
            im1 = axes[idx, 0].imshow(original_mag, cmap='jet', vmin=0, vmax=vmax)
            axes[idx, 0].set_title(f'Frame {frame_idx}: Original Flow')
            axes[idx, 0].axis('off')
            plt.colorbar(im1, ax=axes[idx, 0])
            
            im2 = axes[idx, 1].imshow(camera_mag, cmap='jet', vmin=0, vmax=vmax)
            axes[idx, 1].set_title(f'Camera Flow (inliers={comp_result["inliers"]})')
            axes[idx, 1].axis('off')
            plt.colorbar(im2, ax=axes[idx, 1])
            
            im3 = axes[idx, 2].imshow(residual_mag, cmap='jet', vmin=0, vmax=vmax)
            axes[idx, 2].set_title(f'Residual Flow (mean={residual_mag.mean():.2f})')
            axes[idx, 2].axis('off')
            plt.colorbar(im3, ax=axes[idx, 2])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_video_report(self, result: Dict) -> str:
        """Generate video analysis report"""
        temporal_stats = result['temporal_stats']
        unified_dynamics = result['unified_dynamics']
        classification = result['dynamics_classification']
        
        report = f"""
{'='*70}
Video Dynamics Assessment Report
{'='*70}

Basic Information:
- Total frames: {len(result['frame_results'])}
- Camera compensation: {'Enabled' if result.get('camera_compensation_enabled') else 'Disabled'}

{'='*70}
Unified Dynamics Score
{'='*70}

Score: {unified_dynamics['unified_dynamics_score']:.3f} / 1.000
Scene Type: {unified_dynamics['scene_type']}
Classification: {classification['description']} ({classification['category']})
Confidence: {unified_dynamics['confidence']:.3f}

Typical Examples: {', '.join(classification['typical_examples'])}

{'='*70}
Detailed Interpretation
{'='*70}

{unified_dynamics['interpretation']}

{'='*70}
Technical Metrics
{'='*70}

Mean Score: {temporal_stats['mean_score']:.6f}
Score Std Dev: {temporal_stats['std_score']:.6f}
Max Score: {temporal_stats['max_score']:.6f}
Min Score: {temporal_stats['min_score']:.6f}

Mean Magnitude: {temporal_stats['mean_magnitude']:.6f}
Temporal Stability: {temporal_stats['temporal_stability']:.3f}

Static Region Ratio: {temporal_stats['mean_static_ratio']:.1%}
Dynamic Region Ratio: {temporal_stats['mean_dynamic_ratio']:.1%}

"""
        
        # Add camera compensation stats
        if result.get('camera_compensation_enabled'):
            comp_stats = self._calculate_camera_compensation_stats(result.get('camera_compensation_results', []))
            if comp_stats:
                report += f"""
{'='*70}
Camera Compensation Statistics
{'='*70}

Success Rate: {comp_stats['success_rate']:.1%} ({comp_stats['successful_frames']}/{comp_stats['total_frames']})
Mean Inliers: {comp_stats['mean_inliers']:.1f} ± {comp_stats['std_inliers']:.1f}
Mean Match Ratio: {comp_stats['mean_match_ratio']:.1%} ± {comp_stats['std_match_ratio']:.1%}

"""
        
        return report


def process_single_video(processor, video_path, output_dir, camera_fov, expected_label=None):
    """Process single video"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    
    print(f"\n{'='*70}")
    print(f"Processing: {video_name}")
    if expected_label is not None:
        print(f"Expected label: {expected_label}")
    print(f"{'='*70}")
    
    try:
        frames = processor.load_video(video_path)
        camera_matrix = processor.estimate_camera_matrix(frames[0].shape, camera_fov)
        result = processor.process_video(frames, camera_matrix, video_output_dir)
        
        temporal_stats = result['temporal_stats']
        unified = result['unified_dynamics']
        
        video_result = {
            'video_name': video_name,
            'video_path': video_path,
            'status': 'success',
            'frame_count': len(frames),
            'unified_dynamics_score': unified['unified_dynamics_score'],
            'scene_type': unified['scene_type'],
            'classification': result['dynamics_classification'],
            'confidence': unified['confidence'],
            'temporal_stats': temporal_stats,
            'output_dir': video_output_dir,
            'full_result': result
        }
        
        # BadCase detection if label provided
        if expected_label is not None:
            badcase_result = processor.badcase_analyzer.analyze_with_details(result, expected_label)
            video_result['expected_label'] = expected_label
            video_result['is_badcase'] = badcase_result['is_badcase']
            video_result['badcase_type'] = badcase_result.get('badcase_type', 'normal')
            
            if badcase_result['is_badcase']:
                print(f"Warning: BadCase detected")
                print(f"  Type: {badcase_result['badcase_type']}")
        
        return video_result
        
    except Exception as e:
        print(f"Error: Processing failed - {e}")
        return {
            'video_name': video_name,
            'video_path': video_path,
            'status': 'failed',
            'error': str(e)
        }


def batch_process_videos(processor, input_dir, output_dir, camera_fov, badcase_labels=None):
    """Batch process videos"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    video_files = list(set(video_files))
    
    if not video_files:
        print(f"Error: No video files found in {input_dir}")
        return []
    
    video_files.sort()
    print(f"\nFound {len(video_files)} video files")
    if badcase_labels:
        print(f"Loaded {len(badcase_labels)} labels")
    
    results = []
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        expected_label = None
        if badcase_labels is not None:
            expected_label = badcase_labels.get(video_name, 'dynamic')
        
        print(f"\nProgress: {i}/{len(video_files)}")
        result = process_single_video(processor, video_path, output_dir, camera_fov, expected_label)
        results.append(result)
    
    # Save batch summary
    save_batch_summary(results, output_dir)
    
    if badcase_labels is not None:
        batch_summary = processor.badcase_analyzer.generate_batch_summary(results)
        processor.badcase_analyzer.save_batch_report(batch_summary, results, output_dir)
        return batch_summary
    else:
        return results


def save_batch_summary(results, output_dir):
    """Save batch processing summary"""
    summary_path = os.path.join(output_dir, 'batch_summary.txt')
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Batch Video Processing Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total videos: {len(results)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failed_count}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Detailed Results\n")
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            f.write(f"Video: {result['video_name']}\n")
            if result['status'] == 'success':
                f.write(f"  Status: Success\n")
                f.write(f"  Frames: {result['frame_count']}\n")
                f.write(f"  Score: {result['unified_dynamics_score']:.3f}\n")
                f.write(f"  Scene: {result['scene_type']}\n")
                f.write(f"  Category: {result['classification']['category']}\n")
                f.write(f"  Output: {result['output_dir']}\n")
            else:
                f.write(f"  Status: Failed\n")
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")
    
    # Save JSON
    json_summary_path = os.path.join(output_dir, 'batch_summary.json')
    json_safe_results = []
    for result in results:
        json_safe_result = {
            'video_name': result['video_name'],
            'video_path': result['video_path'],
            'status': result['status']
        }
        
        if result['status'] == 'success':
            json_safe_result.update({
                'frame_count': result['frame_count'],
                'unified_dynamics_score': result['unified_dynamics_score'],
                'scene_type': result['scene_type'],
                'classification': result['classification'],
                'confidence': result['confidence'],
                'output_dir': result['output_dir']
            })
        else:
            json_safe_result['error'] = result['error']
        
        json_safe_results.append(json_safe_result)
    
    with open(json_summary_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch summary saved to: {summary_path}")


def load_expected_labels(label_file: str) -> Dict:
    """Load expected labels from file"""
    labels = {}
    if label_file.endswith('.json'):
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    return labels


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified Video Dynamics Assessment')
    parser.add_argument('--input', '-i', required=True,
                       help='Input video file or directory')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory')
    parser.add_argument('--raft_model', '-m', default="pretrained_models/raft-things.pth",
                       help='RAFT model path')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--frame_skip', type=int, default=1,
                       help='Frame skip interval')
    parser.add_argument('--device', default='cuda',
                       help='Computing device (cuda/cpu)')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='Camera field of view (degrees)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode')
    parser.add_argument('--config', default='balanced',
                       choices=['strict', 'balanced', 'lenient'],
                       help='Configuration preset')
    parser.add_argument('--badcase-labels', '-l', default=None,
                       help='Expected labels file (JSON) for BadCase detection')
    parser.add_argument('--no-viz', dest='visualize', action='store_false',
                       help='Disable visualization')
    parser.add_argument('--no-camera-comp', dest='camera_compensation', action='store_false',
                       help='Disable camera compensation')
    parser.set_defaults(visualize=True, camera_compensation=True)
    
    args = parser.parse_args()
    
    # Load badcase labels if provided
    badcase_labels = None
    if args.badcase_labels:
        print(f"Loading expected labels: {args.badcase_labels}")
        badcase_labels = load_expected_labels(args.badcase_labels)
        print(f"Loaded {len(badcase_labels)} labels")
    
    # Create processor
    processor = VideoProcessor(
        raft_model_path=args.raft_model,
        device=args.device,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        enable_visualization=args.visualize,
        enable_camera_compensation=args.camera_compensation,
        config_preset=args.config
    )
    
    # Batch processing
    if args.batch:
        if not os.path.isdir(args.input):
            raise ValueError(f"Batch mode requires directory path: {args.input}")
        
        results = batch_process_videos(processor, args.input, args.output, args.fov, badcase_labels)
        
        print(f"\n{'='*70}")
        print(f"Batch processing complete!")
        print(f"{'='*70}")
        
        if isinstance(results, dict) and 'total_videos' in results:
            # BadCase mode
            print(f"Total videos: {results['total_videos']}")
            print(f"Successful: {results['successful']}")
            print(f"BadCase count: {results['badcase_count']}")
        else:
            # Normal mode
            success_count = sum(1 for r in results if r['status'] == 'success')
            print(f"Total: {len(results)} videos")
            print(f"Successful: {success_count}")
        
        print(f"\nResults saved to: {args.output}")
        print(f"{'='*70}")
    
    # Single video processing
    else:
        if os.path.isfile(args.input):
            frames = processor.load_video(args.input)
        elif os.path.isdir(args.input):
            frames = processor.extract_frames_from_images(args.input)
            if len(frames) == 0:
                raise ValueError(f"No valid images found in: {args.input}")
        else:
            raise ValueError(f"Invalid input path: {args.input}")
        
        camera_matrix = processor.estimate_camera_matrix(frames[0].shape, args.fov)
        result = processor.process_video(frames, camera_matrix, args.output)
        
        temporal_stats = result['temporal_stats']
        unified = result['unified_dynamics']
        
        print(f"\nAnalysis complete!")
        print(f"Unified dynamics score: {unified['unified_dynamics_score']:.3f}")
        print(f"Scene type: {unified['scene_type']}")
        print(f"Classification: {result['dynamics_classification']['category']}")
        print(f"Detailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
