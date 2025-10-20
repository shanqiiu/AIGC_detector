# -*- coding: utf-8 -*-
"""
Unified Dynamics Calculator (Refactored)
Auto-adapts to static and dynamic scenes, outputs unified 0-1 dynamics score

Key improvements:
1. Calculates both static and dynamic region metrics
2. Auto-detects scene type
3. Uses unified scoring standard (0-1), no forced segmentation
4. Can filter "dynamic scenes with low motion"
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


class UnifiedDynamicsCalculator:
    """Unified Dynamics Calculator - Auto-adapts to scene types"""
    
    def __init__(self, 
                 static_threshold=0.002,
                 subject_threshold=0.005,
                 use_normalized_flow=True,
                 scene_auto_detect=True):
        """
        Initialize calculator
        
        Args:
            static_threshold: Static region detection threshold (normalized)
            subject_threshold: Subject region detection threshold (normalized)
            use_normalized_flow: Whether to use resolution normalization
            scene_auto_detect: Whether to auto-detect scene type
        """
        self.static_threshold = static_threshold
        self.subject_threshold = subject_threshold
        self.use_normalized_flow = use_normalized_flow
        self.scene_auto_detect = scene_auto_detect
    
    def calculate_unified_dynamics(self, 
                                   flows: List[np.ndarray], 
                                   images: List[np.ndarray],
                                   camera_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate unified dynamics score
        
        Args:
            flows: List of optical flows (residual flows after camera compensation)
            images: List of images
            camera_matrix: Camera intrinsic matrix (reserved for future use)
            
        Returns:
            Dictionary containing unified dynamics score and details
        """
        
        if len(flows) == 0 or len(images) < 2:
            return self._get_empty_result()
        
        # Analyze each frame
        frame_results = []
        for i, flow in enumerate(flows):
            normalization_factor = 1.0
            if self.use_normalized_flow:
                h, w = images[i].shape[:2]
                normalization_factor = np.sqrt(h**2 + w**2)
            
            frame_result = self._analyze_single_frame(
                flow, images[i].shape, normalization_factor
            )
            frame_results.append(frame_result)
        
        # Determine scene type
        scene_type = self._determine_scene_type(frame_results)
        
        # Calculate final unified score
        unified_score = self._calculate_final_score(frame_results, scene_type)
        
        # Calculate temporal statistics
        temporal_stats = self._calculate_temporal_stats(frame_results, scene_type)
        
        # Classify dynamics level
        classification = self._classify_dynamics(unified_score, scene_type)
        
        return {
            'unified_dynamics_score': unified_score,
            'scene_type': scene_type,
            'classification': classification,
            'frame_results': frame_results,
            'temporal_stats': temporal_stats,
            'interpretation': self._generate_interpretation(
                unified_score, scene_type, temporal_stats
            )
        }
    
    def _analyze_single_frame(self, 
                             flow: np.ndarray, 
                             image_shape: Tuple, 
                             normalization_factor: float) -> Dict:
        """
        Analyze single frame - calculate both static and dynamic region metrics
        
        Args:
            flow: Optical flow (H, W, 2)
            image_shape: Image shape
            normalization_factor: Normalization factor
            
        Returns:
            Single frame analysis result
        """
        
        # Calculate flow magnitude
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # Normalize
        if self.use_normalized_flow and normalization_factor > 0:
            flow_magnitude_norm = flow_magnitude / normalization_factor
        else:
            flow_magnitude_norm = flow_magnitude
        
        # Detect static and dynamic regions
        static_mask = flow_magnitude_norm < self.static_threshold
        dynamic_mask = flow_magnitude_norm > self.subject_threshold
        
        # Morphological processing for dynamic regions
        kernel = np.ones((7, 7), np.uint8)
        dynamic_mask_cleaned = cv2.morphologyEx(
            dynamic_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        dynamic_mask_cleaned = cv2.morphologyEx(
            dynamic_mask_cleaned, cv2.MORPH_OPEN, kernel
        )
        dynamic_mask = dynamic_mask_cleaned.astype(bool)
        
        # Calculate static region metrics (for static scenes)
        static_metrics = self._calculate_region_metrics(
            flow, static_mask, normalization_factor, 'static'
        )
        
        # Calculate dynamic region metrics (for dynamic scenes)
        dynamic_metrics = self._calculate_region_metrics(
            flow, dynamic_mask, normalization_factor, 'dynamic'
        )
        
        # Calculate global metrics
        global_metrics = self._calculate_global_metrics(
            flow_magnitude_norm, static_mask, dynamic_mask
        )
        
        return {
            'static_metrics': static_metrics,
            'dynamic_metrics': dynamic_metrics,
            'global_metrics': global_metrics,
            'static_ratio': float(np.sum(static_mask) / static_mask.size),
            'dynamic_ratio': float(np.sum(dynamic_mask) / dynamic_mask.size)
        }
    
    def _calculate_region_metrics(self, 
                                  flow: np.ndarray, 
                                  mask: np.ndarray, 
                                  normalization_factor: float,
                                  region_type: str) -> Dict:
        """
        Calculate region metrics
        
        Args:
            flow: Optical flow
            mask: Region mask
            normalization_factor: Normalization factor
            region_type: Region type ('static' or 'dynamic')
            
        Returns:
            Region metrics dictionary
        """
        
        if not np.any(mask):
            return {
                'mean_magnitude': 0.0,
                'max_magnitude': 0.0,
                'std_magnitude': 0.0,
                'median_magnitude': 0.0,
                'score': 0.0,
                'pixel_count': 0
            }
        
        # Extract region flow
        flow_x = flow[:, :, 0][mask]
        flow_y = flow[:, :, 1][mask]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # Normalize
        if self.use_normalized_flow and normalization_factor > 0:
            magnitude = magnitude / normalization_factor
        
        # Calculate statistics
        mean_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        std_mag = np.std(magnitude)
        median_mag = np.median(magnitude)
        
        # Calculate score based on region type
        if region_type == 'static':
            # Static region: expect small residual flow
            score = mean_mag + 0.5 * std_mag
        else:
            # Dynamic region: focus on motion intensity
            score = mean_mag + 0.8 * std_mag + 0.3 * max_mag
        
        return {
            'mean_magnitude': float(mean_mag),
            'max_magnitude': float(max_mag),
            'std_magnitude': float(std_mag),
            'median_magnitude': float(median_mag),
            'score': float(score),
            'pixel_count': int(np.sum(mask))
        }
    
    def _calculate_global_metrics(self, 
                                  flow_magnitude_norm: np.ndarray,
                                  static_mask: np.ndarray,
                                  dynamic_mask: np.ndarray) -> Dict:
        """Calculate global metrics"""
        
        global_mean = np.mean(flow_magnitude_norm)
        global_std = np.std(flow_magnitude_norm)
        global_max = np.max(flow_magnitude_norm)
        
        motion_coverage = np.sum(dynamic_mask) / dynamic_mask.size
        
        consistency = 1.0 - (global_std / (global_mean + 1e-6))
        consistency = max(0.0, min(1.0, consistency))
        
        return {
            'global_mean_magnitude': float(global_mean),
            'global_std_magnitude': float(global_std),
            'global_max_magnitude': float(global_max),
            'motion_coverage': float(motion_coverage),
            'consistency_score': float(consistency)
        }
    
    def _determine_scene_type(self, frame_results: List[Dict]) -> str:
        """
        Determine scene type
        
        Logic:
        - If dynamic region ratio is high and motion is obvious -> dynamic scene
        - If static region residual is very small -> static scene
        - Otherwise decide based on which indicator is stronger
        
        Returns:
            'static' or 'dynamic'
        """
        
        if not self.scene_auto_detect:
            return 'dynamic'
        
        # Calculate average metrics
        avg_dynamic_ratio = np.mean([r['dynamic_ratio'] for r in frame_results])
        avg_dynamic_score = np.mean([r['dynamic_metrics']['score'] for r in frame_results])
        avg_static_score = np.mean([r['static_metrics']['score'] for r in frame_results])
        avg_motion_coverage = np.mean([r['global_metrics']['motion_coverage'] for r in frame_results])
        
        # Detection logic
        if avg_dynamic_ratio > 0.15 and avg_dynamic_score > 0.01:
            return 'dynamic'
        
        if avg_motion_coverage > 0.2 and avg_dynamic_score > 0.008:
            return 'dynamic'
        
        if avg_static_score < 0.003:
            return 'static'
        
        if avg_dynamic_score > avg_static_score * 2.5:
            return 'dynamic'
        else:
            return 'static'
    
    def _calculate_final_score(self, 
                               frame_results: List[Dict], 
                               scene_type: str) -> float:
        """
        Calculate final unified score (linear mapping)
        
        Key improvement:
        - Uses full 0-1 range
        - Scene type only determines which region to focus on
        - Score purely reflects dynamics level
        
        Returns:
            Dynamics score between 0-1
        """
        
        if scene_type == 'static':
            scores = [r['static_metrics']['score'] for r in frame_results]
            mean_score = np.mean(scores)
            unified_score = self._normalize_static_score(mean_score)
            
        else:  # dynamic
            scores = [r['dynamic_metrics']['score'] for r in frame_results]
            mean_score = np.mean(scores)
            unified_score = self._normalize_dynamic_score(mean_score)
        
        return float(np.clip(unified_score, 0.0, 1.0))
    
    def _normalize_static_score(self, raw_score: float) -> float:
        """
        Normalize static scene score (linear mapping)
        
        Mapping table:
        raw_score | unified_score | meaning
        0.0001   -> 0.00         | perfectly still
        0.001    -> 0.10         | extremely low residual
        0.003    -> 0.25         | low residual
        0.005    -> 0.35         | medium residual
        0.01     -> 0.55         | high residual
        0.02     -> 0.75         | very high residual
        0.05+    -> 1.00         | extreme residual/anomaly
        """
        
        if raw_score < 0.001:
            return raw_score / 0.001 * 0.10
        elif raw_score < 0.003:
            return 0.10 + (raw_score - 0.001) / 0.002 * 0.15
        elif raw_score < 0.005:
            return 0.25 + (raw_score - 0.003) / 0.002 * 0.10
        elif raw_score < 0.01:
            return 0.35 + (raw_score - 0.005) / 0.005 * 0.20
        elif raw_score < 0.02:
            return 0.55 + (raw_score - 0.01) / 0.01 * 0.20
        else:
            return min(0.75 + (raw_score - 0.02) / 0.03 * 0.25, 1.0)
    
    def _normalize_dynamic_score(self, raw_score: float) -> float:
        """
        Normalize dynamic scene score (linear mapping)
        
        Mapping table:
        raw_score | unified_score | meaning
        0.005    -> 0.10         | subject almost still (can be filtered!)
        0.008    -> 0.20         | subject slight motion
        0.015    -> 0.35         | subject normal motion
        0.025    -> 0.50         | subject active motion
        0.04     -> 0.65         | subject fast motion
        0.08     -> 0.85         | subject intense motion
        0.15+    -> 1.00         | subject extreme motion
        """
        
        if raw_score < 0.008:
            return raw_score / 0.008 * 0.20
        elif raw_score < 0.015:
            return 0.20 + (raw_score - 0.008) / 0.007 * 0.15
        elif raw_score < 0.025:
            return 0.35 + (raw_score - 0.015) / 0.01 * 0.15
        elif raw_score < 0.04:
            return 0.50 + (raw_score - 0.025) / 0.015 * 0.15
        elif raw_score < 0.08:
            return 0.65 + (raw_score - 0.04) / 0.04 * 0.20
        else:
            return min(0.85 + (raw_score - 0.08) / 0.07 * 0.15, 1.0)
    
    def _calculate_temporal_stats(self, 
                                  frame_results: List[Dict], 
                                  scene_type: str) -> Dict:
        """Calculate temporal statistics"""
        
        if scene_type == 'static':
            scores = [r['static_metrics']['score'] for r in frame_results]
            mean_mag = np.mean([r['static_metrics']['mean_magnitude'] for r in frame_results])
        else:
            scores = [r['dynamic_metrics']['score'] for r in frame_results]
            mean_mag = np.mean([r['dynamic_metrics']['mean_magnitude'] for r in frame_results])
        
        return {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'mean_magnitude': float(mean_mag),
            'temporal_stability': float(1.0 / (1.0 + np.std(scores))),
            'mean_static_ratio': float(np.mean([r['static_ratio'] for r in frame_results])),
            'mean_dynamic_ratio': float(np.mean([r['dynamic_ratio'] for r in frame_results]))
        }
    
    def _classify_dynamics(self, score: float, scene_type: str) -> Dict:
        """
        Classify dynamics level
        
        Args:
            score: Unified dynamics score (0-1)
            scene_type: Scene type
            
        Returns:
            Classification dictionary
        """
        
        if score < 0.15:
            category = 'pure_static'
            description = 'Pure Static'
            if scene_type == 'static':
                examples = ['Buildings', 'Sculptures', 'Static objects']
            else:
                examples = ['Person standing still', 'Sitting', 'Almost no motion']
        elif score < 0.35:
            category = 'low_dynamic'
            description = 'Low Dynamic'
            if scene_type == 'static':
                examples = ['Flags waving', 'Leaves in breeze']
            else:
                examples = ['Slow movement', 'Small gestures', 'Minor adjustments']
        elif score < 0.60:
            category = 'medium_dynamic'
            description = 'Medium Dynamic'
            if scene_type == 'static':
                examples = ['Noticeable vibration', 'Swaying']
            else:
                examples = ['Walking', 'Daily activities', 'Normal gestures']
        elif score < 0.85:
            category = 'high_dynamic'
            description = 'High Dynamic'
            if scene_type == 'static':
                examples = ['Strong vibration', 'Fast swaying']
            else:
                examples = ['Running', 'Dancing', 'Active movements']
        else:
            category = 'extreme_dynamic'
            description = 'Extreme Dynamic'
            if scene_type == 'static':
                examples = ['Severe anomaly', 'Equipment failure']
            else:
                examples = ['Fast dancing', 'Sports', 'Fighting scenes']
        
        return {
            'category': category,
            'category_id': ['pure_static', 'low_dynamic', 'medium_dynamic', 
                          'high_dynamic', 'extreme_dynamic'].index(category),
            'description': description,
            'typical_examples': examples,
            'scene_type': scene_type
        }
    
    def _generate_interpretation(self, 
                                score: float, 
                                scene_type: str,
                                temporal_stats: Dict) -> str:
        """Generate textual interpretation"""
        
        classification = self._classify_dynamics(score, scene_type)
        
        if scene_type == 'static':
            scene_desc = "Static scene (buildings, still objects)"
            metric_desc = f"Static region residual flow mean: {temporal_stats['mean_magnitude']:.6f}"
        else:
            scene_desc = "Dynamic scene (people, animals)"
            metric_desc = f"Subject region flow intensity mean: {temporal_stats['mean_magnitude']:.6f}"
        
        interpretation = f"""
[Scene Type] {scene_desc}
[Dynamics Score] {score:.3f} / 1.000
[Dynamics Level] {classification['description']} ({classification['category']})
[Typical Examples] {', '.join(classification['typical_examples'])}

[Technical Metrics]
- {metric_desc}
- Temporal stability: {temporal_stats['temporal_stability']:.3f}
- Static region ratio: {temporal_stats['mean_static_ratio']:.1%}
- Dynamic region ratio: {temporal_stats['mean_dynamic_ratio']:.1%}
"""
        return interpretation.strip()
    
    def _get_empty_result(self) -> Dict:
        """Return empty result"""
        return {
            'unified_dynamics_score': 0.0,
            'scene_type': 'unknown',
            'classification': {
                'category': 'unknown',
                'category_id': -1,
                'description': 'Invalid data',
                'typical_examples': [],
                'scene_type': 'unknown'
            },
            'frame_results': [],
            'temporal_stats': {},
            'interpretation': 'Cannot analyze: input data is empty'
        }
