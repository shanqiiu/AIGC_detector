# -*- coding: utf-8 -*-
"""
ͳһ��̬�ȼ����� (�ع���)
�Զ���Ӧ��̬�����Ͷ�̬���������ͳһ�� 0-1 ��̬�ȷ���

���ĸĽ���
1. ͬʱ���㾲̬����Ͷ�̬����ָ��
2. �Զ��жϳ�������
3. ʹ��ͳһ�����ֱ�׼��0-1��������ǿ�Ʒֶ�
4. ����ɸѡ��"��̬�����ж�̬�Ⱥܵ�"����Ƶ
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


class UnifiedDynamicsCalculator:
    """ͳһ��̬�ȼ����� - �Զ���Ӧ��������"""
    
    def __init__(self, 
                 static_threshold=0.002,   # ��̬��������ֵ����һ����
                 subject_threshold=0.005,  # ������������ֵ����һ����
                 use_normalized_flow=True,
                 scene_auto_detect=True):
        """
        ��ʼ��������
        
        Args:
            static_threshold: ��̬��������ֵ
            subject_threshold: ������������ֵ
            use_normalized_flow: �Ƿ�ʹ�÷ֱ��ʹ�һ��
            scene_auto_detect: �Ƿ��Զ���ⳡ������
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
        ����ͳһ��̬�ȷ���
        
        Args:
            flows: �����б� (�������������Ĳв����)
            images: ͼ���б�
            camera_matrix: ����ڲξ��󣨱���������չ��
            
        Returns:
            ����ͳһ��̬�ȷ�������ϸ��Ϣ���ֵ�
        """
        
        if len(flows) == 0 or len(images) < 2:
            return self._get_empty_result()
        
        # ����ÿһ֡
        frame_results = []
        for i, flow in enumerate(flows):
            # ��һ������
            normalization_factor = 1.0
            if self.use_normalized_flow:
                h, w = images[i].shape[:2]
                normalization_factor = np.sqrt(h**2 + w**2)
            
            # ������֡
            frame_result = self._analyze_single_frame(
                flow, images[i].shape, normalization_factor
            )
            frame_results.append(frame_result)
        
        # ���������ж�
        scene_type = self._determine_scene_type(frame_results)
        
        # ��������ͳһ����
        unified_score = self._calculate_final_score(frame_results, scene_type)
        
        # ����ʱ��ͳ��
        temporal_stats = self._calculate_temporal_stats(frame_results, scene_type)
        
        # ���ද̬�ȵȼ�
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
        ������֡��ͬʱ���㾲̬����Ͷ�̬����ָ��
        
        Args:
            flow: ���� (H, W, 2)
            image_shape: ͼ����״
            normalization_factor: ��һ������
            
        Returns:
            ��֡�������
        """
        
        # �����������
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # ��һ��
        if self.use_normalized_flow and normalization_factor > 0:
            flow_magnitude_norm = flow_magnitude / normalization_factor
        else:
            flow_magnitude_norm = flow_magnitude
        
        # ��⾲̬����Ͷ�̬����
        static_mask = flow_magnitude_norm < self.static_threshold
        dynamic_mask = flow_magnitude_norm > self.subject_threshold
        
        # ��̬ѧ����̬���������ڽ�����
        kernel = np.ones((7, 7), np.uint8)
        dynamic_mask_cleaned = cv2.morphologyEx(
            dynamic_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        dynamic_mask_cleaned = cv2.morphologyEx(
            dynamic_mask_cleaned, cv2.MORPH_OPEN, kernel
        )
        dynamic_mask = dynamic_mask_cleaned.astype(bool)
        
        # ���㾲̬����ָ�꣨���ھ�̬������
        static_metrics = self._calculate_region_metrics(
            flow, static_mask, normalization_factor, 'static'
        )
        
        # ���㶯̬����ָ�꣨���ڶ�̬������
        dynamic_metrics = self._calculate_region_metrics(
            flow, dynamic_mask, normalization_factor, 'dynamic'
        )
        
        # ����ȫ��ָ��
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
        ��������ָ��
        
        Args:
            flow: ����
            mask: ��������
            normalization_factor: ��һ������
            region_type: �������� ('static' �� 'dynamic')
            
        Returns:
            ����ָ���ֵ�
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
        
        # ��ȡ�������
        flow_x = flow[:, :, 0][mask]
        flow_y = flow[:, :, 1][mask]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # ��һ��
        if self.use_normalized_flow and normalization_factor > 0:
            magnitude = magnitude / normalization_factor
        
        # ����ͳ����
        mean_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        std_mag = np.std(magnitude)
        median_mag = np.median(magnitude)
        
        # �����������ͼ������
        if region_type == 'static':
            # ��̬���������в����ԽСԽ��
            # �ۺϿ��Ǿ�ֵ�ͱ�׼��
            score = mean_mag + 0.5 * std_mag
        else:
            # ��̬���򣺹�ע�˶�ǿ��
            # �ۺϿ��Ǿ�ֵ����׼��ͷ�ֵ
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
        """����ȫ��ָ��"""
        
        # ȫ��ͳ��
        global_mean = np.mean(flow_magnitude_norm)
        global_std = np.std(flow_magnitude_norm)
        global_max = np.max(flow_magnitude_norm)
        
        # �˶�������
        motion_coverage = np.sum(dynamic_mask) / dynamic_mask.size
        
        # һ���Է���
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
        �жϳ�������
        
        �����߼���
        - �����̬����ռ�ȸ��Ҷ�̬ǿ������ �� ��̬����
        - �����̬����Ĳв��С �� ��̬����
        - ��������ĸ�ָ����������ж�
        
        Returns:
            'static' �� 'dynamic'
        """
        
        if not self.scene_auto_detect:
            # ��������Զ���⣬Ĭ��Ϊ��̬����
            return 'dynamic'
        
        # ͳ��ָ��
        avg_dynamic_ratio = np.mean([r['dynamic_ratio'] for r in frame_results])
        avg_dynamic_score = np.mean([r['dynamic_metrics']['score'] for r in frame_results])
        avg_static_score = np.mean([r['static_metrics']['score'] for r in frame_results])
        avg_motion_coverage = np.mean([r['global_metrics']['motion_coverage'] for r in frame_results])
        
        # �ж��߼�
        # ����1: ���ԵĶ�̬����
        if avg_dynamic_ratio > 0.15 and avg_dynamic_score > 0.01:
            return 'dynamic'
        
        # ����2: �˶������ʸ�
        if avg_motion_coverage > 0.2 and avg_dynamic_score > 0.008:
            return 'dynamic'
        
        # ����3: ��̬����в��С
        if avg_static_score < 0.003:
            return 'static'
        
        # ����4: �����ĸ�ָ�������
        if avg_dynamic_score > avg_static_score * 2.5:
            return 'dynamic'
        else:
            return 'static'
    
    def _calculate_final_score(self, 
                               frame_results: List[Dict], 
                               scene_type: str) -> float:
        """
        ��������ͳһ������ʹ������ӳ�䣩
        
        �ؼ��Ľ���
        - ʹ�������� 0-1 ��Χ
        - ��������ֻ������ע�ĸ�����
        - �������ⷴӳ��̬�̶�
        
        Returns:
            0-1 ֮��Ķ�̬�ȷ���
        """
        
        if scene_type == 'static':
            # ��̬��������ע��̬����Ĳв�
            scores = [r['static_metrics']['score'] for r in frame_results]
            mean_score = np.mean(scores)
            unified_score = self._normalize_static_score(mean_score)
            
        else:  # dynamic
            # ��̬��������ע��̬������˶�ǿ��
            scores = [r['dynamic_metrics']['score'] for r in frame_results]
            mean_score = np.mean(scores)
            unified_score = self._normalize_dynamic_score(mean_score)
        
        return float(np.clip(unified_score, 0.0, 1.0))
    
    def _normalize_static_score(self, raw_score: float) -> float:
        """
        ��һ����̬��������������ӳ�䣩
        
        ��̬�����Ĳв����ӳ�䣺
        raw_score | unified_score | ����
        0.0001   -> 0.00         | ������ֹ
        0.001    -> 0.10         | ���Ͳв�
        0.003    -> 0.25         | �Ͳв�
        0.005    -> 0.35         | �еȲв�
        0.01     -> 0.55         | �ϸ߲в�
        0.02     -> 0.75         | �߲в�
        0.05+    -> 1.00         | ���߲в�/�쳣
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
        ��һ����̬��������������ӳ�䣩
        
        ��̬�������������ӳ�䣺
        raw_score | unified_score | ����
        0.005    -> 0.10         | ���弸������ �� ��ɸѡ��
        0.008    -> 0.20         | ������΢����
        0.015    -> 0.35         | ������������
        0.025    -> 0.50         | �����Ծ����
        0.04     -> 0.65         | ������ٶ���
        0.08     -> 0.85         | ������Ҷ���
        0.15+    -> 1.00         | ���弫�Ⱦ���
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
        """����ʱ��ͳ��"""
        
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
        ���ද̬�ȵȼ�
        
        Args:
            score: ͳһ��̬�ȷ��� (0-1)
            scene_type: ��������
            
        Returns:
            ������Ϣ�ֵ�
        """
        
        if score < 0.15:
            category = 'pure_static'
            description = '����̬'
            if scene_type == 'static':
                examples = ['��ȫ��ֹ�Ľ���', '����', '�̶��������ľ���']
            else:
                examples = ['���Ｘ������', '����', '��ֹվ��']
        elif score < 0.35:
            category = 'low_dynamic'
            description = '�Ͷ�̬'
            if scene_type == 'static':
                examples = ['��΢��', 'Ʈ��������', '΢���е���Ҷ']
            else:
                examples = ['�����ƶ�', '��΢��������', '�ֲ�С����']
        elif score < 0.60:
            category = 'medium_dynamic'
            description = '�еȶ�̬'
            if scene_type == 'static':
                examples = ['������', '�ϴ����ҡ��']
            else:
                examples = ['��������', '�ճ��', '��ͨ����']
        elif score < 0.85:
            category = 'high_dynamic'
            description = '�߶�̬'
            if scene_type == 'static':
                examples = ['������', '����ҡ��']
            else:
                examples = ['�ܲ�', '����', '���趯��']
        else:
            category = 'extreme_dynamic'
            description = '���߶�̬'
            if scene_type == 'static':
                examples = ['�����쳣�˶�', '�豸����']
            else:
                examples = ['�����赸', '�����˶�', '�򶷳���']
        
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
        """�������ֽ���"""
        
        classification = self._classify_dynamics(score, scene_type)
        
        if scene_type == 'static':
            scene_desc = "��̬����������������ȣ�"
            metric_desc = f"��̬����в������ֵ: {temporal_stats['mean_magnitude']:.6f}"
        else:
            scene_desc = "��̬�������������ȣ�"
            metric_desc = f"�����������ǿ�Ⱦ�ֵ: {temporal_stats['mean_magnitude']:.6f}"
        
        interpretation = f"""
���������͡�{scene_desc}
����̬�ȷ�����{score:.3f} / 1.000
����̬�ȼ���{classification['description']} ({classification['category']})
������ʾ����{', '.join(classification['typical_examples'])}

������ָ�꡿
- {metric_desc}
- ʱ���ȶ���: {temporal_stats['temporal_stability']:.3f}
- ��̬����ռ��: {temporal_stats['mean_static_ratio']:.1%}
- ��̬����ռ��: {temporal_stats['mean_dynamic_ratio']:.1%}
"""
        return interpretation.strip()
    
    def _get_empty_result(self) -> Dict:
        """���ؿս��"""
        return {
            'unified_dynamics_score': 0.0,
            'scene_type': 'unknown',
            'classification': {
                'category': 'unknown',
                'category_id': -1,
                'description': '��Ч����',
                'typical_examples': [],
                'scene_type': 'unknown'
            },
            'frame_results': [],
            'temporal_stats': {},
            'interpretation': '�޷���������������Ϊ��'
        }

