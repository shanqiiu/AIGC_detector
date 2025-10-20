# -*- coding: utf-8 -*-
"""
统一动态度计算器 (重构版)
自动适应静态场景和动态场景，输出统一的 0-1 动态度分数

核心改进：
1. 同时计算静态区域和动态区域指标
2. 自动判断场景类型
3. 使用统一的评分标准（0-1），不再强制分段
4. 可以筛选出"动态场景中动态度很低"的视频
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


class UnifiedDynamicsCalculator:
    """统一动态度计算器 - 自动适应场景类型"""
    
    def __init__(self, 
                 static_threshold=0.002,   # 静态区域检测阈值（归一化）
                 subject_threshold=0.005,  # 主体区域检测阈值（归一化）
                 use_normalized_flow=True,
                 scene_auto_detect=True):
        """
        初始化计算器
        
        Args:
            static_threshold: 静态区域检测阈值
            subject_threshold: 主体区域检测阈值
            use_normalized_flow: 是否使用分辨率归一化
            scene_auto_detect: 是否自动检测场景类型
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
        计算统一动态度分数
        
        Args:
            flows: 光流列表 (经过相机补偿后的残差光流)
            images: 图像列表
            camera_matrix: 相机内参矩阵（保留用于扩展）
            
        Returns:
            包含统一动态度分数和详细信息的字典
        """
        
        if len(flows) == 0 or len(images) < 2:
            return self._get_empty_result()
        
        # 分析每一帧
        frame_results = []
        for i, flow in enumerate(flows):
            # 归一化因子
            normalization_factor = 1.0
            if self.use_normalized_flow:
                h, w = images[i].shape[:2]
                normalization_factor = np.sqrt(h**2 + w**2)
            
            # 分析单帧
            frame_result = self._analyze_single_frame(
                flow, images[i].shape, normalization_factor
            )
            frame_results.append(frame_result)
        
        # 场景类型判断
        scene_type = self._determine_scene_type(frame_results)
        
        # 计算最终统一分数
        unified_score = self._calculate_final_score(frame_results, scene_type)
        
        # 计算时序统计
        temporal_stats = self._calculate_temporal_stats(frame_results, scene_type)
        
        # 分类动态度等级
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
        分析单帧，同时计算静态区域和动态区域指标
        
        Args:
            flow: 光流 (H, W, 2)
            image_shape: 图像形状
            normalization_factor: 归一化因子
            
        Returns:
            单帧分析结果
        """
        
        # 计算光流幅度
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # 归一化
        if self.use_normalized_flow and normalization_factor > 0:
            flow_magnitude_norm = flow_magnitude / normalization_factor
        else:
            flow_magnitude_norm = flow_magnitude
        
        # 检测静态区域和动态区域
        static_mask = flow_magnitude_norm < self.static_threshold
        dynamic_mask = flow_magnitude_norm > self.subject_threshold
        
        # 形态学处理动态区域（连接邻近区域）
        kernel = np.ones((7, 7), np.uint8)
        dynamic_mask_cleaned = cv2.morphologyEx(
            dynamic_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        dynamic_mask_cleaned = cv2.morphologyEx(
            dynamic_mask_cleaned, cv2.MORPH_OPEN, kernel
        )
        dynamic_mask = dynamic_mask_cleaned.astype(bool)
        
        # 计算静态区域指标（用于静态场景）
        static_metrics = self._calculate_region_metrics(
            flow, static_mask, normalization_factor, 'static'
        )
        
        # 计算动态区域指标（用于动态场景）
        dynamic_metrics = self._calculate_region_metrics(
            flow, dynamic_mask, normalization_factor, 'dynamic'
        )
        
        # 计算全局指标
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
        计算区域指标
        
        Args:
            flow: 光流
            mask: 区域掩码
            normalization_factor: 归一化因子
            region_type: 区域类型 ('static' 或 'dynamic')
            
        Returns:
            区域指标字典
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
        
        # 提取区域光流
        flow_x = flow[:, :, 0][mask]
        flow_y = flow[:, :, 1][mask]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # 归一化
        if self.use_normalized_flow and normalization_factor > 0:
            magnitude = magnitude / normalization_factor
        
        # 计算统计量
        mean_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        std_mag = np.std(magnitude)
        median_mag = np.median(magnitude)
        
        # 根据区域类型计算分数
        if region_type == 'static':
            # 静态区域：期望残差光流越小越好
            # 综合考虑均值和标准差
            score = mean_mag + 0.5 * std_mag
        else:
            # 动态区域：关注运动强度
            # 综合考虑均值、标准差和峰值
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
        """计算全局指标"""
        
        # 全局统计
        global_mean = np.mean(flow_magnitude_norm)
        global_std = np.std(flow_magnitude_norm)
        global_max = np.max(flow_magnitude_norm)
        
        # 运动覆盖率
        motion_coverage = np.sum(dynamic_mask) / dynamic_mask.size
        
        # 一致性分数
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
        判断场景类型
        
        核心逻辑：
        - 如果动态区域占比高且动态强度明显 → 动态场景
        - 如果静态区域的残差很小 → 静态场景
        - 否则根据哪个指标更显著来判断
        
        Returns:
            'static' 或 'dynamic'
        """
        
        if not self.scene_auto_detect:
            # 如果禁用自动检测，默认为动态场景
            return 'dynamic'
        
        # 统计指标
        avg_dynamic_ratio = np.mean([r['dynamic_ratio'] for r in frame_results])
        avg_dynamic_score = np.mean([r['dynamic_metrics']['score'] for r in frame_results])
        avg_static_score = np.mean([r['static_metrics']['score'] for r in frame_results])
        avg_motion_coverage = np.mean([r['global_metrics']['motion_coverage'] for r in frame_results])
        
        # 判断逻辑
        # 条件1: 明显的动态主体
        if avg_dynamic_ratio > 0.15 and avg_dynamic_score > 0.01:
            return 'dynamic'
        
        # 条件2: 运动覆盖率高
        if avg_motion_coverage > 0.2 and avg_dynamic_score > 0.008:
            return 'dynamic'
        
        # 条件3: 静态区域残差很小
        if avg_static_score < 0.003:
            return 'static'
        
        # 条件4: 根据哪个指标更显著
        if avg_dynamic_score > avg_static_score * 2.5:
            return 'dynamic'
        else:
            return 'static'
    
    def _calculate_final_score(self, 
                               frame_results: List[Dict], 
                               scene_type: str) -> float:
        """
        计算最终统一分数（使用线性映射）
        
        关键改进：
        - 使用完整的 0-1 范围
        - 场景类型只决定关注哪个区域
        - 分数纯粹反映动态程度
        
        Returns:
            0-1 之间的动态度分数
        """
        
        if scene_type == 'static':
            # 静态场景：关注静态区域的残差
            scores = [r['static_metrics']['score'] for r in frame_results]
            mean_score = np.mean(scores)
            unified_score = self._normalize_static_score(mean_score)
            
        else:  # dynamic
            # 动态场景：关注动态区域的运动强度
            scores = [r['dynamic_metrics']['score'] for r in frame_results]
            mean_score = np.mean(scores)
            unified_score = self._normalize_dynamic_score(mean_score)
        
        return float(np.clip(unified_score, 0.0, 1.0))
    
    def _normalize_static_score(self, raw_score: float) -> float:
        """
        归一化静态场景分数（线性映射）
        
        静态场景的残差光流映射：
        raw_score | unified_score | 含义
        0.0001   -> 0.00         | 完美静止
        0.001    -> 0.10         | 极低残差
        0.003    -> 0.25         | 低残差
        0.005    -> 0.35         | 中等残差
        0.01     -> 0.55         | 较高残差
        0.02     -> 0.75         | 高残差
        0.05+    -> 1.00         | 极高残差/异常
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
        归一化动态场景分数（线性映射）
        
        动态场景的主体光流映射：
        raw_score | unified_score | 含义
        0.005    -> 0.10         | 主体几乎不动 ← 可筛选！
        0.008    -> 0.20         | 主体轻微动作
        0.015    -> 0.35         | 主体正常动作
        0.025    -> 0.50         | 主体活跃动作
        0.04     -> 0.65         | 主体快速动作
        0.08     -> 0.85         | 主体剧烈动作
        0.15+    -> 1.00         | 主体极度剧烈
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
        """计算时序统计"""
        
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
        分类动态度等级
        
        Args:
            score: 统一动态度分数 (0-1)
            scene_type: 场景类型
            
        Returns:
            分类信息字典
        """
        
        if score < 0.15:
            category = 'pure_static'
            description = '纯静态'
            if scene_type == 'static':
                examples = ['完全静止的建筑', '雕塑', '固定相机拍摄的静物']
            else:
                examples = ['人物几乎不动', '静坐', '静止站立']
        elif score < 0.35:
            category = 'low_dynamic'
            description = '低动态'
            if scene_type == 'static':
                examples = ['轻微振动', '飘动的旗帜', '微风中的树叶']
            else:
                examples = ['缓慢移动', '轻微调整姿势', '手部小动作']
        elif score < 0.60:
            category = 'medium_dynamic'
            description = '中等动态'
            if scene_type == 'static':
                examples = ['明显振动', '较大幅度摇动']
            else:
                examples = ['正常行走', '日常活动', '普通手势']
        elif score < 0.85:
            category = 'high_dynamic'
            description = '高动态'
            if scene_type == 'static':
                examples = ['剧烈振动', '快速摇动']
            else:
                examples = ['跑步', '跳舞', '挥舞动作']
        else:
            category = 'extreme_dynamic'
            description = '极高动态'
            if scene_type == 'static':
                examples = ['严重异常运动', '设备故障']
            else:
                examples = ['快速舞蹈', '体育运动', '打斗场面']
        
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
        """生成文字解释"""
        
        classification = self._classify_dynamics(score, scene_type)
        
        if scene_type == 'static':
            scene_desc = "静态场景（建筑、静物等）"
            metric_desc = f"静态区域残差光流均值: {temporal_stats['mean_magnitude']:.6f}"
        else:
            scene_desc = "动态场景（人物、动物等）"
            metric_desc = f"主体区域光流强度均值: {temporal_stats['mean_magnitude']:.6f}"
        
        interpretation = f"""
【场景类型】{scene_desc}
【动态度分数】{score:.3f} / 1.000
【动态等级】{classification['description']} ({classification['category']})
【典型示例】{', '.join(classification['typical_examples'])}

【技术指标】
- {metric_desc}
- 时序稳定性: {temporal_stats['temporal_stability']:.3f}
- 静态区域占比: {temporal_stats['mean_static_ratio']:.1%}
- 动态区域占比: {temporal_stats['mean_dynamic_ratio']:.1%}
"""
        return interpretation.strip()
    
    def _get_empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'unified_dynamics_score': 0.0,
            'scene_type': 'unknown',
            'classification': {
                'category': 'unknown',
                'category_id': -1,
                'description': '无效数据',
                'typical_examples': [],
                'scene_type': 'unknown'
            },
            'frame_results': [],
            'temporal_stats': {},
            'interpretation': '无法分析：输入数据为空'
        }

