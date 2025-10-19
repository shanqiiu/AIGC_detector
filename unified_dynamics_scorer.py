# -*- coding: utf-8 -*-
"""
统一动态度评分器 - Unified Dynamics Scorer
整合多维度指标，输出0-1标准化动态度分数
适用于所有类型的视频（静态物体到动态人物）
"""

import numpy as np
from typing import Dict, List, Optional
import warnings


class UnifiedDynamicsScorer:
    """
    统一动态度评分器
    
    输出标准化分数: 0.0 (纯静态) ~ 1.0 (高动态)
    
    整合指标:
    - 光流幅度 (运动强度)
    - 静态区域比例 (场景组成)
    - 时序稳定性 (时间一致性)
    - 空间一致性 (运动均匀性)
    - 相机补偿效果 (多视角场景)
    """
    
    def __init__(self,
                 mode: str = 'auto',
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None):
        """
        初始化评分器
        
        Args:
            mode: 评估模式
                - 'auto': 自动检测场景类型
                - 'static_scene': 静态场景模式（有相机运动）
                - 'dynamic_scene': 动态场景模式（人物运动）
            weights: 各指标权重，如 {'flow': 0.4, 'spatial': 0.3, ...}
            thresholds: 归一化阈值，用于sigmoid函数
        """
        self.mode = mode
        
        # 默认权重配置
        self.default_weights = {
            'flow_magnitude': 0.35,      # 光流幅度（最重要）
            'spatial_coverage': 0.25,    # 运动覆盖范围
            'temporal_variation': 0.20,  # 时序变化
            'spatial_consistency': 0.10, # 空间一致性
            'camera_factor': 0.10        # 相机补偿因子
        }
        
        self.weights = weights if weights is not None else self.default_weights
        
        # 默认归一化阈值（用于sigmoid函数）
        self.default_thresholds = {
            'flow_low': 1.0,      # 低动态阈值（像素/帧）
            'flow_mid': 5.0,      # 中等动态阈值
            'flow_high': 15.0,    # 高动态阈值
            'static_ratio': 0.5,  # 静态区域判断阈值
        }
        
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds
    
    def calculate_unified_score(self, 
                                temporal_result: Dict,
                                camera_compensation_enabled: bool = False) -> Dict:
        """
        计算统一动态度分数
        
        Args:
            temporal_result: video_processor返回的时序分析结果
            camera_compensation_enabled: 是否启用了相机补偿
            
        Returns:
            {
                'unified_dynamics_score': 0.0-1.0,  # 统一动态度分数
                'scene_type': 'static'/'dynamic',   # 场景类型
                'confidence': 0.0-1.0,              # 置信度
                'component_scores': {...},          # 各维度分数
                'interpretation': str               # 文字解释
            }
        """
        
        # 提取基础统计
        temporal_stats = temporal_result['temporal_stats']
        frame_results = temporal_result['frame_results']
        
        # 1. 检测场景类型
        scene_type = self._detect_scene_type(temporal_stats, camera_compensation_enabled)
        
        # 2. 计算各维度分数
        component_scores = self._calculate_component_scores(
            temporal_stats, frame_results, camera_compensation_enabled, scene_type
        )
        
        # 3. 加权融合得到最终分数
        unified_score = self._weighted_fusion(component_scores, scene_type)
        
        # 4. 计算置信度
        confidence = self._calculate_confidence(component_scores, temporal_stats)
        
        # 5. 生成解释
        interpretation = self._generate_interpretation(
            unified_score, scene_type, component_scores
        )
        
        return {
            'unified_dynamics_score': float(unified_score),
            'scene_type': scene_type,
            'confidence': float(confidence),
            'component_scores': component_scores,
            'interpretation': interpretation,
            'normalization_params': {
                'mode': self.mode,
                'detected_scene': scene_type,
                'weights_used': self.weights
            }
        }
    
    def _detect_scene_type(self, 
                          temporal_stats: Dict,
                          camera_comp_enabled: bool) -> str:
        """检测场景类型"""
        
        if self.mode == 'static_scene':
            return 'static'
        elif self.mode == 'dynamic_scene':
            return 'dynamic'
        
        # auto模式：自动检测
        mean_static_ratio = temporal_stats['mean_static_ratio']
        
        # 如果启用了相机补偿且静态比例高，判定为静态场景
        if camera_comp_enabled and mean_static_ratio > self.thresholds['static_ratio']:
            return 'static'
        else:
            return 'dynamic'
    
    def _calculate_component_scores(self,
                                    temporal_stats: Dict,
                                    frame_results: List[Dict],
                                    camera_comp_enabled: bool,
                                    scene_type: str) -> Dict:
        """计算各维度分数"""
        
        # 1. 光流幅度分数（核心指标）
        flow_score = self._calculate_flow_magnitude_score(
            temporal_stats, camera_comp_enabled, scene_type
        )
        
        # 2. 空间覆盖分数（运动区域占比）
        spatial_score = self._calculate_spatial_coverage_score(temporal_stats)
        
        # 3. 时序变化分数（运动的时间变化）
        temporal_score = self._calculate_temporal_variation_score(temporal_stats)
        
        # 4. 空间一致性分数（运动的空间均匀性）
        consistency_score = self._calculate_spatial_consistency_score(temporal_stats)
        
        # 5. 相机补偿因子（如果启用）
        camera_score = self._calculate_camera_factor(
            temporal_stats, camera_comp_enabled
        )
        
        return {
            'flow_magnitude': flow_score,
            'spatial_coverage': spatial_score,
            'temporal_variation': temporal_score,
            'spatial_consistency': consistency_score,
            'camera_factor': camera_score
        }
    
    def _calculate_flow_magnitude_score(self,
                                        temporal_stats: Dict,
                                        camera_comp_enabled: bool,
                                        scene_type: str) -> float:
        """
        计算光流幅度分数
        
        对于静态场景：使用dynamics_score（补偿后的残差）
        对于动态场景：使用原始光流幅度
        """
        
        if scene_type == 'static' and camera_comp_enabled:
            # 静态场景：使用补偿后的动态度分数
            raw_value = temporal_stats['mean_dynamics_score']
            # 归一化：0.5 -> 0.0, 2.0 -> 0.5, 5.0 -> 1.0
            score = self._sigmoid_normalize(
                raw_value,
                threshold=self.thresholds['flow_mid'],
                steepness=0.5
            )
        else:
            # 动态场景：估算原始光流幅度
            # 如果有全局统计，使用它；否则用dynamics_score作为代理
            raw_value = temporal_stats.get('mean_flow_magnitude', 
                                          temporal_stats['mean_dynamics_score'] * 2)
            
            # 归一化：5 -> 0.3, 15 -> 0.7, 30 -> 1.0
            score = self._sigmoid_normalize(
                raw_value,
                threshold=self.thresholds['flow_high'],
                steepness=0.3
            )
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_spatial_coverage_score(self, temporal_stats: Dict) -> float:
        """
        计算空间覆盖分数
        
        动态区域占比越大，分数越高
        """
        dynamic_ratio = 1.0 - temporal_stats['mean_static_ratio']
        
        # 已经是0-1范围，直接使用
        return float(np.clip(dynamic_ratio, 0.0, 1.0))
    
    def _calculate_temporal_variation_score(self, temporal_stats: Dict) -> float:
        """
        计算时序变化分数
        
        动态度的时间变化越大，说明运动越丰富
        """
        std_dynamics = temporal_stats['std_dynamics_score']
        
        # 归一化：标准差越大，变化越丰富
        # 0.1 -> 0.1, 0.5 -> 0.5, 2.0 -> 0.9
        score = self._sigmoid_normalize(std_dynamics, threshold=1.0, steepness=1.0)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_spatial_consistency_score(self, temporal_stats: Dict) -> float:
        """
        计算空间一致性分数
        
        一致性低 -> 运动不均匀 -> 更动态
        """
        consistency = temporal_stats['mean_consistency_score']
        
        # 反转：一致性低代表更动态
        score = 1.0 - consistency
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_camera_factor(self,
                                 temporal_stats: Dict,
                                 camera_comp_enabled: bool) -> float:
        """
        计算相机补偿因子
        
        如果启用且成功率低，说明场景可能更动态
        """
        if not camera_comp_enabled:
            return 0.5  # 中性值
        
        # 如果有相机补偿统计
        if 'camera_compensation_stats' in temporal_stats:
            comp_stats = temporal_stats['camera_compensation_stats']
            success_rate = comp_stats.get('success_rate', 0.5)
            
            # 成功率低 -> 更动态
            score = 1.0 - success_rate
            return float(np.clip(score, 0.0, 1.0))
        
        return 0.5
    
    def _sigmoid_normalize(self, 
                          value: float,
                          threshold: float = 5.0,
                          steepness: float = 0.5) -> float:
        """
        Sigmoid归一化函数
        
        将任意范围的值映射到0-1
        
        Args:
            value: 输入值
            threshold: 中点阈值（sigmoid的x0）
            steepness: 陡峭度（越大越陡）
        """
        return 1.0 / (1.0 + np.exp(-steepness * (value - threshold)))
    
    def _weighted_fusion(self,
                        component_scores: Dict,
                        scene_type: str) -> float:
        """
        加权融合各维度分数
        
        根据场景类型调整权重
        """
        
        # 根据场景类型调整权重
        if scene_type == 'static':
            # 静态场景：更关注残差光流和时序稳定性
            adjusted_weights = {
                'flow_magnitude': 0.40,
                'spatial_coverage': 0.20,
                'temporal_variation': 0.15,
                'spatial_consistency': 0.15,
                'camera_factor': 0.10
            }
        else:
            # 动态场景：更关注光流幅度和空间覆盖
            adjusted_weights = {
                'flow_magnitude': 0.45,
                'spatial_coverage': 0.30,
                'temporal_variation': 0.15,
                'spatial_consistency': 0.05,
                'camera_factor': 0.05
            }
        
        # 加权求和
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, score in component_scores.items():
            weight = adjusted_weights.get(key, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            unified_score = weighted_sum / total_weight
        else:
            unified_score = 0.5
        
        return float(np.clip(unified_score, 0.0, 1.0))
    
    def _calculate_confidence(self,
                             component_scores: Dict,
                             temporal_stats: Dict) -> float:
        """
        计算置信度
        
        基于:
        - 时序稳定性（越稳定越可信）
        - 各维度分数的一致性（分数越接近越可信）
        """
        
        # 1. 时序稳定性贡献
        stability = temporal_stats.get('temporal_stability', 0.5)
        
        # 2. 分数一致性：计算各维度分数的标准差
        scores = list(component_scores.values())
        score_std = np.std(scores)
        consistency = 1.0 / (1.0 + score_std)  # 标准差越小，一致性越高
        
        # 3. 融合
        confidence = 0.6 * stability + 0.4 * consistency
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_interpretation(self,
                                unified_score: float,
                                scene_type: str,
                                component_scores: Dict) -> str:
        """生成文字解释"""
        
        # 动态度等级
        if unified_score < 0.2:
            level = "极低动态（纯静态）"
            emoji = "??"
        elif unified_score < 0.4:
            level = "低动态"
            emoji = "?"
        elif unified_score < 0.6:
            level = "中等动态"
            emoji = "?"
        elif unified_score < 0.8:
            level = "高动态"
            emoji = "?"
        else:
            level = "极高动态"
            emoji = "?"
        
        # 场景类型描述
        scene_desc = "静态场景（相机运动）" if scene_type == 'static' else "动态场景（物体运动）"
        
        # 主要贡献因子
        max_component = max(component_scores, key=component_scores.get)
        component_names = {
            'flow_magnitude': '光流幅度',
            'spatial_coverage': '运动覆盖',
            'temporal_variation': '时序变化',
            'spatial_consistency': '空间一致性',
            'camera_factor': '相机因子'
        }
        main_factor = component_names.get(max_component, max_component)
        
        interpretation = f"""
{emoji} 动态度: {unified_score:.3f} ({level})
场景类型: {scene_desc}
主要贡献: {main_factor} ({component_scores[max_component]:.3f})

分数解释:
- 0.0-0.2: 纯静态物体（如建筑、雕塑）
- 0.2-0.4: 轻微运动（如飘动的旗帜）
- 0.4-0.6: 中等运动（如行走的人）
- 0.6-0.8: 活跃运动（如跑步、舞蹈）
- 0.8-1.0: 剧烈运动（如快速舞蹈、体育运动）
"""
        
        return interpretation.strip()
    
    def batch_calculate(self,
                       results: List[Dict],
                       camera_comp_enabled: bool = False) -> Dict:
        """
        批量计算多个视频的统一动态度分数
        
        Args:
            results: 多个视频的temporal_result列表
            camera_comp_enabled: 是否启用相机补偿
            
        Returns:
            批量统计结果
        """
        
        scores = []
        scene_types = []
        confidences = []
        
        for result in results:
            unified_result = self.calculate_unified_score(
                result, camera_comp_enabled
            )
            scores.append(unified_result['unified_dynamics_score'])
            scene_types.append(unified_result['scene_type'])
            confidences.append(unified_result['confidence'])
        
        return {
            'scores': scores,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'scene_type_distribution': {
                'static': scene_types.count('static'),
                'dynamic': scene_types.count('dynamic')
            },
            'mean_confidence': float(np.mean(confidences))
        }


class DynamicsClassifier:
    """
    动态度分类器
    
    基于统一动态度分数进行分类
    """
    
    def __init__(self,
                 thresholds: Optional[Dict[str, float]] = None):
        """
        初始化分类器
        
        Args:
            thresholds: 分类阈值字典
        """
        self.default_thresholds = {
            'pure_static': 0.15,      # < 0.15: 纯静态
            'low_dynamic': 0.35,      # 0.15-0.35: 低动态
            'medium_dynamic': 0.65,   # 0.35-0.65: 中等动态
            'high_dynamic': 0.85,     # 0.65-0.85: 高动态
            # >= 0.85: 极高动态
        }
        
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds
    
    def classify(self, unified_score: float) -> Dict:
        """
        分类动态度
        
        Returns:
            {
                'category': str,           # 类别名称
                'category_id': int,        # 类别ID (0-4)
                'description': str,        # 描述
                'typical_examples': list   # 典型例子
            }
        """
        
        if unified_score < self.thresholds['pure_static']:
            return {
                'category': 'pure_static',
                'category_id': 0,
                'description': '纯静态物体',
                'typical_examples': ['建筑物', '雕塑', '静止的风景']
            }
        elif unified_score < self.thresholds['low_dynamic']:
            return {
                'category': 'low_dynamic',
                'category_id': 1,
                'description': '低动态场景',
                'typical_examples': ['飘动的旗帜', '缓慢移动的云', '微风中的树叶']
            }
        elif unified_score < self.thresholds['medium_dynamic']:
            return {
                'category': 'medium_dynamic',
                'category_id': 2,
                'description': '中等动态场景',
                'typical_examples': ['行走的人', '慢跑', '日常活动']
            }
        elif unified_score < self.thresholds['high_dynamic']:
            return {
                'category': 'high_dynamic',
                'category_id': 3,
                'description': '高动态场景',
                'typical_examples': ['跑步', '跳舞', '体育运动']
            }
        else:
            return {
                'category': 'extreme_dynamic',
                'category_id': 4,
                'description': '极高动态场景',
                'typical_examples': ['快速舞蹈', '激烈运动', '打斗场面']
            }
    
    def get_binary_label(self, 
                        unified_score: float,
                        threshold: float = 0.5) -> int:
        """
        二分类：静态(0) vs 动态(1)
        
        Args:
            unified_score: 统一动态度分数
            threshold: 分类阈值
            
        Returns:
            0 (静态) 或 1 (动态)
        """
        return 1 if unified_score >= threshold else 0

