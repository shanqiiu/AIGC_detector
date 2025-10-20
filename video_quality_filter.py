# -*- coding: utf-8 -*-
"""
视频质量筛选器
用于筛选不符合质量标准的视频
"""

import numpy as np
from typing import List, Dict, Optional


class VideoQualityFilter:
    """视频质量筛选器"""
    
    def __init__(self):
        """初始化筛选器"""
        pass
    
    def filter_low_dynamics_in_dynamic_scenes(self, 
                                             results: List[Dict], 
                                             threshold: float = 0.3) -> List[Dict]:
        """
        筛选动态场景中动态度过低的视频
        
        应用场景：
        - 检测"演员没有按要求做动作"的视频
        - 检测"动作幅度不足"的视频
        - 质量控制：确保动态视频的动作明显
        
        Args:
            results: 视频分析结果列表
            threshold: 动态度阈值（默认0.3）
            
        Returns:
            筛选出的低动态度视频列表
        """
        
        filtered = []
        for result in results:
            # 条件：动态场景 AND 动态度低于阈值
            if (result.get('scene_type') == 'dynamic' and 
                result.get('unified_dynamics_score', 1.0) < threshold):
                filtered.append({
                    'video_name': result.get('video_name', 'unknown'),
                    'video_path': result.get('video_path', ''),
                    'score': result.get('unified_dynamics_score', 0.0),
                    'scene_type': result.get('scene_type', 'unknown'),
                    'classification': result.get('classification', {}),
                    'reason': f"动态场景但动作幅度过小 (score={result.get('unified_dynamics_score', 0.0):.3f} < {threshold})",
                    'recommendation': '建议重新拍摄或增加动作幅度'
                })
        
        return filtered
    
    def filter_high_static_anomaly(self, 
                                   results: List[Dict], 
                                   threshold: float = 0.5) -> List[Dict]:
        """
        筛选静态场景中异常运动过高的视频
        
        应用场景：
        - 检测"相机抖动严重"的视频
        - 检测"相机补偿失败"的视频
        - 质量控制：确保静态视频的稳定性
        
        Args:
            results: 视频分析结果列表
            threshold: 异常阈值（默认0.5）
            
        Returns:
            筛选出的高异常视频列表
        """
        
        filtered = []
        for result in results:
            # 条件：静态场景 AND 动态度高于阈值
            if (result.get('scene_type') == 'static' and 
                result.get('unified_dynamics_score', 0.0) > threshold):
                filtered.append({
                    'video_name': result.get('video_name', 'unknown'),
                    'video_path': result.get('video_path', ''),
                    'score': result.get('unified_dynamics_score', 0.0),
                    'scene_type': result.get('scene_type', 'unknown'),
                    'classification': result.get('classification', {}),
                    'reason': f"静态场景但残差过大 (score={result.get('unified_dynamics_score', 0.0):.3f} > {threshold})",
                    'recommendation': '建议检查相机稳定性或重新拍摄'
                })
        
        return filtered
    
    def filter_by_score_range(self, 
                             results: List[Dict],
                             min_score: float = 0.0,
                             max_score: float = 1.0,
                             scene_type: Optional[str] = None) -> List[Dict]:
        """
        按分数范围筛选视频
        
        Args:
            results: 视频分析结果列表
            min_score: 最小分数
            max_score: 最大分数
            scene_type: 场景类型过滤（None表示不过滤）
            
        Returns:
            符合条件的视频列表
        """
        
        filtered = []
        for result in results:
            score = result.get('unified_dynamics_score', -1)
            s_type = result.get('scene_type', 'unknown')
            
            # 检查分数范围
            if score < min_score or score > max_score:
                continue
            
            # 检查场景类型
            if scene_type is not None and s_type != scene_type:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def filter_by_category(self, 
                          results: List[Dict],
                          categories: List[str]) -> List[Dict]:
        """
        按动态度分类筛选视频
        
        Args:
            results: 视频分析结果列表
            categories: 分类列表，如 ['pure_static', 'low_dynamic']
            
        Returns:
            符合分类的视频列表
        """
        
        filtered = []
        for result in results:
            classification = result.get('classification', {})
            category = classification.get('category', 'unknown')
            
            if category in categories:
                filtered.append(result)
        
        return filtered
    
    def generate_filter_report(self, 
                              all_results: List[Dict],
                              filtered_results: List[Dict],
                              filter_name: str) -> str:
        """
        生成筛选报告
        
        Args:
            all_results: 所有视频结果
            filtered_results: 筛选后的结果
            filter_name: 筛选器名称
            
        Returns:
            报告文本
        """
        
        total_count = len(all_results)
        filtered_count = len(filtered_results)
        filter_rate = filtered_count / total_count if total_count > 0 else 0
        
        report = f"""
{'='*70}
视频质量筛选报告 - {filter_name}
{'='*70}

总视频数: {total_count}
筛选出: {filtered_count} 个视频
筛选率: {filter_rate:.1%}

{'='*70}
筛选结果详情
{'='*70}

"""
        
        for i, result in enumerate(filtered_results, 1):
            report += f"{i}. {result['video_name']}\n"
            report += f"   分数: {result['score']:.3f}\n"
            report += f"   场景: {result['scene_type']}\n"
            report += f"   原因: {result['reason']}\n"
            if 'recommendation' in result:
                report += f"   建议: {result['recommendation']}\n"
            report += "\n"
        
        return report
    
    def get_quality_statistics(self, results: List[Dict]) -> Dict:
        """
        获取质量统计信息
        
        Args:
            results: 视频分析结果列表
            
        Returns:
            统计信息字典
        """
        
        if not results:
            return {}
        
        scores = [r.get('unified_dynamics_score', 0) for r in results]
        
        # 场景类型分布
        scene_types = {}
        for r in results:
            st = r.get('scene_type', 'unknown')
            scene_types[st] = scene_types.get(st, 0) + 1
        
        # 分类分布
        categories = {}
        for r in results:
            cat = r.get('classification', {}).get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_videos': len(results),
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'scene_type_distribution': scene_types,
            'category_distribution': categories
        }

