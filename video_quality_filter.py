# -*- coding: utf-8 -*-
"""
��Ƶ����ɸѡ��
����ɸѡ������������׼����Ƶ
"""

import numpy as np
from typing import List, Dict, Optional


class VideoQualityFilter:
    """��Ƶ����ɸѡ��"""
    
    def __init__(self):
        """��ʼ��ɸѡ��"""
        pass
    
    def filter_low_dynamics_in_dynamic_scenes(self, 
                                             results: List[Dict], 
                                             threshold: float = 0.3) -> List[Dict]:
        """
        ɸѡ��̬�����ж�̬�ȹ��͵���Ƶ
        
        Ӧ�ó�����
        - ���"��Աû�а�Ҫ��������"����Ƶ
        - ���"�������Ȳ���"����Ƶ
        - �������ƣ�ȷ����̬��Ƶ�Ķ�������
        
        Args:
            results: ��Ƶ��������б�
            threshold: ��̬����ֵ��Ĭ��0.3��
            
        Returns:
            ɸѡ���ĵͶ�̬����Ƶ�б�
        """
        
        filtered = []
        for result in results:
            # ��������̬���� AND ��̬�ȵ�����ֵ
            if (result.get('scene_type') == 'dynamic' and 
                result.get('unified_dynamics_score', 1.0) < threshold):
                filtered.append({
                    'video_name': result.get('video_name', 'unknown'),
                    'video_path': result.get('video_path', ''),
                    'score': result.get('unified_dynamics_score', 0.0),
                    'scene_type': result.get('scene_type', 'unknown'),
                    'classification': result.get('classification', {}),
                    'reason': f"��̬�������������ȹ�С (score={result.get('unified_dynamics_score', 0.0):.3f} < {threshold})",
                    'recommendation': '����������������Ӷ�������'
                })
        
        return filtered
    
    def filter_high_static_anomaly(self, 
                                   results: List[Dict], 
                                   threshold: float = 0.5) -> List[Dict]:
        """
        ɸѡ��̬�������쳣�˶����ߵ���Ƶ
        
        Ӧ�ó�����
        - ���"�����������"����Ƶ
        - ���"�������ʧ��"����Ƶ
        - �������ƣ�ȷ����̬��Ƶ���ȶ���
        
        Args:
            results: ��Ƶ��������б�
            threshold: �쳣��ֵ��Ĭ��0.5��
            
        Returns:
            ɸѡ���ĸ��쳣��Ƶ�б�
        """
        
        filtered = []
        for result in results:
            # ��������̬���� AND ��̬�ȸ�����ֵ
            if (result.get('scene_type') == 'static' and 
                result.get('unified_dynamics_score', 0.0) > threshold):
                filtered.append({
                    'video_name': result.get('video_name', 'unknown'),
                    'video_path': result.get('video_path', ''),
                    'score': result.get('unified_dynamics_score', 0.0),
                    'scene_type': result.get('scene_type', 'unknown'),
                    'classification': result.get('classification', {}),
                    'reason': f"��̬�������в���� (score={result.get('unified_dynamics_score', 0.0):.3f} > {threshold})",
                    'recommendation': '����������ȶ��Ի���������'
                })
        
        return filtered
    
    def filter_by_score_range(self, 
                             results: List[Dict],
                             min_score: float = 0.0,
                             max_score: float = 1.0,
                             scene_type: Optional[str] = None) -> List[Dict]:
        """
        ��������Χɸѡ��Ƶ
        
        Args:
            results: ��Ƶ��������б�
            min_score: ��С����
            max_score: ������
            scene_type: �������͹��ˣ�None��ʾ�����ˣ�
            
        Returns:
            ������������Ƶ�б�
        """
        
        filtered = []
        for result in results:
            score = result.get('unified_dynamics_score', -1)
            s_type = result.get('scene_type', 'unknown')
            
            # ��������Χ
            if score < min_score or score > max_score:
                continue
            
            # ��鳡������
            if scene_type is not None and s_type != scene_type:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def filter_by_category(self, 
                          results: List[Dict],
                          categories: List[str]) -> List[Dict]:
        """
        ����̬�ȷ���ɸѡ��Ƶ
        
        Args:
            results: ��Ƶ��������б�
            categories: �����б��� ['pure_static', 'low_dynamic']
            
        Returns:
            ���Ϸ������Ƶ�б�
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
        ����ɸѡ����
        
        Args:
            all_results: ������Ƶ���
            filtered_results: ɸѡ��Ľ��
            filter_name: ɸѡ������
            
        Returns:
            �����ı�
        """
        
        total_count = len(all_results)
        filtered_count = len(filtered_results)
        filter_rate = filtered_count / total_count if total_count > 0 else 0
        
        report = f"""
{'='*70}
��Ƶ����ɸѡ���� - {filter_name}
{'='*70}

����Ƶ��: {total_count}
ɸѡ��: {filtered_count} ����Ƶ
ɸѡ��: {filter_rate:.1%}

{'='*70}
ɸѡ�������
{'='*70}

"""
        
        for i, result in enumerate(filtered_results, 1):
            report += f"{i}. {result['video_name']}\n"
            report += f"   ����: {result['score']:.3f}\n"
            report += f"   ����: {result['scene_type']}\n"
            report += f"   ԭ��: {result['reason']}\n"
            if 'recommendation' in result:
                report += f"   ����: {result['recommendation']}\n"
            report += "\n"
        
        return report
    
    def get_quality_statistics(self, results: List[Dict]) -> Dict:
        """
        ��ȡ����ͳ����Ϣ
        
        Args:
            results: ��Ƶ��������б�
            
        Returns:
            ͳ����Ϣ�ֵ�
        """
        
        if not results:
            return {}
        
        scores = [r.get('unified_dynamics_score', 0) for r in results]
        
        # �������ͷֲ�
        scene_types = {}
        for r in results:
            st = r.get('scene_type', 'unknown')
            scene_types[st] = scene_types.get(st, 0) + 1
        
        # ����ֲ�
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

