# -*- coding: utf-8 -*-
"""
BadCase检测器 - 筛选劣质AIGC生成视频

检测两类劣质视频：
1. 期望静态但动态度高（如建筑物抖动）
2. 期望动态但动态度低（如演唱会大屏幕静止）
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import json


class BadCaseDetector:
    """
    劣质视频检测器
    
    核心逻辑：比较期望动态度和实际动态度，检测不匹配情况
    """
    
    def __init__(self,
                 mismatch_threshold: float = 0.3,
                 confidence_threshold: float = 0.6):
        """
        初始化BadCase检测器
        
        Args:
            mismatch_threshold: 不匹配阈值（期望与实际的差异）
            confidence_threshold: 最低置信度要求
        """
        self.mismatch_threshold = mismatch_threshold
        self.confidence_threshold = confidence_threshold
    
    def detect_badcase(self,
                      actual_score: float,
                      expected_label: Union[str, float],
                      confidence: float = 1.0,
                      video_info: Optional[Dict] = None) -> Dict:
        """
        检测单个视频的BadCase
        
        Args:
            actual_score: 实际动态度分数（0-1）
            expected_label: 期望标签
                - 'static' 或 0: 期望静态
                - 'dynamic' 或 1: 期望动态
                - 0.0-1.0: 期望的具体动态度分数
            confidence: 检测结果的置信度
            video_info: 视频额外信息（可选）
            
        Returns:
            {
                'is_badcase': bool,
                'badcase_type': str,
                'mismatch_score': float,
                'severity': str,
                'description': str,
                'suggestion': str
            }
        """
        
        # 转换期望标签为数值
        expected_score = self._parse_expected_label(expected_label)
        
        # 计算不匹配程度
        mismatch_score = abs(actual_score - expected_score)
        
        # 判断是否为BadCase
        is_badcase = (mismatch_score >= self.mismatch_threshold and 
                     confidence >= self.confidence_threshold)
        
        # 确定BadCase类型
        badcase_type = self._classify_badcase_type(
            actual_score, expected_score, mismatch_score
        )
        
        # 评估严重程度
        severity = self._evaluate_severity(mismatch_score)
        
        # 生成描述和建议
        description = self._generate_description(
            actual_score, expected_score, badcase_type, video_info
        )
        
        suggestion = self._generate_suggestion(badcase_type, mismatch_score)
        
        return {
            'is_badcase': is_badcase,
            'badcase_type': badcase_type,
            'mismatch_score': float(mismatch_score),
            'severity': severity,
            'expected_score': float(expected_score),
            'actual_score': float(actual_score),
            'confidence': float(confidence),
            'description': description,
            'suggestion': suggestion
        }
    
    def _parse_expected_label(self, label: Union[str, float]) -> float:
        """解析期望标签为分数"""
        if isinstance(label, str):
            label_lower = label.lower()
            if label_lower in ['static', 'still', '静态']:
                return 0.0
            elif label_lower in ['dynamic', 'moving', '动态']:
                return 1.0
            else:
                raise ValueError(f"未知的标签类型: {label}")
        else:
            return float(np.clip(label, 0.0, 1.0))
    
    def _classify_badcase_type(self,
                               actual: float,
                               expected: float,
                               mismatch: float) -> str:
        """分类BadCase类型"""
        
        if mismatch < self.mismatch_threshold:
            return 'normal'  # 正常，非BadCase
        
        # 期望静态但实际动态
        if expected < 0.3 and actual > 0.5:
            return 'static_to_dynamic'  # 类型A
        
        # 期望动态但实际静态
        if expected > 0.7 and actual < 0.4:
            return 'dynamic_to_static'  # 类型B
        
        # 期望中等动态，但偏差大
        if 0.3 <= expected <= 0.7:
            if actual > expected + 0.3:
                return 'over_dynamic'
            elif actual < expected - 0.3:
                return 'under_dynamic'
        
        # 一般不匹配
        return 'mismatch'
    
    def _evaluate_severity(self, mismatch: float) -> str:
        """评估严重程度"""
        if mismatch < self.mismatch_threshold:
            return 'normal'
        elif mismatch < 0.4:
            return 'mild'      # 轻微
        elif mismatch < 0.6:
            return 'moderate'  # 中等
        else:
            return 'severe'    # 严重
    
    def _generate_description(self,
                             actual: float,
                             expected: float,
                             badcase_type: str,
                             video_info: Optional[Dict]) -> str:
        """生成BadCase描述"""
        
        if badcase_type == 'normal':
            return "视频质量正常，动态度符合期望"
        
        video_name = video_info.get('name', '未知视频') if video_info else '未知视频'
        
        descriptions = {
            'static_to_dynamic': f"劣质视频：{video_name} 期望静态（{expected:.2f}）但实际高动态（{actual:.2f}）。"
                                f"可能原因：物体抖动、飘移、相机补偿失败。",
            
            'dynamic_to_static': f"劣质视频：{video_name} 期望动态（{expected:.2f}）但实际低动态（{actual:.2f}）。"
                                f"可能原因：人物僵硬、大屏幕静止、生成失败。",
            
            'over_dynamic': f"劣质视频：{video_name} 动态度过高（期望{expected:.2f}，实际{actual:.2f}）。"
                           f"可能原因：过度运动、抖动、不稳定。",
            
            'under_dynamic': f"劣质视频：{video_name} 动态度过低（期望{expected:.2f}，实际{actual:.2f}）。"
                            f"可能原因：运动不足、卡顿、僵硬。",
            
            'mismatch': f"视频质量异常：{video_name} 动态度不匹配（期望{expected:.2f}，实际{actual:.2f}）。"
        }
        
        return descriptions.get(badcase_type, "未知类型")
    
    def _generate_suggestion(self, badcase_type: str, mismatch: float) -> str:
        """生成改进建议"""
        
        suggestions = {
            'static_to_dynamic': 
                "建议：\n"
                "1. 检查视频稳定性，是否存在抖动\n"
                "2. 验证相机补偿是否正常工作\n"
                "3. 查看可视化结果，定位异常运动区域\n"
                "4. 考虑重新生成视频",
            
            'dynamic_to_static': 
                "建议：\n"
                "1. 检查人物动作是否生成正确\n"
                "2. 查看关键帧，确认是否存在静止画面\n"
                "3. 检查大屏幕等应动态区域是否正常\n"
                "4. 考虑调整生成参数或重新生成",
            
            'over_dynamic': 
                "建议：\n"
                "1. 检查是否有异常抖动或飘移\n"
                "2. 降低生成运动幅度参数\n"
                "3. 增强视频稳定性处理",
            
            'under_dynamic': 
                "建议：\n"
                "1. 检查运动生成是否充分\n"
                "2. 增加运动幅度参数\n"
                "3. 确认关键帧运动连续性",
            
            'normal': "无需改进，质量符合预期",
            
            'mismatch': "建议人工review，确认是否需要重新生成"
        }
        
        return suggestions.get(badcase_type, "建议人工检查")
    
    def batch_detect(self,
                    results: List[Dict],
                    expected_labels: List[Union[str, float]],
                    video_names: Optional[List[str]] = None) -> Dict:
        """
        批量检测BadCase
        
        Args:
            results: 视频处理结果列表（每个包含unified_dynamics）
            expected_labels: 期望标签列表（与results对应）
            video_names: 视频名称列表（可选）
            
        Returns:
            {
                'badcase_list': [...],
                'badcase_count': int,
                'badcase_rate': float,
                'severity_distribution': {...},
                'type_distribution': {...}
            }
        """
        
        if len(results) != len(expected_labels):
            raise ValueError("结果数量和标签数量不匹配")
        
        badcase_list = []
        severity_count = {'mild': 0, 'moderate': 0, 'severe': 0}
        type_count = {}
        
        for i, (result, expected) in enumerate(zip(results, expected_labels)):
            # 获取实际分数和置信度
            unified = result.get('unified_dynamics', {})
            actual_score = unified.get('unified_dynamics_score', 0.5)
            confidence = unified.get('confidence', 0.0)
            
            # 准备视频信息
            video_info = {
                'name': video_names[i] if video_names and i < len(video_names) else f'video_{i}',
                'index': i
            }
            
            # 检测BadCase
            detection = self.detect_badcase(
                actual_score, expected, confidence, video_info
            )
            
            # 如果是BadCase，加入列表
            if detection['is_badcase']:
                detection['video_name'] = video_info['name']
                detection['video_index'] = i
                badcase_list.append(detection)
                
                # 统计
                severity_count[detection['severity']] = severity_count.get(detection['severity'], 0) + 1
                type_count[detection['badcase_type']] = type_count.get(detection['badcase_type'], 0) + 1
        
        total_videos = len(results)
        badcase_count = len(badcase_list)
        badcase_rate = badcase_count / total_videos if total_videos > 0 else 0.0
        
        return {
            'badcase_list': badcase_list,
            'badcase_count': badcase_count,
            'total_videos': total_videos,
            'badcase_rate': float(badcase_rate),
            'severity_distribution': severity_count,
            'type_distribution': type_count,
            'normal_count': total_videos - badcase_count
        }
    
    def generate_badcase_report(self, batch_result: Dict) -> str:
        """生成BadCase报告"""
        
        report = f"""
{'='*70}
劣质视频检测报告 (BadCase Detection Report)
{'='*70}

总体统计:
- 总视频数: {batch_result['total_videos']}
- BadCase数量: {batch_result['badcase_count']}
- BadCase比例: {batch_result['badcase_rate']:.1%}
- 正常视频数: {batch_result['normal_count']}

严重程度分布:
- 轻微 (Mild): {batch_result['severity_distribution'].get('mild', 0)}
- 中等 (Moderate): {batch_result['severity_distribution'].get('moderate', 0)}
- 严重 (Severe): {batch_result['severity_distribution'].get('severe', 0)}

BadCase类型分布:
"""
        
        for btype, count in batch_result['type_distribution'].items():
            type_names = {
                'static_to_dynamic': '期望静态→实际动态',
                'dynamic_to_static': '期望动态→实际静态',
                'over_dynamic': '动态度过高',
                'under_dynamic': '动态度过低',
                'mismatch': '一般不匹配'
            }
            report += f"- {type_names.get(btype, btype)}: {count}\n"
        
        report += f"\n{'='*70}\nBadCase详细列表:\n{'='*70}\n\n"
        
        # 按严重程度排序
        sorted_badcases = sorted(
            batch_result['badcase_list'],
            key=lambda x: {'severe': 3, 'moderate': 2, 'mild': 1}.get(x['severity'], 0),
            reverse=True
        )
        
        for i, badcase in enumerate(sorted_badcases, 1):
            report += f"{i}. {badcase['video_name']}\n"
            report += f"   类型: {badcase['badcase_type']}\n"
            report += f"   严重程度: {badcase['severity']}\n"
            report += f"   期望动态度: {badcase['expected_score']:.3f}\n"
            report += f"   实际动态度: {badcase['actual_score']:.3f}\n"
            report += f"   不匹配度: {badcase['mismatch_score']:.3f}\n"
            report += f"   描述: {badcase['description']}\n"
            report += f"   {badcase['suggestion']}\n"
            report += f"\n"
        
        return report
    
    def filter_badcases(self,
                       batch_result: Dict,
                       severity_levels: Optional[List[str]] = None,
                       badcase_types: Optional[List[str]] = None) -> List[Dict]:
        """
        筛选特定类型的BadCase
        
        Args:
            batch_result: batch_detect的返回结果
            severity_levels: 严重程度过滤（如['moderate', 'severe']）
            badcase_types: BadCase类型过滤（如['static_to_dynamic']）
            
        Returns:
            筛选后的BadCase列表
        """
        
        filtered = batch_result['badcase_list']
        
        if severity_levels:
            filtered = [b for b in filtered if b['severity'] in severity_levels]
        
        if badcase_types:
            filtered = [b for b in filtered if b['badcase_type'] in badcase_types]
        
        return filtered


class BadCaseAnalyzer:
    """
    BadCase分析器 - 提供更详细的分析和诊断
    """
    
    def __init__(self):
        self.detector = BadCaseDetector()
    
    def analyze_with_details(self,
                            result: Dict,
                            expected_label: Union[str, float]) -> Dict:
        """
        详细分析单个视频的BadCase
        
        包含组件分数、时序分析等详细信息
        """
        
        unified = result.get('unified_dynamics', {})
        actual_score = unified.get('unified_dynamics_score', 0.5)
        confidence = unified.get('confidence', 0.0)
        component_scores = unified.get('component_scores', {})
        
        # 基础检测
        detection = self.detector.detect_badcase(
            actual_score, expected_label, confidence
        )
        
        # 如果是BadCase，添加详细诊断
        if detection['is_badcase']:
            diagnosis = self._diagnose_root_cause(
                component_scores,
                result.get('temporal_stats', {}),
                detection['badcase_type']
            )
            detection['diagnosis'] = diagnosis
        
        return detection
    
    def _diagnose_root_cause(self,
                            component_scores: Dict,
                            temporal_stats: Dict,
                            badcase_type: str) -> Dict:
        """诊断根本原因"""
        
        diagnosis = {
            'primary_issue': '',
            'contributing_factors': [],
            'detailed_analysis': {}
        }
        
        # 找出异常的维度
        if badcase_type == 'static_to_dynamic':
            # 期望静态但动态度高，检查哪个维度贡献最大
            max_component = max(component_scores, key=component_scores.get)
            diagnosis['primary_issue'] = f"{max_component} 异常偏高"
            
            if component_scores.get('flow_magnitude', 0) > 0.6:
                diagnosis['contributing_factors'].append('光流幅度过大（可能有抖动或飘移）')
            
            if component_scores.get('spatial_coverage', 0) > 0.5:
                diagnosis['contributing_factors'].append('运动区域覆盖广（全局性运动）')
            
            if component_scores.get('camera_factor', 0) > 0.5:
                diagnosis['contributing_factors'].append('相机补偿失败率高（特征匹配问题）')
        
        elif badcase_type == 'dynamic_to_static':
            # 期望动态但动态度低
            diagnosis['primary_issue'] = '整体运动不足'
            
            if component_scores.get('flow_magnitude', 0) < 0.3:
                diagnosis['contributing_factors'].append('光流幅度过小（运动幅度不足）')
            
            if component_scores.get('spatial_coverage', 0) < 0.4:
                diagnosis['contributing_factors'].append('运动区域覆盖小（局部静止）')
            
            if component_scores.get('temporal_variation', 0) < 0.3:
                diagnosis['contributing_factors'].append('时序变化小（运动单调或静止）')
        
        # 添加时序统计诊断
        if temporal_stats:
            diagnosis['detailed_analysis'] = {
                'mean_dynamics': temporal_stats.get('mean_dynamics_score', 0),
                'std_dynamics': temporal_stats.get('std_dynamics_score', 0),
                'stability': temporal_stats.get('temporal_stability', 0),
                'static_ratio': temporal_stats.get('mean_static_ratio', 0)
            }
        
        return diagnosis
    
    def generate_batch_summary(self, results: List[Dict]) -> Dict:
        """
        生成批量处理的BadCase统计摘要
        
        Args:
            results: 视频处理结果列表，每个包含:
                - video_name: 视频名称
                - video_path: 视频路径
                - status: 处理状态
                - is_badcase: 是否为BadCase
                - badcase_type: BadCase类型
                - severity: 严重程度
                - mismatch_score: 不匹配度
                - expected_label: 期望标签
                - actual_score: 实际动态度
                - confidence: 置信度
                
        Returns:
            统计摘要字典
        """
        
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {'error': '没有成功处理的视频'}
        
        # 统计BadCase
        badcases = [r for r in successful_results if r.get('is_badcase', False)]
        normal = [r for r in successful_results if not r.get('is_badcase', False)]
        
        # 按类型统计
        type_count = {}
        severity_count = {'mild': 0, 'moderate': 0, 'severe': 0}
        
        for bc in badcases:
            btype = bc.get('badcase_type', 'unknown')
            type_count[btype] = type_count.get(btype, 0) + 1
            severity = bc.get('severity', 'normal')
            if severity in severity_count:
                severity_count[severity] += 1
        
        return {
            'total_videos': len(results),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'badcase_count': len(badcases),
            'normal_count': len(normal),
            'badcase_rate': len(badcases) / len(successful_results) if successful_results else 0.0,
            'type_distribution': type_count,
            'severity_distribution': severity_count,
            'badcase_list': badcases,
            'normal_list': normal
        }
    
    def save_batch_report(self, summary: Dict, results: List[Dict], output_dir: str):
        """
        保存批量处理的完整报告
        
        统一保存：文本报告 + JSON结果 + BadCase视频列表
        
        Args:
            summary: generate_batch_summary() 生成的统计摘要
            results: 原始结果列表
            output_dir: 输出目录
        """
        import os
        
        # 1. 保存文本报告
        report_path = os.path.join(output_dir, 'badcase_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("劣质视频检测总结 (BadCase Detection Summary)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"总视频数: {summary['total_videos']}\n")
            f.write(f"成功处理: {summary['successful']}\n")
            f.write(f"处理失败: {summary['failed']}\n")
            f.write(f"\nBadCase数量: {summary['badcase_count']}\n")
            f.write(f"正常视频数: {summary['normal_count']}\n")
            f.write(f"BadCase比例: {summary['badcase_rate']:.1%}\n\n")
            
            f.write("="*70 + "\n")
            f.write("BadCase类型分布:\n")
            f.write("="*70 + "\n")
            type_names = {
                'static_to_dynamic': '期望静态→实际动态（如建筑抖动）',
                'dynamic_to_static': '期望动态→实际静态（如屏幕静止）',
                'over_dynamic': '动态度过高',
                'under_dynamic': '动态度过低'
            }
            for btype, count in summary['type_distribution'].items():
                f.write(f"  {type_names.get(btype, btype)}: {count}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("严重程度分布:\n")
            f.write("="*70 + "\n")
            for severity, count in summary['severity_distribution'].items():
                f.write(f"  {severity}: {count}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("BadCase详细列表:\n")
            f.write("="*70 + "\n\n")
            
            # 按严重程度排序
            sorted_badcases = sorted(
                summary['badcase_list'],
                key=lambda x: {'severe': 3, 'moderate': 2, 'mild': 1}.get(x.get('severity', 'normal'), 0),
                reverse=True
            )
            
            for i, bc in enumerate(sorted_badcases, 1):
                f.write(f"{i}. {bc['video_name']}\n")
                f.write(f"   类型: {bc['badcase_type']}\n")
                f.write(f"   严重程度: {bc['severity']}\n")
                f.write(f"   期望: {bc['expected_label']}\n")
                f.write(f"   实际动态度: {bc['actual_score']:.3f}\n")
                f.write(f"   不匹配度: {bc['mismatch_score']:.3f}\n")
                f.write(f"   置信度: {bc['confidence']:.1%}\n")
                f.write(f"\n")
        
        # 2. 保存JSON（精简版）
        json_path = os.path.join(output_dir, 'badcase_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            summary_compact = {
                'total_videos': summary['total_videos'],
                'successful': summary['successful'],
                'failed': summary['failed'],
                'badcase_count': summary['badcase_count'],
                'normal_count': summary['normal_count'],
                'badcase_rate': summary['badcase_rate'],
                'type_distribution': summary['type_distribution'],
                'severity_distribution': summary['severity_distribution'],
                'badcase_list': [
                    {k: v for k, v in bc.items() if k != 'full_result'}
                    for bc in summary['badcase_list']
                ]
            }
            json.dump(summary_compact, f, indent=2, ensure_ascii=False)
        
        # 3. 保存BadCase视频列表（便于筛选）
        badcase_videos_path = os.path.join(output_dir, 'badcase_videos.txt')
        sorted_badcases = sorted(
            summary['badcase_list'],
            key=lambda x: {'severe': 3, 'moderate': 2, 'mild': 1}.get(x.get('severity', 'normal'), 0),
            reverse=True
        )
        with open(badcase_videos_path, 'w', encoding='utf-8') as f:
            for bc in sorted_badcases:
                f.write(f"{bc['video_path']}\n")
        
        print(f"\nBadCase总结已保存:")
        print(f"  - 文本报告: {report_path}")
        print(f"  - JSON结果: {json_path}")
        print(f"  - 视频列表: {badcase_videos_path}")
    
    def export_badcase_list(self,
                           batch_result: Dict,
                           output_path: str,
                           format: str = 'json'):
        """导出BadCase列表"""
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, indent=2, ensure_ascii=False)
        
        elif format == 'txt':
            report = self.detector.generate_badcase_report(batch_result)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'video_name', 'badcase_type', 'severity',
                    'expected_score', 'actual_score', 'mismatch_score'
                ])
                
                for bc in batch_result['badcase_list']:
                    writer.writerow([
                        bc['video_name'],
                        bc['badcase_type'],
                        bc['severity'],
                        bc['expected_score'],
                        bc['actual_score'],
                        bc['mismatch_score']
                    ])
        
        print(f"BadCase列表已导出到: {output_path}")


class QualityFilter:
    """
    质量过滤器 - 基于BadCase检测筛选视频
    """
    
    def __init__(self,
                 accept_mismatch: float = 0.3,
                 min_confidence: float = 0.6):
        """
        初始化过滤器
        
        Args:
            accept_mismatch: 可接受的不匹配度
            min_confidence: 最低置信度要求
        """
        self.detector = BadCaseDetector(
            mismatch_threshold=accept_mismatch,
            confidence_threshold=min_confidence
        )
    
    def filter_videos(self,
                     video_results: List[Tuple[str, Dict, Union[str, float]]],
                     keep_mode: str = 'good') -> Tuple[List[str], List[str]]:
        """
        过滤视频
        
        Args:
            video_results: [(video_path, result, expected_label), ...]
            keep_mode: 
                - 'good': 保留正常视频，过滤BadCase
                - 'bad': 保留BadCase，过滤正常视频
                - 'all': 返回所有视频（仅标记）
                
        Returns:
            (kept_videos, filtered_videos)
        """
        
        kept = []
        filtered = []
        
        for video_path, result, expected in video_results:
            unified = result.get('unified_dynamics', {})
            actual_score = unified.get('unified_dynamics_score', 0.5)
            confidence = unified.get('confidence', 0.0)
            
            detection = self.detector.detect_badcase(
                actual_score, expected, confidence
            )
            
            is_badcase = detection['is_badcase']
            
            if keep_mode == 'good':
                if not is_badcase:
                    kept.append(video_path)
                else:
                    filtered.append(video_path)
            
            elif keep_mode == 'bad':
                if is_badcase:
                    kept.append(video_path)
                else:
                    filtered.append(video_path)
            
            else:  # 'all'
                kept.append((video_path, is_badcase))
        
        return kept, filtered

