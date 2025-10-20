# -*- coding: utf-8 -*-
"""Quick test script for new system"""

import numpy as np
from unified_dynamics_calculator import UnifiedDynamicsCalculator
from video_quality_filter import VideoQualityFilter
from dynamics_config import get_config


def test_unified_calculator():
    """����ͳһ��̬�ȼ�����"""
    print("\n" + "="*70)
    print("����1: ͳһ��̬�ȼ�����")
    print("="*70)
    
    # ����������
    calculator = UnifiedDynamicsCalculator(
        static_threshold=0.002,
        subject_threshold=0.005,
        use_normalized_flow=True
    )
    
    # ģ���������
    h, w = 720, 1280
    num_frames = 10
    
    # ����1: ��̬�������в��С��
    print("\n����1: ��̬������������")
    flows_static = [np.random.randn(h, w, 2) * 0.5 for _ in range(num_frames)]
    images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(num_frames + 1)]
    
    result1 = calculator.calculate_unified_dynamics(flows_static, images)
    print(f"  ��������: {result1['scene_type']}")
    print(f"  ��̬�ȷ���: {result1['unified_dynamics_score']:.3f}")
    print(f"  ����: {result1['classification']['description']}")
    
    # ����2: ��̬�����������˶����ԣ�
    print("\n����2: ��̬���������")
    flows_dynamic = []
    for _ in range(num_frames):
        flow = np.random.randn(h, w, 2) * 0.5
        # ������Ե������˶�����
        flow[300:400, 500:700, :] += np.random.randn(100, 200, 2) * 15.0
        flows_dynamic.append(flow)
    
    result2 = calculator.calculate_unified_dynamics(flows_dynamic, images)
    print(f"  ��������: {result2['scene_type']}")
    print(f"  ��̬�ȷ���: {result2['unified_dynamics_score']:.3f}")
    print(f"  ����: {result2['classification']['description']}")
    
    # ����3: ��̬������������С
    print("\n����3: ��̬������������С")
    flows_low_dynamic = []
    for _ in range(num_frames):
        flow = np.random.randn(h, w, 2) * 0.5
        # �����΢�������˶�
        flow[300:400, 500:700, :] += np.random.randn(100, 200, 2) * 5.0
        flows_low_dynamic.append(flow)
    
    result3 = calculator.calculate_unified_dynamics(flows_low_dynamic, images)
    print(f"  ��������: {result3['scene_type']}")
    print(f"  ��̬�ȷ���: {result3['unified_dynamics_score']:.3f}")
    print(f"  ����: {result3['classification']['description']}")
    print(f"  ? ���Ա�ɸѡ��������������0.3��: {result3['unified_dynamics_score'] < 0.3}")


def test_quality_filter():
    """��������ɸѡ��"""
    print("\n" + "="*70)
    print("����2: ����ɸѡ��")
    print("="*70)
    
    # ģ����Ƶ���
    mock_results = [
        {
            'video_name': 'video1_static_perfect.mp4',
            'video_path': 'videos/video1.mp4',
            'unified_dynamics_score': 0.08,
            'scene_type': 'static',
            'classification': {'category': 'pure_static', 'description': '����̬'}
        },
        {
            'video_name': 'video2_dynamic_low.mp4',
            'video_path': 'videos/video2.mp4',
            'unified_dynamics_score': 0.18,
            'scene_type': 'dynamic',
            'classification': {'category': 'low_dynamic', 'description': '�Ͷ�̬'}
        },
        {
            'video_name': 'video3_dynamic_normal.mp4',
            'video_path': 'videos/video3.mp4',
            'unified_dynamics_score': 0.52,
            'scene_type': 'dynamic',
            'classification': {'category': 'medium_dynamic', 'description': '�еȶ�̬'}
        },
        {
            'video_name': 'video4_static_anomaly.mp4',
            'video_path': 'videos/video4.mp4',
            'unified_dynamics_score': 0.68,
            'scene_type': 'static',
            'classification': {'category': 'high_dynamic', 'description': '�߶�̬'}
        },
    ]
    
    # ����ɸѡ��
    quality_filter = VideoQualityFilter()
    
    # ����1: ɸѡ��̬�����ж�̬�ȹ��͵���Ƶ
    print("\nɸѡ1: ��̬�����ж�̬�ȹ��͵���Ƶ����ֵ=0.3��")
    low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
        mock_results,
        threshold=0.3
    )
    print(f"  ɸѡ�� {len(low_dynamic_videos)} ����Ƶ:")
    for video in low_dynamic_videos:
        print(f"    - {video['video_name']}: {video['score']:.3f}")
    
    # ����2: ɸѡ��̬�������쳣���ߵ���Ƶ
    print("\nɸѡ2: ��̬�������쳣���ߵ���Ƶ����ֵ=0.5��")
    high_anomaly_videos = quality_filter.filter_high_static_anomaly(
        mock_results,
        threshold=0.5
    )
    print(f"  ɸѡ�� {len(high_anomaly_videos)} ����Ƶ:")
    for video in high_anomaly_videos:
        print(f"    - {video['video_name']}: {video['score']:.3f}")
    
    # ����3: ��������Χɸѡ
    print("\nɸѡ3: ������0.1-0.4֮�����Ƶ")
    range_filtered = quality_filter.filter_by_score_range(
        mock_results,
        min_score=0.1,
        max_score=0.4
    )
    print(f"  ɸѡ�� {len(range_filtered)} ����Ƶ:")
    for video in range_filtered:
        print(f"    - {video['video_name']}: {video['unified_dynamics_score']:.3f}")
    
    # ����4: ����ͳ��
    print("\nͳ��: ������������")
    stats = quality_filter.get_quality_statistics(mock_results)
    print(f"  ����Ƶ��: {stats['total_videos']}")
    print(f"  ƽ������: {stats['score_statistics']['mean']:.3f}")
    print(f"  �������ͷֲ�: {stats['scene_type_distribution']}")
    print(f"  ��̬�ȼ��ֲ�: {stats['category_distribution']}")


def test_config_loading():
    """�������ü���"""
    print("\n" + "="*70)
    print("����3: ���ü���")
    print("="*70)
    
    for preset in ['strict', 'balanced', 'lenient']:
        config = get_config(preset)
        print(f"\n{preset.upper()} ģʽ:")
        print(f"  ��̬��ֵ: {config['detection']['static_threshold']}")
        print(f"  ������ֵ: {config['detection']['subject_threshold']}")
        print(f"  �Ͷ�̬ɸѡ��ֵ: {config['quality_filter']['low_dynamic_in_dynamic_scene']}")


def test_score_mapping():
    """���Է���ӳ��"""
    print("\n" + "="*70)
    print("����4: ����ӳ����֤")
    print("="*70)
    
    calculator = UnifiedDynamicsCalculator()
    
    # ���Ծ�̬������ӳ��
    print("\n��̬��������ӳ��:")
    static_raw_scores = [0.0001, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    for raw_score in static_raw_scores:
        unified_score = calculator._normalize_static_score(raw_score)
        print(f"  raw={raw_score:.4f} �� unified={unified_score:.3f}")
    
    # ���Զ�̬������ӳ��
    print("\n��̬��������ӳ��:")
    dynamic_raw_scores = [0.005, 0.008, 0.015, 0.025, 0.04, 0.08, 0.15]
    for raw_score in dynamic_raw_scores:
        unified_score = calculator._normalize_dynamic_score(raw_score)
        print(f"  raw={raw_score:.4f} �� unified={unified_score:.3f}")


def main():
    """�������в���"""
    print("\n" + "="*70)
    print("��ϵͳ���ٲ���")
    print("="*70)
    
    try:
        test_unified_calculator()
        print("\n? ����1ͨ��")
    except Exception as e:
        print(f"\n? ����1ʧ��: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_quality_filter()
        print("\n? ����2ͨ��")
    except Exception as e:
        print(f"\n? ����2ʧ��: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_config_loading()
        print("\n? ����3ͨ��")
    except Exception as e:
        print(f"\n? ����3ʧ��: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_score_mapping()
        print("\n? ����4ͨ��")
    except Exception as e:
        print(f"\n? ����4ʧ��: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("������ɣ�")
    print("="*70)


if __name__ == '__main__':
    main()

