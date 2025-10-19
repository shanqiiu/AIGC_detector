# -*- coding: utf-8 -*-
"""
����ͳһ��̬������ϵͳ
"""

import os
import sys
import numpy as np
from video_processor import VideoProcessor
from unified_dynamics_scorer import UnifiedDynamicsScorer, DynamicsClassifier


def test_scorer_initialization():
    """������������ʼ��"""
    print("\n" + "="*70)
    print("����1: ��������ʼ��")
    print("="*70)
    
    # Ĭ�ϳ�ʼ��
    scorer1 = UnifiedDynamicsScorer()
    assert scorer1.mode == 'auto'
    print("? Ĭ�ϳ�ʼ���ɹ�")
    
    # �Զ���ģʽ
    scorer2 = UnifiedDynamicsScorer(mode='static_scene')
    assert scorer2.mode == 'static_scene'
    print("? �Զ���ģʽ��ʼ���ɹ�")
    
    # �Զ���Ȩ��
    custom_weights = {'flow_magnitude': 0.5}
    scorer3 = UnifiedDynamicsScorer(weights=custom_weights)
    assert scorer3.weights['flow_magnitude'] == 0.5
    print("? �Զ���Ȩ�س�ʼ���ɹ�")
    
    return True


def test_classifier():
    """���Զ�̬�ȷ�����"""
    print("\n" + "="*70)
    print("����2: ��̬�ȷ�����")
    print("="*70)
    
    classifier = DynamicsClassifier()
    
    # ���Բ�ͬ�����ķ���
    test_cases = [
        (0.05, 'pure_static', 0),
        (0.25, 'low_dynamic', 1),
        (0.50, 'medium_dynamic', 2),
        (0.75, 'high_dynamic', 3),
        (0.95, 'extreme_dynamic', 4)
    ]
    
    for score, expected_category, expected_id in test_cases:
        result = classifier.classify(score)
        assert result['category'] == expected_category
        assert result['category_id'] == expected_id
        print(f"? ���� {score:.2f} -> {expected_category} (ID:{expected_id})")
    
    # ���Զ�����
    assert classifier.get_binary_label(0.3) == 0  # ��̬
    assert classifier.get_binary_label(0.7) == 1  # ��̬
    print("? ���������ͨ��")
    
    return True


def test_with_demo_data():
    """ʹ��demo���ݲ�����������"""
    print("\n" + "="*70)
    print("����3: �������̲��ԣ�ʹ��demo���ݣ�")
    print("="*70)
    
    demo_dir = "demo_data"
    if not os.path.exists(demo_dir):
        print("? ������demo_dataĿ¼������")
        return True
    
    # ����������
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=True
    )
    
    print("  - �����������ɹ�")
    
    # ����ͼ��
    frames = processor.extract_frames_from_images(demo_dir)
    
    if len(frames) < 2:
        print("? ������ͼ����������")
        return True
    
    print(f"  - ����֡��: {len(frames)}")
    
    # ������Ƶ
    output_dir = "test_output_unified"
    result = processor.process_video(frames[:5], output_dir=output_dir)
    
    # ��֤ͳһ��̬�Ƚ��
    assert 'unified_dynamics' in result, "���Ӧ����unified_dynamics"
    assert 'dynamics_classification' in result, "���Ӧ����dynamics_classification"
    
    unified_dynamics = result['unified_dynamics']
    classification = result['dynamics_classification']
    
    print("\n  ͳһ��̬���������:")
    print(f"  - �ۺϷ���: {unified_dynamics['unified_dynamics_score']:.3f}")
    print(f"  - ��������: {unified_dynamics['scene_type']}")
    print(f"  - ���Ŷ�: {unified_dynamics['confidence']:.1%}")
    print(f"  - ����: {classification['description']}")
    
    # ��ʾ��ά�ȷ���
    print("\n  ��ά�ȷ���:")
    for name, score in unified_dynamics['component_scores'].items():
        print(f"    {name}: {score:.3f}")
    
    # ��֤������Χ
    score = unified_dynamics['unified_dynamics_score']
    assert 0.0 <= score <= 1.0, f"����Ӧ��0-1֮�䣬ʵ��: {score}"
    
    # ��֤���Ŷ�
    conf = unified_dynamics['confidence']
    assert 0.0 <= conf <= 1.0, f"���Ŷ�Ӧ��0-1֮�䣬ʵ��: {conf}"
    
    print("\n? �������̲���ͨ��")
    
    return True


def test_different_scenes():
    """���Բ�ͬ����������"""
    print("\n" + "="*70)
    print("����4: ��ͬ�������ֲ���")
    print("="*70)
    
    scorer = UnifiedDynamicsScorer()
    
    # ģ�⾲̬��������
    static_scene = {
        'temporal_stats': {
            'mean_dynamics_score': 0.5,    # �Ͳв�
            'std_dynamics_score': 0.1,
            'mean_static_ratio': 0.9,      # �߾�̬����
            'std_static_ratio': 0.05,
            'mean_consistency_score': 0.8,
            'temporal_stability': 0.9
        },
        'frame_results': [
            {'static_dynamics': {}, 'global_dynamics': {}}
            for _ in range(10)
        ]
    }
    
    static_result = scorer.calculate_unified_score(static_scene, True)
    static_score = static_result['unified_dynamics_score']
    
    print(f"  ��̬��������: {static_score:.3f}")
    assert static_score < 0.4, f"��̬��������Ӧ<0.4��ʵ��: {static_score}"
    print("  ? ��̬�������ֺ���")
    
    # ģ�⶯̬��������
    dynamic_scene = {
        'temporal_stats': {
            'mean_dynamics_score': 15.0,   # �߹���
            'std_dynamics_score': 3.0,
            'mean_static_ratio': 0.2,      # �;�̬����
            'std_static_ratio': 0.1,
            'mean_consistency_score': 0.3,
            'temporal_stability': 0.6
        },
        'frame_results': [
            {'static_dynamics': {}, 'global_dynamics': {}}
            for _ in range(10)
        ]
    }
    
    dynamic_result = scorer.calculate_unified_score(dynamic_scene, False)
    dynamic_score = dynamic_result['unified_dynamics_score']
    
    print(f"  ��̬��������: {dynamic_score:.3f}")
    assert dynamic_score > 0.5, f"��̬��������Ӧ>0.5��ʵ��: {dynamic_score}"
    print("  ? ��̬�������ֺ���")
    
    # ��֤�������
    assert static_result['scene_type'] == 'static'
    assert dynamic_result['scene_type'] == 'dynamic'
    print("  ? �������ͼ����ȷ")
    
    return True


def test_custom_configuration():
    """�����Զ�������"""
    print("\n" + "="*70)
    print("����5: �Զ������ò���")
    print("="*70)
    
    # �Զ���Ȩ��
    custom_weights = {
        'flow_magnitude': 0.5,
        'spatial_coverage': 0.3,
        'temporal_variation': 0.1,
        'spatial_consistency': 0.05,
        'camera_factor': 0.05
    }
    
    scorer1 = UnifiedDynamicsScorer(weights=custom_weights)
    assert scorer1.weights == custom_weights
    print("? �Զ���Ȩ�����óɹ�")
    
    # �Զ�����ֵ
    custom_thresholds = {
        'flow_low': 0.5,
        'flow_mid': 3.0,
        'flow_high': 10.0,
        'static_ratio': 0.6
    }
    
    scorer2 = UnifiedDynamicsScorer(thresholds=custom_thresholds)
    assert scorer2.thresholds == custom_thresholds
    print("? �Զ�����ֵ���óɹ�")
    
    # �Զ��������ֵ
    custom_class_thresholds = {
        'pure_static': 0.10,
        'low_dynamic': 0.30,
        'medium_dynamic': 0.60,
        'high_dynamic': 0.80
    }
    
    classifier = DynamicsClassifier(thresholds=custom_class_thresholds)
    result = classifier.classify(0.25)
    assert result['category'] == 'low_dynamic'
    print("? �Զ��������ֵ���óɹ�")
    
    return True


def test_edge_cases():
    """���Ա߽����"""
    print("\n" + "="*70)
    print("����6: �߽��������")
    print("="*70)
    
    scorer = UnifiedDynamicsScorer()
    
    # ���˵�ֵ
    extreme_low = {
        'temporal_stats': {
            'mean_dynamics_score': 0.0,
            'std_dynamics_score': 0.0,
            'mean_static_ratio': 1.0,
            'std_static_ratio': 0.0,
            'mean_consistency_score': 1.0,
            'temporal_stability': 1.0
        },
        'frame_results': [
            {'static_dynamics': {}, 'global_dynamics': {}}
        ]
    }
    
    result_low = scorer.calculate_unified_score(extreme_low, True)
    print(f"  ���˵�ֵ����: {result_low['unified_dynamics_score']:.3f}")
    assert result_low['unified_dynamics_score'] >= 0.0
    print("  ? ���˵�ֵ������ȷ")
    
    # ���˸�ֵ
    extreme_high = {
        'temporal_stats': {
            'mean_dynamics_score': 50.0,
            'std_dynamics_score': 10.0,
            'mean_static_ratio': 0.0,
            'std_static_ratio': 0.0,
            'mean_consistency_score': 0.0,
            'temporal_stability': 0.0
        },
        'frame_results': [
            {'static_dynamics': {}, 'global_dynamics': {}}
        ]
    }
    
    result_high = scorer.calculate_unified_score(extreme_high, False)
    print(f"  ���˸�ֵ����: {result_high['unified_dynamics_score']:.3f}")
    assert result_high['unified_dynamics_score'] <= 1.0
    print("  ? ���˸�ֵ������ȷ")
    
    return True


def main():
    """�������в���"""
    print("\n" + "="*70)
    print("ͳһ��̬������ϵͳ����")
    print("="*70)
    
    tests = [
        ("��������ʼ��", test_scorer_initialization),
        ("����������", test_classifier),
        ("�Զ�������", test_custom_configuration),
        ("��ͬ��������", test_different_scenes),
        ("�߽����", test_edge_cases),
        ("��������", test_with_demo_data),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"? {test_name} ʧ��")
        except Exception as e:
            failed += 1
            print(f"? {test_name} �쳣: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("�����ܽ�")
    print("="*70)
    print(f"ͨ��: {passed}/{len(tests)}")
    print(f"ʧ��: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n? ���в���ͨ��!")
        print("\n? ͳһ��̬������ϵͳ�Ѿ���!")
        print("\nʹ�÷���:")
        print("  python video_processor.py -i your_video.mp4 -o output/")
        print("\n�鿴��ϸ�ĵ�:")
        print("  UNIFIED_DYNAMICS_GUIDE.md")
        return 0
    else:
        print("\n? ���ֲ���ʧ��")
        return 1


if __name__ == '__main__':
    sys.exit(main())

