# -*- coding: utf-8 -*-
"""
测试统一动态度评分系统
"""

import os
import sys
import numpy as np
from video_processor import VideoProcessor
from unified_dynamics_scorer import UnifiedDynamicsScorer, DynamicsClassifier


def test_scorer_initialization():
    """测试评分器初始化"""
    print("\n" + "="*70)
    print("测试1: 评分器初始化")
    print("="*70)
    
    # 默认初始化
    scorer1 = UnifiedDynamicsScorer()
    assert scorer1.mode == 'auto'
    print("? 默认初始化成功")
    
    # 自定义模式
    scorer2 = UnifiedDynamicsScorer(mode='static_scene')
    assert scorer2.mode == 'static_scene'
    print("? 自定义模式初始化成功")
    
    # 自定义权重
    custom_weights = {'flow_magnitude': 0.5}
    scorer3 = UnifiedDynamicsScorer(weights=custom_weights)
    assert scorer3.weights['flow_magnitude'] == 0.5
    print("? 自定义权重初始化成功")
    
    return True


def test_classifier():
    """测试动态度分类器"""
    print("\n" + "="*70)
    print("测试2: 动态度分类器")
    print("="*70)
    
    classifier = DynamicsClassifier()
    
    # 测试不同分数的分类
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
        print(f"? 分数 {score:.2f} -> {expected_category} (ID:{expected_id})")
    
    # 测试二分类
    assert classifier.get_binary_label(0.3) == 0  # 静态
    assert classifier.get_binary_label(0.7) == 1  # 动态
    print("? 二分类测试通过")
    
    return True


def test_with_demo_data():
    """使用demo数据测试完整流程"""
    print("\n" + "="*70)
    print("测试3: 完整流程测试（使用demo数据）")
    print("="*70)
    
    demo_dir = "demo_data"
    if not os.path.exists(demo_dir):
        print("? 跳过：demo_data目录不存在")
        return True
    
    # 创建处理器
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=True
    )
    
    print("  - 处理器创建成功")
    
    # 加载图像
    frames = processor.extract_frames_from_images(demo_dir)
    
    if len(frames) < 2:
        print("? 跳过：图像数量不足")
        return True
    
    print(f"  - 加载帧数: {len(frames)}")
    
    # 处理视频
    output_dir = "test_output_unified"
    result = processor.process_video(frames[:5], output_dir=output_dir)
    
    # 验证统一动态度结果
    assert 'unified_dynamics' in result, "结果应包含unified_dynamics"
    assert 'dynamics_classification' in result, "结果应包含dynamics_classification"
    
    unified_dynamics = result['unified_dynamics']
    classification = result['dynamics_classification']
    
    print("\n  统一动态度评估结果:")
    print(f"  - 综合分数: {unified_dynamics['unified_dynamics_score']:.3f}")
    print(f"  - 场景类型: {unified_dynamics['scene_type']}")
    print(f"  - 置信度: {unified_dynamics['confidence']:.1%}")
    print(f"  - 分类: {classification['description']}")
    
    # 显示各维度分数
    print("\n  各维度分数:")
    for name, score in unified_dynamics['component_scores'].items():
        print(f"    {name}: {score:.3f}")
    
    # 验证分数范围
    score = unified_dynamics['unified_dynamics_score']
    assert 0.0 <= score <= 1.0, f"分数应在0-1之间，实际: {score}"
    
    # 验证置信度
    conf = unified_dynamics['confidence']
    assert 0.0 <= conf <= 1.0, f"置信度应在0-1之间，实际: {conf}"
    
    print("\n? 完整流程测试通过")
    
    return True


def test_different_scenes():
    """测试不同场景的评分"""
    print("\n" + "="*70)
    print("测试4: 不同场景评分测试")
    print("="*70)
    
    scorer = UnifiedDynamicsScorer()
    
    # 模拟静态场景数据
    static_scene = {
        'temporal_stats': {
            'mean_dynamics_score': 0.5,    # 低残差
            'std_dynamics_score': 0.1,
            'mean_static_ratio': 0.9,      # 高静态比例
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
    
    print(f"  静态场景分数: {static_score:.3f}")
    assert static_score < 0.4, f"静态场景分数应<0.4，实际: {static_score}"
    print("  ? 静态场景评分合理")
    
    # 模拟动态场景数据
    dynamic_scene = {
        'temporal_stats': {
            'mean_dynamics_score': 15.0,   # 高光流
            'std_dynamics_score': 3.0,
            'mean_static_ratio': 0.2,      # 低静态比例
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
    
    print(f"  动态场景分数: {dynamic_score:.3f}")
    assert dynamic_score > 0.5, f"动态场景分数应>0.5，实际: {dynamic_score}"
    print("  ? 动态场景评分合理")
    
    # 验证场景检测
    assert static_result['scene_type'] == 'static'
    assert dynamic_result['scene_type'] == 'dynamic'
    print("  ? 场景类型检测正确")
    
    return True


def test_custom_configuration():
    """测试自定义配置"""
    print("\n" + "="*70)
    print("测试5: 自定义配置测试")
    print("="*70)
    
    # 自定义权重
    custom_weights = {
        'flow_magnitude': 0.5,
        'spatial_coverage': 0.3,
        'temporal_variation': 0.1,
        'spatial_consistency': 0.05,
        'camera_factor': 0.05
    }
    
    scorer1 = UnifiedDynamicsScorer(weights=custom_weights)
    assert scorer1.weights == custom_weights
    print("? 自定义权重设置成功")
    
    # 自定义阈值
    custom_thresholds = {
        'flow_low': 0.5,
        'flow_mid': 3.0,
        'flow_high': 10.0,
        'static_ratio': 0.6
    }
    
    scorer2 = UnifiedDynamicsScorer(thresholds=custom_thresholds)
    assert scorer2.thresholds == custom_thresholds
    print("? 自定义阈值设置成功")
    
    # 自定义分类阈值
    custom_class_thresholds = {
        'pure_static': 0.10,
        'low_dynamic': 0.30,
        'medium_dynamic': 0.60,
        'high_dynamic': 0.80
    }
    
    classifier = DynamicsClassifier(thresholds=custom_class_thresholds)
    result = classifier.classify(0.25)
    assert result['category'] == 'low_dynamic'
    print("? 自定义分类阈值设置成功")
    
    return True


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*70)
    print("测试6: 边界情况测试")
    print("="*70)
    
    scorer = UnifiedDynamicsScorer()
    
    # 极端低值
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
    print(f"  极端低值分数: {result_low['unified_dynamics_score']:.3f}")
    assert result_low['unified_dynamics_score'] >= 0.0
    print("  ? 极端低值处理正确")
    
    # 极端高值
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
    print(f"  极端高值分数: {result_high['unified_dynamics_score']:.3f}")
    assert result_high['unified_dynamics_score'] <= 1.0
    print("  ? 极端高值处理正确")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("统一动态度评分系统测试")
    print("="*70)
    
    tests = [
        ("评分器初始化", test_scorer_initialization),
        ("分类器功能", test_classifier),
        ("自定义配置", test_custom_configuration),
        ("不同场景评分", test_different_scenes),
        ("边界情况", test_edge_cases),
        ("完整流程", test_with_demo_data),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"? {test_name} 失败")
        except Exception as e:
            failed += 1
            print(f"? {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n? 所有测试通过!")
        print("\n? 统一动态度评分系统已就绪!")
        print("\n使用方法:")
        print("  python video_processor.py -i your_video.mp4 -o output/")
        print("\n查看详细文档:")
        print("  UNIFIED_DYNAMICS_GUIDE.md")
        return 0
    else:
        print("\n? 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())

