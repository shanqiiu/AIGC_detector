# -*- coding: utf-8 -*-
"""Quick test script for new system"""

import numpy as np
from unified_dynamics_calculator import UnifiedDynamicsCalculator
from video_quality_filter import VideoQualityFilter
from dynamics_config import get_config


def test_unified_calculator():
    """测试统一动态度计算器"""
    print("\n" + "="*70)
    print("测试1: 统一动态度计算器")
    print("="*70)
    
    # 创建计算器
    calculator = UnifiedDynamicsCalculator(
        static_threshold=0.002,
        subject_threshold=0.005,
        use_normalized_flow=True
    )
    
    # 模拟光流数据
    h, w = 720, 1280
    num_frames = 10
    
    # 场景1: 静态场景（残差很小）
    print("\n场景1: 静态场景（建筑）")
    flows_static = [np.random.randn(h, w, 2) * 0.5 for _ in range(num_frames)]
    images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(num_frames + 1)]
    
    result1 = calculator.calculate_unified_dynamics(flows_static, images)
    print(f"  场景类型: {result1['scene_type']}")
    print(f"  动态度分数: {result1['unified_dynamics_score']:.3f}")
    print(f"  分类: {result1['classification']['description']}")
    
    # 场景2: 动态场景（主体运动明显）
    print("\n场景2: 动态场景（人物）")
    flows_dynamic = []
    for _ in range(num_frames):
        flow = np.random.randn(h, w, 2) * 0.5
        # 添加明显的主体运动区域
        flow[300:400, 500:700, :] += np.random.randn(100, 200, 2) * 15.0
        flows_dynamic.append(flow)
    
    result2 = calculator.calculate_unified_dynamics(flows_dynamic, images)
    print(f"  场景类型: {result2['scene_type']}")
    print(f"  动态度分数: {result2['unified_dynamics_score']:.3f}")
    print(f"  分类: {result2['classification']['description']}")
    
    # 场景3: 动态场景但动作很小
    print("\n场景3: 动态场景但动作很小")
    flows_low_dynamic = []
    for _ in range(num_frames):
        flow = np.random.randn(h, w, 2) * 0.5
        # 添加轻微的主体运动
        flow[300:400, 500:700, :] += np.random.randn(100, 200, 2) * 5.0
        flows_low_dynamic.append(flow)
    
    result3 = calculator.calculate_unified_dynamics(flows_low_dynamic, images)
    print(f"  场景类型: {result3['scene_type']}")
    print(f"  动态度分数: {result3['unified_dynamics_score']:.3f}")
    print(f"  分类: {result3['classification']['description']}")
    print(f"  ? 可以被筛选出来（分数低于0.3）: {result3['unified_dynamics_score'] < 0.3}")


def test_quality_filter():
    """测试质量筛选器"""
    print("\n" + "="*70)
    print("测试2: 质量筛选器")
    print("="*70)
    
    # 模拟视频结果
    mock_results = [
        {
            'video_name': 'video1_static_perfect.mp4',
            'video_path': 'videos/video1.mp4',
            'unified_dynamics_score': 0.08,
            'scene_type': 'static',
            'classification': {'category': 'pure_static', 'description': '纯静态'}
        },
        {
            'video_name': 'video2_dynamic_low.mp4',
            'video_path': 'videos/video2.mp4',
            'unified_dynamics_score': 0.18,
            'scene_type': 'dynamic',
            'classification': {'category': 'low_dynamic', 'description': '低动态'}
        },
        {
            'video_name': 'video3_dynamic_normal.mp4',
            'video_path': 'videos/video3.mp4',
            'unified_dynamics_score': 0.52,
            'scene_type': 'dynamic',
            'classification': {'category': 'medium_dynamic', 'description': '中等动态'}
        },
        {
            'video_name': 'video4_static_anomaly.mp4',
            'video_path': 'videos/video4.mp4',
            'unified_dynamics_score': 0.68,
            'scene_type': 'static',
            'classification': {'category': 'high_dynamic', 'description': '高动态'}
        },
    ]
    
    # 创建筛选器
    quality_filter = VideoQualityFilter()
    
    # 测试1: 筛选动态场景中动态度过低的视频
    print("\n筛选1: 动态场景中动态度过低的视频（阈值=0.3）")
    low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
        mock_results,
        threshold=0.3
    )
    print(f"  筛选出 {len(low_dynamic_videos)} 个视频:")
    for video in low_dynamic_videos:
        print(f"    - {video['video_name']}: {video['score']:.3f}")
    
    # 测试2: 筛选静态场景中异常过高的视频
    print("\n筛选2: 静态场景中异常过高的视频（阈值=0.5）")
    high_anomaly_videos = quality_filter.filter_high_static_anomaly(
        mock_results,
        threshold=0.5
    )
    print(f"  筛选出 {len(high_anomaly_videos)} 个视频:")
    for video in high_anomaly_videos:
        print(f"    - {video['video_name']}: {video['score']:.3f}")
    
    # 测试3: 按分数范围筛选
    print("\n筛选3: 分数在0.1-0.4之间的视频")
    range_filtered = quality_filter.filter_by_score_range(
        mock_results,
        min_score=0.1,
        max_score=0.4
    )
    print(f"  筛选出 {len(range_filtered)} 个视频:")
    for video in range_filtered:
        print(f"    - {video['video_name']}: {video['unified_dynamics_score']:.3f}")
    
    # 测试4: 质量统计
    print("\n统计: 整体质量分析")
    stats = quality_filter.get_quality_statistics(mock_results)
    print(f"  总视频数: {stats['total_videos']}")
    print(f"  平均分数: {stats['score_statistics']['mean']:.3f}")
    print(f"  场景类型分布: {stats['scene_type_distribution']}")
    print(f"  动态等级分布: {stats['category_distribution']}")


def test_config_loading():
    """测试配置加载"""
    print("\n" + "="*70)
    print("测试3: 配置加载")
    print("="*70)
    
    for preset in ['strict', 'balanced', 'lenient']:
        config = get_config(preset)
        print(f"\n{preset.upper()} 模式:")
        print(f"  静态阈值: {config['detection']['static_threshold']}")
        print(f"  主体阈值: {config['detection']['subject_threshold']}")
        print(f"  低动态筛选阈值: {config['quality_filter']['low_dynamic_in_dynamic_scene']}")


def test_score_mapping():
    """测试分数映射"""
    print("\n" + "="*70)
    print("测试4: 分数映射验证")
    print("="*70)
    
    calculator = UnifiedDynamicsCalculator()
    
    # 测试静态场景的映射
    print("\n静态场景分数映射:")
    static_raw_scores = [0.0001, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    for raw_score in static_raw_scores:
        unified_score = calculator._normalize_static_score(raw_score)
        print(f"  raw={raw_score:.4f} → unified={unified_score:.3f}")
    
    # 测试动态场景的映射
    print("\n动态场景分数映射:")
    dynamic_raw_scores = [0.005, 0.008, 0.015, 0.025, 0.04, 0.08, 0.15]
    for raw_score in dynamic_raw_scores:
        unified_score = calculator._normalize_dynamic_score(raw_score)
        print(f"  raw={raw_score:.4f} → unified={unified_score:.3f}")


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("新系统快速测试")
    print("="*70)
    
    try:
        test_unified_calculator()
        print("\n? 测试1通过")
    except Exception as e:
        print(f"\n? 测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_quality_filter()
        print("\n? 测试2通过")
    except Exception as e:
        print(f"\n? 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_config_loading()
        print("\n? 测试3通过")
    except Exception as e:
        print(f"\n? 测试3失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_score_mapping()
        print("\n? 测试4通过")
    except Exception as e:
        print(f"\n? 测试4失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == '__main__':
    main()

