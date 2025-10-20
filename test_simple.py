# -*- coding: utf-8 -*-
"""Simple test for new unified dynamics system"""

import numpy as np
from unified_dynamics_calculator import UnifiedDynamicsCalculator
from video_quality_filter import VideoQualityFilter
from dynamics_config import get_config


def test_calculator():
    """Test unified dynamics calculator"""
    print("="*70)
    print("Test 1: Unified Dynamics Calculator")
    print("="*70)
    
    calculator = UnifiedDynamicsCalculator(
        static_threshold=0.002,
        subject_threshold=0.005,
        use_normalized_flow=True
    )
    
    # Mock data
    h, w = 720, 1280
    num_frames = 10
    
    # Scene 1: Static scene (small residual)
    print("\nScene 1: Static scene (building)")
    flows_static = [np.random.randn(h, w, 2) * 0.5 for _ in range(num_frames)]
    images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(num_frames + 1)]
    
    result1 = calculator.calculate_unified_dynamics(flows_static, images)
    print(f"  Scene type: {result1['scene_type']}")
    print(f"  Score: {result1['unified_dynamics_score']:.3f}")
    print(f"  Category: {result1['classification']['category']}")
    
    # Scene 2: Dynamic scene (obvious motion)
    print("\nScene 2: Dynamic scene (person)")
    flows_dynamic = []
    for _ in range(num_frames):
        flow = np.random.randn(h, w, 2) * 0.5
        flow[300:400, 500:700, :] += np.random.randn(100, 200, 2) * 15.0
        flows_dynamic.append(flow)
    
    result2 = calculator.calculate_unified_dynamics(flows_dynamic, images)
    print(f"  Scene type: {result2['scene_type']}")
    print(f"  Score: {result2['unified_dynamics_score']:.3f}")
    print(f"  Category: {result2['classification']['category']}")
    
    # Scene 3: Dynamic scene but low motion
    print("\nScene 3: Dynamic scene but low motion")
    flows_low = []
    for _ in range(num_frames):
        flow = np.random.randn(h, w, 2) * 0.5
        flow[300:400, 500:700, :] += np.random.randn(100, 200, 2) * 5.0
        flows_low.append(flow)
    
    result3 = calculator.calculate_unified_dynamics(flows_low, images)
    print(f"  Scene type: {result3['scene_type']}")
    print(f"  Score: {result3['unified_dynamics_score']:.3f}")
    print(f"  Category: {result3['classification']['category']}")
    print(f"  Can be filtered (score < 0.3): {result3['unified_dynamics_score'] < 0.3}")
    
    return True


def test_filter():
    """Test quality filter"""
    print("\n" + "="*70)
    print("Test 2: Video Quality Filter")
    print("="*70)
    
    mock_results = [
        {
            'video_name': 'video1_static.mp4',
            'video_path': 'videos/video1.mp4',
            'unified_dynamics_score': 0.08,
            'scene_type': 'static',
            'classification': {'category': 'pure_static'}
        },
        {
            'video_name': 'video2_dynamic_low.mp4',
            'video_path': 'videos/video2.mp4',
            'unified_dynamics_score': 0.18,
            'scene_type': 'dynamic',
            'classification': {'category': 'low_dynamic'}
        },
        {
            'video_name': 'video3_dynamic_normal.mp4',
            'video_path': 'videos/video3.mp4',
            'unified_dynamics_score': 0.52,
            'scene_type': 'dynamic',
            'classification': {'category': 'medium_dynamic'}
        },
    ]
    
    quality_filter = VideoQualityFilter()
    
    low_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
        mock_results,
        threshold=0.3
    )
    
    print(f"\nFiltered {len(low_videos)} low-dynamic videos:")
    for video in low_videos:
        print(f"  - {video['video_name']}: {video['score']:.3f}")
    
    stats = quality_filter.get_quality_statistics(mock_results)
    print(f"\nStatistics:")
    print(f"  Total: {stats['total_videos']}")
    print(f"  Mean score: {stats['score_statistics']['mean']:.3f}")
    
    return True


def test_config():
    """Test config loading"""
    print("\n" + "="*70)
    print("Test 3: Configuration Loading")
    print("="*70)
    
    for preset in ['strict', 'balanced', 'lenient']:
        config = get_config(preset)
        print(f"\n{preset.upper()} preset:")
        print(f"  static_threshold: {config['detection']['static_threshold']}")
        print(f"  subject_threshold: {config['detection']['subject_threshold']}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("New System Quick Test")
    print("="*70)
    
    try:
        test_calculator()
        print("\nTest 1 PASSED")
    except Exception as e:
        print(f"\nTest 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        test_filter()
        print("\nTest 2 PASSED")
    except Exception as e:
        print(f"\nTest 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        test_config()
        print("\nTest 3 PASSED")
    except Exception as e:
        print(f"\nTest 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("All Tests PASSED!")
    print("="*70)
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

