# -*- coding: utf-8 -*-
"""
Usage Examples for Unified Dynamics Assessment System
"""

import os
from video_processor import VideoProcessor, batch_process_videos
from video_quality_filter import VideoQualityFilter
from dynamics_config import get_config


def example1_single_video():
    """Example 1: Process single video"""
    print("\n" + "="*70)
    print("Example 1: Single Video Processing")
    print("="*70)
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        enable_camera_compensation=True,
        config_preset='balanced'
    )
    
    video_path = "videos/The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    frames = processor.load_video(video_path)
    result = processor.process_video(frames, output_dir='output_examples/single')
    
    print("\n[Results]")
    print(f"Scene type: {result['unified_dynamics']['scene_type']}")
    print(f"Dynamics score: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
    print(f"Category: {result['dynamics_classification']['category']}")
    print(f"Description: {result['dynamics_classification']['description']}")


def example2_batch_with_filtering():
    """Example 2: Batch processing with quality filtering"""
    print("\n" + "="*70)
    print("Example 2: Batch Processing with Filtering")
    print("="*70)
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        config_preset='balanced'
    )
    
    results = batch_process_videos(
        processor,
        'videos/',
        'output_examples/batch',
        camera_fov=60.0
    )
    
    quality_filter = VideoQualityFilter()
    
    # Filter low-motion videos in dynamic scenes
    low_dynamic = quality_filter.filter_low_dynamics_in_dynamic_scenes(
        results,
        threshold=0.3
    )
    
    print(f"\n[Filtering Results]")
    print(f"Total videos: {len(results)}")
    print(f"Low-motion videos found: {len(low_dynamic)}")
    
    for video in low_dynamic:
        print(f"\n  Video: {video['video_name']}")
        print(f"    Score: {video['score']:.3f}")
        print(f"    Reason: {video['reason']}")
    
    # Save filtering report
    if low_dynamic:
        report = quality_filter.generate_filter_report(
            results,
            low_dynamic,
            "Low Motion Filter"
        )
        report_path = 'output_examples/batch/filter_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  Report saved to: {report_path}")


def example3_quality_statistics():
    """Example 3: Get quality statistics"""
    print("\n" + "="*70)
    print("Example 3: Quality Statistics")
    print("="*70)
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        config_preset='balanced'
    )
    
    results = batch_process_videos(
        processor,
        'videos/',
        'output_examples/stats',
        camera_fov=60.0
    )
    
    quality_filter = VideoQualityFilter()
    stats = quality_filter.get_quality_statistics(results)
    
    print("\n[Quality Statistics]")
    print(f"Total videos: {stats['total_videos']}")
    print(f"\nScore statistics:")
    print(f"  Mean: {stats['score_statistics']['mean']:.3f}")
    print(f"  Std: {stats['score_statistics']['std']:.3f}")
    print(f"  Min: {stats['score_statistics']['min']:.3f}")
    print(f"  Max: {stats['score_statistics']['max']:.3f}")
    print(f"  Median: {stats['score_statistics']['median']:.3f}")
    
    print(f"\nScene type distribution:")
    for scene_type, count in stats['scene_type_distribution'].items():
        print(f"  {scene_type}: {count}")
    
    print(f"\nCategory distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")


def example4_config_comparison():
    """Example 4: Compare different configurations"""
    print("\n" + "="*70)
    print("Example 4: Configuration Comparison")
    print("="*70)
    
    video_path = "videos/The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    configs = ['strict', 'balanced', 'lenient']
    comparison = {}
    
    for config_name in configs:
        print(f"\nProcessing with {config_name} config...")
        
        processor = VideoProcessor(
            raft_model_path="pretrained_models/raft-things.pth",
            device='cuda',
            config_preset=config_name
        )
        
        frames = processor.load_video(video_path)
        result = processor.process_video(
            frames,
            output_dir=f'output_examples/compare/{config_name}'
        )
        
        comparison[config_name] = {
            'score': result['unified_dynamics']['unified_dynamics_score'],
            'scene': result['unified_dynamics']['scene_type'],
            'category': result['dynamics_classification']['category']
        }
    
    print("\n" + "="*70)
    print("Configuration Comparison Results")
    print("="*70)
    print(f"{'Config':<12} {'Score':<10} {'Scene':<12} {'Category'}")
    print("-"*70)
    for config_name, res in comparison.items():
        print(f"{config_name:<12} {res['score']:<10.3f} {res['scene']:<12} {res['category']}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Unified Dynamics Assessment - Usage Examples")
    print("="*70)
    
    examples = [
        ("Example 1: Single Video", example1_single_video),
        ("Example 2: Batch with Filtering", example2_batch_with_filtering),
        ("Example 3: Quality Statistics", example3_quality_statistics),
        ("Example 4: Config Comparison", example4_config_comparison),
    ]
    
    for title, func in examples:
        try:
            func()
            print(f"\n[{title}] Completed successfully")
        except Exception as e:
            print(f"\n[{title}] Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()

