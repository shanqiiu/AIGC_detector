# -*- coding: utf-8 -*-
"""
新系统示例脚本
演示重构后的统一动态度评估系统
"""

import os
from video_processor import VideoProcessor, batch_process_videos
from video_quality_filter import VideoQualityFilter
from dynamics_config import get_config, print_config_guide


def example1_single_video():
    """示例1: 处理单个视频"""
    print("\n" + "="*70)
    print("示例1: 处理单个视频")
    print("="*70)
    
    # 创建处理器（使用新计算器）
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        enable_camera_compensation=True,
        use_normalized_flow=True,
        use_new_calculator=True,  # 使用新计算器
        config_preset='balanced'
    )
    
    # 处理视频
    video_path = "videos/The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    frames = processor.load_video(video_path)
    result = processor.process_video(frames, output_dir='output_new/example1')
    
    # 打印结果
    print("\n【分析结果】")
    print(f"场景类型: {result['dynamics_classification']['scene_type']}")
    print(f"动态度分数: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
    print(f"动态等级: {result['dynamics_classification']['description']}")
    print(f"分类: {result['dynamics_classification']['category']}")


def example2_batch_processing():
    """示例2: 批量处理并筛选"""
    print("\n" + "="*70)
    print("示例2: 批量处理并筛选低动态度视频")
    print("="*70)
    
    # 创建处理器
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        enable_camera_compensation=True,
        use_normalized_flow=True,
        use_new_calculator=True,
        config_preset='balanced'
    )
    
    # 批量处理
    results = batch_process_videos(
        processor,
        'videos/',
        'output_new/batch',
        camera_fov=60.0
    )
    
    # 筛选低动态度视频
    quality_filter = VideoQualityFilter()
    
    # 筛选1: 动态场景中动态度过低的视频
    low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
        results,
        threshold=0.3
    )
    
    print(f"\n找到 {len(low_dynamic_videos)} 个动态度过低的视频：")
    for video in low_dynamic_videos:
        print(f"\n视频: {video['video_name']}")
        print(f"  分数: {video['score']:.3f}")
        print(f"  原因: {video['reason']}")
        print(f"  建议: {video['recommendation']}")
    
    # 筛选2: 静态场景中异常过高的视频
    high_anomaly_videos = quality_filter.filter_high_static_anomaly(
        results,
        threshold=0.5
    )
    
    print(f"\n找到 {len(high_anomaly_videos)} 个异常过高的视频：")
    for video in high_anomaly_videos:
        print(f"\n视频: {video['video_name']}")
        print(f"  分数: {video['score']:.3f}")
        print(f"  原因: {video['reason']}")
    
    # 生成筛选报告
    if low_dynamic_videos:
        report = quality_filter.generate_filter_report(
            results,
            low_dynamic_videos,
            "动态场景低动态度筛选"
        )
        
        report_path = 'output_new/batch/low_dynamic_filter_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n筛选报告已保存: {report_path}")


def example3_compare_configs():
    """示例3: 对比不同配置"""
    print("\n" + "="*70)
    print("示例3: 对比不同配置模式")
    print("="*70)
    
    video_path = "videos/The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    configs = ['strict', 'balanced', 'lenient']
    results_comparison = {}
    
    for config_name in configs:
        print(f"\n使用配置: {config_name}")
        
        processor = VideoProcessor(
            raft_model_path="pretrained_models/raft-things.pth",
            device='cuda',
            enable_camera_compensation=True,
            use_normalized_flow=True,
            use_new_calculator=True,
            config_preset=config_name
        )
        
        frames = processor.load_video(video_path)
        result = processor.process_video(
            frames, 
            output_dir=f'output_new/compare/{config_name}'
        )
        
        results_comparison[config_name] = {
            'score': result['unified_dynamics']['unified_dynamics_score'],
            'scene_type': result['dynamics_classification']['scene_type'],
            'category': result['dynamics_classification']['category']
        }
    
    # 打印对比
    print("\n" + "="*70)
    print("配置对比结果")
    print("="*70)
    print(f"{'配置':<10} {'分数':<10} {'场景类型':<12} {'动态等级'}")
    print("-"*70)
    for config_name, result in results_comparison.items():
        print(f"{config_name:<10} {result['score']:<10.3f} {result['scene_type']:<12} {result['category']}")


def example4_quality_statistics():
    """示例4: 获取质量统计"""
    print("\n" + "="*70)
    print("示例4: 批量质量统计")
    print("="*70)
    
    # 创建处理器
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        enable_camera_compensation=True,
        use_normalized_flow=True,
        use_new_calculator=True,
        config_preset='balanced'
    )
    
    # 批量处理
    results = batch_process_videos(
        processor,
        'videos/',
        'output_new/statistics',
        camera_fov=60.0
    )
    
    # 获取统计信息
    quality_filter = VideoQualityFilter()
    stats = quality_filter.get_quality_statistics(results)
    
    print("\n【质量统计】")
    print(f"总视频数: {stats['total_videos']}")
    print(f"\n分数统计:")
    print(f"  均值: {stats['score_statistics']['mean']:.3f}")
    print(f"  标准差: {stats['score_statistics']['std']:.3f}")
    print(f"  最小值: {stats['score_statistics']['min']:.3f}")
    print(f"  最大值: {stats['score_statistics']['max']:.3f}")
    print(f"  中位数: {stats['score_statistics']['median']:.3f}")
    
    print(f"\n场景类型分布:")
    for scene_type, count in stats['scene_type_distribution'].items():
        print(f"  {scene_type}: {count}")
    
    print(f"\n动态等级分布:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")


def main():
    """主函数"""
    
    # 打印配置指南
    print_config_guide()
    
    # 运行示例
    print("\n" + "="*70)
    print("新系统示例演示")
    print("="*70)
    
    # 示例1: 单视频处理
    try:
        example1_single_video()
    except Exception as e:
        print(f"示例1错误: {e}")
    
    # 示例2: 批量处理和筛选
    try:
        example2_batch_processing()
    except Exception as e:
        print(f"示例2错误: {e}")
    
    # 示例3: 配置对比
    try:
        example3_compare_configs()
    except Exception as e:
        print(f"示例3错误: {e}")
    
    # 示例4: 质量统计
    try:
        example4_quality_statistics()
    except Exception as e:
        print(f"示例4错误: {e}")
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)


if __name__ == '__main__':
    main()

