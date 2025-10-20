# -*- coding: utf-8 -*-
"""
��ϵͳʾ���ű�
��ʾ�ع����ͳһ��̬������ϵͳ
"""

import os
from video_processor import VideoProcessor, batch_process_videos
from video_quality_filter import VideoQualityFilter
from dynamics_config import get_config, print_config_guide


def example1_single_video():
    """ʾ��1: ��������Ƶ"""
    print("\n" + "="*70)
    print("ʾ��1: ��������Ƶ")
    print("="*70)
    
    # ������������ʹ���¼�������
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        enable_camera_compensation=True,
        use_normalized_flow=True,
        use_new_calculator=True,  # ʹ���¼�����
        config_preset='balanced'
    )
    
    # ������Ƶ
    video_path = "videos/The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    if not os.path.exists(video_path):
        print(f"��Ƶ�ļ�������: {video_path}")
        return
    
    frames = processor.load_video(video_path)
    result = processor.process_video(frames, output_dir='output_new/example1')
    
    # ��ӡ���
    print("\n�����������")
    print(f"��������: {result['dynamics_classification']['scene_type']}")
    print(f"��̬�ȷ���: {result['unified_dynamics']['unified_dynamics_score']:.3f}")
    print(f"��̬�ȼ�: {result['dynamics_classification']['description']}")
    print(f"����: {result['dynamics_classification']['category']}")


def example2_batch_processing():
    """ʾ��2: ��������ɸѡ"""
    print("\n" + "="*70)
    print("ʾ��2: ��������ɸѡ�Ͷ�̬����Ƶ")
    print("="*70)
    
    # ����������
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        enable_camera_compensation=True,
        use_normalized_flow=True,
        use_new_calculator=True,
        config_preset='balanced'
    )
    
    # ��������
    results = batch_process_videos(
        processor,
        'videos/',
        'output_new/batch',
        camera_fov=60.0
    )
    
    # ɸѡ�Ͷ�̬����Ƶ
    quality_filter = VideoQualityFilter()
    
    # ɸѡ1: ��̬�����ж�̬�ȹ��͵���Ƶ
    low_dynamic_videos = quality_filter.filter_low_dynamics_in_dynamic_scenes(
        results,
        threshold=0.3
    )
    
    print(f"\n�ҵ� {len(low_dynamic_videos)} ����̬�ȹ��͵���Ƶ��")
    for video in low_dynamic_videos:
        print(f"\n��Ƶ: {video['video_name']}")
        print(f"  ����: {video['score']:.3f}")
        print(f"  ԭ��: {video['reason']}")
        print(f"  ����: {video['recommendation']}")
    
    # ɸѡ2: ��̬�������쳣���ߵ���Ƶ
    high_anomaly_videos = quality_filter.filter_high_static_anomaly(
        results,
        threshold=0.5
    )
    
    print(f"\n�ҵ� {len(high_anomaly_videos)} ���쳣���ߵ���Ƶ��")
    for video in high_anomaly_videos:
        print(f"\n��Ƶ: {video['video_name']}")
        print(f"  ����: {video['score']:.3f}")
        print(f"  ԭ��: {video['reason']}")
    
    # ����ɸѡ����
    if low_dynamic_videos:
        report = quality_filter.generate_filter_report(
            results,
            low_dynamic_videos,
            "��̬�����Ͷ�̬��ɸѡ"
        )
        
        report_path = 'output_new/batch/low_dynamic_filter_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nɸѡ�����ѱ���: {report_path}")


def example3_compare_configs():
    """ʾ��3: �ԱȲ�ͬ����"""
    print("\n" + "="*70)
    print("ʾ��3: �ԱȲ�ͬ����ģʽ")
    print("="*70)
    
    video_path = "videos/The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    if not os.path.exists(video_path):
        print(f"��Ƶ�ļ�������: {video_path}")
        return
    
    configs = ['strict', 'balanced', 'lenient']
    results_comparison = {}
    
    for config_name in configs:
        print(f"\nʹ������: {config_name}")
        
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
    
    # ��ӡ�Ա�
    print("\n" + "="*70)
    print("���öԱȽ��")
    print("="*70)
    print(f"{'����':<10} {'����':<10} {'��������':<12} {'��̬�ȼ�'}")
    print("-"*70)
    for config_name, result in results_comparison.items():
        print(f"{config_name:<10} {result['score']:<10.3f} {result['scene_type']:<12} {result['category']}")


def example4_quality_statistics():
    """ʾ��4: ��ȡ����ͳ��"""
    print("\n" + "="*70)
    print("ʾ��4: ��������ͳ��")
    print("="*70)
    
    # ����������
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cuda',
        enable_camera_compensation=True,
        use_normalized_flow=True,
        use_new_calculator=True,
        config_preset='balanced'
    )
    
    # ��������
    results = batch_process_videos(
        processor,
        'videos/',
        'output_new/statistics',
        camera_fov=60.0
    )
    
    # ��ȡͳ����Ϣ
    quality_filter = VideoQualityFilter()
    stats = quality_filter.get_quality_statistics(results)
    
    print("\n������ͳ�ơ�")
    print(f"����Ƶ��: {stats['total_videos']}")
    print(f"\n����ͳ��:")
    print(f"  ��ֵ: {stats['score_statistics']['mean']:.3f}")
    print(f"  ��׼��: {stats['score_statistics']['std']:.3f}")
    print(f"  ��Сֵ: {stats['score_statistics']['min']:.3f}")
    print(f"  ���ֵ: {stats['score_statistics']['max']:.3f}")
    print(f"  ��λ��: {stats['score_statistics']['median']:.3f}")
    
    print(f"\n�������ͷֲ�:")
    for scene_type, count in stats['scene_type_distribution'].items():
        print(f"  {scene_type}: {count}")
    
    print(f"\n��̬�ȼ��ֲ�:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")


def main():
    """������"""
    
    # ��ӡ����ָ��
    print_config_guide()
    
    # ����ʾ��
    print("\n" + "="*70)
    print("��ϵͳʾ����ʾ")
    print("="*70)
    
    # ʾ��1: ����Ƶ����
    try:
        example1_single_video()
    except Exception as e:
        print(f"ʾ��1����: {e}")
    
    # ʾ��2: ���������ɸѡ
    try:
        example2_batch_processing()
    except Exception as e:
        print(f"ʾ��2����: {e}")
    
    # ʾ��3: ���öԱ�
    try:
        example3_compare_configs()
    except Exception as e:
        print(f"ʾ��3����: {e}")
    
    # ʾ��4: ����ͳ��
    try:
        example4_quality_statistics()
    except Exception as e:
        print(f"ʾ��4����: {e}")
    
    print("\n" + "="*70)
    print("����ʾ��������ɣ�")
    print("="*70)


if __name__ == '__main__':
    main()

