# -*- coding: utf-8 -*-
"""
��̬������ϵͳ�����ļ�
�������пɵ���������ֵ
"""

# ============ ������������ ============
FLOW_CONFIG = {
    'use_normalized_flow': True,      # �Ƿ�ʹ�÷ֱ��ʹ�һ��
    'baseline_diagonal': 1469.0,      # ��׼�ֱ��ʶԽ��߳��� (1280x720)
}

# ============ ��̬/��̬��������ֵ ============
DETECTION_THRESHOLDS = {
    'static_threshold': 0.002,        # ��̬��������ֵ����һ����
    'subject_threshold': 0.005,       # ������������ֵ����һ����
    'min_subject_ratio': 0.05,        # ��С����ռ��
}

# ============ ���������ж���ֵ ============
SCENE_CLASSIFICATION = {
    'dynamic_ratio_threshold': 0.15,  # ��̬����ռ����ֵ
    'dynamic_score_threshold': 0.01,  # ��̬������ֵ
    'static_score_threshold': 0.003,  # ��̬������ֵ
    'motion_coverage_threshold': 0.2, # �˶���������ֵ
    'scene_auto_detect': True,        # �Ƿ��Զ���ⳡ������
}

# ============ ��̬�ȷ�����ֵ ============
DYNAMICS_CLASSIFICATION = {
    'pure_static': 0.15,      # < 0.15: ����̬
    'low_dynamic': 0.35,      # 0.15-0.35: �Ͷ�̬
    'medium_dynamic': 0.60,   # 0.35-0.60: �еȶ�̬
    'high_dynamic': 0.85,     # 0.60-0.85: �߶�̬
    # >= 0.85: ���߶�̬
}

# ============ ����ɸѡ��ֵ ============
QUALITY_FILTER_THRESHOLDS = {
    'low_dynamic_in_dynamic_scene': 0.3,   # ��̬�����еĵͶ�̬��ֵ
    'high_anomaly_in_static_scene': 0.5,   # ��̬�����еĸ��쳣��ֵ
}

# ============ ����������� ============
CAMERA_COMPENSATION = {
    'enable': True,                    # �Ƿ������������
    'method': 'homography',            # ��������
    'ransac_thresh': 1.0,              # RANSAC��ֵ�����أ�
    'max_features': 2000,              # �����������
    'good_match_ratio': 0.7,           # ��ƥ�����
}

# ============ ���ӻ����� ============
VISUALIZATION = {
    'enable': True,                    # �Ƿ����ɿ��ӻ�
    'dpi': 300,                        # ͼ��ֱ���
    'key_frame_count': 5,              # �ؼ�֡����
}

# ============ ��Ƶ�������� ============
VIDEO_PROCESSING = {
    'max_frames': None,                # �����֡����None��ʾȫ����
    'frame_skip': 1,                   # ֡��Ծ���
    'device': 'cuda',                  # �����豸
    'raft_model_path': 'pretrained_models/raft-things.pth',  # RAFTģ��·��
}

# ============ ��̬������ӳ����ο��� ============
SCORE_MAPPING_GUIDE = {
    'static_scene': {
        0.00: '������ֹ',
        0.10: '���Ͳв�',
        0.25: '�Ͳв�',
        0.35: '�еȲв�',
        0.55: '�ϸ߲в�',
        0.75: '�߲в�',
        1.00: '���߲в�/�쳣'
    },
    'dynamic_scene': {
        0.10: '���弸������',
        0.20: '������΢����',
        0.35: '������������',
        0.50: '�����Ծ����',
        0.65: '������ٶ���',
        0.85: '������Ҷ���',
        1.00: '���弫�Ⱦ���'
    }
}

# ============ ����Ӧ�ó������� ============
PRESET_CONFIGS = {
    # �ϸ�ģʽ��������Ҫ���
    'strict': {
        'static_threshold': 0.0015,
        'subject_threshold': 0.004,
        'low_dynamic_in_dynamic_scene': 0.35,
        'high_anomaly_in_static_scene': 0.4,
    },
    
    # ����ģʽ�����ܸ�����Ƶ
    'lenient': {
        'static_threshold': 0.003,
        'subject_threshold': 0.008,
        'low_dynamic_in_dynamic_scene': 0.2,
        'high_anomaly_in_static_scene': 0.6,
    },
    
    # ƽ��ģʽ��Ĭ�ϣ�
    'balanced': {
        'static_threshold': 0.002,
        'subject_threshold': 0.005,
        'low_dynamic_in_dynamic_scene': 0.3,
        'high_anomaly_in_static_scene': 0.5,
    }
}


def get_config(preset='balanced'):
    """
    ��ȡ����
    
    Args:
        preset: Ԥ��ģʽ ('strict', 'lenient', 'balanced')
        
    Returns:
        ���������ֵ�
    """
    
    # ��������
    config = {
        'flow': FLOW_CONFIG.copy(),
        'detection': DETECTION_THRESHOLDS.copy(),
        'scene_classification': SCENE_CLASSIFICATION.copy(),
        'dynamics_classification': DYNAMICS_CLASSIFICATION.copy(),
        'quality_filter': QUALITY_FILTER_THRESHOLDS.copy(),
        'camera_compensation': CAMERA_COMPENSATION.copy(),
        'visualization': VISUALIZATION.copy(),
        'video_processing': VIDEO_PROCESSING.copy(),
    }
    
    # Ӧ��Ԥ��
    if preset in PRESET_CONFIGS:
        preset_config = PRESET_CONFIGS[preset]
        config['detection'].update({
            k: v for k, v in preset_config.items() 
            if k in ['static_threshold', 'subject_threshold']
        })
        config['quality_filter'].update({
            k: v for k, v in preset_config.items()
            if k in ['low_dynamic_in_dynamic_scene', 'high_anomaly_in_static_scene']
        })
    
    return config


def print_config_guide():
    """��ӡ����ָ��"""
    
    guide = """
�X�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�[
�U                   ��̬������ϵͳ - ����ָ��                         �U
�^�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�a

��Ԥ��ģʽ��
  1. strict   - �ϸ�ģʽ��������Ҫ��ߣ�ɸѡ������Ƶ
  2. balanced - ƽ��ģʽ��Ĭ�����ã��ʺϴ��������
  3. lenient  - ����ģʽ�����ܸ�����Ƶ��ɸѡ����

���ؼ���ֵ˵����
  ? static_threshold (0.002)
    - ��̬��������ֵ
    - ���ͣ�����������Ϊ�Ǿ�̬
    - ��ߣ�����������Ϊ�Ǿ�̬
  
  ? subject_threshold (0.005)
    - ������������ֵ
    - ���ͣ������׼�⵽��̬����
    - ��ߣ�ֻ������ԵĶ�̬����
  
  ? low_dynamic_in_dynamic_scene (0.3)
    - ɸѡ"��̬�����ж�����С"����Ƶ
    - ���ͣ����ϸ�ɸѡ����
    - ��ߣ������ɣ�ɸѡ����

��ʹ��ʾ����
  from dynamics_config import get_config
  
  # ʹ��Ĭ������
  config = get_config('balanced')
  
  # ʹ���ϸ�ģʽ
  config = get_config('strict')
"""
    
    print(guide)


if __name__ == '__main__':
    print_config_guide()
    
    # ��ӡ��ģʽ�����öԱ�
    print("\n����ģʽ���öԱȡ�")
    for preset in ['strict', 'balanced', 'lenient']:
        config = get_config(preset)
        print(f"\n{preset.upper()} ģʽ:")
        print(f"  ��̬��ֵ: {config['detection']['static_threshold']}")
        print(f"  ������ֵ: {config['detection']['subject_threshold']}")
        print(f"  �Ͷ�̬ɸѡ: {config['quality_filter']['low_dynamic_in_dynamic_scene']}")
        print(f"  ���쳣ɸѡ: {config['quality_filter']['high_anomaly_in_static_scene']}")

