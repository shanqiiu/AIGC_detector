# -*- coding: utf-8 -*-
"""
动态度评估系统配置文件
包含所有可调参数和阈值
"""

# ============ 光流计算配置 ============
FLOW_CONFIG = {
    'use_normalized_flow': True,      # 是否使用分辨率归一化
    'baseline_diagonal': 1469.0,      # 基准分辨率对角线长度 (1280x720)
}

# ============ 静态/动态区域检测阈值 ============
DETECTION_THRESHOLDS = {
    'static_threshold': 0.002,        # 静态区域检测阈值（归一化）
    'subject_threshold': 0.005,       # 主体区域检测阈值（归一化）
    'min_subject_ratio': 0.05,        # 最小主体占比
}

# ============ 场景类型判断阈值 ============
SCENE_CLASSIFICATION = {
    'dynamic_ratio_threshold': 0.15,  # 动态区域占比阈值
    'dynamic_score_threshold': 0.01,  # 动态分数阈值
    'static_score_threshold': 0.003,  # 静态分数阈值
    'motion_coverage_threshold': 0.2, # 运动覆盖率阈值
    'scene_auto_detect': True,        # 是否自动检测场景类型
}

# ============ 动态度分类阈值 ============
DYNAMICS_CLASSIFICATION = {
    'pure_static': 0.15,      # < 0.15: 纯静态
    'low_dynamic': 0.35,      # 0.15-0.35: 低动态
    'medium_dynamic': 0.60,   # 0.35-0.60: 中等动态
    'high_dynamic': 0.85,     # 0.60-0.85: 高动态
    # >= 0.85: 极高动态
}

# ============ 质量筛选阈值 ============
QUALITY_FILTER_THRESHOLDS = {
    'low_dynamic_in_dynamic_scene': 0.3,   # 动态场景中的低动态阈值
    'high_anomaly_in_static_scene': 0.5,   # 静态场景中的高异常阈值
}

# ============ 相机补偿参数 ============
CAMERA_COMPENSATION = {
    'enable': True,                    # 是否启用相机补偿
    'method': 'homography',            # 补偿方法
    'ransac_thresh': 1.0,              # RANSAC阈值（像素）
    'max_features': 2000,              # 最大特征点数
    'good_match_ratio': 0.7,           # 好匹配比例
}

# ============ 可视化配置 ============
VISUALIZATION = {
    'enable': True,                    # 是否生成可视化
    'dpi': 300,                        # 图像分辨率
    'key_frame_count': 5,              # 关键帧数量
}

# ============ 视频处理配置 ============
VIDEO_PROCESSING = {
    'max_frames': None,                # 最大处理帧数（None表示全部）
    'frame_skip': 1,                   # 帧跳跃间隔
    'device': 'cuda',                  # 计算设备
    'raft_model_path': 'pretrained_models/raft-things.pth',  # RAFT模型路径
}

# ============ 动态度评分映射表（参考） ============
SCORE_MAPPING_GUIDE = {
    'static_scene': {
        0.00: '完美静止',
        0.10: '极低残差',
        0.25: '低残差',
        0.35: '中等残差',
        0.55: '较高残差',
        0.75: '高残差',
        1.00: '极高残差/异常'
    },
    'dynamic_scene': {
        0.10: '主体几乎不动',
        0.20: '主体轻微动作',
        0.35: '主体正常动作',
        0.50: '主体活跃动作',
        0.65: '主体快速动作',
        0.85: '主体剧烈动作',
        1.00: '主体极度剧烈'
    }
}

# ============ 典型应用场景配置 ============
PRESET_CONFIGS = {
    # 严格模式：对质量要求高
    'strict': {
        'static_threshold': 0.0015,
        'subject_threshold': 0.004,
        'low_dynamic_in_dynamic_scene': 0.35,
        'high_anomaly_in_static_scene': 0.4,
    },
    
    # 宽松模式：接受更多视频
    'lenient': {
        'static_threshold': 0.003,
        'subject_threshold': 0.008,
        'low_dynamic_in_dynamic_scene': 0.2,
        'high_anomaly_in_static_scene': 0.6,
    },
    
    # 平衡模式（默认）
    'balanced': {
        'static_threshold': 0.002,
        'subject_threshold': 0.005,
        'low_dynamic_in_dynamic_scene': 0.3,
        'high_anomaly_in_static_scene': 0.5,
    }
}


def get_config(preset='balanced'):
    """
    获取配置
    
    Args:
        preset: 预设模式 ('strict', 'lenient', 'balanced')
        
    Returns:
        完整配置字典
    """
    
    # 基础配置
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
    
    # 应用预设
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
    """打印配置指南"""
    
    guide = """
╔════════════════════════════════════════════════════════════════════╗
║                   动态度评估系统 - 配置指南                         ║
╚════════════════════════════════════════════════════════════════════╝

【预设模式】
  1. strict   - 严格模式：对质量要求高，筛选更多视频
  2. balanced - 平衡模式：默认配置，适合大多数场景
  3. lenient  - 宽松模式：接受更多视频，筛选较少

【关键阈值说明】
  ? static_threshold (0.002)
    - 静态区域检测阈值
    - 降低：更多区域被认为是静态
    - 提高：更少区域被认为是静态
  
  ? subject_threshold (0.005)
    - 主体区域检测阈值
    - 降低：更容易检测到动态主体
    - 提高：只检测明显的动态主体
  
  ? low_dynamic_in_dynamic_scene (0.3)
    - 筛选"动态场景中动作过小"的视频
    - 降低：更严格，筛选更多
    - 提高：更宽松，筛选更少

【使用示例】
  from dynamics_config import get_config
  
  # 使用默认配置
  config = get_config('balanced')
  
  # 使用严格模式
  config = get_config('strict')
"""
    
    print(guide)


if __name__ == '__main__':
    print_config_guide()
    
    # 打印各模式的配置对比
    print("\n【各模式配置对比】")
    for preset in ['strict', 'balanced', 'lenient']:
        config = get_config(preset)
        print(f"\n{preset.upper()} 模式:")
        print(f"  静态阈值: {config['detection']['static_threshold']}")
        print(f"  主体阈值: {config['detection']['subject_threshold']}")
        print(f"  低动态筛选: {config['quality_filter']['low_dynamic_in_dynamic_scene']}")
        print(f"  高异常筛选: {config['quality_filter']['high_anomaly_in_static_scene']}")

